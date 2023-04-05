use hashbrown::{HashMap, HashSet};
use kgdata::models::Entity;

use crate::{
    context::AlgoContext,
    error::GramsError,
    index::EntityTraversal,
    literal_matchers::{parsed_text_repr::ParsedTextRepr, LiteralMatcher},
    table::LinkedTable,
};

pub struct MatchedQualifier {
    pub qualifier: String,
    pub qualifier_index: usize,
    pub matched_score: f64,
}

pub struct MatchedStatement {
    pub property: String,
    pub statement_index: usize,
    pub property_matched_score: Option<f64>,
    pub qualifier_matched_scores: Vec<MatchedQualifier>,
}

/// The relationships of an entity that are matched with other values in the table
pub struct MatchedEntRel {
    /// the source entity of which contains found relationships
    pub source_entity_id: String,
    /// the statements that are matched
    pub statements: Vec<MatchedStatement>,
}

/// Potential relationships between two nodes
pub struct PotentialRelationships {
    /// source node id (either cell or an entity), this is not entity id
    pub source_id: usize,
    /// target node id (either cell or an entity), this is not entity id
    pub target_id: usize,
    /// The potential relationships between the two nodes
    pub rels: Vec<MatchedEntRel>,
}

pub struct CellNode {
    pub row: usize,
    pub col: usize,
}

pub struct EntityNode {
    pub entity_id: String,
}

pub enum Node {
    Cell(CellNode),
    Entity(EntityNode),
}

/// A struct for performing data matching within the table to discover potential relationships between cells/context entities
pub struct DataMatching<'t, ET: EntityTraversal> {
    /// The context object that contains the data needed for the algorithm to run for a table
    context: &'t AlgoContext,

    /// A helper to quickly find links between entities
    entity_traversal: &'t mut ET,

    /// The literal matcher to use
    literal_matcher: &'t LiteralMatcher,

    /// The columns to ignore
    ignored_columns: &'t Vec<usize>,

    /// The properties to ignore and not used during **literal** matching
    ignored_props: &'t HashSet<String>,

    /// whether we try to discover relationships between the same entity in same row but different columns
    allow_same_ent_search: bool,

    /// whether we use context entities in the search
    use_context: bool,
}

impl<'t, ET: EntityTraversal> DataMatching<'t, ET> {
    /// Perform data matching within the table to discover potential relationships between items
    ///
    /// # Arguments
    ///
    /// * `table` - The table to perform data matching on
    /// * `table_cells` - The parsed text of the table cells
    /// * `context` - The context object that contains the data needed for the algorithm to run for each table
    /// * `entity_traversal` - A helper to quickly find links between entities
    /// * `ignored_columns` - The columns to ignore
    /// * `ignored_props` - The properties to ignore and not used during **literal** matching
    /// * `allow_same_ent_search` - whether we try to discover relationships between the same entity in same row but different columns
    /// * `use_context` - whether we use context entities in the search
    pub fn match_data(
        table: &LinkedTable,
        table_cells: &Vec<Vec<ParsedTextRepr>>,
        context: &mut AlgoContext,
        entity_traversal: &mut ET,
        literal_matcher: &LiteralMatcher,
        ignored_columns: &Vec<usize>,
        ignored_props: &HashSet<String>,
        allow_same_ent_search: bool,
        use_context: bool,
    ) -> Result<(), GramsError> {
        let (nrows, ncols) = table.shape();
        let mut nodes = Vec::with_capacity(nrows * ncols);
        let mut edges = vec![];

        let search_columns = (0..ncols)
            .filter(|ci| ignored_columns.contains(ci))
            .collect::<Vec<_>>();

        for ri in 0..nrows {
            for ci in 0..ncols {
                nodes.push(Node::Cell(CellNode { row: ri, col: ci }));
            }
        }
        let mut context_entities = Vec::new();
        if use_context && table.context.page_entities.len() > 0 {
            for ent in &table.context.page_entities {
                nodes.push(Node::Entity(EntityNode {
                    entity_id: ent.0.clone(),
                }));
                context_entities.push((nodes.len() - 1, &context.entities[&ent.0]));
            }
        }

        let cell2entities = table
            .links
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        cell.iter()
                            .flat_map(|link| {
                                link.candidates.iter().map(|c| &context.entities[&c.id.0])
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut data_matching = DataMatching {
            context,
            entity_traversal,
            literal_matcher,
            ignored_columns,
            ignored_props,
            allow_same_ent_search,
            use_context,
        };

        for &ci in &search_columns {
            let target_search_columns = search_columns
                .clone()
                .into_iter()
                .filter(|&cj| ci != cj)
                .collect::<Vec<_>>();

            for ri in 0..nrows {
                let source_id = ri * ncols + ci;
                let source_cell = &table_cells[ri][ci];
                let source_entities = &cell2entities[ri][ci];

                for &cj in &target_search_columns {
                    let target_id = ri * ncols + cj;
                    let target_cell = &table_cells[ri][cj];
                    let target_entities = &cell2entities[ri][cj];

                    data_matching.match_pair(
                        source_id,
                        source_entities,
                        Some(source_cell),
                        target_id,
                        target_entities,
                        Some(&target_cell),
                        true,
                        &mut edges,
                    )?;
                }

                for (target_id, target_ent) in &context_entities {
                    data_matching.match_pair(
                        source_id,
                        source_entities,
                        Some(&source_cell),
                        *target_id,
                        &[target_ent],
                        None,
                        true,
                        &mut edges,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Search for any connections between two objects (texts, cells, or entities). This function is a re-implementation of `kg_path_discovery` in Python.
    fn match_pair(
        &mut self,
        source_id: usize,
        source_entities: &[&Entity],
        source_cell: Option<&ParsedTextRepr>,
        target_id: usize,
        target_entities: &[&Entity],
        target_cell: Option<&ParsedTextRepr>,
        bidirection: bool,
        output: &mut Vec<PotentialRelationships>,
    ) -> Result<(), GramsError> {
        let mut rels = Vec::new();
        for source_entity in source_entities {
            for target_ent in target_entities {
                if !self.allow_same_ent_search && source_entity.id == target_ent.id {
                    continue;
                }
                self.match_entity_pair(source_entity, target_ent, &mut rels);
            }
            if let Some(target_cell) = target_cell {
                self.match_entity_text_pair(source_entity, target_cell, &mut rels)?;
            }
        }

        if rels.len() > 0 {
            output.push(
                (PotentialRelationships {
                    source_id,
                    target_id,
                    rels,
                })
                .dedup_rels(),
            );
        }
        if bidirection {
            self.match_pair(
                target_id,
                target_entities,
                target_cell,
                source_id,
                source_entities,
                source_cell,
                false,
                output,
            )?;
        }

        Ok(())
    }

    /// Search for any connections between two entities. This function is a re-implementation of `_path_object_search_v2` in Python.
    #[inline(always)]
    fn match_entity_pair(
        &mut self,
        source: &Entity,
        target: &Entity,
        output: &mut Vec<MatchedEntRel>,
    ) {
        let statements = self
            .entity_traversal
            .iter_props_by_entity(&source.id, &target.id)
            .map(|ms| MatchedStatement {
                property: ms.property.clone(),
                statement_index: ms.statement_index,
                property_matched_score: if ms.is_property_matched {
                    Some(1.0)
                } else {
                    None
                },
                qualifier_matched_scores: ms
                    .qualifiers
                    .iter()
                    .map(|mq| MatchedQualifier {
                        qualifier: mq.qualifier.clone(),
                        qualifier_index: mq.qualifier_index,
                        matched_score: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();

        if statements.len() == 0 {
            return;
        }

        output.push(MatchedEntRel {
            source_entity_id: source.id.clone(),
            statements,
        });
    }

    /// Search for any connections between an entity and a text. This function is a re-implementation of `_path_object_search_v2` in Python.
    #[inline(always)]
    fn match_entity_text_pair(
        &mut self,
        source_entity: &Entity,
        target_cell: &ParsedTextRepr,
        output: &mut Vec<MatchedEntRel>,
    ) -> Result<(), GramsError> {
        let mut matched_statements = Vec::new();
        for (p, stmts) in source_entity.props.iter() {
            if self.ignored_props.contains(p) {
                continue;
            }

            for (stmt_i, stmt) in stmts.iter().enumerate() {
                let mut property_matched_score: Option<f64> = None;
                let mut qualifier_matched_scores = vec![];

                if let Some((_matched_fn, confidence)) =
                    self.literal_matcher
                        .compare(target_cell, &stmt.value, self.context)?
                {
                    property_matched_score = Some(confidence);
                }

                for (q, qvals) in stmt.qualifiers.iter() {
                    for (qual_i, qval) in qvals.iter().enumerate() {
                        if let Some((_matched_fn, confidence)) =
                            self.literal_matcher
                                .compare(target_cell, qval, self.context)?
                        {
                            qualifier_matched_scores.push(MatchedQualifier {
                                qualifier: q.clone(),
                                qualifier_index: qual_i,
                                matched_score: confidence,
                            });
                        }
                    }
                }

                matched_statements.push(MatchedStatement {
                    property: p.clone(),
                    statement_index: stmt_i,
                    property_matched_score,
                    qualifier_matched_scores,
                });
            }
        }

        if matched_statements.len() > 0 {
            output.push(MatchedEntRel {
                source_entity_id: source_entity.id.clone(),
                statements: matched_statements,
            });
        }

        Ok(())
    }
}

impl PotentialRelationships {
    fn dedup_rels(mut self) -> Self {
        let mut dedup_rels = Vec::with_capacity(self.rels.len());
        let mut id2rels = HashMap::with_capacity(self.rels.len());
        for rel in self.rels {
            if !id2rels.contains_key(&rel.source_entity_id) {
                let source_entity_id = rel.source_entity_id.clone();
                dedup_rels.push(rel);
                id2rels.insert(source_entity_id, dedup_rels.len() - 1);
            } else {
                dedup_rels[id2rels[&rel.source_entity_id]]
                    .statements
                    .extend(rel.statements);
            }
        }
        self.rels = dedup_rels;
        self
    }
}

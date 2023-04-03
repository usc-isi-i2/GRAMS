use kgdata::models::Entity;

use crate::{
    index::{EntityTraversal, MatchedStatement},
    table::LinkedTable,
};

pub struct MatchEdge {
    source: usize,
    target: usize,
    matched_entity: String,
    matched_statement: MatchedStatement,
}

struct CellNode {
    row: usize,
    col: usize,
}

struct EntityNode {
    id: String,
}

pub enum Node {
    Cell(CellNode),
    Entity(EntityNode),
}

///
/// Perform data matching on the table
///
/// # Arguments
///
/// * `table` - The table to perform data matching on
/// * `ignore_columns` - The columns to ignore
/// * `allow_same_ent_search` - whether we try to discover relationships between the same entity in same row but different columns
pub fn match_data(table: &LinkedTable, ignore_columns: Vec<usize>, allow_same_ent_search: bool) {
    let (nrows, ncols) = table.shape();
    let mut nodes = Vec::with_capacity(nrows * ncols);
    let mut edges = vec![];

    let search_columns = (0..ncols)
        .filter(|ci| ignore_columns.contains(ci))
        .collect::<Vec<_>>();

    let id2nodes = HashMap::new();
    for ri in 0..nrows {
        for ci in 0..ncols {
            nodes.push(Node::Cell(CellNode { row: ri, col: ci }))
            // id2nodes.insert()
        }
    }

    for ci in search_columns {
        for cj in search_columns {
            if ci == cj {
                continue;
            }

            for ri in 0..nrows {
                kg_path_discovering(
                    table,
                    source_id,
                    source_entities,
                    source_cell,
                    target_id,
                    target_entities,
                    target_cell,
                    entity_traversal,
                    allow_same_ent_search,
                    true,
                    edges,
                )
            }
        }
    }
}

pub fn kg_path_discovering<ET: EntityTraversal>(
    table: &LinkedTable,
    source_id: usize,
    source_entities: Vec<Entity>,
    source_cell: &String,
    target_id: usize,
    target_entities: Vec<Entity>,
    target_cell: &String,
    entity_traversal: &mut ET,
    allow_same_ent_search: bool,
    bidirection: bool,

    matches: &mut Vec<MatchEdge>,
) {
    for source_ent in &source_entities {
        for target_ent in &target_entities {
            if !allow_same_ent_search && source_ent.id == target_ent.id {
                continue;
            }
            matches.extend(_path_object_search_v2(
                source_id,
                target_id,
                source,
                target,
                entity_traversal,
            ));
        }
        matches.extend(_path_value_search(source_ent, source_cell));
    }

    if bidirection {
        kg_path_discovering(
            table,
            target_id,
            target_entities,
            target_cell,
            source_id,
            source_entities,
            source_cell,
            entity_traversal,
            allow_same_ent_search,
            false,
            matches,
        );
    }

    return matches;
}

/// Discover paths between two entities
#[inline(always)]
pub fn _path_object_search_v2<ET: EntityTraversal>(
    source_id: usize,
    target_id: usize,
    source: &Entity,
    target: &Entity,
    entity_traversal: &mut ET,
) -> Vec<MatchEdge> {
    let mut matches = vec![];

    for matched_statement in entity_traversal.iter_props_by_entity(&source.id, &target.id) {
        matches.push(MatchEdge {
            source: source_id,
            target: target_id,
            matched_entity: source.id.clone(),
            matched_statement: matched_statement.clone(),
        })
    }

    matches
}

#[inline(always)]
pub fn _path_value_search(source_id: usize, target_id: usize) {}

use hashbrown::HashSet;
use kgdata::models::Value;
use pyo3::prelude::*;

use super::super::data_matching::{DataMatching, Node, PotentialRelationships};
use crate::context::PyAlgoContext;
use crate::error::into_pyerr;
use crate::index::{IndexTraversal, ObjectHop1Index};
use crate::literal_matchers::parsed_text_repr::ParsedTextRepr;
use crate::literal_matchers::PyLiteralMatcher;
use crate::steps::data_matching::{CellNode, MatchedEntRel, MatchedQualifier, MatchedStatement};
use crate::table::LinkedTable;
use kgdata::macros::unsafe_update_view_lifetime_signature;
use kgdata::{pyiter, pyview};
use postcard::{from_bytes, to_allocvec};

#[pyfunction(name = "matching")]
#[pyo3(signature = (table, table_cells, context, literal_matcher, ignored_columns, ignored_props, allow_same_ent_search = false, allow_ent_matching = true, use_context = true))]
pub fn matching(
    table: &LinkedTable,
    table_cells: Vec<Vec<ParsedTextRepr>>,
    context: &mut PyAlgoContext,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: Vec<usize>,
    ignored_props: Vec<String>,
    allow_same_ent_search: bool,
    allow_ent_matching: bool,
    use_context: bool,
) -> PyResult<PyDataMatchesResult> {
    if !allow_ent_matching {
        context.0.init_object_1hop_index();
    } else {
        context.0.index_object1hop = Some(ObjectHop1Index {
            index: Default::default(),
        });
    }
    let mut traversal = IndexTraversal::from_context(&context.0);
    let (nodes, edges) = DataMatching::match_data(
        table,
        &table_cells,
        &context.0,
        &mut traversal,
        &literal_matcher.0,
        &ignored_columns,
        &HashSet::from_iter(ignored_props),
        allow_same_ent_search,
        allow_ent_matching,
        use_context,
    )
    .map_err(into_pyerr)?;
    Ok(PyDataMatchesResult { nodes, edges })
}

#[pyclass(module = "grams.core.steps.data_matching", name = "DataMatchesResult")]
#[derive(Debug)]
pub struct PyDataMatchesResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<PotentialRelationships>,
}

#[pymethods]
impl PyDataMatchesResult {
    pub fn save(&self, file: &str) -> PyResult<()> {
        let out = to_allocvec(&(&self.nodes, &self.edges)).map_err(into_pyerr)?;
        std::fs::write(file, out).map_err(into_pyerr)
    }

    #[staticmethod]
    pub fn load(file: &str) -> PyResult<PyDataMatchesResult> {
        let (nodes, edges) = from_bytes::<(Vec<Node>, Vec<PotentialRelationships>)>(
            &std::fs::read(file).map_err(into_pyerr)?,
        )
        .map_err(into_pyerr)?;
        Ok(PyDataMatchesResult { nodes, edges })
    }

    pub fn get_n_nodes(&self) -> usize {
        return self.nodes.len();
    }

    pub fn is_cell_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Cell(_) => true,
            _ => false,
        }
    }

    pub fn is_entity_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Entity(_) => true,
            _ => false,
        }
    }

    pub fn get_cell_node(&self, idx: usize) -> PyResult<CellNode> {
        match &self.nodes[idx] {
            Node::Cell(cell) => Ok(cell.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not a cell node")),
        }
    }

    pub fn get_entity_node(&self, idx: usize) -> PyResult<&String> {
        match &self.nodes[idx] {
            Node::Entity(entity) => Ok(&entity.entity_id),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not an entity node")),
        }
    }

    pub fn iter_rels(&self) -> PotentialRelationshipsVecView {
        PotentialRelationshipsVecView::new(&self.edges)
    }
}

pyview!(MatchedQualifierView(module = "grams.core.steps.data_matching", name = "MatchedQualifier", cls = MatchedQualifier) {
    r(qualifier: String),
    c(qualifier_index: usize),
    c(matched_score: f64)
});
pyview!(MatchedStatementView(module = "grams.core.steps.data_matching", name = "MatchedStatement", cls = MatchedStatement) {
    r(property: String),
    c(statement_index: usize),
    c(property_matched_score: Option<f64>),
    iter(iter_qualifier_matched_scores { qualifier_matched_scores: MatchedQualifierVecView })
});
pyview!(MatchedEntRelView(module = "grams.core.steps.data_matching", name = "MatchedEntRelView", cls = MatchedEntRel) {
    r(source_entity_id: String),
    iter(iter_statements { statements: MatchedStatementVecView })
});
pyview!(PotentialRelationshipsView(module = "grams.core.steps.data_matching", name = "PotentialRelationshipsView", cls = PotentialRelationships) {
    c(source_id: usize),
    c(target_id: usize),
    iter(iter_rels { rels: MatchedEntRelVecView })
});
pyiter!(MatchedQualifierVecView(module = "grams.core.steps.data_matching", name = "MatchedQualifierVecView") { MatchedQualifierView: MatchedQualifier });
pyiter!(MatchedStatementVecView(module = "grams.core.steps.data_matching", name = "MatchedStatementVecView") { MatchedStatementView: MatchedStatement });
pyiter!(MatchedEntRelVecView(module = "grams.core.steps.data_matching", name = "MatchedEntRelVecView") { MatchedEntRelView: MatchedEntRel });
pyiter!(PotentialRelationshipsVecView(module = "grams.core.steps.data_matching", name = "PotentialRelationshipsVecView") { PotentialRelationshipsView: PotentialRelationships });

#[pymethods]
impl MatchedEntRelView {
    pub fn get_matched_target_entities(&self, context: &PyAlgoContext) -> PyResult<Vec<String>> {
        let mut target_ent_ids = HashSet::new();
        let ent = &context.0.entities[&self.0.source_entity_id];

        for stmt in &self.0.statements {
            let kgstmt = &ent.props[&stmt.property][stmt.statement_index];
            if stmt.property_matched_score.is_some() {
                if let Value::EntityId(id) = &kgstmt.value {
                    if !target_ent_ids.contains(&id.id) {
                        target_ent_ids.insert(id.id.clone());
                    }
                }
            }

            for qual in &stmt.qualifier_matched_scores {
                if let Value::EntityId(id) =
                    &kgstmt.qualifiers[&qual.qualifier][qual.qualifier_index]
                {
                    if !target_ent_ids.contains(&id.id) {
                        target_ent_ids.insert(id.id.clone());
                    }
                }
            }
        }

        Ok(target_ent_ids.into_iter().collect::<Vec<_>>())
    }
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "data_matching")?;

    m.add_submodule(submodule)?;

    submodule.add_function(wrap_pyfunction!(matching, submodule)?)?;

    submodule.add_class::<PyDataMatchesResult>()?;
    submodule.add_class::<PotentialRelationshipsVecView>()?;
    submodule.add_class::<MatchedEntRelVecView>()?;
    submodule.add_class::<MatchedStatementVecView>()?;
    submodule.add_class::<MatchedQualifierVecView>()?;
    submodule.add_class::<PotentialRelationshipsView>()?;
    submodule.add_class::<MatchedEntRelView>()?;
    submodule.add_class::<MatchedStatementView>()?;
    submodule.add_class::<MatchedQualifierView>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.steps.data_matching", submodule)?;

    Ok(())
}

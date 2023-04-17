use hashbrown::HashSet;
use pyo3::prelude::*;

use super::super::candidate_local_search::candidate_local_search;
use super::super::data_matching::{DataMatching, Node, PotentialRelationships};
use crate::context::PyAlgoContext;
use crate::error::into_pyerr;
use crate::index::IndexTraversal;
use crate::literal_matchers::parsed_text_repr::ParsedTextRepr;
use crate::literal_matchers::PyLiteralMatcher;
use crate::macros::unsafe_update_view_lifetime_signature;
use crate::steps::data_matching::{
    CellNode, EntityNode, MatchedEntRel, MatchedQualifier, MatchedStatement,
};
use crate::table::{Link, LinkedTable};
use crate::{pyiter, pyview, strsim};
use postcard::{from_bytes, to_allocvec};

#[pyfunction(name = "matching")]
pub fn matching(
    table: &LinkedTable,
    table_cells: Vec<Vec<ParsedTextRepr>>,
    context: &mut PyAlgoContext,
    literal_matcher: &PyLiteralMatcher,
    ignored_columns: Vec<usize>,
    ignored_props: Vec<String>,
    allow_same_ent_search: bool,
    use_context: bool,
) -> PyResult<PyDataMatchesResult> {
    context.0.init_object_1hop_index();
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
        use_context,
    )
    .map_err(into_pyerr)?;
    Ok(PyDataMatchesResult { nodes, edges })
}

#[pyclass(module = "grams.core.steps.data_matching", name = "DataMatchesResult")]
pub struct PyDataMatchesResult {
    nodes: Vec<Node>,
    edges: Vec<PotentialRelationships>,
}

#[pymethods]
impl PyDataMatchesResult {
    fn save(&self, file: &str) -> PyResult<()> {
        let out = to_allocvec(&(&self.nodes, &self.edges)).map_err(into_pyerr)?;
        std::fs::write(file, out).map_err(into_pyerr)
    }

    #[staticmethod]
    fn load(file: &str) -> PyResult<PyDataMatchesResult> {
        let (nodes, edges) = from_bytes::<(Vec<Node>, Vec<PotentialRelationships>)>(
            &std::fs::read(file).map_err(into_pyerr)?,
        )
        .map_err(into_pyerr)?;
        Ok(PyDataMatchesResult { nodes, edges })
    }

    fn is_cell_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Cell(_) => true,
            _ => false,
        }
    }

    fn is_entity_node(&self, idx: usize) -> bool {
        match self.nodes[idx] {
            Node::Entity(_) => true,
            _ => false,
        }
    }

    fn get_cell_node(&self, idx: usize) -> PyResult<CellNode> {
        match &self.nodes[idx] {
            Node::Cell(cell) => Ok(cell.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not a cell node")),
        }
    }

    fn get_entity_node(&self, idx: usize) -> PyResult<&String> {
        match &self.nodes[idx] {
            Node::Entity(entity) => Ok(&entity.entity_id),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Not an entity node")),
        }
    }

    fn iter_rels(&self) -> PotentialRelationshipsVecView {
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

use hashbrown::HashSet;
use pyo3::prelude::*;

use super::candidate_local_search::candidate_local_search;
use super::data_matching::{DataMatching, Node, PotentialRelationships};
use crate::context::PyAlgoContext;
use crate::error::into_pyerr;
use crate::index::{EntityTraversal, IndexTraversal};
use crate::literal_matchers::parsed_text_repr::ParsedTextRepr;
use crate::literal_matchers::PyLiteralMatchers;
use crate::strsim;
use crate::table::{Link, LinkedTable};
use postcard::{from_bytes, to_allocvec};

#[pyclass(module = "grams.core.steps", name = "CandidateLocalSearchConfig")]
pub struct PyCandidateLocalSearchConfig {
    pub strsim: String,
    pub threshold: f64,
    pub use_column_name: bool,
    pub use_language: Option<String>,
    pub search_all_columns: bool,
}

#[pymethods]
impl PyCandidateLocalSearchConfig {
    #[new]
    pub fn new(
        strsim: String,
        threshold: f64,
        use_column_name: bool,
        use_language: Option<String>,
        search_all_columns: bool,
    ) -> Self {
        Self {
            strsim,
            threshold,
            use_column_name,
            use_language,
            search_all_columns,
        }
    }
}

#[pyfunction(name = "candidate_local_search")]
pub fn py_candidate_local_search<'t>(
    table: &LinkedTable,
    context: &'t mut PyAlgoContext,
    cfg: &PyCandidateLocalSearchConfig,
) -> PyResult<LinkedTable> {
    context.0.init_object_1hop_index();
    let mut char_tokenizer = strsim::CharacterTokenizer {};
    let mut traversal: Box<dyn EntityTraversal + 't> =
        Box::new(IndexTraversal::from_context(&context.0));

    match cfg.strsim.as_str() {
        "levenshtein" => candidate_local_search(
            table,
            &mut traversal,
            &mut strsim::SeqStrSim::new(&mut char_tokenizer, strsim::Levenshtein::default())
                .map_err(into_pyerr)?,
            cfg.threshold,
            cfg.use_column_name,
            cfg.use_language.as_ref(),
            cfg.search_all_columns,
        )
        .map_err(into_pyerr),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid strsim",
        )),
    }
}

#[pyclass(module = "grams.core.steps", name = "PyDataMatchesResult")]
pub struct PyDataMatchesResult {
    nodes: Vec<Node>,
    edges: Vec<PotentialRelationships>,
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
}

#[pyfunction(name = "data_matching")]
pub fn py_data_matching(
    table: &LinkedTable,
    table_cells: Vec<Vec<ParsedTextRepr>>,
    context: &mut PyAlgoContext,
    literal_matcher: &PyLiteralMatchers,
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

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "steps")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<PyCandidateLocalSearchConfig>()?;
    submodule.add_function(wrap_pyfunction!(py_candidate_local_search, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(py_data_matching, submodule)?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.steps", submodule)?;

    Ok(())
}

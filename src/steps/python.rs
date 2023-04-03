use pyo3::prelude::*;

use crate::context::PyAlgoContext;
use crate::error::into_pyerr;
use crate::index::{EntityTraversal, IndexTraversal};
use crate::strsim;
use crate::table::LinkedTable;

use super::candidate_local_search::candidate_local_search;

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
    let mut char_tokenizer = strsim::CharacterTokenizer {};
    let mut traversal: Box<dyn EntityTraversal + 't> =
        Box::new(IndexTraversal::from_context(&mut context.0));

    match cfg.strsim.as_str() {
        "levenshtein" => candidate_local_search(
            table,
            &mut traversal,
            &mut strsim::StrSimWithValueTokenizer::new(
                &mut char_tokenizer,
                strsim::Levenshtein::default(),
            ),
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

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "steps")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<PyCandidateLocalSearchConfig>()?;
    submodule.add_function(wrap_pyfunction!(py_candidate_local_search, submodule)?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.steps", submodule)?;

    Ok(())
}

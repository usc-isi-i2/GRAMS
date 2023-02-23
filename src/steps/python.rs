use pyo3::prelude::*;

use crate::context::PyAlgoContext;
use crate::error::into_pyerr;
use crate::index::{EntityTraversal, IndexTraversal};
use crate::strsim;
use crate::table::LinkedTable;

use super::augmenting_candidates::augment_candidates;

#[pyclass(module = "grams.core.steps", name = "AugCanConfig")]
pub struct PyAugCanConfig {
    pub strsim: String,
    pub threshold: f64,
    pub use_column_name: bool,
    pub use_language: Option<String>,
}

#[pymethods]
impl PyAugCanConfig {
    #[new]
    pub fn new(
        strsim: String,
        threshold: f64,
        use_column_name: bool,
        use_language: Option<String>,
    ) -> Self {
        Self {
            strsim,
            threshold,
            use_column_name,
            use_language,
        }
    }
}

#[pyfunction(name = "augment_candidates")]
pub fn py_augment_candidates<'t>(
    table: &LinkedTable,
    context: &'t mut PyAlgoContext,
    cfg: &PyAugCanConfig,
) -> PyResult<LinkedTable> {
    let strsim: Box<dyn strsim::StrSim> = match cfg.strsim.as_str() {
        "levenshtein" => Box::new(strsim::Levenshtein::default()),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid strsim",
            ))
        }
    };

    let mut traversal: Box<dyn EntityTraversal + 't> =
        Box::new(IndexTraversal::from_context(&mut context.0));

    augment_candidates(
        table,
        &mut traversal,
        &strsim,
        cfg.threshold,
        cfg.use_column_name,
        cfg.use_language.as_ref(),
    )
    .map_err(into_pyerr)
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "steps")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<PyAugCanConfig>()?;
    submodule.add_function(wrap_pyfunction!(py_augment_candidates, submodule)?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.steps", submodule)?;

    Ok(())
}

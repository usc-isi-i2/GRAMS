use pyo3::prelude::*;

use crate::strsim;
use crate::table::PyLinkedTable;

#[pyclass]
pub struct PyAugCanConfig {
    pub strsim: String,
    pub threshold: f64,
    pub use_column_name: bool,
}

#[pyfunction]
pub fn py_augment_candidates(table: &PyLinkedTable, cfg: &PyAugCanConfig) -> PyResult<()> {
    match &cfg.strsim {
        "levenshtein" => Box::new(strsim::Levenshtein::default()),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::ValueError, _>(
                "Invalid strsim",
            ))
        }
    }
    Ok(())
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

pub mod datagraph;
pub mod error;
pub mod index;
pub mod steps;
pub mod strsim;
pub mod table;

use pyo3::{prelude::*, types::PyList};
use table::PyLinkedTable;

#[pyfunction]
pub fn init_env_logger() -> PyResult<()> {
    env_logger::init();
    Ok(())
}

#[pymodule]
fn core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.setattr("__path__", PyList::empty(py))?;

    m.add_function(wrap_pyfunction!(init_env_logger, m)?)?;
    m.add_class::<PyLinkedTable>()?;

    datagraph::python::register(py, m)?;
    steps::python::register(py, m)?;

    Ok(())
}

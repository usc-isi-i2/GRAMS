pub mod context;
pub mod datagraph;
pub mod db;
pub mod error;
pub mod helper;
pub mod index;
pub mod literal_matchers;
pub mod steps;
pub mod strsim;
pub mod table;

use db::GramsDB;
use pyo3::{prelude::*, types::PyList};

/// Initialize env logger so we can control Rust logging from outside.
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
    m.add_class::<GramsDB>()?;

    table::register(py, m)?;
    datagraph::python::register(py, m)?;
    steps::python::register(py, m)?;

    Ok(())
}

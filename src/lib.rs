pub mod datagraph;
pub mod error;

use pyo3::{prelude::*, types::PyList};

#[pyfunction]
pub fn init_env_logger() -> PyResult<()> {
    env_logger::init();
    Ok(())
}

#[pymodule]
fn kgdata(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.setattr("__path__", PyList::empty(py))?;

    m.add_function(wrap_pyfunction!(init_env_logger, m)?)?;
    datagraph::register(py, m)?;

    Ok(())
}

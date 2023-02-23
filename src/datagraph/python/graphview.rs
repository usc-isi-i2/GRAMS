use crate::datagraph::DGraph;
use crate::error::into_pyerr;
use postcard::{from_bytes, to_allocvec};
use pyo3::{prelude::*, types::PyBytes};

#[pyclass(module = "grams.core.datagraph", name = "DGraph")]
pub struct PyDGraph {
    pub graph: DGraph,
}

#[pymethods]
impl PyDGraph {
    #[staticmethod]
    pub fn from_json(data: &[u8]) -> PyResult<Self> {
        let graph = serde_json::from_slice::<DGraph>(data).map_err(into_pyerr)?;
        Ok(PyDGraph { graph })
    }

    #[staticmethod]
    pub fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let graph = from_bytes::<DGraph>(data).map_err(into_pyerr)?;
        Ok(PyDGraph { graph })
    }

    pub fn to_bytes<'s>(&self, py: Python<'s>) -> PyResult<&'s PyBytes> {
        let out = to_allocvec(&self.graph).map_err(into_pyerr)?;
        Ok(PyBytes::new(py, &out))
    }
}

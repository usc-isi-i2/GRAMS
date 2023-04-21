use crate::datagraph::node::{CellNode, ContextSpan, EntityValueNode, LiteralValueNode, Span};
use crate::datagraph::statement::StatementNode;
use crate::datagraph::{node::DGNode, DGraph};
use crate::error::into_pyerr;
use hashbrown::HashMap;
use kgdata::models::python::value::PyValue;
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

#[pyclass(module = "grams.core.datagraph", name = "DGNode")]
#[derive(Debug, Clone)]
pub struct PyDGNode(pub DGNode);

#[pymethods]
impl PyDGNode {
    #[staticmethod]
    pub fn cell(cell: CellNode) -> Self {
        Self(DGNode::Cell(cell))
    }

    #[staticmethod]
    pub fn literal(literal: LiteralValueNode) -> Self {
        Self(DGNode::LiteralValue(literal))
    }

    #[staticmethod]
    pub fn entity(entity: EntityValueNode) -> Self {
        Self(DGNode::EntityValue(entity))
    }

    #[staticmethod]
    pub fn statement(statement: StatementNode) -> Self {
        Self(DGNode::Statement(statement))
    }
}

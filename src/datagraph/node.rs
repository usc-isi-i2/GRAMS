use super::statement::StatementNode;
use hashbrown::HashMap;
use kgdata::models::{python::value::PyValue, Value};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "grams.core.datagraph", name = "Span")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[pymethods]
impl Span {
    #[new]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[pyclass(module = "grams.core.datagraph", name = "ContextSpan")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ContextSpan {
    pub text: String,
    pub span: Span,
}

#[pymethods]
impl ContextSpan {
    #[new]
    pub fn new(text: String, span: Span) -> Self {
        Self { text, span }
    }
}

#[pyclass(module = "grams.core.datagraph", name = "CellNode")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CellNode {
    pub id: String,
    pub value: String,
    pub column: usize,
    pub row: usize,
    pub entity_ids: Vec<String>,
    pub entity_spans: HashMap<String, Vec<Span>>,
    pub entity_probs: HashMap<String, f64>,
}

#[pymethods]
impl CellNode {
    #[new]
    pub fn new(
        id: String,
        value: String,
        column: usize,
        row: usize,
        entity_ids: Vec<String>,
        entity_spans: Vec<(String, Vec<Span>)>,
        entity_probs: Vec<(String, f64)>,
    ) -> Self {
        Self {
            id,
            value,
            column,
            row,
            entity_ids,
            entity_spans: entity_spans.into_iter().collect::<HashMap<_, _>>(),
            entity_probs: entity_probs.into_iter().collect::<HashMap<_, _>>(),
        }
    }
}

#[pyclass(module = "grams.core.datagraph", name = "LiteralValueNode")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LiteralValueNode {
    pub id: String,
    pub value: Value,
    pub context_span: Option<ContextSpan>,
}

#[pymethods]
impl LiteralValueNode {
    #[new]
    pub fn new(id: String, value: PyValue, context_span: Option<ContextSpan>) -> Self {
        Self {
            id,
            value: value.0,
            context_span,
        }
    }
}

#[pyclass(module = "grams.core.datagraph", name = "EntityValueNode")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntityValueNode {
    pub id: String,
    pub entity_id: String,
    pub entity_prob: f64,
    pub context_span: Option<ContextSpan>,
}

#[pymethods]
impl EntityValueNode {
    #[new]
    pub fn new(
        id: String,
        entity_id: String,
        entity_prob: f64,
        context_span: Option<ContextSpan>,
    ) -> Self {
        Self {
            id,
            entity_id,
            entity_prob,
            context_span,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DGNode {
    Cell(CellNode),
    LiteralValue(LiteralValueNode),
    EntityValue(EntityValueNode),
    Statement(StatementNode),
}

impl DGNode {
    pub fn id(&self) -> &str {
        match self {
            DGNode::Cell(node) => &node.id,
            DGNode::LiteralValue(node) => &node.id,
            DGNode::EntityValue(node) => &node.id,
            DGNode::Statement(node) => &node.id,
        }
    }
}

/// ID of a node in the data graph.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq, Hash, FromPyObject)]
pub struct DGNodeId(pub usize);

impl IntoPy<PyObject> for DGNodeId {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

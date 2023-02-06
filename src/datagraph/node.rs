use super::statement::StatementNode;
use hashbrown::HashMap;
use kgdata::models::python::value::PyValueView;
use kgdata::models::Value;
use pyo3::{prelude::*, types::PyString};
use std::rc::Rc;

#[derive(Clone, Debug)]
#[pyclass]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[pymethods]
impl Span {
    #[new]
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct ContextSpan {
    pub text: String,
    pub span: Span,
}

#[pyclass]
pub struct CellNode {
    pub id: Py<PyString>,
    pub value: String,
    pub column: usize,
    pub row: usize,
    pub entity_ids: Vec<String>,
    pub entity_spans: HashMap<String, Span>,
}

#[pyclass]
pub struct LiteralValueNode {
    pub id: Py<PyString>,
    pub value: Value,
    pub context_span: Option<ContextSpan>,
}

#[pyclass]
pub struct EntityValueNode {
    pub id: Py<PyString>,
    pub entity_id: String,
    pub entity_prob: f64,
    pub context_span: Option<ContextSpan>,
}

pub enum DGNode {
    Cell(CellNode),
    LiteralValue(LiteralValueNode),
    EntityValue(EntityValueNode),
    Statement(StatementNode),
}

#[pyclass]
pub struct PDGNode {
    node: DGNode,
}

#[pymethods]
impl PDGNode {
    pub fn id(&self) -> &Py<PyString> {
        match &self.node {
            DGNode::Cell(node) => &node.id,
            DGNode::LiteralValue(node) => &node.id,
            DGNode::EntityValue(node) => &node.id,
            DGNode::Statement(node) => &node.id,
        }
    }

    #[staticmethod]
    pub fn cell_node(id: Py<PyString>, value: String, column: usize, row: usize) -> Self {
        PDGNode {
            node: DGNode::Cell(CellNode {
                id,
                value,
                column,
                row,
                entity_ids: Vec::new(),
                entity_spans: HashMap::new(),
            }),
        }
    }

    #[staticmethod]
    pub fn literal_value_node(
        id: Py<PyString>,
        value: PyValueView,
        context_span: Option<ContextSpan>,
    ) -> Self {
        PDGNode {
            node: DGNode::LiteralValue(LiteralValueNode {
                id,
                value: value.value.clone(),
                context_span,
            }),
        }
    }

    #[staticmethod]
    pub fn entity_value_node(
        id: Py<PyString>,
        entity_id: String,
        entity_prob: f64,
        context_span: Option<ContextSpan>,
    ) -> Self {
        PDGNode {
            node: DGNode::EntityValue(EntityValueNode {
                id,
                entity_id,
                entity_prob,
                context_span,
            }),
        }
    }
}

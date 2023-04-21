use kgdata::models::Value;
use serde::{Deserialize, Serialize};

use crate::datagraph::node::{ContextSpan, DGNodeId};
use pyo3::prelude::*;

#[derive(FromPyObject, Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct CGNodeId(pub usize);

pub struct ColumnNode {
    pub id: CGNodeId,
    // column name
    pub label: String,
    // column index
    pub column: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LiteralNode {
    pub id: CGNodeId,
    pub value: Value,
    pub context_span: Option<ContextSpan>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntityNode {
    pub id: CGNodeId,
    pub entity_id: String,
    pub context_span: Option<ContextSpan>,
}

pub struct CGStatementNode {
    pub id: CGNodeId,
    pub flow: Vec<CGStatementFlow>,
}

pub enum CGNode {
    Column(ColumnNode),
    Entity(EntityNode),
    Literal(LiteralNode),
    Statement(CGStatementNode),
}

impl CGNode {
    pub fn get_id(&self) -> CGNodeId {
        match self {
            CGNode::Column(node) => node.id,
            CGNode::Entity(node) => node.id,
            CGNode::Literal(node) => node.id,
            CGNode::Statement(node) => node.id,
        }
    }

    pub fn is_column(&self) -> bool {
        match self {
            CGNode::Column(_) => true,
            _ => false,
        }
    }

    pub fn as_column(&self) -> Option<&ColumnNode> {
        match self {
            CGNode::Column(node) => Some(node),
            _ => None,
        }
    }

    pub fn as_statement(&self) -> Option<&CGStatementNode> {
        match self {
            CGNode::Statement(node) => Some(node),
            _ => None,
        }
    }
}

#[pyclass(module = "grams.core.cangraph", name = "CGEdgeFlowSource")]
#[derive(Clone)]
pub struct CGEdgeFlowSource {
    pub dgsource: DGNodeId,
    pub cgsource: CGNodeId,
    pub edgeid: String,
}
#[pyclass(module = "grams.core.cangraph", name = "CGEdgeFlowTarget")]
#[derive(Clone)]
pub struct CGEdgeFlowTarget {
    pub dgtarget: DGNodeId,
    pub cgtarget: CGNodeId,
    pub edgeid: String,
}

#[pyclass(module = "grams.core.cangraph", name = "CGStatementFlow")]
#[derive(Clone)]
pub struct CGStatementFlow {
    pub incoming: CGEdgeFlowSource,
    pub outgoing: CGEdgeFlowTarget,
    pub dg_stmts: Vec<String>,
}

#[pymethods]
impl CGStatementFlow {
    #[new]
    pub fn new(
        incoming: CGEdgeFlowSource,
        outgoing: CGEdgeFlowTarget,
        dg_stmts: Vec<String>,
    ) -> Self {
        CGStatementFlow {
            incoming,
            outgoing,
            dg_stmts,
        }
    }
}

#[pymethods]
impl CGEdgeFlowSource {
    #[new]
    pub fn new(dgsource: usize, cgsource: usize, edgeid: String) -> Self {
        CGEdgeFlowSource {
            dgsource: DGNodeId(dgsource),
            cgsource: CGNodeId(cgsource),
            edgeid,
        }
    }
}

#[pymethods]
impl CGEdgeFlowTarget {
    #[new]
    pub fn new(dgtarget: usize, cgtarget: usize, edgeid: String) -> Self {
        CGEdgeFlowTarget {
            dgtarget: DGNodeId(dgtarget),
            cgtarget: CGNodeId(cgtarget),
            edgeid,
        }
    }
}

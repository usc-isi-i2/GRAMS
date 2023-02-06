use super::statement::StatementNode;
use hashbrown::HashMap;
use kgdata::models::Value;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ContextSpan {
    pub text: String,
    pub span: Span,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CellNode {
    pub id: String,
    pub value: String,
    pub column: usize,
    pub row: usize,
    pub entity_ids: Vec<String>,
    pub entity_spans: HashMap<String, Span>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LiteralValueNode {
    pub id: String,
    pub value: Value,
    pub context_span: Option<ContextSpan>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntityValueNode {
    pub id: String,
    pub entity_id: String,
    pub entity_prob: f64,
    pub context_span: Option<ContextSpan>,
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

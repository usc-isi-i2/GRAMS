use super::statement::StatementNode;
use std::collections::HashMap;
use crate::kgvalue::KGValue;

pub struct Span {
    start: u64,
    end: u64,
}

pub struct ContextSpan {
    text: String,
    span: Span,
}

pub struct CellNode {
    id: String,
    value: String,
    column: u64,
    row: u64,
    entity_ids: Vec<String>,
    entity_spans: HashMap<String, Span>,
}

pub struct LiteralValueNode {
    id: String,
    value: KGValue,
    context_span: Option<ContextSpan>,
}

pub struct EntityValueNode {
    id: String,
    entity_id: String,
    entity_prob: f64,
    context_span: Option<ContextSpan>,
}

pub enum DGNode {
    Cell(CellNode),
    LiteralValue(LiteralValueNode),
    EntityValue(EntityValueNode),
    Statement(StatementNode)
}
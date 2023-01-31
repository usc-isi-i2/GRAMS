use std::collections::{HashMap, HashSet};
use crate::kgvalue::KGValue;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeFlowSource {
    source_id: String,
    edge_id: String,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeFlowTarget {
    target_id: String,
    edge_id: String,
}

pub struct StatementNode {
    id: String,
    // id of the qnode that contains the statement
    entity_id: String,
    // predicate of the statement
    predicate: String,
    // whether this statement actually exist in KG
    is_in_kg: bool,

    // recording which link in the source is connected to the target.
    forward_flow: HashMap<EdgeFlowSource, HashSet<EdgeFlowTarget>>,
    reversed_flow: HashMap<EdgeFlowTarget, HashSet<EdgeFlowSource>>,
    flow: HashMap<(EdgeFlowSource, EdgeFlowTarget), Vec<FlowProvenance>>,
}

pub struct LiteralMatchingFuncArg {
    func: String,
    value: KGValue,
}

pub enum LinkGenMethod {
    FromWikidataLink,
    FromLiteralMatchingFunc(LiteralMatchingFuncArg)
}

pub struct FlowProvenance {
    gen_method: LinkGenMethod,
    prob: f64
}

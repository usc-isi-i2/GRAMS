use kgdata::models::Value;
use pyo3::{types::PyString, Py};
use std::collections::{HashMap, HashSet};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeFlowSource {
    pub source_id: String,
    pub edge_id: String,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeFlowTarget {
    pub target_id: String,
    pub edge_id: String,
}

pub struct StatementNode {
    pub id: Py<PyString>,
    // id of the qnode that contains the statement
    pub entity_id: String,
    // predicate of the statement
    pub predicate: String,
    // whether this statement actually exist in KG
    pub is_in_kg: bool,

    // recording which link in the source is connected to the target.
    pub forward_flow: HashMap<EdgeFlowSource, HashSet<EdgeFlowTarget>>,
    pub reversed_flow: HashMap<EdgeFlowTarget, HashSet<EdgeFlowSource>>,
    pub flow: HashMap<(EdgeFlowSource, EdgeFlowTarget), Vec<FlowProvenance>>,
}

pub struct LiteralMatchingFuncArg {
    pub func: String,
    pub value: Value,
}

pub enum LinkGenMethod {
    FromWikidataLink,
    FromLiteralMatchingFunc(LiteralMatchingFuncArg),
}

pub struct FlowProvenance {
    gen_method: LinkGenMethod,
    prob: f64,
}

use kgdata::models::Value;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[pyclass(module = "grams.core.datagraph", name = "EdgeFlowSource")]
#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct EdgeFlowSource {
    pub source_id: String,
    pub predicate: String,
}

#[pyclass(module = "grams.core.datagraph", name = "EdgeFlowTarget")]
#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct EdgeFlowTarget {
    pub target_id: String,
    pub predicate: String,
}

#[pyclass(module = "grams.core.datagraph", name = "StatementNode")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StatementNode {
    pub id: String,
    // id of the entity that contains the statement
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

#[pyclass(module = "grams.core.datagraph", name = "LiteralMatchingFuncArg")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LiteralMatchingFuncArg {
    pub func: String,
    pub value: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LinkGenMethod {
    FromWikidataLink,
    FromLiteralMatchingFunc(LiteralMatchingFuncArg),
}

#[pyclass(module = "grams.core.datagraph", name = "FlowProvenance")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlowProvenance {
    pub gen_method: LinkGenMethod,
    pub prob: f64,
}

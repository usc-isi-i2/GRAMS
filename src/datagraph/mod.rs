use rustworkx_core::petgraph::{stable_graph::StableGraph, Directed};
use self::node::{DGNode};
use std::collections::HashMap;
pub mod node;
pub mod statement;

pub struct DGEdge {
    id: u64,
    source: String,
    target: String,
    predicate: String,
    is_qualifier: bool,
    is_inferred: bool,
}

pub struct DGraph {
    graph: StableGraph<DGNode, DGEdge, Directed>,
    idmap: HashMap<String, u64>
}


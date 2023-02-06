use self::node::DGNode;

use rustworkx_core::petgraph::stable_graph::{
    EdgeIndex, EdgeIndices, NodeIndex, NodeIndices, StableGraph,
};
use rustworkx_core::petgraph::{visit::EdgeRef, Directed, Direction};
use std::collections::HashMap;
pub mod node;
pub mod python;
pub mod statement;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DGEdge {
    pub id: usize,
    pub source: String,
    pub target: String,
    pub predicate: String,
    pub is_qualifier: bool,
    pub is_inferred: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DGraph {
    pub graph: StableGraph<DGNode, DGEdge, Directed>,
    pub idmap: HashMap<String, NodeIndex>,
}

impl DGraph {
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    pub fn nodes(&self) -> Vec<&DGNode> {
        self.graph
            .node_indices()
            .map(|i| self.graph.node_weight(i).unwrap())
            .collect()
    }

    pub fn iter_nodes(&self) -> DGraphNodeIterator {
        DGraphNodeIterator {
            graph: &self.graph,
            index: self.graph.node_indices(),
        }
    }

    pub fn has_node(&self, nodeid: &str) -> bool {
        self.idmap.contains_key(nodeid)
    }

    pub fn get_node(&self, nodeid: &str) -> Option<&DGNode> {
        if let Some(index) = self.idmap.get(nodeid) {
            return self.graph.node_weight(*index);
        }
        None
    }

    /// Different from our graph APIs in Python that this returns a reference to the node.
    pub fn add_node(&mut self, node: DGNode) -> &DGNode {
        let nodeid = node.id().to_owned();
        let index = self.graph.add_node(node);
        self.idmap.insert(nodeid, index);
        return self.graph.node_weight(index).unwrap();
    }

    pub fn remove_node(&mut self, nid: &str) -> Option<DGNode> {
        if let Some(index) = self.idmap.remove(nid) {
            self.graph.remove_node(index)
        } else {
            None
        }
    }

    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn edges(&self) -> Vec<&DGEdge> {
        self.graph
            .edge_indices()
            .map(|i| self.graph.edge_weight(i).unwrap())
            .collect()
    }

    pub fn iter_edges(&self) -> DGraphEdgeIterator {
        DGraphEdgeIterator {
            graph: &self.graph,
            index: self.graph.edge_indices(),
        }
    }

    pub fn has_edge(&self, eid: usize) -> bool {
        self.graph.edge_weight(EdgeIndex::new(eid)).is_some()
    }

    pub fn get_edge(&self, eid: usize) -> Option<&DGEdge> {
        self.graph.edge_weight(EdgeIndex::new(eid))
    }

    /// Different from our graph APIs in Python that this returns a reference to the node.
    pub fn add_edge(&mut self, edge: DGEdge) -> &DGEdge {
        let edgeid = self
            .graph
            .add_edge(self.idmap[&edge.source], self.idmap[&edge.target], edge);

        let edge = self.graph.edge_weight_mut(edgeid).unwrap();
        edge.id = edgeid.index();
        return edge;
    }

    pub fn remove_edge(&mut self, eid: usize) -> Option<DGEdge> {
        self.graph.remove_edge(EdgeIndex::new(eid))
    }

    pub fn remove_edge_between_nodes(
        &mut self,
        source: &str,
        target: &str,
        predicate: &str,
    ) -> Option<DGEdge> {
        if let Some(edge) = self.get_edge_between_nodes(source, target, predicate) {
            self.graph.remove_edge(EdgeIndex::new(edge.id))
        } else {
            None
        }
    }

    pub fn remove_edges_between_nodes(&mut self, source: &str, target: &str) {
        let uid = self.idmap[source];
        let vid = self.idmap[target];
        while let Some(edgeid) = self.graph.find_edge(uid, vid) {
            self.graph.remove_edge(edgeid);
        }
    }

    pub fn get_edge_between_nodes(
        &self,
        source: &str,
        target: &str,
        predicate: &str,
    ) -> Option<&DGEdge> {
        let uid = self.idmap[source];
        let vid = self.idmap[target];
        let raw_edges = self.graph.edges_directed(uid, Direction::Outgoing);

        for edgeref in raw_edges {
            let edge = edgeref.weight();
            if edgeref.target() == vid && edge.predicate == predicate {
                return Some(edge);
            }
        }
        return None;
    }
}

pub struct DGraphNodeIterator<'a> {
    graph: &'a StableGraph<DGNode, DGEdge, Directed>,
    index: NodeIndices<'a, DGNode>,
}

impl<'a> Iterator for DGraphNodeIterator<'a> {
    type Item = &'a DGNode;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index.next()?;
        Some(self.graph.node_weight(index).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }
}

pub struct DGraphEdgeIterator<'a> {
    graph: &'a StableGraph<DGNode, DGEdge, Directed>,
    index: EdgeIndices<'a, DGEdge>,
}

impl<'a> Iterator for DGraphEdgeIterator<'a> {
    type Item = &'a DGEdge;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index.next()?;
        Some(self.graph.edge_weight(index).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }
}

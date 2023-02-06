use self::node::DGNode;
use self::node::PDGNode;
use pyo3::types::PyString;
use rustworkx_core::petgraph::{stable_graph::NodeIndex, stable_graph::StableGraph, Directed};
use std::borrow::Borrow;
use std::collections::HashMap;
pub mod node;
pub mod statement;
use pyo3::prelude::*;
use pyo3::Py;

#[pyclass]
pub struct DGEdge {
    id: usize,
    source: Py<PyString>,
    target: Py<PyString>,
    predicate: Py<PyString>,
    is_qualifier: bool,
    is_inferred: bool,
}

#[pyclass]
pub struct DGraph {
    graph: StableGraph<Py<PDGNode>, Py<DGEdge>, Directed>,
    idmap: HashMap<String, NodeIndex>,
}

#[pymethods]
impl DGraph {
    fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    fn nodes(&self, py: Python) -> Vec<Py<PDGNode>> {
        self.graph
            .node_indices()
            .map(|i| self.graph.node_weight(i).unwrap().clone_ref(py))
            .collect()
    }

    fn edges(&self, py: Python) -> Vec<Py<DGEdge>> {
        self.graph
            .edge_indices()
            .map(|i| self.graph.edge_weight(i).unwrap().clone_ref(py))
            .collect()
    }

    fn add_node(&mut self, py: Python<'_>, node: Py<PDGNode>) -> PyResult<Py<PyString>> {
        let nodeid = node.borrow(py).id().clone_ref(py);
        let index = self.graph.add_node(node);
        self.idmap
            .insert(nodeid.as_ref(py).to_str()?.to_owned(), index);
        return Ok(nodeid);
    }

    // fn add_edge(&mut self, py: Python, edge: Py<DGEdge>) -> usize {
    //     let mut edge_ref = edge.borrow_mut(py);
    //     let edgeid = self.graph.add_edge(
    //         self.idmap[&edge_ref.source],
    //         self.idmap[&edge_ref.target],
    //         edge.clone_ref(py),
    //     );

    //     edge_ref.id = edgeid.index();
    //     return edgeid.index();
    // }
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "datagraph")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<DGraph>()?;
    submodule.add_class::<PDGNode>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.grams.datagraph", submodule)?;

    Ok(())
}

use crate::{
    cangraph::{CGEdge, CGNode, CGNodeId, CGStatementNode},
    context::AlgoContext,
    datagraph::{
        node::{CellNode, DGNode, DGNodeId},
        python::graphview::PyDGNode,
    },
    db::GramsDB,
    error::GramsError,
    table::LinkedTable,
};
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;

pub struct DGProxy {
    // shape of the table (number of rows, number of columns)
    shape: (usize, usize),
    // nodes in the data graph
    nodes: Vec<DGNode>,
    // mapping from kg nodes in CG to DG's kg nodes
    cg2dg: Vec<Option<DGNodeId>>,
    // mapping from ri * ncols + ci to DG's cell node id
    cells: Vec<DGNodeId>,
}

impl DGProxy {
    pub fn new(table: &LinkedTable, nodes: Vec<PyDGNode>, cg2dg: Vec<Option<DGNodeId>>) -> Self {
        let shape = table.shape();
        let nodes = nodes.into_iter().map(|n| n.0).collect::<Vec<_>>();
        let mut cells = vec![DGNodeId(0); shape.0 * shape.1];

        for (i, node) in nodes.iter().enumerate() {
            if let DGNode::Cell(cell) = &node {
                cells[cell.row * shape.1 + cell.column].0 = i;
            }
        }

        Self {
            shape,
            nodes,
            cg2dg,
            cells,
        }
    }

    #[inline]
    pub fn is_cell_node(&self, id: DGNodeId) -> bool {
        if let DGNode::Cell(_) = self.nodes[id.0] {
            true
        } else {
            false
        }
    }

    pub fn get_node(&self, id: DGNodeId) -> &DGNode {
        &self.nodes[id.0]
    }

    pub fn get_num_entities(&self, id: DGNodeId) -> usize {
        match &self.nodes[id.0] {
            DGNode::Cell(cell) => cell.entity_ids.len(),
            DGNode::EntityValue(_) => 1,
            _ => 0,
        }
    }

    pub fn dg_pair_has_possible_ent_links(
        &self,
        dgu: DGNodeId,
        dgv: DGNodeId,
        is_data_predicate: bool,
    ) -> bool {
        let is_dgu_cell = self.is_cell_node(dgu);
        let is_dgv_cell = self.is_cell_node(dgv);
        if is_dgu_cell && is_dgv_cell {
            // both are cells
            if is_data_predicate {
                // data predicate: source cell must link to some entities to have possible links
                return self.get_num_entities(dgu) > 0;
            } else {
                // object predicate: source cell and target cell must link to some entities to have possible links
                return self.get_num_entities(dgu) > 0 && self.get_num_entities(dgv) > 0;
            }
        } else if is_dgu_cell {
            // the source is cell, the target will be literal/entity value
            // we have link when source cell link to some entities, doesn't depend on type of predicate
            return self.get_num_entities(dgu) > 0;
        } else if is_dgv_cell {
            // the target is cell, the source will be literal/entity value
            if is_data_predicate {
                // data predicate: always has possibe links
                return true;
            } else {
                // object predicate: have link when the target cell link to some entities
                return self.get_num_entities(dgv) > 0;
            }
        }
        // all cells are values, always have link due to how the link is generated in the first place
        return true;
    }

    /// Get DG nodes that are involved in the given CG statement and target
    pub fn get_dg_pairs(
        &self,
        s: &CGStatementNode,
        outedge: &CGEdge,
    ) -> HashSet<(DGNodeId, DGNodeId)> {
        s.flow
            .iter()
            .filter(|flow| {
                flow.outgoing.cgtarget == outedge.target
                    && flow.outgoing.edgeid == outedge.predicate
            })
            .map(|flow| (flow.incoming.dgsource, flow.outgoing.dgtarget))
            .collect::<HashSet<_>>()
    }

    /// This function iterate through each pair of data graph nodes between two candidate graph nodes.
    ///
    /// If both cg nodes are entities, we only have one pair.
    /// If one or all of them are columns, the number of pairs will be the size of the table.
    /// Otherwise, not support iterating between nodes & statements        
    ///
    /// # Arguments
    ///
    /// * `u` - The source candidate graph node
    /// * `v` - The target candidate graph node
    ///
    /// # Returns
    ///
    /// This function returns an iterator of pairs of data graph nodes
    pub fn iter_dg_pair(
        &self,
        u: &CGNode,
        v: &CGNode,
    ) -> Box<dyn Iterator<Item = (DGNodeId, DGNodeId)> + '_> {
        let (nrows, ncols) = self.shape;

        if u.is_column() && v.is_column() {
            let uci = u.as_column().unwrap().column;
            let vci = v.as_column().unwrap().column;

            return Box::new(
                (0..nrows).map(move |i| (self.cells[i * ncols + uci], self.cells[i * ncols + vci])),
            );
        }

        if u.is_column() {
            let uci = u.as_column().unwrap().column;
            let vid = self.cg2dg[v.get_id().0].unwrap();
            return Box::new((0..nrows).map(move |i| (self.cells[i * ncols + uci], vid)));
        }

        if v.is_column() {
            let uid = self.cg2dg[u.get_id().0].unwrap();
            let vci = v.as_column().unwrap().column;

            return Box::new((0..nrows).map(move |i| (uid, self.cells[i * ncols + vci])));
        }

        Box::new(
            [(
                self.cg2dg[u.get_id().0].unwrap(),
                self.cg2dg[v.get_id().0].unwrap(),
            )]
            .into_iter(),
        )
    }
}

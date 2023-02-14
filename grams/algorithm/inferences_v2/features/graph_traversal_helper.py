from __future__ import annotations

from typing import (
    Iterable,
    Tuple,
)
from grams.algorithm.data_graph.dg_graph import DGGraph, DGNode

from grams.algorithm.data_graph import CellNode
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.inputs.linked_table import LinkedTable
from sm.misc.fn_cache import CacheMethod


class GraphTraversalHelper:
    def __init__(self, table: LinkedTable, cg: CGGraph, dg: DGGraph):
        self.nrows = table.size()
        self.cg = cg
        self.dg = dg

    @CacheMethod.cache(CacheMethod.two_object_args)
    def get_rel_dg_pairs(self, s: CGStatementNode, outedge: CGEdge):
        """Get pairs of DG nodes that constitute this relationship inedge -> s -> outedge"""
        return {
            (source_flow.dg_source_id, target_flow.dg_target_id)
            for source_flow, target_flow in s.flow
            if target_flow.sg_target_id == outedge.target
            and target_flow.edge_id == outedge.predicate
        }

    def iter_dg_pair(self, uid: str, vid: str) -> Iterable[Tuple[DGNode, DGNode]]:
        """This function iterate through each pair of data graph nodes between two candidate graph nodes.

        If both cg nodes are entities, we only have one pair.
        If one or all of them are columns, the number of pairs will be the size of the table.
        Otherwise, not support iterating between nodes & statements
        """
        u = self.cg.get_node(uid)
        v = self.cg.get_node(vid)

        if isinstance(u, CGColumnNode) and isinstance(v, CGColumnNode):
            uci = u.column
            vci = v.column
            for ri in range(self.nrows):
                ucell = self.dg.get_node(f"{ri}-{uci}")
                vcell = self.dg.get_node(f"{ri}-{vci}")
                yield ucell, vcell
        elif isinstance(u, CGColumnNode):
            assert isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
            uci = u.column
            vcell = self.dg.get_node(v.id)
            for ri in range(self.nrows):
                ucell = self.dg.get_node(f"{ri}-{uci}")
                yield ucell, vcell
        elif isinstance(v, CGColumnNode):
            assert isinstance(u, (CGEntityValueNode, CGLiteralValueNode))
            vci = v.column
            ucell = self.dg.get_node(u.id)
            for ri in range(self.nrows):
                vcell = self.dg.get_node(f"{ri}-{vci}")
                yield ucell, vcell
        else:
            assert not isinstance(u, CGColumnNode) and not isinstance(v, CGColumnNode)
            yield self.dg.get_node(u.id), self.dg.get_node(v.id)

    def dg_pair_has_possible_ent_links(
        self, dgu: DGNode, dgv: DGNode, is_data_predicate: bool
    ):
        if isinstance(dgu, CellNode) and isinstance(dgv, CellNode):
            # both are cells
            if is_data_predicate:
                # data predicate: source cell must link to some entities to have possible links
                return len(dgu.entity_ids) > 0
            else:
                # object predicate: source cell and target cell must link to some entities to have possible links
                return len(dgu.entity_ids) > 0 and len(dgv.entity_ids) > 0
        elif isinstance(dgu, CellNode):
            # the source is cell, the target will be literal/entity value
            # we have link when source cell link to some entities, doesn't depend on type of predicate
            return len(dgu.entity_ids) > 0
        elif isinstance(dgv, CellNode):
            # the target is cell, the source will be literal/entity value
            if is_data_predicate:
                # data predicate: always has possibe links
                return True
            else:
                # object predicate: have link when the target cell link to some entities
                return len(dgv.entity_ids) > 0
        else:
            # all cells are values, always have link due to how the link is generated in the first place
            return True

from __future__ import annotations

from typing import Iterable, Set, Tuple, cast

from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph import CellNode
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    DGNode,
    EdgeFlowTarget,
    EntityValueNode,
    FromInference_GenArg,
    FromLiteralMatchingFunc_GenArg,
    FromWikidataLink_GenArg,
    LinkGenMethod,
)
from grams.inputs.linked_table import LinkedTable
from sm.misc.fn_cache import CacheMethod


MAX_RANK = 100000


class GraphHelper:
    def __init__(
        self, table: LinkedTable, cg: CGGraph, dg: DGGraph, context: AlgoContext
    ):
        shp = table.shape()
        self.nrows = shp[0]
        self.ncols = shp[1]
        self.cg = cg
        self.dg = dg
        self.context = context
        self.wdentities = context.wdentities

        index2entscore = {}
        index2entrank = {}
        ncols = len(table.table.columns)
        for ri, ci, links in table.links.enumerate_flat_iter():
            ent2score = {}
            ent2rank = {}
            for link in links:
                for j, can in enumerate(link.candidates):
                    entid = str(can.entity_id)
                    ent2score[entid] = max(can.probability, ent2score.get(entid, 0.0))
                    ent2rank[entid] = min(j, ent2rank.get(entid, MAX_RANK))

            index2entscore[ri * ncols + ci] = ent2score
            index2entrank[ri * ncols + ci] = ent2rank
        self.index2entscore = index2entscore
        self.index2entrank = index2entrank

    def get_candidate_entity_score(self, ri: int, ci: int, entid: str) -> float:
        return self.index2entscore[ri * self.ncols + ci].get(entid, 0.0)

    def get_candidate_entity_rank(self, ri: int, ci: int, entid: str) -> int:
        return self.index2entrank[ri * self.ncols + ci].get(entid, MAX_RANK)

    @CacheMethod.cache(CacheMethod.single_literal_arg)
    def get_entity_value_rank(self, entid: str) -> int:
        v = self.dg.get_node(entid)
        assert isinstance(v, EntityValueNode)

        if v.is_context:
            return 0

        best_rank = MAX_RANK
        for sv_edge in self.dg.in_edges(v.id):
            s = self.dg.get_statement_node(sv_edge.source)
            for us_edge in self.dg.in_edges(sv_edge.source):
                u = self.dg.get_node(us_edge.source)
                if isinstance(u, CellNode):
                    best_rank = min(
                        best_rank,
                        self.get_candidate_entity_rank(u.row, u.column, s.qnode_id),
                    )
                # should we?
                # elif isinstance(u, EntityValueNode) and u.is_context:
                #     best_rank = ...
        return best_rank

    def does_statement_from_entity_of_type(self, sid: str, type: str):
        return type in self.get_dg_statement_source_entity_types(sid)

    @CacheMethod.cache(CacheMethod.single_literal_arg)
    def get_dg_statement_source_entity_types(self, sid: str):
        ent = self.wdentities[self.dg.get_statement_node(sid).qnode_id]
        return {stmt.value.as_entity_id_safe() for stmt in ent.props.get("P31", [])}

    @CacheMethod.cache(CacheMethod.single_literal_arg)
    def get_kg_statement(self, sid: str):
        dgs = self.dg.get_statement_node(sid)
        return self.wdentities[dgs.qnode_id].props[dgs.predicate][dgs.statement_index]

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_dg_statement_target_kgentities(
        self, sid: str, target_flow: EdgeFlowTarget
    ) -> Set[str]:
        dgs = self.dg.get_statement_node(sid)
        provs = [
            prov
            for (sflow, tflow), provs in dgs.flow.items()
            if tflow == target_flow
            for prov in provs
        ]

        ents = set()
        non_infer_provs: list[
            FromWikidataLink_GenArg | FromLiteralMatchingFunc_GenArg
        ] = [
            prov.gen_method_arg
            for prov in provs
            if prov.gen_method != LinkGenMethod.FromInference
        ]  # type: ignore
        if len(non_infer_provs) > 0:
            kgstmt = self.wdentities[dgs.qnode_id].props[dgs.predicate][
                dgs.statement_index
            ]
            is_qualifier = dgs.predicate != target_flow.edge_id
            if is_qualifier:
                quals = kgstmt.qualifiers[target_flow.edge_id]
                ents = {
                    q.as_entity_id()
                    for prov in non_infer_provs
                    if (q := quals[prov["value_index"]]).is_entity_id(q)
                }
            else:
                if kgstmt.value.is_entity_id(kgstmt.value):
                    ents = {kgstmt.value.as_entity_id()}
                else:
                    ents = set()
        for prov in provs:
            if prov.gen_method == LinkGenMethod.FromInference:
                prov_gen_method_arg: FromInference_GenArg = prov.gen_method_arg  # type: ignore
                osid, oedge, ovid = prov_gen_method_arg["from_path"][-3:]
                if (
                    osid == sid
                    and oedge == target_flow.edge_id
                    and ovid == target_flow.target_id
                ):
                    continue
                ents.update(
                    self.get_dg_statement_target_kgentities(
                        osid, EdgeFlowTarget(ovid, oedge)
                    )
                )
        return ents

    @CacheMethod.cache(CacheMethod.single_literal_arg)
    def get_entity_types(self, eid: str):
        return {
            stmt.value.as_entity_id_safe()
            for stmt in self.wdentities[eid].props.get("P31", [])
        }

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

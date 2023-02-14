from __future__ import annotations

from typing import (
    Set,
    cast,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    EdgeFlowTarget,
    FromInference_GenArg,
    FromLiteralMatchingFunc_GenArg,
    FromWikidataLink_GenArg,
    LinkGenMethod,
)

from grams.algorithm.candidate_graph.cg_graph import (
    CGGraph,
)
from grams.inputs.linked_table import LinkedTable
from sm.misc.fn_cache import CacheMethod


class GraphHelper:
    def __init__(
        self, table: LinkedTable, cg: CGGraph, dg: DGGraph, context: AlgoContext
    ):
        self.nrows = table.size()
        self.cg = cg
        self.dg = dg
        self.context = context
        self.wdentities = context.wdentities

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

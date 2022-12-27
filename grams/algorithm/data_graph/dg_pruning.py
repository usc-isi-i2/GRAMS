from collections import defaultdict
from grams.algorithm.data_graph.dg_config import DGConfigs
import networkx as nx
from typing import Dict, List, Optional, Tuple, TypedDict, Union, cast

from functools import cmp_to_key
from grams.algorithm.data_graph.dg_graph import (
    CellNode,
    DGEdge,
    DGGraph,
    DGNode,
    EdgeFlowSource,
    EdgeFlowTarget,
    EntityValueNode,
    FlowProvenance,
    LinkGenMethod,
    LiteralValueNode,
    StatementNode,
)
from graph.retworkx import digraph_all_simple_paths


class DGPruning:
    NxDGEdgeAttr = TypedDict("NxDGEdgeAttr", data=DGEdge)
    NxDGEdge = Tuple[str, str, str, NxDGEdgeAttr]

    def __init__(self, dg: DGGraph, cfg: DGConfigs):
        self.dg = dg
        self.cfg = cfg

    def prune_hidden_entities(self):
        """Prune redundant KG entities, which added to the graph via KG discovering and from the context.

        **Step 1:**
        Let:
        - n be an entity node in DG.
        - v is a node connected from n via a property: LEG2: n -> p -> s -> p' -> v, and s does not have other property/qualifier rather than p'

        We made the following heuristics:
        * If there is no other node connect to n, then n is a root node and is from the context. We should not
        prune this node, so just skip it.
        * For all another node ui in U that connects to n via the path: LEG1: ui -> pi -> s' -> pi' -> n, if there is always a better
        path LEG* between ui and v, then we can remove the path LEG2. U contains nodes in cells or context, otherwise
        ui will be an entity to entity that won't be in the final model anyway.
        Note: LEG* is better than LEG2 when it's shorter, also from wikidata link or if not, it must have better match confidence (i.e., better provenance)

        **Step 2:** (?) this is questionable -- controlled by PRUNE_SINGLE_LEAF_ENT flag
        Let n' be an entity node in DG that do not link to other nodes (v doesn't exist).
        We have the following heuristics:
        * If there is no other node connect to it, this is a standable node and should be removed
        * For any node ui that connects to n via the path: LEG1: ui -> pi -> s' -> pi' -> n. If s' doesn't have other properties/qualifiers,
        then we can remove LEG1.

        **Step 3:**
        Let n be an entity/literal node in DG (not from cells & context)
        * If there is only one entity node u that connects to n via the path: LEG1: u -> p -> s -> p' -> n, then we can remove n. If that renders the
        statement s without value, we will remove s as well.

        **Step 3:**
        Finally, if a node is standalone, we should remove it.
        """
        # step 1: prune the second leg paths
        legprime: Dict[Tuple[str, str], Optional[FlowProvenance]] = {}
        rm_legs: List[Tuple[str, EdgeFlowSource, EdgeFlowTarget]] = []
        for n in self.dg.iter_nodes():
            if not isinstance(n, EntityValueNode):
                continue

            if self.dg.in_degree(n.id) == 0:
                # no other node connect to it
                continue

            # get list of grandpa ui (only cells or values in the context), along with their paths to node n.
            grandpa = set()
            for gp in self.iter_grand_parents(n.id):
                if isinstance(gp, CellNode) or (
                    isinstance(gp, (EntityValueNode, LiteralValueNode))
                    and gp.is_context
                ):
                    grandpa.add(gp.id)

            # for _, sid, ns_eid in self.dg.out_edges(nid, keys=True):
            #     stmt: StatementNode = self.dg.nodes[sid]["data"]
            for ns_edge in self.dg.out_edges(n.id):
                stmt = self.dg.get_node(ns_edge.target)
                assert isinstance(stmt, StatementNode)
                stmt_outedges = self.dg.out_edges(stmt.id)
                if len(stmt_outedges) > 1:
                    # this stmt has other qualifiers, so it's not what we are looking for
                    continue

                for sv_outedge in stmt_outedges:
                    v = self.dg.get_node(sv_outedge.target)
                    # got leg 2, now looks for all incoming
                    leg2 = (
                        EdgeFlowSource(n.id, ns_edge.predicate),
                        EdgeFlowTarget(v.id, sv_outedge.predicate),
                    )
                    if not stmt.has_flow(*leg2):
                        continue
                    leg2_provenance = stmt.get_provenance(*leg2)

                    has_better_paths = True
                    for gpid in grandpa:
                        if (gpid, v.id) not in legprime:
                            paths = [
                                (
                                    cast(
                                        StatementNode, self.dg.get_node(path[0].target)
                                    ),
                                    EdgeFlowSource(path[0].source, path[0].predicate),
                                    EdgeFlowTarget(path[1].target, path[1].predicate),
                                )
                                for path in digraph_all_simple_paths(
                                    self.dg,
                                    gpid,
                                    v.id,
                                    cutoff=2,
                                )
                            ]
                            provs = [
                                prov
                                for s, sf, tf in paths
                                if s.has_flow(sf, tf)
                                for prov in s.get_provenance(sf, tf)
                            ]
                            if len(provs) == 0:
                                legprime[gpid, v.id] = None
                            else:
                                legprime[gpid, v.id] = max(
                                    provs,
                                    key=cmp_to_key(
                                        self.specific_pruning_provenance_cmp
                                    ),
                                )
                        best_prov = legprime[gpid, v.id]
                        if (
                            best_prov is None
                            or max(
                                self.specific_pruning_provenance_cmp(
                                    best_prov, leg2_prov
                                )
                                for leg2_prov in leg2_provenance
                            )
                            < 0
                        ):
                            # no better path
                            has_better_paths = False
                            break

                    if has_better_paths:
                        rm_legs.append((stmt.id, leg2[0], leg2[1]))

        # logger.info("#legs: {}", len(rm_legs))
        self.remove_flow(rm_legs)
        # logger.info("# 0-indegree: {}", sum(self.dg.in_degree(uid) == 0 for uid in self.dg.nodes))
        # logger.info("# 0-outdegree: {}", sum(self.dg.out_degree(uid) == 0 for uid in self.dg.nodes))
        # logger.info("# 0-standalone: {}",
        #             sum(self.dg.out_degree(uid) + self.dg.in_degree(uid) == 0 for uid in self.dg.nodes))

        # step 2: prune the first leg paths (temporary disable)
        if self.cfg.PRUNE_SINGLE_LEAF_ENT:
            rm_legs: List[Tuple[str, EdgeFlowSource, EdgeFlowTarget]] = []
            for n in self.dg.iter_nodes():
                if not isinstance(n, EntityValueNode) or self.dg.out_degree(n.id) > 0:
                    continue

                for sn_edge in self.dg.in_edges(n.id):
                    if self.dg.out_degree(sn_edge.source) == 1:
                        # stmt does not have other property/qualifier
                        target_flow = EdgeFlowTarget(n.id, sn_edge.predicate)
                        stmt = self.dg.get_node(sn_edge.source)
                        assert isinstance(stmt, StatementNode)
                        for source_flow, _ in stmt.iter_source_flow(target_flow):
                            rm_legs.append((sn_edge.source, source_flow, target_flow))
            self.remove_flow(rm_legs)

        rm_legs: List[Tuple[str, EdgeFlowSource, EdgeFlowTarget]] = []
        # step 3
        for n in self.dg.iter_nodes():
            if (
                not isinstance(n, (EntityValueNode, LiteralValueNode))
                or self.dg.out_degree(n.id) > 0
                or n.is_context
            ):
                continue
            # for sid, _, sn_eid, sn_edata in self.dg.in_edges(nid, data=True, keys=True):
            for sn_edge in self.dg.in_edges(n.id):
                # for uid, _, us_eid in self.dg.in_edges(sn_edge.source):
                for us_edge in self.dg.in_edges(sn_edge.source):
                    if isinstance(self.dg.get_node(us_edge.source), EntityValueNode):
                        # two consecutive entity nodes, we can remove this link
                        stmt = self.dg.get_node(sn_edge.source)
                        assert isinstance(stmt, StatementNode)
                        if us_edge.predicate == sn_edge.predicate:
                            # the link we are going to remove is the statement value, so we should remove the statement
                            for source_flow, target_flow in stmt.flow:
                                rm_legs.append((stmt.id, source_flow, target_flow))
                        else:
                            target_flow = EdgeFlowTarget(n.id, sn_edge.predicate)
                            for source_flow, _ in stmt.iter_source_flow(target_flow):
                                rm_legs.append((stmt.id, source_flow, target_flow))

        # logger.info("#legs: {}", len(rm_legs))
        self.remove_flow(rm_legs)
        # logger.info("# 0-indegree: {}", sum(self.dg.in_degree(uid) == 0 for uid in self.dg.nodes))
        # logger.info("# 0-outdegree: {}", sum(self.dg.out_degree(uid) == 0 for uid in self.dg.nodes))
        # logger.info("# 0-standalone: {}",
        #             sum(self.dg.out_degree(uid) + self.dg.in_degree(uid) == 0 for uid in self.dg.nodes))
        self.prune_disconnected_nodes()

    def prune_disconnected_nodes(self):
        """This function prune out disconnected nodes that are:
        1. nodes without incoming edges and outgoing edges
        2. statement nodes with no incoming edges or no outgoing edges

        Returns
        -------
        """
        rm_nodes = set()
        for u in self.dg.iter_nodes():
            if isinstance(u, EntityValueNode):
                if self.dg.in_degree(u.id) == 0 and self.dg.out_degree(u.id) == 0:
                    rm_nodes.add(u.id)
            elif isinstance(u, StatementNode):
                if self.dg.in_degree(u.id) == 0 or self.dg.out_degree(u.id) == 0:
                    rm_nodes.add(u.id)
        for uid in rm_nodes:
            self.dg.remove_node(uid)

    def remove_flow(self, flows: List[Tuple[str, EdgeFlowSource, EdgeFlowTarget]]):
        for sid, source_flow, target_flow in flows:
            stmt = cast(StatementNode, self.dg.get_node(sid))
            stmt.untrack_flow(source_flow, target_flow)
            if not stmt.has_source_flow(source_flow):
                self.dg.remove_edge_between_nodes(
                    source_flow.source_id, sid, source_flow.edge_id
                )
            if not stmt.has_target_flow(target_flow):
                self.dg.remove_edge_between_nodes(
                    sid, target_flow.target_id, target_flow.edge_id
                )

    def specific_pruning_provenance_cmp(
        self, prov0: FlowProvenance, prov1: FlowProvenance
    ) -> int:
        # compare provenance, this function only accept
        if prov0.gen_method == LinkGenMethod.FromWikidataLink:
            # always favour from wikidata link
            return 1
        if prov1.gen_method == LinkGenMethod.FromWikidataLink:
            return -1
        # assert prov0.gen_method == prov1.gen_method and prov0.gen_method_arg == prov1.gen_method_arg
        # do not need to check if the two gen method and args are equal, as even if we select the incorrect one
        # we only truncate when the other leg worst than it
        return prov0.prob - prov1.prob  # type: ignore

    def iter_grand_parents(self, nid: str):
        for parent in self.dg.predecessors(nid):
            for grand_parent in self.dg.predecessors(parent.id):
                yield grand_parent

    # def out_edges(self, uid: str) -> Dict[str, List[DGEdge]]:
    #     label2edges = defaultdict(list)
    #     for _, vid, eid, edata in self.dg.out_edges(uid, data=True, keys=True):
    #         label2edges[eid].append(edata["data"])
    #     return label2edges

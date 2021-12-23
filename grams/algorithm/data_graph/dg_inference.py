from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Set
import networkx as nx
from kgdata.wikidata.models import QNode, DataValue, WDProperty, WDClass
from grams.algorithm.data_graph.dg_graph import (
    CellNode,
    DGEdge,
    DGNode,
    DGStatementID,
    EdgeFlowSource,
    EdgeFlowTarget,
    StatementNode,
    DGPath,
    DGPathEdge,
    EntityValueNode,
    DGPathNodeStatement,
    DGPathNodeQNode,
    DGPathNodeLiteralValue,
    DGPathExistingNode,
    FlowProvenance,
)


class KGInference:
    def __init__(
        self,
        dg: nx.MultiDiGraph,
        qnodes: Mapping[str, QNode],
        wdprops: Mapping[str, WDProperty],
    ):
        # mapping from qnode id, and property to a list of statement values (with the corresponding DG statement node if exist)
        # the reason we want to include all is that we want to know if we need to add new statement value or reuse existing value
        self.subkg: Dict[
            Tuple[str, str], List[Tuple[Optional[StatementNode], DataValue]]
        ] = {}
        self.qnodes = qnodes
        self.wdprops = wdprops
        self.dg = dg

        for sid, sdata in dg.nodes(data=True):  # type: ignore
            s: StatementNode = sdata["data"]  # type: ignore
            if not isinstance(s, StatementNode):
                continue

            dgsid = DGStatementID.parse_id(sid)
            self._set_stmt_node(s.qnode_id, dgsid.predicate, dgsid.statement_index, s)

    def infer_subproperty(self):
        """Infer new properties via sub-property of (inheritance)"""
        properties = set()
        qualifiers = set()

        for uid, udata in self.dg.nodes(data=True):  # type: ignore
            u: StatementNode = udata["data"]  # type: ignore
            if isinstance(u, StatementNode):
                continue

            for sid, us_edges in self.dg[u.id].items():
                # add all links to the list of properties
                properties.update(us_edges.keys())

                for vid, sv_edges in self.dg[sid].items():
                    qualifiers.update(
                        (v_eid for v_eid in sv_edges.keys() if v_eid not in us_edges)
                    )

        parent_props = self._build_parent_map(properties)
        parent_qualifiers = self._build_parent_map(qualifiers)

        # list of new properties and qualifiers that we will infer
        new_props = []
        new_qualifiers = []

        for sid, sdata in self.dg.nodes(data=True):  # type: ignore
            s: StatementNode = sdata["data"]  # type: ignore
            if not isinstance(s, StatementNode):
                continue

            # parents of the statement
            parents = set()
            prop = s.predicate
            for uid, _, eid, edata in self.dg.in_edges(sid, data=True, keys=True):
                parents.add(uid)
                assert prop == edata["data"].predicate

            # children that are properties of the statement
            prop_children = set()
            # children that are qualifiers of the statement
            qualifier2children = defaultdict(set)
            for _, vid, eid, edata in self.dg.out_edges(sid, data=True, keys=True):
                e: DGEdge = edata["data"]
                if e.predicate == prop:
                    prop_children.add(vid)
                else:
                    qualifier2children[e.predicate].add(vid)

            stmt_index = DGStatementID.parse_id(sid).statement_index
            if prop in parent_props:
                for parent_prop in parent_props[prop]:
                    for vid in prop_children:
                        for source_flow, flow_provenances in s.iter_source_flow(
                            EdgeFlowTarget(vid, prop)
                        ):
                            new_prop = InferredNewProp(
                                qnode_id=s.qnode_id,
                                new_prop=parent_prop,
                                value=self._get_stmt_value(
                                    s.qnode_id, prop, stmt_index
                                ),
                                source_id=source_flow.source_id,
                                target_id=vid,
                                qualifier_edges=[],
                                flow_provenances=flow_provenances,
                            )
                            new_props.append(new_prop)

            for q, children in qualifier2children.items():
                for pq in parent_qualifiers.get(q, []):
                    for vid in children:
                        for source_flow, provenance in s.iter_source_flow(
                            EdgeFlowTarget(vid, q)
                        ):
                            new_qualifiers.append(
                                InferredNewQualifier(
                                    statement_id=sid,
                                    new_qualifier=pq,
                                    source_id=source_flow.source_id,
                                    property=source_flow.edge_id,
                                    target_id=vid,
                                    flow_provenances=provenance,
                                )
                            )

        self.add_inference(new_props, new_qualifiers)
        return self

    def kg_transitive_inference(self):
        """Infer new relationship based on the transitive property: a -> b -> c => a -> c"""

        # find the list of transitive properties in the graph
        transitive_props = set()
        for uid, vid, eid, edata in self.dg.edges(data=True, keys=True):
            prop = self.wdprops[eid]
            # transitive class
            if "Q18647515" in prop.instanceof:
                transitive_props.add(eid)

        # now start from node u, we find if there is another v connect to u via a transitive property, and another p connect
        # to v with the same property, we don't need to keep the chain going as even if it's longer, we will eventually loop
        # through all item in the chain by looping through nodes in the graph

        chains = []
        for uid, udata in self.dg.nodes(data=True):
            u: DGNode = udata["data"]
            if isinstance(u, StatementNode):
                continue

            for sid, us_edges in self.dg[uid].items():
                stmt: StatementNode = self.dg.nodes[sid]["data"]
                for trans_prop in transitive_props:
                    if trans_prop not in us_edges:
                        continue
                    us_edge: DGEdge = us_edges[trans_prop]["data"]
                    for vid, sv_edges in self.dg[sid].items():
                        if trans_prop not in sv_edges:
                            continue
                        sv_edge: DGEdge = sv_edges[trans_prop]["data"]
                        if not stmt.is_same_flow(us_edge, sv_edge):
                            # don't allow infer new link across rows
                            continue

                        for s2id, vs2_edges in self.dg[vid].items():
                            if trans_prop not in vs2_edges:
                                continue
                            vs2_edge: DGEdge = vs2_edges[trans_prop]["data"]
                            stmt2: StatementNode = self.dg.nodes[s2id]["data"]
                            for v2id, s2v2_edges in self.dg[s2id].items():
                                if trans_prop not in s2v2_edges:
                                    continue
                                s2v2_edge: DGEdge = s2v2_edges[trans_prop]["data"]
                                if not stmt2.is_same_flow(vs2_edge, s2v2_edge):
                                    # don't allow infer new link across rows
                                    continue

                                # we now record the chain
                                chains.append((us_edge, sv_edge, vs2_edge, s2v2_edge))

        # make sure that there is no duplication in the chains
        assert len(chains) == len(
            {
                (
                    us_edge.source,
                    us_edge.predicate,
                    us_edge.target,
                    sv_edge.source,
                    sv_edge.predicate,
                    sv_edge.target,
                    vs2_edge.source,
                    vs2_edge.predicate,
                    vs2_edge.target,
                    s2v2_edge.source,
                    s2v2_edge.predicate,
                    s2v2_edge.target,
                )
                for us_edge, sv_edge, vs2_edge, s2v2_edge in chains
            }
        )

        # generate new property, but qualifiers cannot inherit via transitive inference
        new_props = []
        for us_edge, sv_edge, vs2_edge, s2v2_edge in chains:
            trans_prop = us_edge.predicate
            stmt: StatementNode = self.dg.nodes[us_edge.target]["data"]
            stmt2: StatementNode = self.dg.nodes[vs2_edge.target]["data"]

            prop_value = self._get_stmt_value(
                stmt2.qnode_id,
                trans_prop,
                DGStatementID.parse_id(stmt2.id).statement_index,
            )
            # prop_value = stmt2.qnode.props[trans_prop][DGStatementID.parse_id(stmt2.id).statement_index].value

            # calculating the provenance of the new transitive link. however, I don't know in case we have multiple
            # provenance since it depends on both first leg and second leg. for now, we put an assertion to handle
            # only the case where we have two legs has the same provenance, which make the new transitive link has
            # the same provenance too.
            first_leg_provenances = stmt.get_provenance_by_edge(us_edge, sv_edge)
            second_leg_provenances = stmt2.get_provenance_by_edge(vs2_edge, s2v2_edge)
            assert (
                len(first_leg_provenances) == 1
                and len(second_leg_provenances) == 1
                and first_leg_provenances[0] == second_leg_provenances[0]
            )

            provenances = first_leg_provenances
            new_props.append(
                InferredNewProp(
                    qnode_id=stmt.qnode_id,
                    new_prop=trans_prop,
                    value=prop_value,
                    source_id=us_edge.source,
                    target_id=s2v2_edge.target,
                    qualifier_edges=[],
                    flow_provenances=provenances,
                )
            )

        self.add_inference(new_props, [])
        return self

    def add_inference(
        self,
        new_props: List[InferredNewProp],
        new_qualifiers: List[InferredNewQualifier],
    ):
        """After we run inference, we got a list of new properties and new qualifiers that can update using this function"""
        new_nodes = []
        new_edges = []

        # here we enforce the constraint that there is no cross links
        # between nodes in different rows, this happen because transitive
        # inference generate links cross rows
        new_props = [
            new_prop
            for new_prop in new_props
            if DGEdge.can_link(
                self.dg.nodes[new_prop.source_id]["data"],
                self.dg.nodes[new_prop.target_id]["data"],
            )
        ]
        new_qualifiers = [
            new_qualifier
            for new_qualifier in new_qualifiers
            if DGEdge.can_link(
                self.dg.nodes[new_qualifier.source_id]["data"],
                self.dg.nodes[new_qualifier.target_id]["data"],
            )
        ]

        for new_prop in new_props:
            stmt_exist = None
            prop = new_prop.new_prop

            # search for existing statement in the KG
            self._track_property(new_prop.qnode_id, prop)
            for sprime, value in self.subkg[new_prop.qnode_id, prop]:
                if sprime is not None and new_prop.value == value:
                    stmt_exist = sprime
                    break

            # if the statement exist, re-use it
            if stmt_exist is not None:
                sprime = stmt_exist
            else:
                for stmt_index, (sprime, value) in enumerate(
                    self.subkg[new_prop.qnode_id, prop]
                ):
                    if new_prop.value == value:
                        stmt_id = DGStatementID(
                            new_prop.qnode_id, prop, stmt_index
                        ).get_id()
                        sprime = StatementNode(
                            stmt_id, new_prop.qnode_id, prop, is_in_kg=True
                        )
                        self._set_stmt_node(new_prop.qnode_id, prop, stmt_index, sprime)
                        new_nodes.append(sprime)
                        break
                else:
                    stmt_index = self.get_next_available_stmt_index(
                        new_prop.qnode_id, prop
                    )
                    stmt_id = DGStatementID(
                        new_prop.qnode_id, prop, stmt_index
                    ).get_id()
                    sprime = StatementNode(
                        stmt_id, new_prop.qnode_id, prop, is_in_kg=False
                    )
                    self._add_stmt_value(
                        new_prop.qnode_id, prop, stmt_index, sprime, new_prop.value
                    )
                new_nodes.append(sprime)

            if not self.dg.has_edge(new_prop.source_id, sprime.id, key=prop):
                new_edges.append(
                    DGEdge(
                        source=new_prop.source_id,
                        target=sprime.id,
                        predicate=prop,
                        is_qualifier=False,
                        is_inferred=True,
                    )
                )
            if not self.dg.has_edge(sprime.id, new_prop.target_id, key=prop):
                new_edges.append(
                    DGEdge(
                        source=sprime.id,
                        target=new_prop.target_id,
                        predicate=prop,
                        is_qualifier=False,
                        is_inferred=True,
                    )
                )

            sprime.track_provenance(
                EdgeFlowSource(new_prop.source_id, prop),
                EdgeFlowTarget(new_prop.target_id, prop),
                new_prop.flow_provenances,
            )

            # TODO: we haven't add to add add qualifiers, so we assert we don't have any. fix me!
            assert len(new_prop.qualifier_edges) == 0
            # for qual_edge in new_prop.qualifier_edges:
            #     new_edges.append(DGEdge(source=stmt_id, target=qual_edge.target, predicate=qual_edge.predicate, is_qualifier=True, paths=[], is_inferred=True))

        for new_qual in new_qualifiers:
            stmt: StatementNode = self.dg.nodes[new_qual.statement_id]["data"]
            new_edges.append(
                DGEdge(
                    source=new_qual.statement_id,
                    target=new_qual.target_id,
                    predicate=new_qual.new_qualifier,
                    is_qualifier=True,
                    is_inferred=True,
                )
            )
            stmt.track_provenance(
                EdgeFlowSource(new_qual.source_id, new_qual.property),
                EdgeFlowTarget(new_qual.target_id, new_qual.new_qualifier),
                new_qual.flow_provenances,
            )

        for node in new_nodes:
            self.dg.add_node(node.id, data=node)
        for edge in new_edges:
            self.dg.add_edge(edge.source, edge.target, key=edge.predicate, data=edge)

    def _track_property(self, qnode_id: str, prop: str):
        """Ensure that the subkg has values of the qnode's property"""
        if (qnode_id, prop) not in self.subkg:
            lst = []
            for stmt_i, stmt in enumerate(self.qnodes[qnode_id].props.get(prop, [])):
                lst.append((None, stmt.value))
            self.subkg[qnode_id, prop] = lst

    def _get_stmt_value(self, qnode_id: str, prop: str, stmt_index: int):
        self._track_property(qnode_id, prop)
        return self.subkg[qnode_id, prop][stmt_index][1]

    def _set_stmt_node(
        self, qnode_id: str, prop: str, stmt_index: int, stmt: StatementNode
    ):
        self._track_property(qnode_id, prop)
        assert (
            self.subkg[qnode_id, prop][stmt_index][0] is None
        ), "Cannot override existing value in the KG"
        self.subkg[qnode_id, prop][stmt_index] = (
            stmt,
            self.subkg[qnode_id, prop][stmt_index][1],
        )

    def _add_stmt_value(
        self,
        qnode_id: str,
        prop: str,
        stmt_index: int,
        stmt: StatementNode,
        value: DataValue,
    ):
        self._track_property(qnode_id, prop)
        assert stmt_index == len(self.subkg[qnode_id, prop]), "Can only add new value"
        self.subkg[qnode_id, prop].append((stmt, value))

    def get_next_available_stmt_index(self, qnode_id: str, prop: str):
        return len(self.subkg[qnode_id, prop])

    def _build_parent_map(self, props: Set[str]):
        """Build a map from a property to its parents in the same list"""
        parent_props: Dict[str, List[str]] = {}
        for p1 in props:
            parent_props[p1] = []
            for p2 in props:
                if p1 == p2:
                    continue
                if p2 in self.wdprops[p1].parents_closure:
                    parent_props[p1].append(p2)
            if len(parent_props[p1]) == 0:
                parent_props.pop(p1)
        return parent_props


@dataclass
class InferredNewProp:
    # information to identify the statement for the new prop (whether to reuse it, or create a new one)
    # the qnode of the new statement (the one will have this property)
    qnode_id: str
    # the new property we inferred
    new_prop: str
    # the value associated with the property (use it to compare if the statement exists) - the reason we use
    # the value instead of the whole statement is that sometimes, desire them to match the qualifier does not
    # sense, if the value exist, then some how the algorithm is already match it so it should be okay
    value: DataValue

    # the source nodes in the data graph
    source_id: str
    # the target node in the data graph that will contain the value of the prop, not the statement node
    target_id: str

    # edges that contain the qualifiers that we want to be copied to the new statement as well
    # the flow of the qualifier can be retrieve
    qualifier_edges: List[DGEdge]

    # flow provenance of the property.
    flow_provenances: List[FlowProvenance]


@dataclass
class InferredNewQualifier:
    # the statement node in the data graph that we will add the qualifier to
    statement_id: str
    # the qualifier we are going to add
    new_qualifier: str

    # source id (node that has the statement id)
    source_id: str
    # the property of the statement (the one that connect source and statement)
    property: str
    # the target node of the qualifier
    target_id: str

    # flow provenance of the qualifier
    flow_provenances: List[FlowProvenance]

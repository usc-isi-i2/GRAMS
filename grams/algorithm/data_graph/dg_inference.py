from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Set
from kgdata.wikidata.models import WDEntity, WDValue, WDProperty
from grams.algorithm.data_graph.dg_graph import (
    DGEdge,
    DGGraph,
    DGStatementID,
    EdgeFlowSource,
    EdgeFlowTarget,
    LinkGenMethod,
    StatementNode,
    FlowProvenance,
)


class KGInference:
    def __init__(
        self,
        dg: DGGraph,
        wdentities: Mapping[str, WDEntity],
        wdprops: Mapping[str, WDProperty],
    ):
        # mapping from qnode id, and property to a list of statement values (with the corresponding DG statement node if exist)
        # the reason we want to include all is that we want to know if we need to add new statement value or reuse existing value
        self.subkg: Dict[
            Tuple[str, str], List[Tuple[Optional[StatementNode], WDValue]]
        ] = {}
        self.wdentities = wdentities
        self.wdprops = wdprops
        self.dg = dg

        for s in dg.iter_nodes():
            if not isinstance(s, StatementNode):
                continue

            dgsid = DGStatementID.parse_id(s.id)
            self._set_stmt_node(s.qnode_id, dgsid.predicate, dgsid.statement_index, s)

    def infer_subproperty(self):
        """Infer new properties via sub-property of (inheritance)"""
        properties = set()
        qualifiers = set()

        for u in self.dg.iter_nodes():
            if isinstance(u, StatementNode):
                continue

            for s in self.dg.successors(u.id):
                assert isinstance(s, StatementNode)
                properties.add(s.predicate)

                for v in self.dg.successors(s.id):
                    sv_edges = self.dg.get_edges_between_nodes(s.id, v.id)
                    qualifiers.update(
                        (
                            sv_edge.predicate
                            for sv_edge in sv_edges
                            if sv_edge != s.predicate
                        )
                    )

        parent_props = self._build_parent_map(properties)
        parent_qualifiers = self._build_parent_map(qualifiers)

        # list of new properties and qualifiers that we will infer
        new_props = []
        new_qualifiers = []

        for s in self.dg.iter_nodes():
            if not isinstance(s, StatementNode):
                continue

            # parents of the statement
            # parents = {u.id for u in self.dg.predecessors(s.id)}
            prop = s.predicate

            # children that are properties of the statement
            prop_children = set()
            # children that are qualifiers of the statement
            qualifier2children = defaultdict(set)
            for e in self.dg.out_edges(s.id):
                if e.predicate == prop:
                    prop_children.add(e.target)
                else:
                    qualifier2children[e.predicate].add(e.target)

            stmt_index = DGStatementID.parse_id(s.id).statement_index
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
                                flow_provenances=[
                                    FlowProvenance.from_inference(
                                        method="subproperty",
                                        from_path=[
                                            source_flow.source_id,
                                            prop,
                                            s.id,
                                            prop,
                                            vid,
                                        ],
                                        prob=max(
                                            prov.prob for prov in flow_provenances
                                        ),
                                    )
                                ],
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
                                    statement_id=s.id,
                                    new_qualifier=pq,
                                    source_id=source_flow.source_id,
                                    property=source_flow.edge_id,
                                    target_id=vid,
                                    flow_provenances=[
                                        FlowProvenance.from_inference(
                                            method="subproperty",
                                            from_path=[
                                                source_flow.source_id,
                                                source_flow.edge_id,
                                                s.id,
                                                q,
                                                vid,
                                            ],
                                            prob=max(prov.prob for prov in provenance),
                                        )
                                    ],
                                )
                            )

        self.add_inference(new_props, new_qualifiers)
        return self

    def kg_transitive_inference(self):
        """Infer new relationship based on the transitive property: a -> b -> c => a -> c.

        Note that (a), (b), and (c) are entities, not cell nodes because a cell may contain multiple candidate entities.
        """

        # find the list of transitive properties in the graph
        transitive_props = set()
        for e in self.dg.edges():
            prop = self.wdprops[e.predicate]
            # transitive class
            if "Q18647515" in prop.instanceof:
                transitive_props.add(e.predicate)

        # now start from node u, we find if there is another v connect to u via a transitive property, and another p connect
        # to v with the same property, we don't need to keep the chain going as even if it's longer, we will eventually loop
        # through all item in the chain by looping through nodes in the graph

        chains = []
        for u in self.dg.iter_nodes():
            if isinstance(u, StatementNode):
                continue

            for stmt, us_edges in self.dg.group_out_edges(u.id):
                assert isinstance(stmt, StatementNode)
                for trans_prop in transitive_props:
                    if trans_prop not in us_edges:
                        continue
                    us_edge = us_edges[trans_prop]
                    for v, sv_edges in self.dg.group_out_edges(stmt.id):
                        if trans_prop not in sv_edges:
                            continue
                        sv_edge = sv_edges[trans_prop]
                        if not stmt.is_same_flow(us_edge, sv_edge):
                            # don't allow infer new link across rows
                            continue

                        for stmt2, vs2_edges in self.dg.group_out_edges(v.id):
                            assert isinstance(stmt2, StatementNode)
                            if trans_prop not in vs2_edges:
                                continue
                            vs2_edge: DGEdge = vs2_edges[trans_prop]
                            for v2, s2v2_edges in self.dg.group_out_edges(stmt2.id):
                                if trans_prop not in s2v2_edges:
                                    continue
                                s2v2_edge = s2v2_edges[trans_prop]
                                if not stmt2.is_same_flow(vs2_edge, s2v2_edge):
                                    # don't allow infer new link across rows
                                    continue

                                # we have to make sure that the target entity of stmt and source entity of stmt2 is the same
                                leg1targetentid = self._get_stmt_value(
                                    stmt.qnode_id, trans_prop, stmt.statement_index
                                ).as_entity_id_safe()
                                if leg1targetentid != stmt2.qnode_id:
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
            stmt = self.dg.get_node(us_edge.target)
            stmt2 = self.dg.get_node(vs2_edge.target)
            assert isinstance(stmt, StatementNode) and isinstance(stmt2, StatementNode)

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
            # if not (
            #     len(first_leg_provenances) == 1
            #     and len(second_leg_provenances) == 1
            #     and first_leg_provenances[0] == second_leg_provenances[0]
            # ):
            #     # I found a case where this condition does not hold, that is when
            #     # the legs are from literal matching functions, but when we have literal matching
            #     # functions, it should be literal and typically literal don't have outgoing edges.
            #     # TODO: check this logic again
            #     assert all(
            #         prov.gen_method == LinkGenMethod.FromLiteralMatchingFunc
            #         for prov in first_leg_provenances + second_leg_provenances
            #     )
            #     continue

            # the provenance of this new prop is from the path of the first leg and second leg
            # the probability is the minimum of the two legs
            prob = min(
                max(prov.prob for prov in first_leg_provenances),
                max(prov.prob for prov in second_leg_provenances),
            )
            new_props.append(
                InferredNewProp(
                    qnode_id=stmt.qnode_id,
                    new_prop=trans_prop,
                    value=prop_value,
                    source_id=us_edge.source,
                    target_id=s2v2_edge.target,
                    qualifier_edges=[],
                    flow_provenances=[
                        FlowProvenance.from_inference(
                            method="transitive",
                            from_path=[
                                us_edge.source,
                                us_edge.predicate,
                                sv_edge.source,
                                sv_edge.predicate,
                                vs2_edge.source,
                                vs2_edge.predicate,
                                s2v2_edge.source,
                                s2v2_edge.predicate,
                                s2v2_edge.target,
                            ],
                            prob=prob,
                        )
                    ],
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
                self.dg.get_node(new_prop.source_id),
                self.dg.get_node(new_prop.target_id),
            )
        ]
        new_qualifiers = [
            new_qualifier
            for new_qualifier in new_qualifiers
            if DGEdge.can_link(
                self.dg.get_node(new_qualifier.source_id),
                self.dg.get_node(new_qualifier.target_id),
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
                            stmt_id,
                            new_prop.qnode_id,
                            prop,
                            is_in_kg=True,
                            forward_flow={},
                            reversed_flow={},
                            flow={},
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
                        stmt_id,
                        new_prop.qnode_id,
                        prop,
                        is_in_kg=False,
                        forward_flow={},
                        reversed_flow={},
                        flow={},
                    )
                    self._add_stmt_value(
                        new_prop.qnode_id, prop, stmt_index, sprime, new_prop.value
                    )
                new_nodes.append(sprime)

            if not self.dg.has_edge_between_nodes(
                new_prop.source_id, sprime.id, key=prop
            ):
                new_edges.append(
                    DGEdge(
                        id=-1,  # will be assigned later
                        source=new_prop.source_id,
                        target=sprime.id,
                        predicate=prop,
                        is_qualifier=False,
                        is_inferred=True,
                    )
                )
            if not self.dg.has_edge_between_nodes(
                sprime.id, new_prop.target_id, key=prop
            ):
                new_edges.append(
                    DGEdge(
                        id=-1,  # will be assigned later
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
            stmt = self.dg.get_node(new_qual.statement_id)
            assert isinstance(stmt, StatementNode)
            new_edges.append(
                DGEdge(
                    id=-1,  # will be assigned later
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
            self.dg.add_node(node)
        for edge in new_edges:
            self.dg.add_edge(edge)

    def _track_property(self, qnode_id: str, prop: str):
        """Ensure that the subkg has values of the qnode's property"""
        if (qnode_id, prop) not in self.subkg:
            lst = []
            for stmt_i, stmt in enumerate(
                self.wdentities[qnode_id].props.get(prop, [])
            ):
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
        value: WDValue,
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
                if p2 in self.wdprops[p1].ancestors:
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
    value: WDValue

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

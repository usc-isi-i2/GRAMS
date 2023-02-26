import itertools
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Set, Tuple
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)

import numpy as np

from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDEntity, WDClass, WDProperty
from rdflib import RDFS
from sm.evaluation import sm_metrics
from sm.outputs.semantic_model import (
    ClassNode,
    DataNode,
    LiteralNodeDataType,
    SemanticModel,
    LiteralNode,
    Edge,
)
from sm.namespaces.prelude import WikidataNamespace


class SMNodeType(IntEnum):
    Column = 0
    Class = 1
    Statement = 2
    Entity = 3
    Literal = 4


class WikidataSemanticModelHelper:
    ENTITY_ID = "Q35120"
    ENTITY_LABEL = "Entity (Q35120)"
    ID_PROPS = {str(RDFS.label)}

    def __init__(
        self,
        wdentities: Mapping[str, WDEntity],
        wdentity_labels: Mapping[str, str],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
    ):
        self.wdentities = wdentities
        self.wdentity_labels = wdentity_labels
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wdns = WikidataNamespace.create()

    def norm_sm(self, sm: SemanticModel):
        """ "Normalize the semantic model with the following modifications:
        1. Add readable label to edge and class
        2. Convert direct link (without statement) to have statement except the id props.
        """
        new_sm = sm.deep_copy()
        wdns = self.wdns

        # update readable label
        for n in new_sm.iter_nodes():
            if isinstance(n, ClassNode):
                if wdns.is_abs_uri_qnode(n.abs_uri):
                    n.readable_label = self.get_qnode_label(
                        wdns.get_entity_id(n.abs_uri)
                    )
            elif isinstance(n, LiteralNode):
                if wdns.is_abs_uri_qnode(n.value):
                    n.readable_label = self.get_qnode_label(wdns.get_entity_id(n.value))
        for e in new_sm.iter_edges():
            if e.abs_uri not in self.ID_PROPS:
                e.readable_label = self.get_pnode_label(wdns.get_prop_id(e.abs_uri))

        # convert direct link
        for edge in list(new_sm.iter_edges()):
            if edge.abs_uri in self.ID_PROPS:
                continue
            source = new_sm.get_node(edge.source)
            target = new_sm.get_node(edge.target)

            if (
                not isinstance(source, ClassNode)
                or source.abs_uri != WikidataNamespace.STATEMENT_URI
            ) and (
                not isinstance(target, ClassNode)
                or target.abs_uri != WikidataNamespace.STATEMENT_URI
            ):
                # this is direct link, we replace its edge
                assert len(new_sm.get_edges_between_nodes(source.id, target.id)) == 1
                new_sm.remove_edges_between_nodes(source.id, target.id)
                stmt = ClassNode(
                    abs_uri=WikidataNamespace.STATEMENT_URI,
                    rel_uri=wdns.get_rel_uri(WikidataNamespace.STATEMENT_URI),
                )
                new_sm.add_node(stmt)

                new_sm.add_edge(
                    Edge(
                        source=edge.source,
                        target=stmt.id,
                        abs_uri=edge.abs_uri,
                        rel_uri=edge.rel_uri,
                        approximation=edge.approximation,
                        readable_label=edge.readable_label,
                    )
                )
                new_sm.add_edge(
                    Edge(
                        source=stmt.id,
                        target=edge.target,
                        abs_uri=edge.abs_uri,
                        rel_uri=edge.rel_uri,
                        approximation=edge.approximation,
                        readable_label=edge.readable_label,
                    )
                )
        return new_sm

    @staticmethod
    def minify_sm(sm: SemanticModel):
        """This is a reverse function of `norm_sm`:
        1. Remove an intermediate statement if it doesn't have any qualifiers
        """
        new_sm = sm.copy()

        for n in sm.iter_nodes():
            if (
                isinstance(n, ClassNode)
                and n.abs_uri == WikidataNamespace.STATEMENT_URI
            ):
                inedges = sm.in_edges(n.id)
                outedges = sm.out_edges(n.id)
                if len(outedges) == 1 and outedges[0].abs_uri == inedges[0].abs_uri:
                    # no qualifiers
                    new_sm.remove_node(n.id)
                    for inedge in inedges:
                        assert inedge.abs_uri == outedges[0].abs_uri
                        new_sm.add_edge(
                            Edge(
                                inedge.source,
                                outedges[0].target,
                                inedge.abs_uri,
                                inedge.rel_uri,
                                # just in case user misannotate to not include approximation in both links
                                inedge.approximation or outedges[0].approximation,
                                inedge.readable_label,
                            )
                        )
        return new_sm

    def create_sm(self, table: LinkedTable, cpa: CGGraph, cta: Dict[int, str]):
        """Create a semantic model from outputs of CPA and CTA tasks"""
        sm = SemanticModel()
        classmap = {}  # mapping from column to its class node
        wdns = self.wdns
        for cid, qnode_id in cta.items():
            dnode = DataNode(
                col_index=cid,
                label=table.table.columns[cid].name or "",
            )

            # somehow, they may end-up predict multiple classes, we need to select one
            if qnode_id.find(" ") != -1:
                qnode_id = qnode_id.split(" ")[0]
            curl = wdns.get_entity_abs_uri(qnode_id)

            try:
                cnode_label = self.get_qnode_label(qnode_id)
            except KeyError:
                cnode_label = wdns.get_entity_rel_uri(qnode_id)
            cnode = ClassNode(
                abs_uri=curl,
                rel_uri=wdns.get_entity_rel_uri(qnode_id),
                readable_label=cnode_label,
            )
            sm.add_node(dnode)
            sm.add_node(cnode)
            classmap[dnode.col_index] = cnode.id
            sm.add_edge(
                Edge(
                    source=cnode.id,
                    target=dnode.id,
                    abs_uri=str(RDFS.label),
                    rel_uri=wdns.get_rel_uri(RDFS.label),
                )
            )

        # do a final sweep to add subject columns that are not in CTA
        for unode in cpa.iter_nodes():
            if not isinstance(unode, CGColumnNode):
                continue
            outdegree: int = cpa.out_degree(unode.id)
            if outdegree > 0 and not sm.has_data_node(unode.column):
                # add data node to the graph and use the entity class (all instances belong to this class) to describe this data node
                dnode = DataNode(
                    col_index=unode.column,
                    label=table.table.columns[unode.column].name or "",
                )
                sm.add_node(dnode)

                curl = wdns.get_entity_abs_uri(self.ENTITY_ID)
                cnode_id = sm.add_node(
                    ClassNode(
                        abs_uri=curl,
                        rel_uri=wdns.get_entity_rel_uri(self.ENTITY_ID),
                        readable_label=self.ENTITY_LABEL,
                    )
                )
                classmap[dnode.col_index] = cnode_id

                sm.add_edge(
                    Edge(
                        source=cnode_id,
                        target=dnode.id,
                        abs_uri=str(RDFS.label),
                        rel_uri="rdfs:label",
                    )
                )

        # now add remaining edges and remember to use class node instead of data node
        cpa_idmap = {}
        for edge in cpa.edges():
            unode = cpa.get_node(edge.source)
            vnode = cpa.get_node(edge.target)

            if isinstance(unode, CGColumnNode):
                # outgoing edge is from a class node instead of a data node
                suid = classmap[unode.column]
                source = sm.get_node(suid)
            elif isinstance(unode, CGEntityValueNode):
                if unode.id not in cpa_idmap:
                    source = LiteralNode(
                        value=wdns.get_entity_abs_uri(unode.qnode_id),
                        readable_label=self.get_qnode_label(unode.qnode_id),
                        datatype=LiteralNodeDataType.Entity,
                        is_in_context=any(
                            unode.qnode_id == page_entity_id
                            for page_entity_id in table.context.page_entities
                        ),
                    )
                    cpa_idmap[unode.id] = sm.add_node(source)
                else:
                    source = sm.get_node(cpa_idmap[unode.id])
            else:
                assert isinstance(
                    unode, CGStatementNode
                ), "Outgoing edge can't not be from literal"
                if unode.id not in cpa_idmap:
                    # create a statement node
                    source = ClassNode(
                        abs_uri=wdns.STATEMENT_URI,
                        rel_uri=wdns.get_rel_uri(wdns.STATEMENT_URI),
                    )
                    cpa_idmap[unode.id] = sm.add_node(source)
                else:
                    source = sm.get_node(cpa_idmap[unode.id])

            if isinstance(vnode, CGColumnNode):
                if vnode.column in classmap:
                    target = sm.get_node(classmap[vnode.column])
                elif sm.has_data_node(vnode.column):
                    target = sm.get_data_node(vnode.column)
                elif vnode.id not in cpa_idmap:
                    target = DataNode(
                        col_index=vnode.column,
                        label=table.table.columns[vnode.column].name or "",
                    )
                    cpa_idmap[vnode.id] = sm.add_node(target)
                else:
                    target = sm.get_node(cpa_idmap[vnode.id])
            elif isinstance(vnode, CGEntityValueNode):
                if vnode.id not in cpa_idmap:
                    target = LiteralNode(
                        value=vnode.get_literal_node_value(wdns),
                        readable_label=self.get_qnode_label(vnode.qnode_id),
                        datatype=LiteralNodeDataType.Entity,
                        is_in_context=any(
                            vnode.qnode_id == page_eid
                            for page_eid in table.context.page_entities
                        ),
                    )
                    cpa_idmap[vnode.id] = sm.add_node(target)
                else:
                    target = sm.get_node(cpa_idmap[vnode.id])
            elif isinstance(vnode, CGLiteralValueNode):
                if vnode.id not in cpa_idmap:
                    target = LiteralNode(
                        value=vnode.get_literal_node_value(),
                        readable_label=vnode.label,
                        datatype=LiteralNodeDataType.String,
                    )
                    cpa_idmap[vnode.id] = sm.add_node(target)
                else:
                    target = sm.get_node(cpa_idmap[vnode.id])
            else:
                if vnode.id not in cpa_idmap:
                    # create a statement node
                    target = ClassNode(
                        abs_uri=WikidataNamespace.STATEMENT_URI,
                        rel_uri=wdns.get_rel_uri(WikidataNamespace.STATEMENT_URI),
                    )
                    cpa_idmap[vnode.id] = sm.add_node(target)
                else:
                    target = sm.get_node(cpa_idmap[vnode.id])

            sm.add_edge(
                Edge(
                    source=source.id,
                    target=target.id,
                    abs_uri=wdns.get_prop_abs_uri(edge.predicate),
                    rel_uri=wdns.get_prop_rel_uri(edge.predicate),
                    readable_label=self.get_pnode_label(edge.predicate),
                )
            )

        return sm

    def gen_equivalent_sm(
        self,
        sm: SemanticModel,
        strict: bool = True,
        force_inversion: bool = False,
        limited_invertible_props: Optional[Set[str]] = None,
        incorrect_invertible_props: Optional[Set[str]] = None,
    ):
        """Given a semantic model (not being modified), generate equivalent **normalized** models by inferring inverse properties.

        Currently, we only inverse the properties, not qualifiers.



        Parameters
        ----------
        sm: the input semantic model (original)
        strict: whether to throw exception when target of an inverse property is not a class.
        force_inversion: only work when strict mode is set to false. Without force_inverse, we skip inverse properties,
                       otherwise, we generate an inverse model with a special class: wikibase:DummyClassForInversion
        limited_invertible_props: if provided, only generate inverse properties for these properties.
        incorrect_invertible_props: if provided, skip generating inverse properties for these properties.
        Returns
        -------
        """
        sm = self.norm_sm(sm)
        wdns = self.wdns

        if incorrect_invertible_props is None:
            incorrect_invertible_props = set()

        invertible_stmts: List[ClassNode] = []
        is_class_fn = lambda n1: isinstance(n1, ClassNode) or (
            isinstance(n1, LiteralNode) and wdns.is_abs_uri_qnode(n1.value)
        )

        for n in sm.iter_nodes():
            if isinstance(n, ClassNode) and wdns.is_abs_uri_statement(n.abs_uri):
                inedges = sm.in_edges(n.id)
                outedges = sm.out_edges(n.id)
                # only has one prop
                (prop,) = list({inedge.abs_uri for inedge in inedges})
                pid = wdns.get_prop_id(prop)
                stmt_has_value = False
                for outedge in outedges:
                    if outedge.abs_uri != prop:
                        # assert len(self.wdprops[self.get_prop_id(outedge.abs_uri)].inverse_properties) == 0, "Just to make sure" \
                        #                                                                                    "that qualifiers is not invertable. Otherwise, this algorithm will missing one generated SMs"
                        # character role has an inverse property: performer. They can be used as qualifier so nothing to do here just pass
                        pass
                    else:
                        stmt_has_value = True
                if (
                    len(self.wdprops[pid].inverse_properties) > 0
                    and pid not in incorrect_invertible_props
                    and (
                        limited_invertible_props is None
                        or pid in limited_invertible_props
                    )
                    and stmt_has_value
                ):
                    # invertible property
                    # people seem to misunderstand what inverse_property means in RDF;
                    # inverse doesn't apply to data property but only object property.
                    # so we catch the error here to detect what we should fix.
                    (outedge,) = [
                        outedge for outedge in outedges if outedge.abs_uri == prop
                    ]
                    targets_are_class = is_class_fn(sm.get_node(outedge.target))
                    if targets_are_class:
                        invertible_stmts.append(n)
                    elif strict:
                        raise Exception(f"{pid} is not invertible")
                    elif force_inversion:
                        assert isinstance(
                            sm.get_node(outedge.target), DataNode
                        ), "Clearly the model is wrong, you have an inverse property to a literal node"
                        invertible_stmts.append(n)

        # we have N statement, so we are going to have N! - 1 ways. It's literally a cartesian product
        all_choices = []
        for stmt in invertible_stmts:
            # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
            (inedge,) = sm.in_edges(stmt.id)
            choice: List[Tuple[ClassNode, Optional[str], Optional[str]]] = [
                (stmt, None, None)
            ]
            for invprop in self.wdprops[
                wdns.get_prop_id(inedge.abs_uri)
            ].inverse_properties:
                choice.append(
                    (
                        stmt,
                        wdns.get_prop_abs_uri(invprop),
                        wdns.get_prop_rel_uri(invprop),
                    )
                )
            all_choices.append(choice)

        n_choices = np.prod([len(c) for c in all_choices]) - 1
        if n_choices > 256:
            raise sm_metrics.PermutationExplosion("Too many possible semantic models")

        all_choices_perm: List[
            Tuple[Tuple[ClassNode, Optional[str], Optional[str]]]
        ] = list(itertools.product(*all_choices))
        assert all(
            invprop is None for _, invprop, _ in all_choices_perm[0]
        ), "First choice is always the current semantic model"
        new_sms = [sm]
        for choice_perm in all_choices_perm[1:]:
            new_sm = sm.copy()
            # we now change the statement from original prop to use the inverse prop (change direction)
            # if the invprop is not None
            for stmt, invprop_abs_uri, invprop_rel_uri in choice_perm:
                if invprop_abs_uri is None or invprop_rel_uri is None:
                    continue
                readable_label = self.get_pnode_label(wdns.get_prop_id(invprop_abs_uri))
                # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
                (inedge,) = sm.in_edges(stmt.id)
                # statement must have only one property
                (outedge,) = [
                    outedge
                    for outedge in sm.out_edges(stmt.id)
                    if outedge.abs_uri == inedge.abs_uri
                ]
                assert (
                    len(new_sm.get_edges_between_nodes(inedge.source, stmt.id)) == 1
                    and len(new_sm.get_edges_between_nodes(stmt.id, outedge.target))
                    == 1
                )
                new_sm.remove_edges_between_nodes(inedge.source, stmt.id)
                new_sm.remove_edges_between_nodes(stmt.id, outedge.target)

                target = sm.get_node(outedge.target)
                if not is_class_fn(target):
                    assert isinstance(target, DataNode)
                    dummy_class_node = ClassNode(
                        abs_uri=wdns.DUMMY_CLASS_FOR_INVERSION_URI,
                        rel_uri=wdns.get_rel_uri(wdns.DUMMY_CLASS_FOR_INVERSION_URI),
                    )
                    new_sm.add_node(dummy_class_node)
                    new_sm.add_edge(
                        Edge(
                            source=dummy_class_node.id,
                            target=target.id,
                            abs_uri=str(RDFS.label),
                            rel_uri="rdfs:label",
                        )
                    )
                    outedge_target = dummy_class_node.id
                else:
                    outedge_target = outedge.target
                new_sm.add_edge(
                    Edge(
                        source=outedge_target,
                        target=stmt.id,
                        abs_uri=invprop_abs_uri,
                        rel_uri=invprop_rel_uri,
                        approximation=outedge.approximation,
                        readable_label=readable_label,
                    )
                )
                new_sm.add_edge(
                    Edge(
                        source=stmt.id,
                        target=inedge.source,
                        abs_uri=invprop_abs_uri,
                        rel_uri=invprop_rel_uri,
                        approximation=inedge.approximation,
                        readable_label=readable_label,
                    )
                )
            new_sms.append(new_sm)
        return new_sms

    def get_entity_columns(self, sm: SemanticModel) -> List[int]:
        ent_columns = []
        for dnode in sm.iter_nodes():
            if isinstance(dnode, DataNode):
                inedges = sm.in_edges(dnode.id)
                if len(inedges) == 0:
                    continue
                assert len({edge.abs_uri for edge in inedges}) == 1, inedges
                edge_abs_uri = inedges[0].abs_uri
                if edge_abs_uri in self.ID_PROPS:
                    assert len(inedges) == 1, inedges
                    source = sm.get_node(inedges[0].source)
                    assert isinstance(
                        source, ClassNode
                    ) and not self.wdns.is_abs_uri_statement(source.abs_uri)
                    ent_columns.append(dnode.col_index)
        return ent_columns

    @classmethod
    def is_uri_column(cls, uri: str):
        """Test if an uri is for specifying the column"""
        return uri.startswith("http://example.com/table/")

    @staticmethod
    def get_column_uri(column_index: int):
        return f"http://example.com/table/{column_index}"

    @staticmethod
    def get_column_index(uri: str):
        assert WikidataSemanticModelHelper.is_uri_column(uri)
        return int(uri.replace("http://example.com/table/", ""))

    def extract_claims(
        self, tbl: LinkedTable, sm: SemanticModel, allow_multiple_ent: bool = True
    ):
        """Extract claims from the table given a semantic model.

        If an entity doesn't have link, its id will be null
        """
        # norm the semantic model first
        sm = self.norm_sm(sm)
        wdns = self.wdns
        schemas = {}
        for u in sm.iter_nodes():
            if not isinstance(u, ClassNode) or wdns.is_abs_uri_statement(u.abs_uri):
                continue

            schema = {"props": {}, "subject": None, "sm_node_id": u.id}
            for us_edge in sm.out_edges(u.id):
                if us_edge.abs_uri in self.ID_PROPS:
                    v = sm.get_node(us_edge.target)
                    assert isinstance(v, DataNode)
                    assert schema["subject"] is None
                    schema["subject"] = v.col_index
                    continue

                s = sm.get_node(us_edge.target)
                assert isinstance(s, ClassNode) and wdns.is_abs_uri_statement(s.abs_uri)
                assert wdns.is_abs_uri_property(us_edge.abs_uri)

                pnode = wdns.get_prop_id(us_edge.abs_uri)
                if pnode not in schema["props"]:
                    schema["props"][pnode] = []

                stmt = {
                    "index": len(schema["props"][pnode]),
                    "value": None,
                    "qualifiers": [],
                }
                schema["props"][pnode].append(stmt)
                for sv_edge in sm.out_edges(s.id):
                    v = sm.get_node(sv_edge.target)

                    assert wdns.is_abs_uri_property(sv_edge.abs_uri)
                    if sv_edge.abs_uri == us_edge.abs_uri:
                        assert stmt["value"] is None, "only one property"
                        # this is property
                        if isinstance(v, ClassNode):
                            stmt["value"] = {"type": "classnode", "value": v.id}
                        elif isinstance(v, DataNode):
                            stmt["value"] = {"type": "datanode", "value": v.col_index}
                        else:
                            assert isinstance(v, LiteralNode)
                            stmt["value"] = {"type": "literalnode", "value": v.value}
                    else:
                        # this is qualifier
                        if isinstance(v, ClassNode):
                            stmt["qualifiers"].append(
                                {
                                    "type": "classnode",
                                    "pnode": wdns.get_prop_id(sv_edge.abs_uri),
                                    "value": v.id,
                                }
                            )
                        elif isinstance(v, DataNode):
                            stmt["qualifiers"].append(
                                {
                                    "type": "datanode",
                                    "pnode": wdns.get_prop_id(sv_edge.abs_uri),
                                    "value": v.col_index,
                                }
                            )
                        else:
                            assert isinstance(v, LiteralNode)
                            stmt["qualifiers"].append(
                                {
                                    "type": "literalnode",
                                    "pnode": wdns.get_prop_id(sv_edge.abs_uri),
                                    "value": v.value,
                                }
                            )
            schemas[u.id] = schema

        assert all(
            c.index == ci for ci, c in enumerate(tbl.table.columns)
        ), "Cannot handle table with missing columns yet"

        records = [{} for ri in range(len(tbl.table.columns[0].values))]
        node2ents = {}

        # extract data props first
        for cid, schema in schemas.items():
            ci = schema["subject"]
            col = tbl.table.columns[ci]
            for ri, val in enumerate(col.values):
                # get entities
                qnode_ids = sorted(
                    {
                        entity_id
                        for link in tbl.links[ri][ci]
                        for entity_id in link.entities
                        if link.start < link.end
                    }
                )
                if len(qnode_ids) == 0:
                    # create new entity
                    ents = [
                        {
                            "id": f"{ri}-{ci}",
                            "column": ci,
                            "row": ri,
                            "uri": None,
                            "label": val,
                            "props": {},
                        }
                    ]
                else:
                    ents = [
                        {
                            "id": qnode_id,
                            "uri": wdns.get_entity_abs_uri(qnode_id),
                            "label": self.get_qnode_label(qnode_id),
                            "props": {},
                        }
                        for qnode_id in qnode_ids
                    ]

                if len(ents) > 1:
                    if not allow_multiple_ent:
                        raise Exception("Encounter multiple entities")

                for prop, stmts in schema["props"].items():
                    for ent in ents:
                        assert prop not in ent["props"]
                        ent["props"][prop] = [
                            {"value": None, "qualifiers": {}} for stmt in stmts
                        ]
                        for stmt in stmts:
                            # set statement property
                            if stmt["value"]["type"] == "classnode":
                                # do it in later phase
                                pass
                            elif stmt["value"]["type"] == "datanode":
                                tci = stmt["value"]["value"]
                                ent["props"][prop][stmt["index"]][
                                    "value"
                                ] = tbl.table.columns[tci].values[ri]
                            else:
                                assert stmt["value"]["type"] == "literalnode"
                                ent["props"][prop][stmt["index"]]["value"] = stmt[
                                    "value"
                                ]["value"]

                            # set statement qualifiers
                            for qual in stmt["qualifiers"]:
                                if (
                                    qual["pnode"]
                                    not in ent["props"][prop][stmt["index"]][
                                        "qualifiers"
                                    ]
                                ):
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ] = []
                                if qual["type"] == "classnode":
                                    # do it in later phase
                                    pass
                                elif qual["type"] == "datanode":
                                    tci = qual["value"]
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ].append(tbl.table.columns[tci].values[ri])
                                elif qual["type"] == "literalnode":
                                    ent["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ].append(qual["value"])

                for ent in ents:
                    assert (ent["id"], ci) not in records[ri]
                    records[ri][ent["id"], ci] = ent
                node2ents[schema["sm_node_id"], ri] = [ent for ent in ents]

        for cid, schema in schemas.items():
            ci = schema["subject"]
            col = tbl.table.columns[ci]
            for ri in range(len(col.values)):
                ulst = node2ents[schema["sm_node_id"], ri]
                for prop, stmts in schema["props"].items():
                    for stmt in stmts:
                        if stmt["value"]["type"] == "classnode":
                            vlst = node2ents[stmt["value"]["value"], ri]
                            for u in ulst:
                                assert len(vlst) > 0
                                u["props"][prop][stmt["index"]]["value"] = vlst[0]["id"]
                                if len(vlst) > 1:
                                    # this statement must not have other qualifiers, so that v can be a list
                                    # and we can create extra statement
                                    assert len(stmt["qualifiers"]) == 0
                                    for v in vlst[1:]:
                                        u["props"][prop].append(
                                            {"value": v["id"], "qualifiers": {}}
                                        )
                        for qual in stmt["qualifiers"]:
                            if qual["type"] == "classnode":
                                for u in ulst:
                                    vlst = node2ents[qual["value"], ri]
                                    u["props"][prop][stmt["index"]]["qualifiers"][
                                        qual["pnode"]
                                    ] = [v["id"] for v in vlst]

        new_records = []
        for ri, record in enumerate(records):
            new_record = {}
            for (ent_id, ci), ent in record.items():
                if ent_id in new_record:
                    # merge the entity
                    for pid, _stmts in ent["props"].items():
                        if pid not in new_record[ent_id]["props"]:
                            new_record[ent_id]["props"][pid] = _stmts
                        else:
                            for _stmt in _stmts:
                                if not any(
                                    x == _stmt for x in new_record[ent_id]["props"][pid]
                                ):
                                    new_record[ent_id]["props"][pid].append(_stmt)
                else:
                    new_record[ent_id] = ent
            new_records.append(new_record)
        return records

    def get_qnode_label(self, qid: str):
        """Get WDEntity label from id"""
        if qid in self.wdclasses:
            label = self.wdclasses[qid].label
        elif qid in self.wdentity_labels:
            label = self.wdentity_labels[qid]
        elif qid in self.wdentities:
            label = self.wdentities[qid].label
        else:
            return qid
        return f"{label} ({qid})"

    def get_pnode_label(self, pid: str):
        """Get PNode label from id"""
        if pid not in self.wdprops:
            return pid
        return f"{self.wdprops[pid].label} ({pid})"

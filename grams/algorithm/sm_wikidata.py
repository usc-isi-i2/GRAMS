import itertools

import numpy as np
from enum import IntEnum
from typing import Dict, Set, List
from uuid import uuid4

from rdflib import RDFS

from grams.inputs.linked_table import LinkedTable
from grams.evaluation import sm_metrics
import sm.outputs as O
from kgdata.wikidata.models import QNode, WDProperty, WDClass
from sm.misc.graph import viz_graph
from sm.misc import identity_func


class SMNodeType(IntEnum):
    Column = 0
    Class = 1
    Statement = 2
    Entity = 3
    Literal = 4


class OutOfNamespace(Exception):
    pass


class WDOnt:
    STATEMENT_URI = "http://wikiba.se/ontology#Statement"
    STATEMENT_REL_URI = "wikibase:Statement"

    def __init__(self, qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty]):
        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.get_qid_fn = {
            "Q": identity_func,
            "q": identity_func,
            # http
            "h": self.get_qnode_id
        }
        self.get_pid_fn = {
            "P": identity_func,
            "p": identity_func,
            "h": self.get_prop_id
        }

    @classmethod
    def is_uri_statement(cls, uri: str):
        return uri == 'http://wikiba.se/ontology#Statement'

    @classmethod
    def is_uri_dummy_class(cls, uri: str):
        return uri == 'http://wikiba.se/ontology#DummyClassForInversion'

    @classmethod
    def is_uri_property(cls, uri: str):
        return uri.startswith(f"http://www.wikidata.org/prop/")

    @classmethod
    def is_uri_qnode(cls, uri: str):
        return uri.startswith("http://www.wikidata.org/entity/")

    @classmethod
    def get_qnode_id(cls, uri: str):
        if not cls.is_uri_qnode(uri):
            raise OutOfNamespace(f"{uri} is not in wikidata qnode namespace")
        return uri.replace("http://www.wikidata.org/entity/", "")

    @classmethod
    def get_qnode_uri(cls, qnode_id: str):
        return f"http://www.wikidata.org/entity/{qnode_id}"

    @classmethod
    def get_prop_id(cls, uri: str):
        if not cls.is_uri_property(uri):
            raise OutOfNamespace(f"{uri} is not in wikidata property namespace")
        return uri.replace(f"http://www.wikidata.org/prop/", "")

    @classmethod
    def get_prop_uri(cls, pid: str):
        return f"http://www.wikidata.org/prop/{pid}"

    def get_qnode_label(self, uri_or_id: str):
        qid = self.get_qid_fn[uri_or_id[0]](uri_or_id)
        if qid in self.wdclasses:
            return f"{self.wdclasses[qid].label} ({qid})"
        return f"{self.qnodes[qid].label} ({qid})"

    def get_pnode_label(self, uri_or_id: str):
        pid = self.get_pid_fn[uri_or_id[0]](uri_or_id)
        # TODO: fix me! should not do this
        if pid not in self.wdprops:
            return pid
        return f"{self.wdprops[pid].label} ({pid})"


class WikidataSemanticModelHelper(WDOnt):
    ID_PROPS = {str(RDFS.label)}

    def norm_sm(self, sm: O.SemanticModel):
        """ "Normalize the semantic model with the following modifications:
        1. Add readable label to edge and class
        2. Convert direct link (without statement) to have statement except the id props.
        """
        new_sm = sm.clone()

        # update readable label
        for n in new_sm.iter_nodes():
            if n.is_class_node:
                if self.is_uri_qnode(n.abs_uri):
                    n.readable_label = self.get_qnode_label(n.abs_uri)
            elif n.is_literal_node:
                if self.is_uri_qnode(n.value):
                    n.readable_label = self.get_qnode_label(n.value)
        for e in new_sm.iter_edges():
            if e.abs_uri not in self.ID_PROPS:
                e.readable_label = self.get_pnode_label(e.abs_uri)

        # convert direct link
        for edge in list(new_sm.iter_edges()):
            if edge.abs_uri in self.ID_PROPS:
                continue
            source = new_sm.get_node(edge.source)
            target = new_sm.get_node(edge.target)

            if (not source.is_class_node or (source.is_class_node and source.abs_uri != WDOnt.STATEMENT_URI)) \
                    and (not target.is_class_node or (target.is_class_node and target.abs_uri != WDOnt.STATEMENT_URI)):
                # this is direct link, we replace its edge
                assert len(new_sm.get_edges_between_nodes(source.id, target.id)) == 1
                new_sm.remove_edges_between_nodes(source.id, target.id)
                stmt = O.ClassNode(str(uuid4()), WDOnt.STATEMENT_URI, WDOnt.STATEMENT_REL_URI, False,
                                   "Statement")
                new_sm.add_node(stmt)

                new_sm.add_edge(O.Edge(source=edge.source,
                                       target=stmt.id,
                                       abs_uri=edge.abs_uri, rel_uri=edge.rel_uri,
                                       approximation=edge.approximation, readable_label=edge.readable_label))
                new_sm.add_edge(O.Edge(source=stmt.id,
                                       target=edge.target,
                                       abs_uri=edge.abs_uri, rel_uri=edge.rel_uri,
                                       approximation=edge.approximation, readable_label=edge.readable_label))
        return new_sm

    @staticmethod
    def minify_sm(sm: O.SemanticModel):
        """This is a reverse function of `norm_sm`:
        1. Remove an intermediate statement if it doesn't have any qualifiers
        """
        new_sm = sm.clone()
        for n in sm.iter_nodes():
            if n.is_class_node and n.abs_uri == WDOnt.STATEMENT_URI:
                inedges = sm.incoming_edges(n.id)
                outedges = sm.outgoing_edges(n.id)
                if len(outedges) == 1 and outedges[0].abs_uri == inedges[0].abs_uri:
                    # no qualifiers
                    new_sm.remove_node(n.id)
                    for inedge in inedges:
                        assert inedge.abs_uri == outedges[0].abs_uri
                        new_sm.add_edge(O.Edge(inedge.source, outedges[0].target, inedge.abs_uri, inedge.rel_uri,
                                               # just in case user misannotate to not include approximation in both links
                                               inedge.approximation or outedges[0].approximation,
                                               inedge.readable_label))
        return new_sm

    def gen_equivalent_sm(self, sm: O.SemanticModel, strict: bool = True, force_inversion: bool = False):
        """Given a semantic model (not being modified), generate equivalent models by inferring inverse properties.

        Parameters
        ----------
        sm: the input semantic model (original)
        strict: whether to throw exception when target of an inverse property is not a class.
        force_inversion: only work when strict mode is set to false. Without force_inverse, we skip inverse properties,
                       otherwise, we generate an inverse model with a special class: wikibase:DummyClassForInversion

        Returns
        -------

        """
        """Given an semantic model (not being modified), generate equivalent models
        by inferring inverse properties. Running on strict mode mean it will check if the invertible property is apply
        to a non-class node (column that .

        Currently, we only inverse the properties, not qualifiers.
        """
        sm = self.norm_sm(sm)
        invertible_stmts = []
        is_class_fn = lambda n1: n1.is_class_node or (n1.is_literal_node and self.is_uri_qnode(n1.value))

        for n in sm.iter_nodes():
            if n.is_class_node and WDOnt.is_uri_statement(n.abs_uri):
                inedges = sm.incoming_edges(n.id)
                outedges = sm.outgoing_edges(n.id)
                # only has one prop
                prop, = list({inedge.abs_uri for inedge in inedges})
                pid = self.get_prop_id(prop)
                stmt_has_value = False
                for outedge in outedges:
                    if outedge.abs_uri != prop:
                        # assert len(self.wdprops[self.get_prop_id(outedge.abs_uri)].inverse_properties) == 0, "Just to make sure" \
                        #                                                                                    "that qualifiers is not invertable. Otherwise, this algorithm will missing one generated SMs"
                        # character role has an inverse property: performer. They can be used as qualifier so nothing to do here just pass
                        pass
                    else:
                        stmt_has_value = True
                if len(self.wdprops[pid].inverse_properties) > 0 and stmt_has_value:
                    # invertible property
                    # people seem to misunderstand what inverse_property means in RDF;
                    # inverse doesn't apply to data property but only object property.
                    # so we catch the error here to detect what we should fix.
                    outedge, = [outedge for outedge in outedges if outedge.abs_uri == prop]
                    targets_are_class = is_class_fn(sm.get_node(outedge.target))
                    if targets_are_class:
                        invertible_stmts.append(n)
                    elif strict:
                        raise Exception(f"{pid} is not invertible")
                    elif force_inversion:
                        assert sm.get_node(outedge.target).is_data_node, "Clearly the model is wrong, you have an inverse property to a literal node"
                        invertible_stmts.append(n)

        # we have N statement, so we are going to have N! - 1 ways. It's literally a cartesian product
        all_choices = []
        for stmt in invertible_stmts:
            # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
            inedge, = sm.incoming_edges(stmt.id)
            choice = [(stmt, None, None)]
            for invprop in self.wdprops[self.get_prop_id(inedge.abs_uri)].inverse_properties:
                choice.append((stmt, self.get_prop_uri(invprop), f"p:{invprop}"))
            all_choices.append(choice)

        n_choices = np.prod([len(c) for c in all_choices]) - 1
        if n_choices > 256:
            raise sm_metrics.PermutationExploding("Too many possible semantic models")

        all_choices = list(itertools.product(*all_choices))
        assert all(invprop is None for _, invprop, _ in all_choices[0]), "First choice is always the current semantic model"
        new_sms = [sm]
        for choice in all_choices[1:]:
            new_sm = sm.clone()
            # we now change the statement from original prop to use the inverse prop (change direction)
            # if the invprop is not None
            for stmt, invprop_abs_uri, invprop_rel_uri in choice:
                if invprop_abs_uri is None:
                    continue
                readable_label = self.get_pnode_label(invprop_abs_uri)
                # assume that each statement only has one incoming link! fix the for loop if this assumption doesn't hold
                inedge, = sm.incoming_edges(stmt.id)
                # statement must have only one property
                outedge, = [outedge for outedge in sm.outgoing_edges(stmt.id) if outedge.abs_uri == inedge.abs_uri]
                assert len(new_sm.get_edges_between_nodes(inedge.source, stmt.id)) == 1 and \
                       len(new_sm.get_edges_between_nodes(stmt.id, outedge.target)) == 1
                new_sm.remove_edges_between_nodes(inedge.source, stmt.id)
                new_sm.remove_edges_between_nodes(stmt.id, outedge.target)

                target = sm.get_node(outedge.target)
                if not is_class_fn(target):
                    assert target.is_data_node
                    dummy_class_node = O.ClassNode(str(uuid4()), abs_uri='http://wikiba.se/ontology#DummyClassForInversion',
                                                   rel_uri='wikibase:DummyClassForInversion')
                    new_sm.add_node(dummy_class_node)
                    new_sm.add_edge(O.Edge(source=dummy_class_node.id, target=target.id, abs_uri=str(RDFS.label),
                                       rel_uri="rdfs:label"))
                    outedge_target = dummy_class_node.id
                else:
                    outedge_target = outedge.target
                new_sm.add_edge(O.Edge(source=outedge_target, target=stmt.id,
                                       abs_uri=invprop_abs_uri, rel_uri=invprop_rel_uri,
                                       approximation=outedge.approximation, readable_label=readable_label))
                new_sm.add_edge(O.Edge(source=stmt.id, target=inedge.source,
                                       abs_uri=invprop_abs_uri, rel_uri=invprop_rel_uri,
                                       approximation=inedge.approximation, readable_label=readable_label))
            new_sms.append(new_sm)
        return new_sms

    def viz_sm(self, sm: O.SemanticModel, outdir, graph_id) -> 'WikidataSemanticModelHelper':
        colors = {
            SMNodeType.Column: dict(fill='#ffd666', stroke='#874d00'),
            SMNodeType.Class: dict(fill="#b7eb8f", stroke="#135200"),
            SMNodeType.Statement: dict(fill="#d9d9d9", stroke="#434343"),
            SMNodeType.Entity: dict(fill="#C6E5FF", stroke="#5B8FF9"),
            SMNodeType.Literal: dict(fill="#C6E5FF", stroke="#5B8FF9"),
        }

        def node_fn(uid, udata):
            u: O.Node = udata['data']
            nodetype = self.get_node_type(u)
            if isinstance(u, O.DataNode):
                label = u.label
            elif isinstance(u, O.ClassNode):
                if u.abs_uri == WDOnt.STATEMENT_URI:
                    label = ""
                elif u.readable_label is None:
                    if self.is_uri_qnode(u.abs_uri):
                        label = self.get_qnode_label(u.abs_uri)
                    else:
                        label = u.label
                else:
                    label = u.readable_label
            else:
                label = u.readable_label

            return {
                "label": label,
                "style": colors[nodetype],
                "labelCfg": {
                    "style": {
                        "fill": "black",
                        "background": {
                            "padding": [4, 4, 4, 4],
                            "radius": 3,
                            **colors[nodetype]
                        }
                    }
                },
            }

        def edge_fn(eid, edata):
            edge: O.Edge = edata['data']
            label = edge.readable_label
            if label is None:
                if edge.abs_uri not in self.ID_PROPS:
                    label = self.get_pnode_label(edge.abs_uri)
                else:
                    label = edge.label

            return {
                "label": label
            }

        viz_graph(sm.g, node_fn, edge_fn, outdir, graph_id)
        return self

    def get_node_type(self, n: O.Node):
        if isinstance(n, O.LiteralNode):
            if self.is_uri_qnode(n.value):
                return SMNodeType.Entity
            return SMNodeType.Literal
        if isinstance(n, O.DataNode) or (isinstance(n, O.ClassNode) and self.is_uri_column(n.abs_uri)):
            return SMNodeType.Column
        if isinstance(n, O.ClassNode):
            if n.abs_uri == WDOnt.STATEMENT_URI:
                return SMNodeType.Statement
            return SMNodeType.Class
        raise Exception("Unreachable!")

    def get_entity_columns(self, sm: O.SemanticModel) -> List[int]:
        ent_columns = []
        for dnode in sm.iter_nodes():
            if dnode.is_data_node:
                inedges = sm.incoming_edges(dnode.id)
                if len(inedges) == 0:
                    continue
                assert len({edge.abs_uri for edge in inedges}) == 1, inedges
                edge_abs_uri = inedges[0].abs_uri
                if edge_abs_uri in self.ID_PROPS:
                    assert len(inedges) == 1, inedges
                    source = sm.get_node(inedges[0].source)
                    assert not self.is_uri_statement(source.abs_uri)
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

    def extract_claims(self, tbl: LinkedTable, sm: O.SemanticModel, allow_multiple_ent: bool = True):
        """Extract claims from the table given a semantic model.

        If an entity doesn't have link, its id will be null
        """
        # norm the semantic model first
        sm = self.norm_sm(sm)
        schemas = {}
        for u in sm.iter_nodes():
            if not u.is_class_node or WDOnt.is_uri_statement(u.abs_uri):
                continue

            schema = {"props": {}, "subject": None, "sm_node_id": u.id}
            for us_edge in sm.outgoing_edges(u.id):
                if us_edge.abs_uri in self.ID_PROPS:
                    v = sm.get_node(us_edge.target)
                    assert v.is_data_node
                    assert schema['subject'] is None
                    schema['subject'] = v.col_index
                    continue

                s = sm.get_node(us_edge.target)
                assert s.is_class_node and WDOnt.is_uri_statement(s.abs_uri)
                assert WDOnt.is_uri_property(us_edge.abs_uri)

                pnode = WDOnt.get_prop_id(us_edge.abs_uri)
                if pnode not in schema['props']:
                    schema['props'][pnode] = []

                stmt = {
                    "index": len(schema['props'][pnode]),
                    'value': None,
                    'qualifiers': []
                }
                schema['props'][pnode].append(stmt)
                for sv_edge in sm.outgoing_edges(s.id):
                    v = sm.get_node(sv_edge.target)

                    assert WDOnt.is_uri_property(sv_edge.abs_uri)
                    if sv_edge.abs_uri == us_edge.abs_uri:
                        assert stmt['value'] is None, "only one property"
                        # this is property
                        if v.is_class_node:
                            stmt['value'] = {'type': 'classnode', 'value': v.id}
                        elif v.is_data_node:
                            stmt['value'] = {'type': 'datanode', 'value': v.col_index}
                        else:
                            assert v.is_literal_node
                            stmt['value'] = {'type': 'literalnode', 'value': v.value}
                    else:
                        # this is qualifier
                        if v.is_class_node:
                            stmt['qualifiers'].append({'type': 'classnode', 'pnode': WDOnt.get_prop_id(sv_edge.abs_uri), 'value': v.id})
                        elif v.is_data_node:
                            stmt['qualifiers'].append({'type': 'datanode', 'pnode': WDOnt.get_prop_id(sv_edge.abs_uri), 'value': v.col_index})
                        else:
                            assert v.is_literal_node
                            stmt['qualifiers'].append({'type': 'literalnode', 'pnode': WDOnt.get_prop_id(sv_edge.abs_uri), 'value': v.value})
            schemas[u.id] = schema

        assert all(c.index == ci for ci, c in enumerate(tbl.table.columns)), "Cannot handle table with missing columns yet"

        records = [{} for ri in range(len(tbl.table.columns[0].values))]
        node2ents = {}

        # extract data props first
        for cid, schema in schemas.items():
            ci = schema['subject']
            col = tbl.table.columns[ci]
            for ri, val in enumerate(col.values):
                # get entities
                qnode_ids = sorted({e.qnode_id for e in tbl.links[ri][ci] if e.qnode_id is not None and e.start < e.end})
                if len(qnode_ids) == 0:
                    # create new entity
                    ents = [{
                        "id": f"{ri}-{ci}",
                        "column": ci,
                        "row": ri,
                        "uri": None,
                        "label": val,
                        "props": {}
                    }]
                else:
                    ents = [
                        {
                            "id": qnode_id, "uri": WDOnt.get_qnode_uri(qnode_id),
                            "label": self.qnodes[qnode_id].label, "props": {}
                        }
                        for qnode_id in qnode_ids
                    ]

                if len(ents) > 1:
                    if not allow_multiple_ent:
                        raise Exception("Encounter multiple entities")

                for prop, stmts in schema['props'].items():
                    for ent in ents:
                        assert prop not in ent['props']
                        ent['props'][prop] = [{'value': None, 'qualifiers': {}} for stmt in stmts]
                        for stmt in stmts:
                            # set statement property
                            if stmt['value']['type'] == 'classnode':
                                # do it in later phase
                                pass
                            elif stmt['value']['type'] == 'datanode':
                                tci = stmt['value']['value']
                                ent['props'][prop][stmt['index']]['value'] = tbl.table.columns[tci].values[ri]
                            else:
                                assert stmt['value']['type'] == 'literalnode'
                                ent['props'][prop][stmt['index']]['value'] = stmt['value']['value']

                            # set statement qualifiers
                            for qual in stmt['qualifiers']:
                                if qual['pnode'] not in ent['props'][prop][stmt['index']]['qualifiers']:
                                    ent['props'][prop][stmt['index']]['qualifiers'][qual['pnode']] = []
                                if qual['type'] == 'classnode':
                                    # do it in later phase
                                    pass
                                elif qual['type'] == 'datanode':
                                    tci = qual['value']
                                    ent['props'][prop][stmt['index']]['qualifiers'][qual['pnode']].append(tbl.table.columns[tci].values[ri])
                                elif qual['type'] == 'literalnode':
                                    ent['props'][prop][stmt['index']]['qualifiers'][qual['pnode']].append(qual['value'])

                for ent in ents:
                    assert (ent['id'], ci) not in records[ri]
                    records[ri][ent['id'], ci] = ent
                node2ents[schema['sm_node_id'], ri] = [ent for ent in ents]

        for cid, schema in schemas.items():
            ci = schema['subject']
            col = tbl.table.columns[ci]
            for ri in range(len(col.values)):
                ulst = node2ents[schema['sm_node_id'], ri]
                for prop, stmts in schema['props'].items():
                    for stmt in stmts:
                        if stmt['value']['type'] == 'classnode':
                            vlst = node2ents[stmt['value']['value'], ri]
                            for u in ulst:
                                assert len(vlst) > 0
                                u['props'][prop][stmt['index']]['value'] = vlst[0]['id']
                                if len(vlst) > 1:
                                    # this statement must not have other qualifiers, so that v can be a list
                                    # and we can create extra statement
                                    assert len(stmt['qualifiers']) == 0
                                    for v in vlst[1:]:
                                        u['props'][prop].append({'value': v['id'], 'qualifiers': {}})
                        for qual in stmt['qualifiers']:
                            if qual['type'] == 'classnode':
                                for u in ulst:
                                    vlst = node2ents[qual['value'], ri]
                                    u['props'][prop][stmt['index']]['qualifiers'][qual['pnode']] = [v['id'] for v in vlst]

        new_records = []
        for ri, record in enumerate(records):
            new_record = {}
            for (ent_id, ci), ent in record.items():
                if ent_id in new_record:
                    # merge the entity
                    for pid, _stmts in ent['props'].items():
                        if pid not in new_record[ent_id]['props']:
                            new_record[ent_id]['props'][pid] = _stmts
                        else:
                            for _stmt in _stmts:
                                if not any(x == _stmt for x in new_record[ent_id]['props'][pid]):
                                    new_record[ent_id]['props'][pid].append(_stmt)
                else:
                    new_record[ent_id] = ent
            new_records.append(new_record)
        return records




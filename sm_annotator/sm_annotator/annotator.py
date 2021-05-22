from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, TypedDict, Tuple, Union

import orjson
from IPython.core.display import display, Javascript
from ipywidgets import HTML

from grams.kg_data.wikidatamodels import QNode, WDClass, WDProperty, DataValue
from sm_annotator.annotator_assistant import AnnotatorAssistant, Resource
from sm_annotator.base_app import BaseApp
from sm_annotator.ontology_api import OntologyAPI
from sm_annotator.slider import SliderApp
from grams.algorithm.helpers import reorder2tree, IndirectDictAccess
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper, WDOnt, OutOfNamespace
import grams.outputs as O
import grams.inputs as I
import grams.misc as M


class Session:

    def __init__(self, id: str, is_curated: bool, note: str, table: LinkedTable, graphs: List[O.SemanticModel]):
        self.id = id
        self.is_curated = is_curated
        self.note = note
        self.table = table
        self.graphs = graphs

        # use it for querying records
        self.table_records_index = set(range(table.size()))
        self.column2name = {}
        cname2freq = Counter(c.name for c in table.table.columns)
        for c in table.table.columns:
            if cname2freq[c.name] > 1:
                self.column2name[c.index] = f"{c.name} ({c.index})"
            else:
                self.column2name[c.index] = c.name

    def to_json(self):
        return {
            "version": 2,
            "table_id": self.table.id,
            "semantic_models": [sm.to_json() for sm in self.graphs],
            "is_curated": self.is_curated,
            "note": self.note,
        }

class FilterOp(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class EntityTypeFilter:
    column_index: int
    # qnode that is the class
    class_id: str
    operator: FilterOp


@dataclass
class RelFilter:
    # the column we want to filter
    column_index: int
    # the endpoint, source or target or wildcard (both none): string for entity & int for column index
    source_endpoint: Optional[Union[int, str]]
    target_endpoint: Optional[Union[int, str]]
    # [0] is property, [1] is qualifier
    props: Tuple[str, str]
    operator: FilterOp


class CellTypeFilterOp(Enum):
    HAS_LINK = "hasLink"
    HAS_ENTITY = "hasEntity"
    HAS_NO_LINK = "noLink"
    HAS_NO_ENTITY = "noEntity"


@dataclass
class CellTypeFilter:
    column_index: int
    op: CellTypeFilterOp



class _Annotator(BaseApp):
    pass


class Annotator(_Annotator):
    def __init__(self,
                 qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty],
                 savedir: str, eshost: str, username: str, password: str,
                 dev: bool = False, assistant: Optional[AnnotatorAssistant] = None):
        super().__init__("annotator", dev)

        self.savedir = Path(savedir)
        self.savedir.mkdir(exist_ok=True, parents=True)

        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.username = username
        self.password = password

        self.ont: OntologyAPI = OntologyAPI(eshost, wdclasses, wdprops)
        self.wdont = WDOnt(qnodes, wdclasses, wdprops)
        self.assistant = assistant
        self.wdclass_parents: Dict[str, Set[str]] = IndirectDictAccess(self.wdclasses, attrgetter("parents_closure"))
        self.cache_id2label: Dict[str, str] = {}

    def annotate(self, id: str, tbl: LinkedTable):
        self.cache_id2label = {}
        infile = M.get_latest_path(self.savedir / id / "version.json")
        if infile is None:
            sms = []
            is_curated = False
            note = ""
        else:
            data = M.deserialize_json(infile)
            assert data['version'] == 2
            sms = [O.SemanticModel.from_json(sm) for sm in data['semantic_models']]
            is_curated = data['is_curated']
            note = data['note']

        self.session = Session(id, is_curated, note, tbl, sms)
        if len(sms) == 0:
            self.add_default_nodes_to_sm(O.SemanticModel())
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "wait_for_client_ready",
            },
            {
                "type": "set_props",
                "props": {
                    "log": {
                        "isCurated": self.session.is_curated,
                        "note": self.session.note,
                    },
                    "table": self.serialize_table_schema(),
                    "graphs": [self.serialize_sm(graph) for graph in self.session.graphs],
                    "entities": {},
                    "assistant": {
                        "id": self.session.table.id
                    },
                    "currentGraphIndex": 0,
                    "wdOntology": {
                        "username": self.username,
                        "password": self.password
                    },
                }
            },
            {
                "type": "exec_func",
                "func": "app.tableFetchData",
                "args": []
            }
        ]).decode())

    def save_annotation(self):
        assert len(self.session.graphs) > 0
        (self.savedir / self.session.id).mkdir(exist_ok=True)
        outfile = M.get_incremental_path(self.savedir / self.session.id / "version.json")
        with open(outfile, "wb") as f:
            f.write(orjson.dumps(self.session.to_json(), option=orjson.OPT_INDENT_2))

    @_Annotator.register_handler("/table")
    def fetch_table_data(self, params: dict):
        table = self.session.table
        start = params['offset']
        end = start + params['limit']
        filters = [
            EntityTypeFilter(
                item['columnId'],
                WDOnt.get_qnode_id(item['uri']),
                FilterOp(item['op']))
            for item in params['typeFilters']
        ]
        filters += [
            RelFilter(
                int(item['columnId']),
                (WDOnt.get_qnode_id(item['endpoint']) if isinstance(item['endpoint'], str) else item['endpoint']) if
                    item['direction'] == 'incoming' and item['endpoint'] != "*" else None,
                (WDOnt.get_qnode_id(item['endpoint']) if isinstance(item['endpoint'], str) else item['endpoint']) if
                    item['direction'] == 'outgoing' and item['endpoint'] != "*" else None,
                (WDOnt.get_prop_id(item['pred1']), WDOnt.get_prop_id(item['pred2'])),
                FilterOp(item['op'])
            )
            for item in params['relFilters']
        ]
        filters += [
            CellTypeFilter(int(column_id), CellTypeFilterOp(op))
            for column_id, op in params['linkFilters'].items()
        ]

        tbl_size = table.size()
        if len(filters) == 0:
            total = tbl_size
            row_index = range(start, min(total, end))
        else:
            includes = set()
            excludes = set()
            for filter in filters:
                ci = filter.column_index

                if isinstance(filter, CellTypeFilter):
                    for ri in range(table.size()):
                        links = table.links[ri][ci]
                        if filter.op == CellTypeFilterOp.HAS_LINK:
                            if len(links) == 0:
                                excludes.add(ri)
                        elif filter.op == CellTypeFilterOp.HAS_NO_LINK:
                            if len(links) > 0:
                                excludes.add(ri)
                        elif filter.op == CellTypeFilterOp.HAS_ENTITY:
                            if all(link.qnode_id is None for link in links):
                                excludes.add(ri)
                        else:
                            assert filter.op == CellTypeFilterOp.HAS_NO_ENTITY
                            if not all(link.qnode_id is None for link in links):
                                excludes.add(ri)
                    continue

                if filter.operator == FilterOp.EXCLUDE:
                    sat_condition = excludes
                else:
                    sat_condition = includes

                if isinstance(filter, RelFilter):
                    if filter.source_endpoint is not None:
                        assert filter.target_endpoint is None
                        # query for source
                        res = self.assistant.get_row_indices(table, filter.source_endpoint, filter.column_index,
                                                             filter.props)
                    elif filter.target_endpoint is not None:
                        assert filter.source_endpoint is None
                        # query for the target
                        res = self.assistant.get_row_indices(table, filter.column_index, filter.target_endpoint,
                                                             filter.props)
                    else:
                        res = set()
                        for ri in range(table.size()):
                            links = table.links[ri][ci]
                            sat = False
                            for link in links:
                                if link.qnode_id is None:
                                    continue
                                qnode = self.qnodes[link.qnode_id]
                                if filter.props[0] not in qnode.props:
                                    continue
                                if filter.props[0] == filter.props[1]:
                                    # statement value
                                    sat = True
                                else:
                                    for stmt in qnode.props[filter.props[0]]:
                                        if filter.props[1] in stmt.qualifiers:
                                            sat = True
                                            break
                                if sat:
                                    break
                            if sat:
                                res.add(ri)
                    sat_condition.update(res)
                else:
                    assert isinstance(filter, EntityTypeFilter)
                    for ri in range(table.size()):
                        links = table.links[ri][ci]
                        sat = False
                        for link in links:
                            if link.qnode_id is None:
                                continue
                            qnode = self.qnodes[link.qnode_id]
                            for stmt in qnode.props.get("P31", []):
                                classid = stmt.value.as_qnode_id()
                                if classid not in self.wdclasses:
                                    continue
                                clsnode = self.wdclasses[classid]
                                # this entity is the same class, or its type is child of the desired class not parents
                                if classid == filter.class_id or filter.class_id in clsnode.parents_closure:
                                    sat = True
                                    break
                            if sat:
                                break
                        if sat:
                            sat_condition.add(ri)

            if len(includes) == 0 and len(excludes) > 0:
                includes = self.session.table_records_index
            row_index = sorted(includes.difference(excludes))
            total = len(row_index)
            row_index = row_index[start:min(total, end)]

        rows = []
        for ri in row_index:
            row = []
            for ci in range(len(table.table.columns)):
                row.append(self.serialize_table_cell(ri, ci))

            rows.append({
                "data": row,
                "rowId": ri
            })
        return {"rows": rows, "total": total}

    @_Annotator.register_handler("/entities")
    def get_entity(self, params: dict):
        return {
            uri: self.serialize_qnode(WDOnt.get_qnode_id(uri), full=True)
            for uri in params['uris']
        }
    
    @_Annotator.register_handler("/ontology/class")
    def search_ont_class(self, params: dict):
        return self.ont.search_class(params['query'])

    @_Annotator.register_handler("/ontology/predicate")
    def search_ont_property(self, params: dict):
        return self.ont.search_predicate(params['query'])

    @_Annotator.register_handler("/save")
    def save_annotation_handler(self, params: dict):
        self.session.note = params['note']
        self.session.is_curated = params['isCurated']
        self.session.graphs = [self.deserialize_sm(g['nodes'], g['edges']) for g in params['graphs']]
        self.save_annotation()

    @_Annotator.register_handler("/assistant/column")
    def suggest_column(self, params: dict):
        ci = params['columnIndex']
        table = self.session.table
        column = table.table.columns[ci]

        # compute statistics
        stats = {
            'entities/linked row': [],
            "links/row": [],
            "avg %link surface": [],
            "# rows": table.size(),
        }
        qnode2types = {}
        for ri, value in enumerate(column.values):
            nchars = len(value)
            links = table.links[ri][ci]
            n_covered_chars = sum(l.end - l.start for l in links)
            n_qnodes = sum(l.qnode_id is not None for l in links)
            if len(links) > 0:
                stats['entities/linked row'].append(n_qnodes)
            stats['links/row'].append(len(links))
            stats['avg %link surface'].append(n_covered_chars / max(nchars, 1e-7))

            for l in links:
                if l.qnode_id is None:
                    continue
                qnode = self.qnodes[l.qnode_id]
                qnode2types[l.qnode_id] = list({stmt.value.as_qnode_id() for stmt in qnode.props.get("P31", [])})

        stats['# unique qnodes'] = len(qnode2types)
        for k, v in stats.items():
            if isinstance(v, list):
                if len(v) == 0:
                    stats[k] = "0%"
                else:
                    stats[k] = f"{sum(v) / max(len(v), 1e-7):.2f}%"

        supertype_freq = defaultdict(int)

        for lst in qnode2types.values():
            etypes = set()
            for item in lst:
                etypes.add(item)
                etypes.update(self.wdclasses[item].parents_closure)
            for c in etypes:
                supertype_freq[c] += 1

        flat_type_hierarchy = reorder2tree(list(supertype_freq.keys()), self.wdclass_parents)
        qnode2children = {}

        def traversal(tree, path):
            if tree.id not in qnode2children:
                qnode2children[tree.id] = set()
            for parent_tree in path:
                qnode2children[parent_tree.id].add(WDOnt.get_qnode_uri(tree.id))

        flat_type_hierarchy.preorder(traversal)

        flat_type_hierarchy = flat_type_hierarchy.update_score(lambda n: supertype_freq[n.id]).sort(reverse=True)
        flat_type_hierarchy = [
            {
                "uri": WDOnt.get_qnode_uri(u.id),
                "label": self.get_qnode_label(u.id),
                "duplicated": u.duplicated,
                "depth": u.depth,
                "freq": supertype_freq[u.id],
                "percentage": supertype_freq[u.id] / len(qnode2types)
            }
            for u in flat_type_hierarchy.get_flatten_hierarchy(dedup=True)
        ]

        resp = {
            "stats": stats,
            "flattenTypeHierarchy": flat_type_hierarchy,
            "type2children": {WDOnt.get_qnode_uri(k): list(v) for k, v in qnode2children.items()},
        }

        if self.assistant is not None:
            relationships = self.assistant.get_column_relationships(table, ci)
            if relationships is not None:
                for rels in relationships.values():
                    for i, rel in enumerate(rels):
                        rels[i] = dict(
                            endpoint=rel.endpoint,
                            predicates=rel.predicates,
                            freq=rel.freq,
                        )
                resp['relationships'] = relationships
        return resp

    def add_default_nodes_to_sm(self, sm: O.SemanticModel):
        table = self.session.table
        column2name = self.session.column2name
        for col in table.table.columns:
            if not any(n.is_data_node and n.col_index == col.index for n in sm.iter_nodes()):
                # no column in the model, add missing columns
                dnodeid = f'd-{col.index}'
                assert not sm.has_node(dnodeid)
                sm.add_node(O.DataNode(f"d-{col.index}", col.index, column2name[col.index]))

        if table.context.page_qnode is not None:
            context_nodeid = f"context-{table.context.page_qnode}"
            sm.add_node(O.LiteralNode(
                context_nodeid,
                WDOnt.get_qnode_uri(table.context.page_qnode),
                f"{self.qnodes[table.context.page_qnode].label} ({table.context.page_qnode})",
                True,
                O.LiteralNodeDataType.Entity
            ))
        return sm

    def deserialize_sm(self, nodes: List[dict], edges: List[dict]) -> O.SemanticModel:
        sm = O.SemanticModel()
        for n in nodes:
            if n['isDataNode']:
                sm.add_node(O.DataNode(n['id'], n['columnId'], self.session.column2name[n['columnId']]))
            elif n['isClassNode']:
                if WikidataSemanticModelHelper.is_uri_qnode(n['uri']):
                    rel_uri = f"wd:{WDOnt.get_qnode_id(n['uri'])}"
                else:
                    # TODO: very sketchy here since label is not rel uri. fix me
                    rel_uri = n['label']
                sm.add_node(
                    O.ClassNode(n['id'], n['uri'], rel_uri, n['approximation']))
            else:
                assert n['isLiteralNode']
                sm.add_node(
                    O.LiteralNode(n['id'], n['uri'], n['label'], n['isInContext'], O.LiteralNodeDataType(n['datatype'])))

        for e in edges:
            if WikidataSemanticModelHelper.is_uri_property(e['uri']):
                rel_uri = f"p:{WDOnt.get_prop_id(e['uri'])}"
            else:
                rel_uri = e['label']
            sm.add_edge(O.Edge(e['source'], e['target'], e['uri'], rel_uri, e['approximation']))
        return sm

    def serialize_sm(self, sm: O.SemanticModel):
        nodes = []
        for n in sm.iter_nodes():
            if n.is_class_node:
                nodes.append({
                    "id": n.id,
                    "uri": n.abs_uri,
                    "label": self.get_qnode_label(n.abs_uri) or n.rel_uri,
                    "approximation": n.approximation,
                    "isClassNode": True,
                    "isDataNode": False,
                    "isLiteralNode": False,
                })
            elif n.is_data_node:
                nodes.append({
                    "id": n.id,
                    "label": self.session.column2name[n.col_index],
                    "isClassNode": False,
                    "isDataNode": True,
                    "isLiteralNode": False,
                    "columnId": n.col_index,
                })
            else:
                nodes.append({
                    "id": n.id,
                    "uri": n.value if n.datatype == O.LiteralNodeDataType.Entity else "",
                    "label": n.label,
                    "isClassNode": False,
                    "isDataNode": False,
                    "isLiteralNode": True,
                    "isInContext": n.is_in_context,
                    "datatype": n.datatype.value
                })

        return {
            "tableID": self.session.table.id,
            "nodes": nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "uri": e.abs_uri,
                    "label": self.get_pnode_label(e.abs_uri) or e.rel_uri,
                    "approximation": e.approximation
                }
                for e in sm.iter_edges()
            ]
        }

    def serialize_table_schema(self):
        """Get schema of the table"""
        table = self.session.table
        columns = [{"title": "", "dataIndex": "rowId"}]
        for col in table.table.columns:
            columns.append({
                "title": col.name,
                "columnId": col.index,
                "dataIndex": ["data", col.index],
            })
        return {
            "id": table.id,
            "columns": columns,
            "rowKey": "rowId",
            "totalRecords": table.size(),
            "metadata": {
                "title": table.context.page_title,
                "url": table.context.page_url,
                "entity": {"uri": WDOnt.get_qnode_uri(table.context.page_qnode), "label": self.get_qnode_label(table.context.page_qnode)} if table.context.page_qnode is not None else None
            }
        }

    def serialize_table_cell(self, ri: int, ci: int):
        table = self.session.table
        value = table.table.columns[ci].values[ri]
        qnodes_metadata = {}
        for link in table.links[ri][ci]:
            if link.qnode_id is not None:
                qnodes_metadata[WDOnt.get_qnode_uri(link.qnode_id)] = self.serialize_qnode(link.qnode_id)

        return {
            "value": value,
            "links": [
                {
                    "start": link.start, "end": link.end,
                    "href": link.url,
                    "entity": WDOnt.get_qnode_uri(link.qnode_id) if link.qnode_id is not None else None
                }
                for link in table.links[ri][ci]
            ],
            "metadata": {
                "entities": qnodes_metadata
            }
        }

    def serialize_qnode(self, qnode_id: str, full: bool = False):
        wdclass_parents = self.wdclass_parents
        qnode = self.qnodes[qnode_id]
        ent = {
            "uri": WDOnt.get_qnode_uri(qnode_id),
            "label": str(self.get_qnode_label(qnode_id)),
        }
        try:
            # get hierarchy
            if "P31" in qnode.props:
                forest = reorder2tree(
                    [stmt.value.as_qnode_id() for stmt in qnode.props.get("P31", [])],
                    wdclass_parents)
                hierarchy = [{"uri": WDOnt.get_qnode_uri(x.id),
                              "label": self.get_qnode_label(x.id),
                              "depth": x.depth}
                             for x in forest.get_flatten_hierarchy()]
            else:
                hierarchy = []

            ent["types"] = hierarchy
            if not full:
                return ent

            props = {}
            for p, stmts in qnode.props.items():
                # if p == 'P31':
                #     # no need to dump the class in here, and also because we just don't have the class in the qnode store
                #     continue

                ser_stmts = []
                for stmt in stmts:
                    ser_stmts.append({
                        "value": self.serialize_datavalue(stmt.value),
                        "qualifiers": {
                            qid: {
                                "uri": WDOnt.get_prop_uri(qid),
                                "label": self.get_pnode_label(qid),
                                "values": [self.serialize_datavalue(qual) for qual in quals]
                            }
                            for qid, quals in stmt.qualifiers.items()
                        }
                    })
                puri = WDOnt.get_prop_uri(p)
                props[puri] = {
                    "uri": puri,
                    "label": self.get_pnode_label(puri),
                    "values": ser_stmts
                }
            ent['props'] = props
            ent['description'] = str(qnode.description)
        except KeyError:
            print(f"Error while obtaining information of a qnode: {qnode_id}")
            raise

        return ent

    def serialize_datavalue(self, val: DataValue):
        if val.is_string():
            return val.as_string()
        if val.is_qnode():
            tmpid = val.as_qnode_id()
            # an exception that they sometime has property instead of qnode id???
            # this is probably an error, but we need to make this code robust
            if tmpid.startswith("P"):
                label = self.get_pnode_label(tmpid)
                uri = WDOnt.get_prop_uri(tmpid)
            elif tmpid.startswith("Q"):
                label = self.get_qnode_label(tmpid)
                uri = WDOnt.get_qnode_uri(tmpid)
            else:
                label = "<error>"
                uri = f"http://www.wikidata.org/wiki/{tmpid}"
            return {"uri": uri, "label": label}
        if val.is_time():
            return val.value['time']
        if val.is_quantity():
            return val.value['amount']
        if val.is_mono_lingual_text():
            return val.value['text']
        if val.is_globe_coordinate():
            return f"%s:%s" % (val.value['latitude'], val.value['longitude'])

        raise NotImplementedError()

    def get_qnode_label(self, uri_or_id: str):
        if uri_or_id not in self.cache_id2label:
            try:
                label = self.wdont.get_qnode_label(uri_or_id)
            except Exception as e:
                if str(e).find("is not in wikidata qnode namespace") == -1:
                    raise
                else:
                    label = None
            self.cache_id2label[uri_or_id] = label
        return self.cache_id2label[uri_or_id]

    def get_pnode_label(self, uri_or_id: str):
        if uri_or_id not in self.cache_id2label:
            try:
                label = self.wdont.get_pnode_label(uri_or_id)
            except OutOfNamespace:
                label = None
            self.cache_id2label[uri_or_id] = label
        return self.cache_id2label[uri_or_id]


class BatchAnnotator(SliderApp):

    def __init__(self, annotator: Annotator, dev: bool = False):
        super().__init__(annotator, annotator.annotate, dev)

    def batch_annotate(self, tables_with_ids: List[Tuple[str, str, LinkedTable]], start_index: int=0):
        self.set_data([
            dict(description=description, args=(table_id, table))
            for table_id, description, table in tables_with_ids
        ], start_index)

import networkx as nx
from dataclasses import dataclass

from typing import Optional, TypedDict, Union, List, Tuple, Dict

from grams.kg_data.wikidatamodels import QNode, WDProperty
from grams.inputs.linked_table import W2WTable
from grams.algorithm.semantic_graph import SGNode, SGStatementNode
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper, WDOnt

Resource = TypedDict("Resource", dict(uri=str, label=str))


@dataclass
class ColumnRelationship:
    # dict for entity, number for column index
    endpoint: Union[Resource, int]
    predicates: Tuple[Resource, Resource]
    freq: int


ColumnRelationshipResult = TypedDict("ColumnRelationshipResult",
                                     dict(incoming=List[ColumnRelationship], outgoing=List[ColumnRelationship]))


# noinspection PyMethodMayBeStatic
class AnnotatorAssistant:

    def get_row_indices(self, table: W2WTable, source_node: Union[int, str], target_node: Union[int, str],
                        links: Tuple[str, str]):
        # return the list of indices
        return set()

    def get_column_relationships(self, table: W2WTable, column_index: str) -> Optional[ColumnRelationshipResult]:
        # get relationships of columns, should return {
        # "incoming": {endpoint, endpoint_type, predicates, freq}[]
        # "outgoing": {endpoint, endpoint_type, predicates, freq}[]
        # }
        return None


class DummyAnnotatorAssistant(AnnotatorAssistant):

    def get_column_relationships(self, table: W2WTable, column_index: str) -> Optional[ColumnRelationshipResult]:
        return {
            "incoming": [
                ColumnRelationship(
                    endpoint=0,
                    predicates=(
                        {"uri": WDOnt.get_prop_uri("P585"), "label": "Point in time (P585)"},),
                    freq=10
                ),
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": WDOnt.get_prop_uri("P710"), "label": "participant (P710)"},),
                    freq=5
                ),
                ColumnRelationship(
                    endpoint={"uri": WDOnt.get_qnode_uri("Q5"), "label": "United States (Q5)"},
                    predicates=({"uri": WDOnt.get_prop_uri("P585"), "label": "Point in time (P585)"},),
                    freq=10
                )
            ],
            "outgoing": [
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": WDOnt.get_prop_uri("P585"), "label": "Point in time (P585)"},),
                    freq=10
                ),
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": WDOnt.get_prop_uri("P710"), "label": "participant (P710)"},),
                    freq=5
                ),
                ColumnRelationship(
                    endpoint={"uri": WDOnt.get_qnode_uri("Q5"), "label": "United States (Q5)"},
                    predicates=({"uri": WDOnt.get_prop_uri("P585"), "label": "Point in time (P585)"},),
                    freq=10
                )
            ]
        }


class GRAMSAnnotatorAssistant(AnnotatorAssistant):

    def __init__(self, inputs: List[TypedDict("input", table=W2WTable, dg=nx.MultiDiGraph, sg=nx.MultiDiGraph)], qnodes: Dict[str, QNode],
                 wdprops: Dict[str, WDProperty]):
        self.qnodes = qnodes
        self.wdprops = wdprops

        self.wdont = WDOnt(self.qnodes, {}, self.wdprops)

        self.table2g = {}
        for e in inputs:
            self.table2g[e['table'].id] = (e['dg'], e['sg'])

    def get_row_indices(self, table: W2WTable, source_node: Union[int, str], target_node: Union[int, str], links: Tuple[str, str]):
        assert isinstance(source_node, int) or isinstance(target_node, int), "Can only get row index for at least one column"
        if isinstance(source_node, str):
            if table.context.page_qnode == source_node:
                # TODO: get context node id, is there any better way to do it?
                source_id = f"ent:{source_node}"
            else:
                source_id = f"ent:{source_node}"
        else:
            source_id = f"column-{source_node}"

        if isinstance(target_node, str):
            # TODO: fix me after we fix the data graph
            if table.context.page_qnode == target_node:
                # TODO: get context node id, is there any better way to do it?
                target_id = f"ent:{target_node}"
            else:
                target_id = f"ent:{target_node}"
        else:
            target_id = f"column-{target_node}"

        dg, sg = self.table2g[table.id]
        rows = set()
        for paths in nx.all_simple_edge_paths(sg, source_id, target_id, cutoff=2):
            assert len(paths) == 2
            uid, sid, eid = paths[0]
            _, vid, peid = paths[1]

            if not (eid == links[0] and peid == links[1]):
                continue

            stmt: SGStatementNode = sg.nodes[sid]['data']
            for (source_flow, target_flow), stmt2prov in stmt.flow.items():
                if target_flow.sg_target_id == vid and target_flow.edge_id == peid:
                    dv = dg.nodes[target_flow.dg_target_id]['data']
                    if dv.is_cell:
                        rows.add(dv.row)
                    else:
                        du = dg.nodes[source_flow.dg_source_id]['data']
                        assert du.is_cell
                        rows.add(du.row)
        return rows

    def get_column_relationships(self, table: W2WTable, column_index: str) -> Optional[ColumnRelationshipResult]:
        dg, sg = self.table2g[table.id]
        node_id = f"column-{column_index}"

        incomings = []
        for sid, _, eid, edata in sg.in_edges(node_id, data=True, keys=True):
            # get statement
            s = sg.nodes[sid]['data']
            inedges = list(sg.in_edges(sid, data=True, keys=True))
            assert s.is_statement, f"Error in the semantic graph for column {column_index} (statement {sid})"
            assert len(inedges) == 1, f"Error in the semantic graph for column {column_index} (statement {sid})"
            uid, _, peid, pedata = inedges[0]

            freq = s.compute_freq(None, None, edata['data'], is_unique_freq=False)
            u = sg.nodes[uid]['data']
            if u.is_column:
                endpoint = u.column
            else:
                if u.qnode_id[0] == 'P':
                    # bug here
                    endpoint = {"uri": WDOnt.get_prop_uri(u.qnode_id), "label": self.wdont.get_pnode_label(u.qnode_id)}
                else:
                    endpoint = {"uri": WDOnt.get_qnode_uri(u.qnode_id), "label": self.wdont.get_qnode_label(u.qnode_id)}

            incomings.append(ColumnRelationship(
                endpoint=endpoint,
                predicates=(
                    {"uri": WDOnt.get_prop_uri(peid), "label": self.wdont.get_pnode_label(peid)},
                    {"uri": WDOnt.get_prop_uri(eid), "label": self.wdont.get_pnode_label(eid)},
                ),
                freq=freq
            ))

        outgoings = []
        for _, sid, peid, pedata in sg.out_edges(node_id, data=True, keys=True):
            # get statement
            s = sg.nodes[sid]['data']
            outedges = list(sg.out_edges(sid, data=True, keys=True))
            assert s.is_statement, f"Error in the semantic graph for column {column_index} (statement {sid})"

            for _, vid, eid, edata in outedges:
                freq = s.compute_freq(None, None, edata['data'], is_unique_freq=False)
                v = sg.nodes[vid]['data']
                if v.is_column:
                    endpoint = v.column
                else:
                    assert v.is_value
                    if v.is_entity_value:
                        if v.qnode_id[0] == 'P':
                            # bug here
                            endpoint = {"uri": WDOnt.get_prop_uri(v.qnode_id), "label": self.wdont.get_pnode_label(v.qnode_id)}
                        else:
                            endpoint = {"uri": WDOnt.get_qnode_uri(v.qnode_id), "label": self.wdont.get_qnode_label(v.qnode_id)}
                    else:
                        # not support yet
                        assert v.is_literal_value
                        continue

                outgoings.append(ColumnRelationship(
                    endpoint=endpoint,
                    predicates=(
                        {"uri": WDOnt.get_prop_uri(peid), "label": self.wdont.get_pnode_label(peid)},
                        {"uri": WDOnt.get_prop_uri(eid), "label": self.wdont.get_pnode_label(eid)},
                    ),
                    freq=freq
                ))

        return dict(incoming=incomings, outgoing=outgoings)


if __name__ == '__main__':
    from sm_unk.prelude import I, O, D, A, M
    from sm_unk.dev.wikitable2wikidata.prelude import *
    from sm_unk.config import *

    dataset_dir = HOME_DIR / "wikitable2wikidata/500tables"
    
    tables = [W2WTable.from_dbpedia_table(Table.deser_str(r)) for r in M.deserialize_lines(dataset_dir / "tables.jl.gz")]
    tables = {x.table.metadata.table_id: x for x in tables}
    tables = [tables[tbl_id] for tbl_id in M.deserialize_lines(dataset_dir / "predictions_order.txt", trim=True)]
    tbl_context = M.deserialize_json(dataset_dir / "context.json.gz")

    curated_model_index = [16]
    annotate_data = []
    assistant_data = []
    for i in curated_model_index:
        tbl = tables[i]
        fname = tbl.get_friendly_fs_id()
        context = tbl_context.get(tbl.id, [])
        
        description = f"<b>table.index</b> = {i}. " + " > ".join([f"<b>[h{h['level']}]</b> {h['header'].strip()}" for h in context])
        sg, dg = M.deserialize_pkl(dataset_dir / f"experiments/v_all/cache/init_sg/a{i:03d}_{fname}.pkl")[1:]
        annotate_data.append((fname, description, tbl))
        assistant_data.append({"table": tbl, "sg": sg, "dg": dg})
    
    assistant = GRAMSAnnotatorAssistant(assistant_data, {}, {})
    print(assistant.get_row_indices(tbl, 2, 'Q18239264', ('P39', 'P39')))
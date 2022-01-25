from collections import defaultdict
from typing import Dict, List, Mapping
from grams.algorithm.data_graph.dg_graph import DGGraph

import networkx as nx
import numpy as np

from grams.algorithm.data_graph import CellNode
from grams.algorithm.literal_matchers import TextParser
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode, CGGraph
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import QNode, WDProperty, WDQuantityPropertyStats


class TypeFeatureExtraction:
    Freq = "FrequencyOfType"
    FreqOverRow = "FreqOfTypeOverRow"

    def __init__(
        self,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        qnodes: Mapping[str, QNode],
        wdprops: Mapping[str, WDProperty],
        wd_num_prop_stats: Mapping[str, WDQuantityPropertyStats],
    ):
        self.table = table
        self.cg = cg
        self.dg = dg
        self.qnodes = qnodes
        self.wdprops = wdprops
        self.wd_num_prop_stats = wd_num_prop_stats
        self.text_parser = TextParser()

        # self.transitive_props = [p.id for p in self.wdprops.values() if p.is_transitive()]
        self.hierarchy_props = {"P131", "P276"}

    def extract_features(self):
        freq_type = {}
        freq_over_row = {}
        cell2qnodes = {}
        column2types = {}

        # for uid, udata in self.cg.nodes(data=True):
        for u in self.cg.iter_nodes():
            # u: SGColumnNode = udata["data"]
            if not isinstance(u, CGColumnNode):
                continue

            # cells in this column
            cells: List[CellNode] = [self.dg.get_cell_node(cid) for cid in u.nodes]
            covered_fractions = [
                sum(
                    span.length for spans in cell.qnodes_span.values() for span in spans
                )
                / max(len(cell.value), 1)
                for cell in cells
                if len(cell.qnode_ids) > 0
            ]
            if len(covered_fractions) == 0:
                continue
            avg_covered_fractions = np.mean(covered_fractions)
            if avg_covered_fractions < 0.8:
                continue

            for cell in cells:
                self.add_merge_qnodes(cell, cell2qnodes)

            type2freq = defaultdict(int)
            for cell in cells:
                classes = {}
                for qnode, prob in cell2qnodes[cell.id]:
                    for stmt in qnode.props.get("P31", []):
                        classes[stmt.value.as_entity_id()] = max(
                            prob, classes.get(stmt.value.as_entity_id(), 0)
                        )
                for c, prob in classes.items():
                    type2freq[c] += prob

            for c, freq in type2freq.items():
                freq_type[u.id, c] = freq
                freq_over_row[u.id, c] = freq / self.table.size()
            column2types[u.id] = list(type2freq.keys())

        return {
            self.Freq: freq_type,
            self.FreqOverRow: freq_over_row,
            "_column_to_types": column2types,
        }

    def add_merge_qnodes(self, cell: CellNode, cell2qnodes: Dict[str, List[QNode]]):
        # merge qnodes that are sub of each other
        # attempt to merge qnodes (spatial) if they are contained in each other
        # we should go even higher order
        assert cell.id not in cell2qnodes

        if len(cell.qnode_ids) > 1:
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            ignore_qnodes = set()
            for q0_id in cell.qnode_ids:
                q0 = self.qnodes[q0_id]
                vals = {
                    stmt.value.as_entity_id()
                    for p in self.hierarchy_props
                    for stmt in q0.props.get(p, [])
                }
                for q1_id in cell.qnode_ids:
                    if q0_id == q1_id:
                        continue
                    if q1_id in vals:
                        # q0 is inside q1, ignore q1
                        ignore_qnodes.add(q1_id)
            qnode_lst = [
                self.qnodes[q_id]
                for q_id in cell.qnode_ids
                if q_id not in ignore_qnodes
            ]
        elif len(cell.qnode_ids) > 0:
            qnode_lst = [self.qnodes[cell.qnode_ids[0]]]
        else:
            qnode_lst = []

        qnode2prob = {}
        for link in self.table.links[cell.row][cell.column]:
            for c in link.candidates:
                qnode2prob[c.entity_id] = max(
                    qnode2prob.get(c.entity_id, 0), c.probability
                )

        cell2qnodes[cell.id] = [(qnode, qnode2prob[qnode.id]) for qnode in qnode_lst]

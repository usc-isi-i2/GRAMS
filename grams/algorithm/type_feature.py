from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from grams.algorithm.data_graph import CellNode
from grams.algorithm.literal_match import TextParser
from grams.algorithm.semantic_graph import SGColumnNode
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import QNode, WDProperty, WDQuantityPropertyStats


class TypeFeatureExtraction:
    Freq = "FrequencyOfType"
    FreqOverRow = "FreqOfTypeOverRow"

    def __init__(self,
                 table: LinkedTable, sg: nx.MultiDiGraph, dg: nx.MultiDiGraph,
                 qnodes: Dict[str, QNode], wdprops: Dict[str, WDProperty],
                 wd_num_prop_stats: Dict[str, WDQuantityPropertyStats]):
        self.table = table
        self.sg = sg
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

        for uid, udata in self.sg.nodes(data=True):
            u: SGColumnNode = udata['data']
            if not u.is_column:
                continue

            # cells in this column
            cells: List[CellNode] = [self.dg.nodes[cid]['data'] for cid in u.nodes]
            covered_fractions = [
                sum(span.length for spans in cell.qnodes_span.values() for span in spans) / max(len(cell.value), 1)
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
                classes = set()
                for qnode in cell2qnodes[cell.id]:
                    for stmt in qnode.props.get("P31", []):
                        classes.add(stmt.value.as_qnode_id())
                for c in classes:
                    type2freq[c] += 1

            for c, freq in type2freq.items():
                freq_type[uid, c] = freq
                freq_over_row[uid, c] = freq / self.table.size()
            column2types[uid] = list(type2freq.keys())

        return {
            self.Freq: freq_type,
            self.FreqOverRow: freq_over_row,
            "_column_to_types": column2types
        }
    
    def add_merge_qnodes(self, cell: CellNode, cell2qnodes: Dict[str, List[QNode]]):
        # merge qnodes that are sub of each other
        # attempt to merge qnodes (spatial) if they are contained in each other
        # we should go even higher order
        if len(cell.qnode_ids) > 1:
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            ignore_qnodes = set()
            for q0_id in cell.qnode_ids:
                q0 = self.qnodes[q0_id]
                vals = {
                    stmt.value.as_qnode_id()
                    for p in self.hierarchy_props
                    for stmt in q0.props.get(p, [])
                }
                for q1_id in cell.qnode_ids:
                    if q0_id == q1_id:
                        continue
                    if q1_id in vals:
                        # q0 is inside q1, ignore q1
                        ignore_qnodes.add(q1_id)
            cell2qnodes[cell.id] = [self.qnodes[q_id] for q_id in cell.qnode_ids if q_id not in ignore_qnodes]
        elif len(cell.qnode_ids) > 0:
            cell2qnodes[cell.id] = [self.qnodes[cell.qnode_ids[0]]]
        else:
            cell2qnodes[cell.id] = []
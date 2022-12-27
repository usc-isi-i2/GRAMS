from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from grams.algorithm.data_graph.dg_graph import DGGraph
from kgdata.wikidata.deprecated.wdclass import WDClass

import networkx as nx
import numpy as np

from grams.algorithm.data_graph import CellNode
from grams.algorithm.literal_matchers import TextParser
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode, CGGraph
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDEntity, WDProperty, WDQuantityPropertyStats


class TypeFeatureExtraction:
    Freq = "TypeFreq"
    FreqOverRow = "TypeFreqOverRow"
    FreqInheritOverRow = "TypeFreqInheritOverRow"
    HeaderSimilarity = "TypeHeaderSimilarity"

    def __init__(
        self,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        qnodes: Mapping[str, WDEntity],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_num_prop_stats: Mapping[str, WDQuantityPropertyStats],
        text_parser: TextParser,
        sim_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.table = table
        self.cg = cg
        self.dg = dg
        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wd_num_prop_stats = wd_num_prop_stats
        self.text_parser = text_parser
        self.sim_fn = sim_fn

        # self.transitive_props = [p.id for p in self.wdprops.values() if p.is_transitive()]
        self.hierarchy_props = {"P131", "P276"}

    def extract_features(self):
        freq_type = {}
        freq_over_row = {}
        freq_inherit_over_row = {}
        cell2qnodes: Dict[str, List[Tuple[WDEntity, float]]] = {}
        column2types = {}
        header_sim_types = {}

        # for uid, udata in self.cg.nodes(data=True):
        for u in self.cg.iter_nodes():
            # u: SGColumnNode = udata["data"]
            if not isinstance(u, CGColumnNode):
                continue

            # cells in this column
            cells: List[CellNode] = [self.dg.get_cell_node(cid) for cid in u.nodes]
            covered_fractions = [
                sum(
                    span.length
                    for spans in cell.entity_spans.values()
                    for span in spans
                )
                / max(len(cell.value), 1)
                for cell in cells
                if len(cell.entity_ids) > 0
            ]
            if len(covered_fractions) == 0:
                continue
            avg_covered_fractions = np.mean(covered_fractions)
            if avg_covered_fractions < 0.8:
                continue

            for cell in cells:
                self.add_merge_qnodes(cell, cell2qnodes)

            # calculate type to frequency
            type2freq = defaultdict(float)
            inherit_type2freq = defaultdict(float)

            for cell in cells:
                classes = {}
                for qnode, prob in cell2qnodes[cell.id]:
                    for stmt in qnode.props.get("P31", []):
                        classes[stmt.value.as_entity_id()] = max(
                            prob, classes.get(stmt.value.as_entity_id(), 0)
                        )
                all_classes = self.retrieve_parents(classes, 1)
                for c, prob in classes.items():
                    type2freq[c] += prob
                for c, prob in all_classes.items():
                    inherit_type2freq[c] += prob

            for c, freq in type2freq.items():
                freq_type[u.id, c] = freq
                freq_over_row[u.id, c] = freq / self.table.size()
            for c, freq in inherit_type2freq.items():
                freq_inherit_over_row[u.id, c] = freq / self.table.size()
            column2types[u.id] = list(inherit_type2freq.keys())
            if len(u.label) > 0 and self.sim_fn is not None:
                for c, freq in inherit_type2freq.items():
                    header_sim_types[u.id, c] = self.sim_fn(
                        u.label, self.wdclasses[c].label
                    )
        return {
            self.Freq: freq_type,
            self.FreqOverRow: freq_over_row,
            self.FreqInheritOverRow: freq_inherit_over_row,
            self.HeaderSimilarity: header_sim_types,
            "_column_to_types": column2types,
        }

    def retrieve_parents(self, classes: Dict[str, float], distance: int):
        """Get parents of classes within distance from the node.
        Distance 1 is the parent, Distance 2 is the grandparent.
        Return the classes and their parents
        """
        if distance <= 0:
            return classes
        output = classes.copy()
        for klass, prob in classes.items():
            for parent_klass in self.wdclasses[klass].parents:
                output[parent_klass] = max(output.get(parent_klass, 0), prob)
        if distance > 1:
            return self.retrieve_parents(output, distance - 1)
        return output

    def add_merge_qnodes(
        self, cell: CellNode, cell2qnodes: Dict[str, List[Tuple[WDEntity, float]]]
    ):
        # merge qnodes that are sub of each other
        # attempt to merge qnodes (spatial) if they are contained in each other
        # we should go even higher order
        assert cell.id not in cell2qnodes

        if len(cell.entity_ids) > 1:
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            ignore_qnodes = set()
            for q0_id in cell.entity_ids:
                q0 = self.qnodes[q0_id]
                vals = {
                    stmt.value.as_entity_id()
                    for p in self.hierarchy_props
                    for stmt in q0.props.get(p, [])
                }
                for q1_id in cell.entity_ids:
                    if q0_id == q1_id:
                        continue
                    if q1_id in vals:
                        # q0 is inside q1, ignore q1
                        ignore_qnodes.add(q1_id)
            qnode_lst = [
                self.qnodes[q_id]
                for q_id in cell.entity_ids
                if q_id not in ignore_qnodes
            ]
        elif len(cell.entity_ids) > 0:
            qnode_lst = [self.qnodes[cell.entity_ids[0]]]
        else:
            qnode_lst = []

        qnode2prob = {}
        for link in self.table.links[cell.row][cell.column]:
            for c in link.candidates:
                qnode2prob[c.entity_id] = max(
                    qnode2prob.get(c.entity_id, 0), c.probability
                )

        cell2qnodes[cell.id] = [(qnode, qnode2prob[qnode.id]) for qnode in qnode_lst]

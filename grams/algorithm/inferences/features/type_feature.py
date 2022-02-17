from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.psl_lib import IDMap
from kgdata.wikidata.models.wdclass import WDClass
from sm.misc import CacheMethod
import networkx as nx
import numpy as np

from grams.algorithm.data_graph import CellNode
from grams.algorithm.literal_matchers import TextParser
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode, CGGraph
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import QNode, WDProperty, WDQuantityPropertyStats


class TypeFeaturesConfigs:
    EXTENDED_DISTANCE: int = 1


class TypeFeatures:
    INSTANCE_OF = "P31"
    HIERARCHY_PROPS = {"P131", "P276"}

    def __init__(
        self,
        idmap: IDMap,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        qnodes: Mapping[str, QNode],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_num_prop_stats: Mapping[str, WDQuantityPropertyStats],
        sim_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.idmap = idmap
        self.table = table
        self.cg = cg
        self.dg = dg
        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wd_num_prop_stats = wd_num_prop_stats

    def extract_features(self, features: List[str]) -> Dict[str, list]:
        # gather the list of entity columns (columns will be tagged with type)
        tagged_columns = []
        for u in self.cg.iter_nodes():
            if not isinstance(u, CGColumnNode):
                continue

            cells = self.get_cells(u)

            # using heuristic to determine if we should tag this column
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

            tagged_columns.append(u)

        # extract features
        feat_data = {}
        for feat in features:
            fn = getattr(self, feat)
            feat_data[feat] = [v for u in tagged_columns for v in fn(u)]
        return feat_data

    def EXTENDED_TYPE_FREQ_OVER_ROW(
        self, u: CGColumnNode
    ) -> List[Tuple[str, str, float]]:
        extended_type_freq = self.get_extended_type_freq(u)
        output = []
        n_rows = self.table.size()
        for c, freq in extended_type_freq.items():
            output.append((self.idmap.m(u.id), self.idmap.m(c), freq / n_rows))
        return output

    def HAS_SUB_TYPE(self, u: CGColumnNode):
        output = set()
        classes = set(self.get_extended_type_freq(u).keys())
        for c in classes:
            for pc in self.wdclasses[c].parents:
                if pc in classes:
                    output.add((self.idmap.m(u.id), self.idmap.m(pc)))
        return list(output)

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_type_freq(self, u: CGColumnNode) -> Dict[str, float]:
        """Calculating frequency of types in a column.
        Each time a type appears in a cell, instead of counting 1, we count its probability
        """
        cells = self.get_cells(u)
        cell2qnodes = self.get_cell_to_qnodes(u)

        type2freq = {}

        for cell in cells:
            classes = {}
            for qnode, prob in cell2qnodes[cell.id]:
                for stmt in qnode.props.get(self.INSTANCE_OF, []):
                    classes[stmt.value.as_entity_id()] = max(
                        prob, classes.get(stmt.value.as_entity_id(), 0)
                    )

            for c, prob in classes.items():
                if c not in type2freq:
                    type2freq[c] = 0.0
                type2freq[c] += prob

        return type2freq

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_extended_type_freq(self, u: CGColumnNode) -> Dict[str, float]:
        """Calculating frequency of types in a column.
        Each time a type appears in a cell, instead of counting 1, we count its probability
        """
        cells = self.get_cells(u)
        cell2qnodes = self.get_cell_to_qnodes(u)

        extended_type2freq = {}

        for cell in cells:
            classes = {}
            for qnode, prob in cell2qnodes[cell.id]:
                for stmt in qnode.props.get(self.INSTANCE_OF, []):
                    classes[stmt.value.as_entity_id()] = max(
                        prob, classes.get(stmt.value.as_entity_id(), 0)
                    )
            classes = self._get_extended_type_freq_of_cell(
                classes, TypeFeaturesConfigs.EXTENDED_DISTANCE
            )
            for c, prob in classes.items():
                if c not in extended_type2freq:
                    extended_type2freq[c] = 0.0
                extended_type2freq[c] += prob

        return extended_type2freq

    def _get_extended_type_freq_of_cell(self, classes: Dict[str, float], distance: int):
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
            return self._get_extended_type_freq_of_cell(output, distance - 1)
        return output

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_cells(self, u: CGColumnNode) -> List[CellNode]:
        """Get cells of a column node"""
        return [self.dg.get_cell_node(cid) for cid in u.nodes]

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_cell_to_qnodes(
        self, u: CGColumnNode
    ) -> Dict[str, List[Tuple[QNode, float]]]:
        """Get a mapping to associated qnodes of a cell"""
        cells = self.get_cells(u)
        cell2qnodes = {}
        for cell in cells:
            self._add_merge_qnodes(cell, cell2qnodes)
        return cell2qnodes

    def _add_merge_qnodes(
        self, cell: CellNode, cell2qnodes: Dict[str, List[Tuple[QNode, float]]]
    ):
        """merge qnodes that are sub of each other
        attempt to merge qnodes (spatial) if they are contained in each other
        we should go even higher order
        """
        assert cell.id not in cell2qnodes
        if len(cell.qnode_ids) > 1:
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            ignore_qnodes = set()
            for q0_id in cell.qnode_ids:
                q0 = self.qnodes[q0_id]
                vals = {
                    stmt.value.as_entity_id()
                    for p in self.HIERARCHY_PROPS
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

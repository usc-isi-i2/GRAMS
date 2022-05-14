from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode, CGGraph
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.inputs.linked_table import LinkedTable
from graph.retworkx.api import digraph_all_simple_paths

import sm.misc as M
import networkx as nx


class KnownTargets:
    """Post-processing that selects relationships and types of given columns"""

    instance = None

    def __init__(
        self,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        edge_probs: Dict[Tuple[str, str, str], float],
        cta_probs: Dict[int, Dict[str, float]],
        threshold: float,
        cpa_targets: Dict[str, List[Tuple[int, int]]],
        cta_targets: Dict[str, List[int]],
    ):
        self.table = table
        self.cg = cg
        self.dg = dg
        self.edge_probs = edge_probs
        self.cta_probs = cta_probs
        self.threshold = threshold

        self.cpa_targets = cpa_targets
        self.cta_targets = cta_targets

    def get_result(self):
        column2node = {
            u.column: u.id for u in self.cg.iter_nodes() if isinstance(u, CGColumnNode)
        }

        selected_edges = {}

        edge_probs = {e: p for e, p in self.edge_probs.items() if p >= self.threshold}
        for sci, tci in self.cpa_targets.get(self.table.id, []):
            uid = column2node[sci]
            vid = column2node[tci]

            if not self.cg.has_node(uid) or not self.cg.has_node(vid):
                continue

            paths = digraph_all_simple_paths(self.cg, uid, vid, cutoff=2)
            paths = [
                p
                for p in paths
                if (p[0].source, p[0].target, p[0].predicate) in edge_probs
                and (p[1].source, p[1].target, p[1].predicate) in edge_probs
            ]
            if len(paths) == 0:
                continue

            e1, e2 = max(
                paths,
                key=lambda p: edge_probs[p[0].source, p[0].target, p[0].predicate]
                + edge_probs[p[1].source, p[1].target, p[1].predicate],
            )

            for e in [e1, e2]:
                triple = e.source, e.target, e.predicate
                selected_edges[triple] = self.edge_probs[triple]

        pred_cpa = self.cg.subgraph_from_edge_triples(selected_edges.keys())
        pred_cta = {
            ci: max(self.cta_probs[ci].items(), key=itemgetter(1))[0]
            for ci in self.cta_targets.get(self.table.id, [])
            if ci in self.cta_probs
        }

        return pred_cpa, pred_cta

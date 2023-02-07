from grams.algorithm.helpers import K
from grams.algorithm.postprocessing.common import add_context, ensure_valid_statements
from grams.algorithm.postprocessing.config import PostprocessingConfig
from graph.interface import BaseNode
import networkx as nx
from typing import Dict, List, Optional, Tuple
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEdgeTriple,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.psl_lib import PSLModel
from grams.inputs.linked_table import LinkedTable


class MinimumArborescence:
    def __init__(
        self,
        table: LinkedTable,
        cg: CGGraph,
        edge_probs: Dict[CGEdgeTriple, float],
        threshold: float,
    ):
        self.table = table
        self.cg = cg
        self.edge_probs = edge_probs
        self.threshold = threshold

    def add_pseudo_root(self, cg: CGGraph, pseudo_root_id: str = "__root__") -> CGGraph:
        """Add a pseudo root node and edges from non-statement nodes to the pseudo root
        so that we can always have a directed rooted tree

        Args:
            cg: the graph to impute
            pseudo_root_id: the id of the pseudo root node
        """
        newcg = cg.copy()
        assert newcg.has_node(pseudo_root_id) is False

        newcg.add_node(
            CGColumnNode(id=pseudo_root_id, label="pseudo root", column=-1, nodes=set())
        )

        for node in cg.iter_nodes():
            if isinstance(node, CGStatementNode):
                continue

            newcg.add_edge(
                CGEdge(pseudo_root_id, node.id, predicate="null", features={})
            )

        return newcg

    def get_result(self) -> CGGraph:
        """Select edges that forms a tree"""
        edge_probs = {e: p for e, p in self.edge_probs.items() if p >= self.threshold}
        subcg = self.cg.subgraph_from_edge_triples(edge_probs.keys())
        subcg.remove_dangling_statement()
        subcg.remove_standalone_nodes()

        if subcg.num_edges() == 0:
            return subcg

        # for each node that is not a statement, add a common root node and edges to it
        pseudo_root_id = "__root__"
        new_subcg = self.add_pseudo_root(subcg, pseudo_root_id)

        # find a minimum arborescence
        nxcg = nx.MultiDiGraph()
        edge_weights = {e: 1.0 / p for e, p in edge_probs.items()}  # p > 0.5
        total_weight = sum(edge_weights.values())

        for edge in new_subcg.out_edges(pseudo_root_id):
            edge_weights[edge.source, edge.target, edge.predicate] = total_weight + 1

        for edge in new_subcg.iter_edges():
            nxcg.add_edge(
                edge.source,
                edge.target,
                key=edge.key,
                weight=edge_weights[edge.source, edge.target, edge.predicate],
                edgekey=edge.key,
            )

        try:
            nxtree = nx.algorithms.tree.branchings.minimum_spanning_arborescence(
                nxcg, attr="weight", preserve_attrs=True
            )
        except nx.NetworkXException as e:
            assert str(e) == "No minimum spanning arborescence in G.", str(e)
            raise NoMinimumArborescenceException() from e

        # remove the pseudo root node
        nxtree.remove_node(pseudo_root_id)

        # convert tree to nxtree
        tree_edges = [
            (uid, vid, ekey) for uid, vid, ekey in nxtree.edges(data="edgekey")
        ]
        tree = subcg.subgraph_from_edge_triples(tree_edges)
        tree.remove_dangling_statement()

        clean_entities(tree)

        if PostprocessingConfig.INCLUDE_CONTEXT:
            add_context(subcg, tree, edge_probs)

        ensure_valid_statements(subcg, tree, create_if_not_exists=True)

        # fmt: off
        # from graph.viz.graphviz import draw
        # draw(graph=tree, filename="/tmp/graphviz/a204.png", **CGGraph.graphviz_props())
        # draw(graph=tree, filename="/tmp/graphviz/g25.png", **CGGraph.graphviz_props())
        # fmt: on
        return tree


class NoMinimumArborescenceException(Exception):
    pass


def clean_entities(tree: CGGraph):
    for u in tree.nodes():
        if not isinstance(u, (CGEntityValueNode, CGLiteralValueNode)):
            continue

        if u.is_in_context:
            continue

        if tree.in_degree(u.id) == 0:
            if tree.out_degree(u.id) == 1:
                tree.remove_node(u.id)
            continue

        (svedge,) = tree.in_edges(u.id)
        stmt = tree.get_node(svedge.source)
        assert isinstance(stmt, CGStatementNode)

        (usedge,) = tree.in_edges(stmt.id)
        if usedge.predicate != svedge.predicate:
            # qualifier
            tree.remove_node(u.id)
            continue

        # property
        if tree.out_degree(stmt.id) == 1:
            tree.remove_node(u.id)
            tree.remove_node(stmt.id)

    tree.remove_dangling_statement()

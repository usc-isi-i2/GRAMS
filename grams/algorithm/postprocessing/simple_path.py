from typing import Dict, Optional, Callable, List, Tuple
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEdgeTriple,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from itertools import chain, combinations
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.inputs.linked_table import LinkedTable
from operator import itemgetter
from graph.retworkx.api import digraph_all_simple_paths


class PostProcessingSimplePath:
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

    def get_result(self) -> CGGraph:
        """Select a simple path between nodes."""
        selected_edges = [e for e, p in self.edge_probs.items() if p >= self.threshold]
        subcg = self.cg.subgraph_from_edge_triples(selected_edges)
        subcg.remove_dangling_statement()
        if subcg.num_edges() == 0:
            # empty graph
            return subcg

        # ranking, prefer the shorter path, when there are multiple shorter path, select the one with the higher probs.
        return self.keep_one_simple_path_between_important_nodes(
            subcg, self.select_shorter_path, both_direction=True
        )

    def select_shorter_path(self, paths: List[List[CGEdgeTriple]]):
        """Prefer shorter path. When there are multiple shorter path, select the one with higher prob"""
        paths = sorted(paths, key=len)
        paths = [path for path in paths if len(path) == len(paths[0])]
        if len(paths) == 1:
            return paths[0]

        # multiple shorter paths
        path_probs = []
        for path in paths:
            path_prob = 0
            for uid, vid, eid in path:
                # edge = self.cg.get_edge_between_nodes(uid, vid, eid)
                path_prob += self.edge_probs[uid, vid, eid]
            path_probs.append(path_prob)
        path, path_prob = max(zip(paths, path_probs), key=itemgetter(1))
        return path

    def keep_one_simple_path_between_important_nodes(
        self,
        cg: CGGraph,
        ranking_fn: Optional[
            Callable[[List[List[CGEdgeTriple]]], List[CGEdgeTriple]]
        ] = None,
        both_direction: bool = True,
    ):
        """
        Let important nodes be columns and context values. If there is more than one path between the important nodes,
        we only keep one. Note that this function doesn't remove a literal value if it is value of a property of a statement
        that appear in the chosen path.

        The default ranking function is select the path with shorter length. Supply your own function for better result.
        The both direction function will select the correct one
        """
        # figure out the important nodes
        important_nodes = set()
        for u in cg.nodes():
            if isinstance(u, CGColumnNode) or (
                isinstance(u, (CGEntityValueNode, CGLiteralValueNode))
                and u.is_in_context
            ):
                important_nodes.add(u.id)

        if ranking_fn is None:
            ranking_fn = lambda paths: sorted(paths, key=lambda path: len(path))[0]

        # select one path if there are multiple edges
        selected_edges = set()
        for uid, vid in combinations(important_nodes, 2):
            forward_paths = [
                [(e.source, e.target, e.predicate) for e in path]
                for path in digraph_all_simple_paths(cg, uid, vid, cutoff=6)
                if not any(edge.source in important_nodes for edge in path[1:])
            ]
            backward_paths = [
                [(e.source, e.target, e.predicate) for e in path]
                for path in digraph_all_simple_paths(cg, vid, uid, cutoff=6)
                if not any(edge.source in important_nodes for edge in path[1:])
            ]
            if both_direction:
                if len(forward_paths) + len(backward_paths) > 0:
                    selected_edges = selected_edges.union(
                        ranking_fn(forward_paths + backward_paths)
                    )
            else:
                if len(forward_paths) > 0:
                    selected_edges = selected_edges.union(ranking_fn(forward_paths))
                if len(backward_paths) > 0:
                    selected_edges = selected_edges.union(ranking_fn(backward_paths))

        # new_sg = SemanticGraphConstructor.get_sg_subgraph(sg, selected_edges)
        new_cg = cg.subgraph_from_edge_triples(selected_edges)

        # add missing statement values
        for stmt in new_cg.nodes():
            if not isinstance(stmt, CGStatementNode):
                continue

            # ((uid, _, eid),) = new_cg.in_edges(stmt.id)
            (us_edge,) = new_cg.in_edges(stmt.id)
            if not any(
                sv_edge.predicate == us_edge.predicate
                for sv_edge in new_cg.out_edges(stmt.id)
            ):
                # we do not have the statement property in the graph. get it from the original graph
                (sv_edge,) = [
                    e for e in cg.out_edges(stmt.id) if e.predicate == us_edge.predicate
                ]
                if not new_cg.has_node(sv_edge.target):
                    new_cg.add_node(cg.get_node(sv_edge.target).clone())
                new_cg.add_edge(sv_edge.clone())

        return new_cg

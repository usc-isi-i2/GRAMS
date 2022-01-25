from typing import Optional, Callable, List, Tuple
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from itertools import chain, combinations

from graph.retworkx.api import digraph_all_simple_paths


def keep_one_simple_path_between_important_nodes(
    cg: CGGraph,
    ranking_fn: Optional[
        Callable[[List[List[Tuple[str, str, str]]]], List[Tuple[str, str, str]]]
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
            isinstance(u, (CGEntityValueNode, CGLiteralValueNode)) and u.is_in_context
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

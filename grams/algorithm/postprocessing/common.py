from typing import Dict, List, Tuple, TypedDict
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from operator import itemgetter


_AddContextPathType = TypedDict(
    "_Path", {"path": Tuple[CGEdge, CGEdge], "score": float}
)


def ensure_valid_statements(
    original_cg: CGGraph,
    pruned_cg: CGGraph,
    create_if_not_exists: bool = False,
    safe_creation: bool = True,
):
    """Ensure that the pruned graph has valid statements, which have statement property.

    NOTE: this function may modify the pruned graph.

    Args:
        original_cg: the original candidate graph
        pruned_cg: the pruned candidate graph -- must have the same id as the original cg
        create_if_not_exists: if True, create a new target of statement property if it does not exist
        safe_creation: if True, we check to make sure the new statement target is not in the graph
    """
    for s in pruned_cg.nodes():
        if not isinstance(s, CGStatementNode):
            continue

        (inedge,) = pruned_cg.in_edges(s.id)
        for outedge in original_cg.out_edges(s.id):
            if outedge.predicate == inedge.predicate:
                # there is a statement prop in the original candidate graph
                # if it is not in the tree, we need to add it back
                if pruned_cg.has_edge_between_nodes(
                    outedge.source, outedge.target, outedge.predicate
                ):
                    continue

                if not create_if_not_exists:
                    raise Exception(
                        f"The property of the statement {s.id} is not in the graph"
                    )

                if safe_creation:
                    assert not pruned_cg.has_node(
                        outedge.target
                    ), "The target should not in the grpah as we do not have the link"
                    target = original_cg.get_node(outedge.target).clone()
                    assert isinstance(
                        target, (CGEntityValueNode, CGLiteralValueNode)
                    ), "The only reason why we don't have statement value is it is an entity/literal"
                    pruned_cg.add_node(target)
                elif not pruned_cg.has_node(outedge.target):
                    target = original_cg.get_node(outedge.target).clone()
                    pruned_cg.add_node(target)

                pruned_cg.add_edge(outedge.clone())


def add_context(
    origin_cg: CGGraph,
    pruned_cg: CGGraph,
    edge_probs: Dict[Tuple[str, str, str], float],
):
    """Add back the context node or entity that are appeared in the original graph but not in the pruned graph.

    Note: this function may modify the pruned graph.

    Args:
        origin_cg: the original candidate graph
        pruned_cg: the pruned candidate graph -- must have the same id as the original cg
        edge_probs: containing edges' probabilities
    """
    for v in origin_cg.iter_nodes():
        if (
            not isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
            or not v.is_in_context
            or pruned_cg.has_node(v.id)
        ):
            continue

        # find the paths that connect the vid to the tree and select the one with highest score and do not create cycle
        paths: List[_AddContextPathType] = []
        for sv_edge in origin_cg.in_edges(v.id):
            if sv_edge.predicate == "P31":  # ignore instanceof
                continue
            for us_edge in origin_cg.in_edges(sv_edge.source):
                if not pruned_cg.has_node(us_edge.source):
                    continue
                paths.append(
                    {
                        "path": (
                            us_edge,
                            sv_edge,
                        ),
                        "score": edge_probs[
                            (us_edge.source, us_edge.target, us_edge.predicate)
                        ]
                        + edge_probs[
                            (sv_edge.source, sv_edge.target, sv_edge.predicate)
                        ],
                    }
                )

        paths = sorted(paths, key=itemgetter("score"), reverse=True)

        if len(paths) == 0:
            continue

        us_edge, sv_edge = paths[0]["path"]
        pruned_cg.add_node(v.clone())
        if not pruned_cg.has_node(sv_edge.source):
            s = origin_cg.get_node(sv_edge.source)
            pruned_cg.add_node(s.clone())
        assert not pruned_cg.has_edge_between_nodes(
            sv_edge.source, sv_edge.target, sv_edge.predicate
        )
        pruned_cg.add_edge(sv_edge.clone())

        if not pruned_cg.has_edge_between_nodes(
            us_edge.source, us_edge.target, us_edge.predicate
        ):
            pruned_cg.add_edge(us_edge.clone())

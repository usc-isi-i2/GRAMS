from typing import Dict, Optional, Callable, List, Tuple, TypedDict, cast
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
from grams.algorithm.postprocessing.simple_path import PostProcessingSimplePath
from grams.algorithm.psl_solver import PSLConfigs
from grams.inputs.linked_table import LinkedTable
from operator import itemgetter
from graph.interface import EdgeTriple
from graph.retworkx.api import dag_longest_path, digraph_all_simple_paths
from steiner_tree.bank import BankSolver
from steiner_tree.bank.struct import NoSingleRootException, Solution


_AddContextPathType = TypedDict(
    "_Path", {"path": Tuple[CGEdge, CGEdge], "score": float}
)


class PostProcessingSteinerTree:
    def __init__(
        self,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        edge_probs: Dict[CGEdgeTriple, float],
        threshold: float,
    ):
        self.table = table
        self.cg = cg
        self.dg = dg
        self.edge_probs = edge_probs
        self.threshold = threshold

    def get_result(self) -> CGGraph:
        """Select edges that forms a tree"""
        # first step is to remove
        selected_edges = [e for e, p in self.edge_probs.items() if p >= self.threshold]
        subcg = self.cg.subgraph_from_edge_triples(selected_edges)
        subcg.remove_dangling_statement()
        subcg.remove_standalone_nodes()

        # terminal nodes are columns node, entity nodes are adding later
        terminal_nodes = {
            u.id for u in subcg.iter_nodes() if isinstance(u, CGColumnNode)
        }
        if len(terminal_nodes) == 0:
            # if the tree does not have any column, we essentially don't predict anything
            # so we return an empty graph
            return CGGraph()

        norm_edge_probs = self.normalize_probs(eps=0.001)
        solver = BankSolver(
            original_graph=subcg,
            terminal_nodes=terminal_nodes,
            top_k_st=50,
            top_k_path=50,
            weight_fn=lambda e: 1.0
            / max(
                1e-7,
                norm_edge_probs[e.source, e.target, e.predicate],
            ),
            solution_cmp_fn=self.compare_solutions,
        )

        try:
            trees, _solutions = solver.run()
        except NoSingleRootException:
            # we don't have any solutions from bank
            trees = []

        if len(trees) == 0:
            # all back to use simple path
            pp = PostProcessingSimplePath(
                self.table, self.cg, self.dg, self.edge_probs, self.threshold
            )
            return pp.get_result()

        tree = cast(CGGraph, trees[0])
        tree.remove_dangling_statement()
        # add back statement property if missing into to ensure a correct model
        for s in tree.nodes():
            if not isinstance(s, CGStatementNode):
                continue
            (in_edge,) = tree.in_edges(s.id)
            for out_edge in subcg.out_edges(s.id):
                if out_edge.predicate == in_edge.predicate:
                    # there is a statement prop in the sub candidate graph
                    # if it's not in the tree, we need to add it back
                    if not tree.has_node(out_edge.target):
                        ent = subcg.get_node(out_edge.target)
                        assert isinstance(
                            ent, (CGEntityValueNode, CGLiteralValueNode)
                        ), f"The only reason why we don't have statement value is it is an literal"
                        tree.add_node(ent.clone())
                        tree.add_edge(out_edge.clone())
                elif not tree.has_edge_between_nodes(
                    out_edge.source, out_edge.target, out_edge.predicate
                ):
                    v = subcg.get_node(out_edge.target)
                    if (
                        isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
                        and v.is_in_context
                    ):
                        # we should have this as PSL think it's correct
                        # only add context nodes
                        if not tree.has_node(v.id):
                            tree.add_node(v.clone())
                        tree.add_edge(out_edge.clone())
        if PSLConfigs.POSTPROCESSING_STEINER_TREE_FORCE_ADDING_CONTEXT:
            # add back the context node or entity that are appeared in the psl results but not in the predicted tree
            for v in subcg.iter_nodes():
                if (
                    not isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
                    or not v.is_in_context
                    or tree.has_node(v.id)
                ):
                    continue

                # find the paths that connect the vid to the tree and select the one with highest score and do not create cycle
                paths: List[_AddContextPathType] = []
                for sv_edge in subcg.in_edges(v.id):
                    if sv_edge.predicate == "P31":
                        continue
                    for us_edge in subcg.in_edges(sv_edge.source):
                        if not tree.has_node(us_edge.source):
                            continue
                        paths.append(
                            {
                                "path": (
                                    us_edge,
                                    sv_edge,
                                ),
                                "score": norm_edge_probs[
                                    (us_edge.source, us_edge.target, us_edge.predicate)
                                ]
                                + norm_edge_probs[
                                    (sv_edge.source, sv_edge.target, sv_edge.predicate)
                                ],
                            }
                        )

                paths = sorted(paths, key=itemgetter("score"), reverse=True)
                # TODO: filter out the path that will create cycle

                if len(paths) == 0:
                    continue

                us_edge, sv_edge = paths[0]["path"]
                tree.add_node(v.clone())
                if not tree.has_node(sv_edge.source):
                    s = subcg.get_node(sv_edge.source)
                    tree.add_node(s.clone())
                assert not tree.has_edge_between_nodes(
                    sv_edge.source, sv_edge.target, sv_edge.predicate
                )
                tree.add_edge(sv_edge.clone())

                if not tree.has_edge_between_nodes(
                    us_edge.source, us_edge.target, us_edge.predicate
                ):
                    tree.add_edge(us_edge.clone())
        return tree

    def compare_solutions(self, a: Solution, b: Solution) -> int:
        """Comparing two solutions, -1 (smaller) means a better solution -- we are solving minimum steiner tree"""
        a_weight = a.weight / max(a.num_edges, 1)
        b_weight = b.weight / max(b.num_edges, 1)

        if a_weight < b_weight:
            return -1
        if a_weight > b_weight:
            return 1
        # equal weight, prefer the one with shorter depth
        if not hasattr(a, "depth"):
            setattr(a, "depth", len(dag_longest_path(a.graph)))
        if not hasattr(b, "depth"):
            setattr(b, "depth", len(dag_longest_path(b.graph)))
        return getattr(a, "depth") - getattr(b, "depth")

    def normalize_probs(self, eps: float = 0.001):
        """The prob. score can be noisy (e.g., from PSL), i.e., equal edge may have
        slightly different scores. This function groups values that are close
        within the range [-eps, +eps] together, and replace them with the average value
        """
        norm_edge_probs = {}
        lst = sorted(
            [x for x in self.edge_probs.items() if x[1] >= self.threshold],
            key=itemgetter(1),
        )
        clusters = []
        pivot = 1
        clusters = [[lst[0]]]
        while pivot < len(lst):
            x = lst[pivot - 1][1]
            y = lst[pivot][1]
            if (y - x) <= eps:
                # same clusters
                clusters[-1].append(lst[pivot])
            else:
                # different clusters
                clusters.append([lst[pivot]])
            pivot += 1
        for cluster in clusters:
            avg_prob = sum([x[1] for x in cluster]) / len(cluster)
            for k, _prob in cluster:
                norm_edge_probs[k] = avg_prob
        return norm_edge_probs

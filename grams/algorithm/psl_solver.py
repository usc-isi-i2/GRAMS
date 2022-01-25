import copy
import math
import os
import time
import uuid
from functools import cmp_to_key
from itertools import chain
from multiprocessing import Pool
from operator import attrgetter, itemgetter, xor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from grams.algorithm.postprocessing.simple_path import (
    keep_one_simple_path_between_important_nodes,
)
from graph.retworkx.api import dag_longest_path

import networkx as nx
import sm.misc as M
from steiner_tree.bank import (
    NoSingleRootException,
    Solution,
    BankSolver,
)
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import DGGraph, DGNode
from grams.algorithm.link_feature import LinkFeatureExtraction
from grams.algorithm.semtab2020 import SemTab2020PostProcessing
from grams.algorithm.type_feature import TypeFeatureExtraction
from grams.inputs.linked_table import LinkedTable
from hugedict.parallel.parallel import Parallel
from kgdata.wikidata.models import QNode, WDClass, WDProperty, WDQuantityPropertyStats
from loguru import logger
from networkx.exception import NetworkXUnfeasible
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from tqdm.auto import tqdm

global_objects = {}


class IDMap:
    def __init__(self, counter: int = 0):
        self.counter = counter
        self.map = {}
        self.invert_map = {}

    def add_keys(self, keys: Iterable[Any]):
        for key in keys:
            new_key = f"i-{self.counter}"
            self.map[key] = new_key
            self.invert_map[new_key] = key
            self.counter += 1
        return self

    def m(self, key):
        """Get a new key from old key"""
        return self.map[key]

    def im(self, new_key):
        """Get the old key from the new key"""
        return self.invert_map[new_key]


class FakeIDMap(IDMap):
    def m(self, key):
        return key

    def im(self, new_key):
        return new_key


class PSLConfigs:
    POSTPROCESSING_METHOD = "steiner_tree"
    POSTPROCESSING_STEINER_TREE_FORCE_ADDING_CONTEXT: bool = True


PSLRunParallelArgs = TypedDict(
    "PSL.RunParallelArgs",
    table=LinkedTable,
    datagraph=DGGraph,
    semanticgraph=CGGraph,
)
_AddContextPathType = TypedDict(
    "_Path", {"path": Tuple[CGEdge, CGEdge], "score": float}
)


class PSLSteinerTreeSolver:
    LinkNegPrior = "NegPrior"
    LinkNegParentPropPrior = "NegParentPropPrior"
    CascadingError = "CascadingError"
    FreqLinkOverRow = LinkFeatureExtraction.FreqOverRow
    FreqLinkOverEntRow = LinkFeatureExtraction.FreqOverEntRow
    FreqLinkOverPosLink = LinkFeatureExtraction.FreqOverPossibleLink
    FreqLinkUnmatchOverEntRow = LinkFeatureExtraction.FreqUnmatchOverEntRow
    FreqLinkUnmatchOverPossibleLink = LinkFeatureExtraction.FreqUnmatchOverPossibleLink
    LinkDataTypeMismatch = LinkFeatureExtraction.DataTypeMismatch
    LinkHeaderSimilarity = LinkFeatureExtraction.HeaderSimilarity
    LinkNotFuncDep = LinkFeatureExtraction.NotFuncDep

    TypeNegPrior = "TypeNegPrior"
    FreqTypeOverRow = TypeFeatureExtraction.FreqOverRow
    TypeMustInPropRange = "TypeMustInPropRange"
    TypeOnlyOneConstraint = "TypeOnlyOneConstraint"

    def __init__(
        self,
        qnodes: Mapping[str, QNode],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_numprop_stats: Mapping[str, WDQuantityPropertyStats],
        disable_rules: Set[str] = None,
        sim_fn: Optional[Callable[[str, str], float]] = None,
        cache_dir: Optional[str] = None,
        postprocessing_method: str = None,
        enable_logging: bool = False,
    ):
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wd_numprop_stats = wd_numprop_stats
        self.qnodes = qnodes
        self.temp_dir = f"/tmp/psl-python-{str(uuid.uuid4()).replace('-', '')}"
        self.sim_fn = sim_fn
        # use this cache dir to catch the extraction result
        self.cache_dir = Path(cache_dir) if cache_dir is not None else cache_dir
        self.postprocessing_method = postprocessing_method
        assert self.postprocessing_method in {
            "select_simplepath",
            "steiner_tree",
            "external:semtab2020",
        }, self.postprocessing_method
        # if self.postprocessing_method == 'external:semtab2020':
        #     self.postprocessing_fn = SemTab2020PostProcessing()

        # blacklist some rules if they do not apply to particular domain (synthetic dataset)
        all_rules = {
            self.LinkNegPrior,
            self.TypeNegPrior,
            self.LinkNegParentPropPrior,
            self.FreqLinkOverRow,
            self.FreqLinkOverEntRow,
            self.FreqLinkOverPosLink,
            self.FreqLinkUnmatchOverEntRow,
            self.FreqLinkUnmatchOverPossibleLink,
            self.LinkHeaderSimilarity,
            self.LinkDataTypeMismatch,
            self.FreqTypeOverRow,
            self.LinkNotFuncDep,
        }
        self.disable_rules = set(disable_rules) if disable_rules is not None else set()
        assert all(r in all_rules for r in self.disable_rules)

        self.enable_logging = enable_logging

        # load the model
        self.get_model()

    def get_model(self):
        """Construct the PSL model"""
        model = Model("st-solver")

        # ##################################################################
        # add predicates
        self.link_pos_feats = [
            r
            for r in [
                self.FreqLinkOverRow,
                self.FreqLinkOverEntRow,
                self.FreqLinkOverPosLink,
                self.LinkHeaderSimilarity,
            ]
            if r not in self.disable_rules
        ]
        self.link_neg_feats = [
            r
            for r in [
                self.FreqLinkUnmatchOverEntRow,
                self.FreqLinkUnmatchOverPossibleLink,
                self.LinkDataTypeMismatch,
            ]
            if r not in self.disable_rules
        ]
        self.link_structure_feats = [
            r for r in [self.LinkNotFuncDep] if r not in self.disable_rules
        ]
        self.link_all_feats = (
            self.link_pos_feats + self.link_neg_feats + self.link_structure_feats
        )
        self.type_pos_feats = [
            r for r in [self.FreqTypeOverRow] if r not in self.disable_rules
        ]
        self.type_all_feats = self.type_pos_feats

        model.add_predicate(Predicate("CanRel", closed=True, size=3))
        model.add_predicate(Predicate("CanType", closed=True, size=2))
        model.add_predicate(Predicate("Rel", closed=False, size=3))
        model.add_predicate(Predicate("Type", closed=False, size=2))
        model.add_predicate(Predicate("SubProp", closed=True, size=2))
        model.add_predicate(Predicate("NotRange", closed=True, size=2))
        model.add_predicate(Predicate("NotStatement", closed=True, size=1))
        model.add_predicate(Predicate("Statement", closed=True, size=1))

        for feat in self.link_all_feats:
            model.add_predicate(Predicate(f"RelFeature_{feat}", closed=True, size=3))
        for feat in self.type_all_feats:
            model.add_predicate(Predicate(f"TypeFeature_{feat}", closed=True, size=2))

        # ##################################################################
        # add rules
        feat_weights = {
            self.LinkNegPrior: 1,
            self.TypeNegPrior: 1,
            self.LinkNegParentPropPrior: 0.1,
            self.CascadingError: 2,
            self.TypeMustInPropRange: 2,
            self.FreqLinkOverRow: 2,
            self.FreqLinkOverEntRow: 2,
            self.FreqLinkOverPosLink: 2,
            self.FreqLinkUnmatchOverEntRow: 2,
            self.FreqLinkUnmatchOverPossibleLink: 2,
            self.LinkHeaderSimilarity: 2,
            self.LinkDataTypeMismatch: 2,
            self.FreqTypeOverRow: 2,
            self.LinkNotFuncDep: 100,
        }
        self.rules = {}

        # rules apply for only link
        for feat in self.link_pos_feats:
            self.rules[feat] = Rule(
                f"CanRel(N1, N2, P) & RelFeature_{feat}(N1, N2, P) -> Rel(N1, N2, P)",
                weighted=True,
                squared=True,
                weight=feat_weights[feat],
            )
        for feat in self.link_neg_feats:
            self.rules[feat] = Rule(
                f"CanRel(N1, N2, P) & RelFeature_{feat}(N1, N2, P) -> ~Rel(N1, N2, P)",
                weighted=True,
                weight=feat_weights[feat],
                squared=True,
            )

        self.rules[self.LinkNotFuncDep] = Rule(
            f"CanRel(N1, N2, P) & CanRel(N2, N3, P2) & NotStatement(N2) & Rel(N1, N2, P) & RelFeature_{self.LinkNotFuncDep}(N2, N3, P2) -> ~Rel(N2, N3, P2)",
            weighted=True,
            weight=feat_weights[self.LinkNotFuncDep],
            squared=True,
        )

        # give a small negative weight to the parent property if there is a more specific child property
        # some how use CanRel is a much better option than Rel, perhaps this reflect how the parameters is set
        self.rules[self.LinkNegParentPropPrior] = [
            Rule(
                f"CanRel(N1, S, P) & Statement(S) & CanRel(S, N2, P) & CanRel(N1, S2, PP) & Statement(S2) & CanRel(S2, N2, PP) & SubProp(P, PP) -> ~Rel(N1, S2, PP)",
                weighted=True,
                weight=feat_weights[self.LinkNegParentPropPrior],
                squared=True,
            ),
            Rule(
                f"CanRel(N1, S, P) & Statement(S) & CanRel(S, N2, P) & CanRel(N1, S2, PP) & Statement(S2) & CanRel(S2, N2, PP) & SubProp(P, PP) -> ~Rel(S2, N2, PP)",
                weighted=True,
                weight=feat_weights[self.LinkNegParentPropPrior],
                squared=True,
            ),
        ]

        # default negative prior
        self.rules[self.LinkNegPrior] = Rule(
            "~Rel(N1, N2, P)",
            weighted=True,
            weight=feat_weights[self.LinkNegPrior],
            squared=True,
        )
        self.rules[self.TypeNegPrior] = Rule(
            "~Type(N, P)",
            weighted=True,
            weight=feat_weights[self.TypeNegPrior],
            squared=True,
        )
        self.rules[self.CascadingError] = [
            Rule(
                f"CanRel(N0, S, P) & Statement(S) & CanRel(S, N1, P) & CanRel(S, N2, Q) & N1 != N2 & ~Rel(S, N1, P) -> ~Rel(S, N2, Q)",
                weighted=True,
                weight=feat_weights[self.CascadingError],
                squared=True,
            ),
            Rule(
                f"CanRel(N0, S, P) & Statement(S) & CanRel(S, N1, P) & ~Rel(S, N1, P) -> ~Rel(N0, S, P)",
                weighted=True,
                weight=feat_weights[self.CascadingError],
                squared=True,
            ),
            Rule(
                f"CanRel(N0, S, P) & Statement(S) & CanRel(S, N1, P) & ~Rel(N0, S, P) -> ~Rel(S, N1, P)",
                weighted=True,
                weight=feat_weights[self.CascadingError],
                squared=True,
            ),
        ]

        # self.rules[self.TypeMustInPropRange] = Rule(
        #     "CanRel(S, N1, P) & Statement(S) & CanType(N1, T) & NotRange(P, T) & Rel(S, N1, P) -> ~Type(N1, T)",
        #     weighted=True, weight=feat_weights[self.TypeMustInPropRange], squared=True)
        self.rules[self.FreqTypeOverRow] = Rule(
            f"CanType(N, T) & TypeFeature_{self.FreqTypeOverRow}(N, T) -> Type(N, T)",
            weighted=True,
            weight=feat_weights[self.FreqTypeOverRow],
            squared=True,
        )

        for rule_id, rules in self.rules.items():
            if rule_id in self.disable_rules:
                continue
            if isinstance(rules, list):
                for rule in rules:
                    model.add_rule(rule)
            else:
                model.add_rule(rules)

        # set the model and done
        self.model = model

    def infer(self, data: Dict[str, list]):
        """Run inference and get back the result.
        Note: Check the model to see the list of predicate we need to pass the data to.
        """
        if len(self.model.get_predicates()) != len(data) + 2:
            preds = {x.upper() for x in data.keys()}
            miss_preds = [
                p
                for p in self.model.get_predicates()
                if p not in preds and p not in {"REL", "TYPE"}
            ]
            raise Exception(
                f"Data in all predicates must be set. Missing {','.join(miss_preds)} predicates"
            )
        # TODO: fix me! temporary allow cantype to be empty
        can_be_empty_predicates = {
            "SubProp",
            f"RelFeature_{self.LinkDataTypeMismatch}",
            "NotRange",
            "CanType",
            f"TypeFeature_{self.FreqTypeOverRow}",
            f"RelFeature_{self.LinkNotFuncDep}",
        }
        RelPredicate = (
            self.model.get_predicate("Rel")
            .clear_data()
            .add_data(Partition.TARGETS, data["CanRel"])
        )
        if len(data["CanType"]) == 0:
            TypePredicate = self.model.get_predicate("Type").clear_data()
        else:
            TypePredicate = (
                self.model.get_predicate("Type")
                .clear_data()
                .add_data(Partition.TARGETS, data["CanType"])
            )
        for pred, pred_data in data.items():
            if pred in can_be_empty_predicates and len(pred_data) == 0:
                self.model.get_predicate(pred).clear_data()
            else:
                self.model.get_predicate(pred).clear_data().add_data(
                    Partition.OBSERVATIONS, pred_data
                )

        infer_resp = self.model.infer(
            logger=None if self.enable_logging else False,
            additional_cli_optons=["--h2path", os.path.join(self.temp_dir, "h2")],
            temp_dir=str(self.temp_dir),
        )  # , cleanup_temp=False)
        return {
            "links": {
                (r[0], r[1], r[2]): r["truth"]
                for ri, r in infer_resp[RelPredicate].iterrows()
            },
            "types": {
                (r[0], r[1]): r["truth"]
                for ri, r in infer_resp[TypePredicate].iterrows()
            },
        }

    def run(self, r: PSLRunParallelArgs, threshold: float = 0.5):
        table, sg, dg = r["table"], r["semanticgraph"], r["datagraph"]
        pred_with_probs = self.solve(table, sg, dg)
        link_probs = pred_with_probs["links"]
        cta = pred_with_probs["types"]
        sg = self.solve_post_process(table, sg, dg, link_probs, threshold)
        return link_probs, sg, cta

    def run_with_parallel(
        self,
        inputs: List[PSLRunParallelArgs],
        threshold: float = 0.5,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ):
        global global_objects
        predictions = self.solve_parallel(
            [(r["table"], r["semanticgraph"], r["datagraph"]) for r in inputs],
            batch_size=batch_size,
            show_progress=show_progress,
        )

        global_objects["PSLSteinerTreeSolver"] = self
        # DO this cause we don't know why semtab postprocessing create that issue...
        is_parallel = self.postprocessing_method != "external:semtab2020"
        sgs = M.parallel_map(
            PSLSteinerTreeSolver._solve_post_process_wrapper,
            [
                (
                    r["table"],
                    r["semanticgraph"],
                    r["datagraph"],
                    predwprobs["links"],
                    threshold,
                )
                for r, predwprobs in zip(inputs, predictions)
            ],
            show_progress=show_progress,
            progress_desc="psl: post-process",
            is_parallel=is_parallel,
        )
        cta = [predwprobs["types"] for r, predwprobs in zip(inputs, predictions)]
        return sgs, cta

    def solve_parallel(
        self,
        inputs: List[Tuple[LinkedTable, CGGraph, DGGraph]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ):
        global global_objects
        global_objects["cache_dir"] = self.cache_dir
        global_objects["wdprops"] = self.wdprops
        global_objects["qnodes"] = self.qnodes
        global_objects["wd_numprop_stats"] = self.wd_numprop_stats
        global_objects["sim_fn"] = self.sim_fn

        # create a mapping from id to index so that we can re-assign the result later.
        results = Parallel().map(
            PSLSteinerTreeSolver._extract_features_wrapper,
            list(enumerate(inputs)),
            show_progress=show_progress,
            progress_desc="psl: extract features",
        )

        if batch_size is None:
            batch_size = len(inputs)
            iter = range(1)
        else:
            iter = range(0, len(inputs), batch_size)
            if show_progress:
                iter = tqdm(
                    iter,
                    total=math.ceil(len(inputs) / batch_size),
                    desc="psl: inference",
                )

        if self.cache_dir is not None:
            infdir = self.cache_dir / "inferences"
            infdir.mkdir(exist_ok=True, parents=True)
            get_cache_file = lambda tbl, index: (
                infdir / f"a{index:03d}_{tbl.get_friendly_fs_id()}.pkl"
            )
        else:
            get_cache_file = lambda tbl, index: None

        predictions = []
        for i in iter:
            input_batch = inputs[i : i + batch_size]
            feature_batch = results[i : i + batch_size]
            cache_files = [
                get_cache_file(table, j)
                for j, (table, sg, dg) in enumerate(inputs[i : i + batch_size], start=i)
            ]
            if not all(x is not None and x.exists() for x in cache_files):
                data, idmap = self.extract_predicate_data(
                    input_batch, feature_batch, is_parallel=True
                )
                try:
                    infer_resp = self.infer(data)
                except:
                    logger.exception("Error while processing batch: {}", i)
                    raise
                table_infer_link_resp = {}
                table_infer_type_resp = {}
                for (uid, vid, eid), prob in infer_resp["links"].items():
                    table_id, uid = idmap.im(uid)
                    assert table_id == idmap.im(vid)[0]
                    vid = idmap.im(vid)[1]

                    if table_id not in table_infer_link_resp:
                        table_infer_link_resp[table_id] = {}
                    table_infer_link_resp[table_id][uid, vid, eid] = prob
                for (uid, classid), prob in infer_resp["types"].items():
                    table_id, uid = idmap.im(uid)
                    if table_id not in table_infer_type_resp:
                        table_infer_type_resp[table_id] = {}
                    if uid not in table_infer_type_resp[table_id]:
                        table_infer_type_resp[table_id][uid] = {}
                    table_infer_type_resp[table_id][uid][classid] = prob

                batch_predictions = []
                for (table, sg, dg), cache_file in zip(input_batch, cache_files):
                    pred = {
                        "id": table.id,
                        "links": table_infer_link_resp.get(table.id, {}),
                        "types": table_infer_type_resp.get(table.id, {}),
                    }
                    if cache_file is not None:
                        M.serialize_pkl(pred, cache_file)
                    batch_predictions.append(pred)
            else:
                batch_predictions = [
                    M.deserialize_pkl(cache_file) for cache_file in cache_files  # type: ignore -- type checker not smart enough
                ]

            # set the prob. for debugging purpose
            for (table, sg, dg), pred in zip(input_batch, batch_predictions):
                assert table.id == pred["id"]
                for k, v in pred["links"].items():
                    sg.get_edge_between_nodes(*k).features["prob"] = v
            predictions += batch_predictions

        return predictions

    def solve(self, table: LinkedTable, sg: CGGraph, dg: DGGraph):
        if sg.num_edges() == 0:
            return {"links": {}, "types": {}}

        data, idmap = self.extract_predicate_data(
            [(table, sg, dg)], is_parallel=False, show_progress=False
        )

        def unpack_mappedid(key):
            table_id, real_key = idmap.im(key)
            assert table_id == table.id
            return real_key

        infer_resp = self.infer(data)
        infer_resp_types = {}
        for (uid, classid), prob in infer_resp["types"].items():
            _ouid = unpack_mappedid(uid)
            if _ouid not in infer_resp_types:
                infer_resp_types[_ouid] = {}
            infer_resp_types[_ouid][classid] = prob

        infer_resp["links"] = {
            (unpack_mappedid(uid), unpack_mappedid(vid), eid): prob
            for (uid, vid, eid), prob in infer_resp["links"].items()
        }
        infer_resp["types"] = infer_resp_types

        # set the prob. for debugging purpose
        link_feat_extractor = LinkFeatureExtraction(
            table, sg, dg, self.qnodes, self.wdprops, self.wd_numprop_stats, self.sim_fn
        )
        link_feat_extractor.add_debug_info(link_feat_extractor.extract_features())
        for k, prob in infer_resp["links"].items():
            sg.get_edge_between_nodes(*k).features["prob"] = prob

        return infer_resp

    def solve_post_process(
        self,
        table,
        sg,
        dg,
        pred_with_probs: Dict[Tuple[str, str, str], float],
        threshold,
    ):
        if self.postprocessing_method == "select_simplepath":
            return self.postprocessing_select_simplepath(
                table, sg, dg, pred_with_probs, threshold
            )
        if self.postprocessing_method == "steiner_tree":
            return self.postprocessing_steiner_tree(
                table, sg, dg, pred_with_probs, threshold
            )
        if self.postprocessing_method == "external:semtab2020":
            # return self.postprocessing_fn.solve_post_process(table, sg, dg, pred_with_probs, threshold)
            return SemTab2020PostProcessing.get_instance().solve_post_process(
                table, sg, dg, pred_with_probs, threshold
            )
        assert False, self.postprocessing_method

    def remove_dangling_statement(self, cg: CGGraph):
        ids = set()
        for s in list(cg.iter_nodes()):
            if isinstance(s, CGStatementNode) and (
                cg.in_degree(s.id) == 0 or cg.out_degree(s.id) == 0
            ):
                ids.add(s.id)
        for id in ids:
            cg.remove_node(id)

    def postprocessing_select_simplepath(
        self,
        table: LinkedTable,
        sg: CGGraph,
        dg: DGGraph,
        pred_with_probs: Dict[Tuple[str, str, str], float],
        threshold,
    ):
        def select_shorter_path(paths: List[List[Tuple[str, str, str]]]):
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
                    edge = sg.get_edge_between_nodes(uid, vid, eid)
                    path_prob += pred_with_probs[uid, vid, eid]
                path_probs.append(path_prob)
            path, path_prob = max(zip(paths, path_probs), key=itemgetter(1))
            return path

        pred_edges = [k for k, v in pred_with_probs.items() if v >= threshold]
        steiner_tree = sg.subgraph_from_edge_triples(pred_edges)
        self.remove_dangling_statement(steiner_tree)

        if steiner_tree.num_edges() == 0:
            # empty graph
            return steiner_tree

        return keep_one_simple_path_between_important_nodes(
            steiner_tree, select_shorter_path, both_direction=True
        )

    def postprocessing_steiner_tree(
        self,
        table,
        sg: CGGraph,
        dg: DGGraph,
        pred_with_probs: Dict[Tuple[str, str, str], float],
        threshold,
    ):
        pred_edges = [k for k, v in pred_with_probs.items() if v >= threshold]
        steiner_tree = sg.subgraph_from_edge_triples(pred_edges)
        self.remove_dangling_statement(steiner_tree)

        if steiner_tree.num_edges() == 0:
            # empty graph
            return steiner_tree

        # normalizing the prob score so that we can compare the weight between graph accurately
        norm_pred_probs = {}
        lst = sorted(
            (x for x in pred_with_probs.items() if x[1] >= threshold), key=itemgetter(1)
        )
        eps = 0.001
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
            for k, prob in cluster:
                norm_pred_probs[k] = avg_prob

        # print(lst)
        # print(norm_pred_probs)

        def get_solution_weight(sol: Solution):
            if sol.num_edges == 0:
                return 0.0
            return sol.weight / sol.num_edges

        def cmp_bank_solution(sol_a: Solution, sol_b: Solution):
            # since PSL inference is not exact but optimization, there is always a delta different in their final
            # calculation, which we need to compensate for.
            # Empirically, the number is very small ~0.001 in a range of [0 - 1] for each edge
            # we need to count the number of edge, but have to do it this way since bank solver
            # do some optimization to reduce the number of nodes & edges
            # diff = abs((sol_a.weight / sol_a.get_n_edges()) - (sol_b.weight / sol_b.get_n_edges()))
            # if diff > 0.002:
            sol_a_weight = get_solution_weight(sol_a)
            sol_b_weight = get_solution_weight(sol_b)

            if sol_a_weight < sol_b_weight:
                return -1
            if sol_a_weight > sol_b_weight:
                return 1

            if not hasattr(sol_a, "depth"):
                sol_a.depth = len(dag_longest_path(sol_a.graph))
            if not hasattr(sol_b, "depth"):
                sol_b.depth = len(dag_longest_path(sol_b.graph))
            return sol_a.depth - sol_b.depth

        terminal_nodes = {
            u.id for u in steiner_tree.iter_nodes() if isinstance(u, CGColumnNode)
        }
        # terminal_nodes = SemanticGraphConstructor.st_terminal_nodes(steiner_tree)
        # terminal_nodes = set()
        # for uid, udata in steiner_tree.nodes(data=True):
        #     u: SGNode = udata["data"]
        #     if u.is_column:
        #         terminal_nodes.add(uid)
        #     elif u.is_value and u.is_in_context:
        #         terminal_nodes.add(uid)
        # TODO: this is a temporary fix for case where steiner tree does not contain any column
        if len(terminal_nodes) == 0:
            # TODO: fix me
            return steiner_tree

        bank_solver = BankSolver(
            steiner_tree,
            terminal_nodes,
            top_k_st=50,
            top_k_path=50,
            weight_fn=lambda e: 1.0
            / max(
                1e-7,
                norm_pred_probs[e.source, e.target, e.predicate],
            ),
            solution_cmp_fn=cmp_bank_solution,
        )
        try:
            candidate_sts, solutions = bank_solver.run()
        except NoSingleRootException:
            # fallback
            return self.postprocessing_select_simplepath(
                table, sg, dg, pred_with_probs, threshold
            )
        except NetworkXUnfeasible:
            # TODO: fix me
            assert table.id == "Diamond_League"
            # fallback
            return self.postprocessing_select_simplepath(
                table, sg, dg, pred_with_probs, threshold
            )
        # TODO fix me when the bank algorithm return empty result
        if len(candidate_sts) == 0:
            return self.postprocessing_select_simplepath(
                table, sg, dg, pred_with_probs, threshold
            )

        pred_tree = candidate_sts[0]
        # remove statements that do not connected from any nodes
        for s in pred_tree.nodes():
            if isinstance(s, CGStatementNode) and pred_tree.in_degree(s.id) == 0:
                pred_tree.remove_node(s.id)
        for s in pred_tree.nodes():
            if not isinstance(s, CGStatementNode):
                continue
            assert pred_tree.in_degree(s.id) == 1
            (in_edge,) = pred_tree.in_edges(s.id)
            for e in steiner_tree.out_edges(s.id):
                if e.predicate == in_edge.predicate:
                    # statement prop
                    if not pred_tree.has_node(e.target):
                        ent = steiner_tree.get_node(e.target)
                        assert isinstance(
                            ent, (CGEntityValueNode, CGLiteralValueNode)
                        ), f"The only reason why we don't have statement value is it is an literal"
                        pred_tree.add_node(ent.clone())
                        pred_tree.add_edge(e.clone())
                else:
                    v = steiner_tree.get_node(e.target)
                    if (
                        isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
                        and v.is_in_context
                    ):
                        # we should have this as PSL think it's correct
                        # TODO: the entity can be in the tree before if it's needed to connect
                        # two nodes, but we need to check it
                        if pred_tree.has_node(v.id):
                            assert pred_tree.has_edge_between_nodes(
                                s.id, v.id, e.predicate
                            )
                        else:
                            # assert not pred_tree.has_node(v.id)
                            pred_tree.add_node(v.clone())
                            pred_tree.add_edge(e.clone())

        if PSLConfigs.POSTPROCESSING_STEINER_TREE_FORCE_ADDING_CONTEXT:
            # add back the context node or entity that are appeared in the psl results but not in the predicted tree
            for v in steiner_tree.iter_nodes():
                if (
                    not isinstance(v, (CGEntityValueNode, CGLiteralValueNode))
                    or not v.is_in_context
                ):
                    continue
                if pred_tree.has_node(v.id):
                    continue

                # find the paths that connect the vid to the tree and select the one with highest score and do not create cycle
                paths: List[_AddContextPathType] = []
                for sv_edge in steiner_tree.in_edges(v.id):
                    if sv_edge.predicate == "P31":
                        continue
                    for us_edge in steiner_tree.in_edges(sv_edge.source):
                        if not pred_tree.has_node(us_edge.source):
                            continue
                        paths.append(
                            {
                                "path": (
                                    us_edge,
                                    sv_edge,
                                ),
                                "score": pred_with_probs[
                                    (us_edge.source, us_edge.target, us_edge.predicate)
                                ]
                                + pred_with_probs[
                                    (sv_edge.source, sv_edge.target, sv_edge.predicate)
                                ],
                            }
                        )

                paths = sorted(paths, key=itemgetter("score"), reverse=True)
                # TODO: filter out the path that will create cycle

                if len(paths) == 0:
                    continue

                us_edge, sv_edge = paths[0]["path"]
                pred_tree.add_node(v.clone())
                if not pred_tree.has_node(sv_edge.source):
                    s = steiner_tree.get_node(sv_edge.source)
                    pred_tree.add_node(s.clone())
                assert not pred_tree.has_edge_between_nodes(
                    sv_edge.source, sv_edge.target, sv_edge.predicate
                )
                pred_tree.add_edge(sv_edge.clone())

                if not pred_tree.has_edge_between_nodes(
                    us_edge.source, us_edge.target, us_edge.predicate
                ):
                    pred_tree.add_edge(us_edge.clone())

        # TODO: uncomment for debugging
        # print(candidate_sts)
        # if there are more than one candidate steiner tree of same weight, select the one with shorter height
        # candidate_sts = [item for item in candidate_sts if item[1] == candidate_sts[0][1]]
        # candidate_st = sorted(candidate_sts, key=lambda g: len(nx.algorithms.dag.dag_longest_path(g[0])))[0][0]

        # env.viz_sg(steiner_tree, "after_PSL")
        # for i, (g, sol) in enumerate(zip(candidate_sts, solutions)):
        #     # g = SemanticGraphConstructor.get_sg_subgraph(sg, list(g.edges(keys=True)))
        #     (precision, recall, f1), oracle_sg = env.eval_steiner_tree(table_index, g)
        #     print(f"\t i={i:02d} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} score={get_solution_weight(sol)} depth={len(nx.algorithms.dag.dag_longest_path(g))}")
        #     env.viz_sg(g, f"st_{i:02d}")
        # # viz = lambda x, y: env.viz_sg(SemanticGraphConstructor.get_sg_subgraph(sg, list(x.edges(keys=True))), y)
        # for i in range(len(candidate_sts)):
        #     env.viz_sg(candidate_sts[i][0], f"st_{i:02d}")

        # bank_solver._get_graph_weight(candidate_sts[0][0])
        return pred_tree

    def train_setup(self, inputs: List[Tuple[LinkedTable, CGGraph, DGGraph, dict]]):
        data, idmap = self.extract_predicate_data(
            inputs=[(x[0], x[1], x[2]) for x in inputs], features=[x[3] for x in inputs]
        )

        assert (
            len(self.model.get_predicates()) == len(data) + 2
        ), "Data in all predicates must be set"
        can_be_empty_predicates = {"SubProp", f"RelFeature_{self.LinkDataTypeMismatch}"}
        self.model.get_predicate("Rel").clear_data().add_data(
            Partition.TARGETS, data["CanRel"]
        )
        self.model.get_predicate("Type").clear_data().add_data(
            Partition.TARGETS, data["CanType"]
        )
        for pred, pred_data in data.items():
            if pred in can_be_empty_predicates and len(pred_data) == 0:
                self.model.get_predicate(pred).clear_data()
            else:
                self.model.get_predicate(pred).clear_data().add_data(
                    Partition.OBSERVATIONS, pred_data
                )

        # clean up the temporary directory first. this code is takening from the PSL model.infer method
        if Path(self.temp_dir).exists():
            self.model._cleanup_temp(str(self.temp_dir))
        # this write out the data file and rules_file
        logger, temp_dir, data_file_path, rules_file_path = self.model._prep_run(
            logger=None if self.enable_logging else False, temp_dir=str(self.temp_dir)
        )
        cli_options = []
        cli_options.append("--infer")
        inferred_dir = os.path.join(temp_dir, Model.CLI_INFERRED_OUTPUT_DIR)
        cli_options.append("--output")
        cli_options.append(inferred_dir)

        self.train_args = dict(
            data_file_path=data_file_path,
            rules_file_path=rules_file_path,
            cli_options=cli_options,
            psl_config={},
            jvm_options=[],
            logger=logger,
            inferred_dir=inferred_dir,
            temp_dir=temp_dir,
        )
        return idmap

    def train_set_parameters(self, rules: Dict[str, float]):
        for rule_id, rule_weight in rules.items():
            if isinstance(self.rules[rule_id], list):
                for rule in self.rules[rule_id]:
                    rule.set_weight(rule_weight)
            else:
                self.rules[rule_id].set_weight(rule_weight)
        self.model._write_rules(self.train_args["temp_dir"])

    def train_eval(self, idmap: IDMap) -> Dict[str, Dict[Tuple[str, str, str], float]]:
        table_infer_resp = {}
        RelPredicate = self.model.get_predicate("Rel")

        self.model._run_psl(
            self.train_args["data_file_path"],
            self.train_args["rules_file_path"],
            self.train_args["cli_options"],
            self.train_args["psl_config"],
            self.train_args["jvm_options"],
            self.train_args["logger"],
        )
        infer_resp = self.model._collect_inference_results(
            self.train_args["inferred_dir"]
        )

        for ri, r in infer_resp[RelPredicate].iterrows():
            uid, vid, eid, prob = r[0], r[1], r[2], r["truth"]
            table_id, uid = idmap.im(uid)
            assert table_id == idmap.im(vid)[0]
            vid = idmap.im(vid)[1]

            if table_id not in table_infer_resp:
                table_infer_resp[table_id] = {}
            table_infer_resp[table_id][uid, vid, eid] = prob
        return table_infer_resp

    def extract_predicate_data(
        self,
        inputs: List[Tuple[LinkedTable, CGGraph, DGGraph]],
        features: Optional[List[dict]] = None,
        idmap=None,
        is_parallel: Union[str, bool] = "auto",
        show_progress: bool = False,
    ):
        """Extract predicates data from the list of tables, sgs, and dgs. If the features do not provided, it will
        be created automatically.

        If parallel is 'auto' we run parallel when length of inputs > 1
        """
        global global_objects
        if idmap is None:
            idmap = IDMap()

        if features is None:
            is_parallel = len(inputs) > 1 if is_parallel == "auto" else is_parallel
            global_objects["wdprops"] = self.wdprops
            global_objects["qnodes"] = self.qnodes
            global_objects["wd_numprop_stats"] = self.wd_numprop_stats
            global_objects["sim_fn"] = self.sim_fn

            if is_parallel:
                global_objects["cache_dir"] = self.cache_dir
            else:
                global_objects["cache_dir"] = None
            pp = Parallel()
            features = pp.map(
                PSLSteinerTreeSolver._extract_features_wrapper,
                list(enumerate(inputs)),
                show_progress=show_progress,
                progress_desc="psl: extract features",
                is_parallel=is_parallel,
            )
        assert features is not None
        data = {
            "CanRel": [],
            "CanType": [],
            "NotStatement": [],
            "Statement": [],
            "SubProp": set(),
            "NotRange": set(),
        }
        for feat in self.link_all_feats:
            data[f"RelFeature_{feat}"] = []
        for feat in self.type_all_feats:
            data[f"TypeFeature_{feat}"] = []
        for (table, sg, dg), result in zip(inputs, features):
            table_id = table.id
            idmap.add_keys([(table_id, u.id) for u in sg.iter_nodes()])
            data["CanRel"] += [
                (
                    idmap.m((table_id, edge.source)),
                    idmap.m((table_id, edge.target)),
                    edge.predicate,
                )
                for edge in sg.iter_edges()
            ]
            data[f"CanType"] += [
                (idmap.m((table_id, uid)), classid)
                for uid, classid in result["type_feats"][TypeFeatureExtraction.Freq]
            ]
            data["NotStatement"] += [
                (idmap.m((table_id, u.id)),)
                for u in sg.iter_nodes()
                if not isinstance(u, CGStatementNode)
            ]
            data["Statement"] += [
                (idmap.m((table_id, u.id)),)
                for u in sg.iter_nodes()
                if isinstance(u, CGStatementNode)
            ]

            props = {edge.predicate for edge in sg.iter_edges()}
            for p in props:
                for pp in props:
                    if p != pp and pp in self.wdprops[p].parents_closure:
                        data["SubProp"].add((p, pp))

            for uid, class_ids in result["type_feats"]["_column_to_types"].items():
                # find list of incoming edges
                incoming_props = {edge.predicate for edge in sg.in_edges(uid)}
                incoming_props = [self.wdprops[eid] for eid in incoming_props]
                for class_id in class_ids:
                    if class_id not in self.wdclasses:
                        continue
                    parents_closure = self.wdclasses[class_id].parents_closure
                    for p in incoming_props:
                        if len(p.subjects) > 0 and not any(
                            class_id == subj or subj in parents_closure
                            for subj in p.subjects
                        ):
                            data["NotRange"].add((p.id, class_id))

            for feat in self.link_all_feats:
                data[f"RelFeature_{feat}"] += [
                    (idmap.m((table_id, uid)), idmap.m((table_id, vid)), eid, prob)
                    for (uid, vid, eid), prob in result["link_feats"][feat].items()
                ]
            for feat in self.type_all_feats:
                data[f"TypeFeature_{feat}"] += [
                    (idmap.m((table_id, uid)), classid, prob)
                    for (uid, classid), prob in result["type_feats"][feat].items()
                ]
        return data, idmap

    @staticmethod
    def _solve_post_process_wrapper(args):
        global global_objects
        table, sg, dg, pred_with_probs, threshold = args
        return global_objects["PSLSteinerTreeSolver"].solve_post_process(
            table, sg, dg, pred_with_probs, threshold
        )

    @staticmethod
    def _update_global_objects(**kwargs):
        global global_objects
        for k, v in kwargs.items():
            global_objects[k] = v

    @staticmethod
    def _extract_features_wrapper(
        args: Tuple[int, Tuple[LinkedTable, CGGraph, DGGraph]]
    ):
        """Wrap the function that need to pass some big objects through a global objects"""
        global global_objects
        wdprops: Mapping[str, WDProperty] = global_objects["wdprops"]
        wd_numprop_stats = global_objects["wd_numprop_stats"]
        qnodes: Mapping[str, QNode] = global_objects["qnodes"]
        cache_dir = global_objects["cache_dir"]
        sim_fn = global_objects["sim_fn"]

        index, (table, sg, dg) = args

        if cache_dir is not None:
            (cache_dir / "features").mkdir(exist_ok=True, parents=True)
            filename = table.get_friendly_fs_id()
            filepath = cache_dir / "features" / f"a{index:03d}_{filename}.pkl"
            if filepath.exists():
                return M.deserialize_pkl(filepath)
        else:
            filepath = None

        start = time.time()
        link_feat_extractor = LinkFeatureExtraction(
            table, sg, dg, qnodes, wdprops, wd_numprop_stats, sim_fn
        )
        type_feat_extractor = TypeFeatureExtraction(
            table, sg, dg, qnodes, wdprops, wd_numprop_stats
        )

        link_feats = link_feat_extractor.extract_features()
        type_feats = type_feat_extractor.extract_features()

        exectime = time.time() - start
        props = {edge.predicate for edge in sg.iter_edges()}
        sub_props = set()
        for p in props:
            for pp in props:
                if p != pp and pp in wdprops[p].parents_closure:
                    sub_props.add((p, pp))
        prop_ranges = set()

        cache_content = {
            "link_feats": link_feats,
            "type_feats": type_feats,
            "sub_props": sub_props,
            "exec_time": exectime,
        }
        if filepath is not None:
            M.serialize_pkl(cache_content, filepath)
        return cache_content

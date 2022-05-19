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

import networkx as nx
import sm.misc as M
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import DGGraph, DGNode
from grams.algorithm.deprecated.link_feature import LinkFeatureExtraction

# from grams.algorithm.postprocessing.semtab2020 import SemTab2020PostProcessing
from grams.algorithm.deprecated.type_feature import TypeFeatureExtraction
from grams.inputs.linked_table import LinkedTable

# from grams.algorithm.postprocessing.simple_path import (
#     keep_one_simple_path_between_important_nodes,
# )
from graph.retworkx.api import dag_longest_path
from hugedict.parallel.parallel import Parallel
from kgdata.wikidata.models import (
    WDEntity,
    WDClass,
    WDProperty,
    WDQuantityPropertyStats,
)
from loguru import logger
from networkx.exception import NetworkXUnfeasible
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from steiner_tree.bank import BankSolver, NoSingleRootException, Solution
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


class PSLInference:
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
    TypeNegParentPrior = "TypeNegParentPrior"
    TypeHeaderSimilarity = TypeFeatureExtraction.HeaderSimilarity
    FreqTypeOverRow = TypeFeatureExtraction.FreqOverRow
    FreqTypeInheritOverRow = TypeFeatureExtraction.FreqInheritOverRow
    TypeMustInPropRange = "TypeMustInPropRange"
    TypeOnlyOneConstraint = "TypeOnlyOneConstraint"

    def __init__(
        self,
        qnodes: Mapping[str, WDEntity],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_numprop_stats: Mapping[str, WDQuantityPropertyStats],
        disable_rules: Set[str] = None,
        sim_fn: Optional[Callable[[str, str], float]] = None,
        cache_dir: Optional[str] = None,
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
            r
            for r in [self.FreqTypeInheritOverRow, self.TypeHeaderSimilarity]
            if r not in self.disable_rules
        ]
        self.type_all_feats = self.type_pos_feats

        model.add_predicate(Predicate("CanRel", closed=True, size=3))
        model.add_predicate(Predicate("CanType", closed=True, size=2))
        model.add_predicate(Predicate("Rel", closed=False, size=3))
        model.add_predicate(Predicate("Type", closed=False, size=2))
        model.add_predicate(Predicate("SubProp", closed=True, size=2))
        model.add_predicate(Predicate("SubType", closed=True, size=2))
        model.add_predicate(Predicate("HasSubType", closed=True, size=2))
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
            self.TypeNegParentPrior: 0.1,
            self.TypeHeaderSimilarity: 0.1,
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
            self.FreqTypeInheritOverRow: 2,
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

        # give a small negative weight to the parent type
        self.rules[self.TypeNegParentPrior] = [
            Rule(
                # f"CanType(N, T1) & CanType(N, T2) & SubType(T1, T2) -> ~Type(N, T2)",
                "CanType(N, T) & HasSubType(N, T) -> ~Type(N, T)",
                weighted=True,
                weight=feat_weights[self.TypeNegParentPrior],
                squared=True,
            )
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
        for feat in self.type_pos_feats:
            self.rules[feat] = Rule(
                f"CanType(N, T) & TypeFeature_{feat}(N, T) -> Type(N, T)",
                weighted=True,
                weight=feat_weights[feat],
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

        # print rules (for debugging)
        # print("=" * 10, "Rules")
        # for rule_id, rules in self.rules.items():
        #     if rule_id in self.disable_rules:
        #         continue
        #     if isinstance(rules, list):
        #         for rule in rules:
        #             print(f"{rule_id}: {rule._rule_body}")
        #     else:
        #         print(f"{rule_id}: {rules._rule_body}")
        # print("=" * 10)

        # set the model and done
        self.model = model

    def infer(self, data: Dict[str, list]):
        """Run inference and get back the result.
        Note: Check the model to see the list of predicate we need to pass the data to.
        """
        preds = {x.upper() for x in data.keys()}
        miss_preds = [
            p
            for p in self.model.get_predicates()
            if p not in preds and p not in {"REL", "TYPE"}
        ]
        if len(miss_preds) > 0:
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
            f"TypeFeature_{self.FreqTypeInheritOverRow}",
            f"TypeFeature_{self.TypeHeaderSimilarity}",
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
            cleanup_temp=False,
        )
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

    def run(self, table: LinkedTable, dg: DGGraph, cg: CGGraph):
        inferred_result = self.solve(table, cg, dg)
        cpa = inferred_result["links"]
        cta = inferred_result["types"]
        cta = {int(ci.replace("column-", "")): classes for ci, classes in cta.items()}
        return cpa, cta

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
            global_objects["wdclasses"] = self.wdclasses
            global_objects["qnodes"] = self.qnodes
            global_objects["wd_numprop_stats"] = self.wd_numprop_stats
            global_objects["sim_fn"] = self.sim_fn

            if is_parallel:
                global_objects["cache_dir"] = self.cache_dir
            else:
                global_objects["cache_dir"] = None
            pp = Parallel()
            features = pp.map(
                PSLInference._extract_features_wrapper,
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
            "SubType": set(),
            "HasSubType": set(),
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
                class_ids = set(class_ids)
                for class_id in class_ids:
                    for parent_class_id in self.wdclasses[class_id].parents:
                        if parent_class_id in class_ids:
                            data["SubType"].add((class_id, parent_class_id))
                            data["HasSubType"].add(
                                (idmap.m((table_id, uid)), parent_class_id)
                            )

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
        wdclasses: Mapping[str, WDClass] = global_objects["wdclasses"]
        wd_numprop_stats = global_objects["wd_numprop_stats"]
        qnodes: Mapping[str, WDEntity] = global_objects["qnodes"]
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
            table, sg, dg, qnodes, wdclasses, wdprops, wd_numprop_stats, sim_fn
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

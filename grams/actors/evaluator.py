from __future__ import annotations
import copy
from collections import defaultdict
from operator import attrgetter
from typing import *
from typing import List

from hugedict.types import HugeMutableMapping
from kgdata.wikidata.models import WDEntity, WDEntityLabel
from kgdata.wikidata.models import WDProperty, WDClass

import networkx as nx
import pandas as pd
import sm.misc as M
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGNode,
    CGStatementNode,
)
from grams.algorithm.helpers import IndirectDictAccess
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.inputs.linked_table import Link, LinkedTable
from grams.main import GRAMS, Annotation
from loguru import logger
from sm.dataset import Example
from sm.evaluation import sm_metrics
from sm.evaluation.cpa_cta_metrics import _cpa_transformation, _get_cta, cpa, cta
from sm.evaluation.hierarchy_scoring_fn import HierarchyScoringFn
from sm.misc.funcs import DictProxy
from sm.outputs.semantic_model import LiteralNode, SemanticModel
from sm.prelude import M, O


class Evaluator:
    """Object to evaluate performance of predicted models. Must provide a
    list of examples in order to init the score functions
    """

    def __init__(
        self,
        qnodes: HugeMutableMapping[str, WDEntity],
        qnode_labels: Mapping[str, WDEntityLabel],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
    ):
        self.qnodes = qnodes.cache()
        self.all_wdclasses = wdclasses
        self.all_wdprops = wdprops

        self.wdclasses = {}
        self.wdprops = {}
        self.sm_helper = WikidataSemanticModelHelper(
            self.qnodes,
            IndirectDictAccess(qnode_labels, attrgetter("label")),
            wdclasses,
            wdprops,
        )
        self.id_props = self.sm_helper.ID_PROPS
        self.wdns = self.sm_helper.wdns
        # caching
        self.example2equivsms = {}

    def cpa_cta(self, example: Example[LinkedTable], sm: O.SemanticModel):
        equiv_sms = self.get_equiv_sms(example)
        annotated_sm = self.sm_helper.norm_sm(sm)
        cpa_sm, cpa_output = max(
            [
                (
                    equiv_sm,
                    cpa(equiv_sm, annotated_sm, self.id_props, self.prop_score_fn),
                )
                for equiv_sm in equiv_sms
            ],
            key=lambda x: x[1].f1 if x[1] is not None else 0.0,
        )
        cta_output = max(
            [
                cta(equiv_sm, annotated_sm, self.id_props, self.class_score_fn)
                for equiv_sm in equiv_sms
            ],
            key=attrgetter("f1"),
        )

        df = pd.DataFrame(
            [
                {
                    "id": "cpa",
                    "precision": cpa_output.precision,
                    "recall": cpa_output.recall,
                    "f1": cpa_output.f1,
                },
                {
                    "id": "cta",
                    "precision": cta_output.precision,
                    "recall": cta_output.recall,
                    "f1": cta_output.f1,
                },
            ]
        )

        cpa_gold_sm = cpa_sm.deep_copy()
        cpa_pred_sm = annotated_sm
        _cpa_transformation(cpa_gold_sm, self.id_props)
        _cpa_transformation(cpa_pred_sm, self.id_props)

        return {
            "gold_sm": cpa_sm.copy(),
            "cpa_gold_sm": cpa_gold_sm,
            "cpa_pred_sm": cpa_pred_sm,
            "df": df,
            "cpa": cpa_output,
            "cta": cta_output,
        }

    def get_equiv_sms(self, example: Example[LinkedTable]) -> List[O.SemanticModel]:
        if example.table.id not in self.example2equivsms:
            equiv_sms = [
                equiv_sm
                for esm in example.sms
                for equiv_sm in self.sm_helper.gen_equivalent_sm(
                    esm, strict=False, incorrect_invertible_props={"P571", "P582"}
                )
            ]
            self.example2equivsms[example.table.id] = equiv_sms
        return self.example2equivsms[example.table.id]

    def update_score_fns(
        self, sms: List[O.SemanticModel], cgs: Optional[List[CGGraph]] = None
    ):
        """This function is expected to be called before all other functions is called"""
        update_wdprops = False
        update_wdclasses = False

        for sm in sms:
            for edge in sm.iter_edges():
                if self.wdns.is_abs_uri_property(edge.abs_uri):
                    pid = self.wdns.get_prop_id(edge.abs_uri)
                    if pid not in self.wdprops:
                        self.wdprops[pid] = self.all_wdprops[pid]
                        update_wdprops = True
            for node in sm.iter_nodes():
                if isinstance(node, O.ClassNode) and self.wdns.is_abs_uri_qnode(
                    node.abs_uri
                ):
                    qid = self.wdns.get_entity_id(node.abs_uri)
                    if qid not in self.wdclasses:
                        self.wdclasses[qid] = self.all_wdclasses[qid]
                        update_wdclasses = True

        for cg in cgs or []:
            for edge in cg.iter_edges():
                pid = edge.predicate
                assert self.wdns.is_valid_id(pid)
                if pid not in self.wdprops:
                    self.wdprops[pid] = self.all_wdprops[pid]
                    update_wdprops = True

        if update_wdprops:
            self.prop_score_fn = HierarchyScoringFn.construct(
                items=list(self.wdprops.keys()),
                get_item_parents=lambda x: self.all_wdprops[x].parents,
                get_item_uri=self.wdns.get_prop_abs_uri,
            )
        if update_wdclasses:
            self.class_score_fn = HierarchyScoringFn.construct(
                items=list(self.wdclasses.keys()),
                get_item_parents=lambda x: self.all_wdclasses[x].parents,
                get_item_uri=self.wdns.get_entity_abs_uri,
            )

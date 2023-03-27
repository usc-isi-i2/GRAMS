from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union
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
from grams.evaluation.scoring_fn import ItemType, get_hierarchy_scoring_fn
from grams.inputs.linked_table import CandidateEntityId, ExtendedLink, LinkedTable

from kgdata.wikidata.models import WDEntity, WDEntityLabel
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdproperty import WDProperty
from ned.metrics import inkb_eval_table
import numpy as np
from sm.dataset import Example
from sm.evaluation.prelude import (
    sm_metrics,
    CTAEvalOutput,
    _cpa_transformation,
    _get_cta,
    cta,
    PrecisionRecallF1,
)
from sm.misc.fn_cache import CacheMethod
from sm.misc.matrix import Matrix
from sm.outputs.semantic_model import (
    SemanticModel,
    Node as SMNode,
    ClassNode,
    DataNode,
    LiteralNode,
)


@dataclass
class WrappedSemanticModel:
    sm: SemanticModel
    is_normalized: bool = False
    is_cpa_transformed: bool = False


@dataclass
class CPAEvalRes:
    result: PrecisionRecallF1
    gold_sm: SemanticModel
    cpa_gold_sm: SemanticModel
    cpa_pred_sm: SemanticModel


class Evaluator:
    def __init__(
        self,
        entities: Mapping[str, WDEntity],
        entity_labels: Mapping[str, WDEntityLabel],
        classes: Mapping[str, WDClass],
        props: Mapping[str, WDProperty],
        cache_dir: Path,
    ):
        self.sm_helper = WikidataSemanticModelHelper(
            entities,
            IndirectDictAccess(entity_labels, attrgetter("label")),
            classes,
            props,
        )
        self.wdns = self.sm_helper.wdns
        self.class_scoring_fn = get_hierarchy_scoring_fn(
            cache_dir / "class_distance.sqlite", classes, ItemType.CLASS
        )
        self.prop_scoring_fn = get_hierarchy_scoring_fn(
            cache_dir / "prop_distance.sqlite", props, ItemType.PROPERTY
        )

    def cpa(self, example: Example[LinkedTable], sm: SemanticModel) -> CPAEvalRes:
        """Calculate the CPA score. The code is borrowed from: sm.evaluation.cpa_cta_metrics.cpa to adapt with this class API that uses WrappedSemanticModel"""
        gold_sms = self.get_example_gold_sms(example)
        cpa_pred_sm = self.convert_sm_for_cpa(self.norm_sm(sm))

        output = None
        best_cpa_gold_sm = None
        best_gold_sm = None

        for gold_sm in gold_sms:
            cpa_gold_sm = self.convert_sm_for_cpa(gold_sm)
            res = sm_metrics.precision_recall_f1(
                gold_sm=cpa_gold_sm.sm,
                pred_sm=cpa_pred_sm.sm,
                scoring_fn=self.prop_scoring_fn,
            )
            if output is None or res.f1 > output.f1:
                output = res
                best_cpa_gold_sm = cpa_gold_sm
                best_gold_sm = gold_sm

        assert (
            output is not None
            and best_cpa_gold_sm is not None
            and best_gold_sm is not None
        )

        return CPAEvalRes(
            result=output,
            gold_sm=best_gold_sm.sm,
            cpa_gold_sm=best_cpa_gold_sm.sm,
            cpa_pred_sm=cpa_pred_sm.sm,
        )

    def cta(self, example: Example[LinkedTable], sm: SemanticModel) -> CTAEvalOutput:
        gold_sms = self.get_example_gold_sms(example)
        pred_sm = self.norm_sm(sm)

        cta_output = max(
            [
                cta(
                    gold_sm.sm,
                    pred_sm.sm,
                    self.sm_helper.ID_PROPS,
                    self.class_scoring_fn,
                )
                for gold_sm in gold_sms
            ],
            key=attrgetter("f1"),
        )
        return cta_output

    def cea(self, example: Example[LinkedTable], k: Optional[Sequence[int]] = None):
        def convert_gold_ents(links: list[ExtendedLink]):
            out: set[str] = set()
            for link in links:
                out.update(link.entities)
            return out

        def convert_pred_ents(links: list[ExtendedLink]):
            out: list[CandidateEntityId] = list()
            for link in links:
                out.extend(link.candidates)
            # python sort is stable
            out = sorted(out, key=lambda x: x.probability, reverse=True)
            return [str(c.entity_id) for c in out]

        gold_ents = example.table.links.map(convert_gold_ents)
        pred_ents = example.table.links.map(convert_pred_ents)

        perf, cm = inkb_eval_table(gold_ents, pred_ents, k)
        return {"value": perf, "confusion_matrix": cm}

    def cpa_at_k(
        self,
        example: Example[LinkedTable],
        cg: CGGraph,
        edge_probs: dict[tuple[str, str, str], float],
        k: Optional[Union[int, Sequence[Optional[int]]]] = None,
    ):
        """Compute CPA performance @K between nodes (such as columns or entities) presented only in the ground truth.

        To compute top K, we group by pairs of non-statement nodes participating in the main property of the n-ary relationships.
        Then, the properties used between the pair are considered ground truth. If the model predicts any of the properties,
        it is considered correct. This implies that inversed properties are allowed, and we assume that the properties of the pair
        cannot be qualifiers in another relationship.

        The qualifiers of any relationships that the pair has are also grouped by one of the nodes in the pair and the target node of
        the considered qualifier.

        Because of the assumption that two nodes cannot be both participating in relationships that it is part of a statement property
        or part of a statement qualifier, it is equivalent to treating n-ary relationships as binary relationships.

        Args:
            example:
            cg:
            edge_probs: probability of each edge in the predicted candidate graph
        """

        def get_sm_node_id(u: SMNode):
            if isinstance(u, DataNode):
                return u.col_index
            if isinstance(u, LiteralNode):
                return u.value
            raise Exception("Unreachable")

        def get_cg_node_id(u: CGNode):
            if isinstance(u, CGColumnNode):
                return u.column
            if isinstance(u, CGEntityValueNode):
                return u.get_literal_node_value(self.wdns)
            if isinstance(u, CGLiteralValueNode):
                return u.get_literal_node_value()
            raise Exception("Unreachable")

        gold_sms = self.get_example_gold_sms(example)

        # mapping between a pair of column to its index
        rel2index: dict[tuple[int | str, str | int], int] = {}
        gold_rels: list[set[str]] = []

        for sm in gold_sms:
            cpa_sm = self.convert_sm_for_cpa(sm).sm
            for node in cpa_sm.iter_nodes():
                if (
                    not isinstance(node, ClassNode)
                    or node.abs_uri != self.wdns.STATEMENT_URI
                ):
                    continue

                (inedge,) = cpa_sm.in_edges(node.id)
                outedges = cpa_sm.out_edges(node.id)

                uid = get_sm_node_id(cpa_sm.get_node(inedge.source))

                for outedge in outedges:
                    vid = get_sm_node_id(cpa_sm.get_node(outedge.target))
                    if (uid, vid) not in rel2index:
                        rel2index[uid, vid] = len(rel2index)
                        gold_rels.append(set())
                    gold_rels[rel2index[uid, vid]].add(
                        self.wdns.get_prop_id(outedge.abs_uri)
                    )

        tmp_pred_rels: list[dict[str, float]] = [{} for _ in range(len(rel2index))]

        for s in cg.iter_nodes():
            if not isinstance(s, CGStatementNode):
                continue
            (inedge,) = cg.in_edges(s.id)
            outedges = cg.out_edges(s.id)

            uid = get_cg_node_id(cg.get_node(inedge.source))
            for outedge in outedges:
                vid = get_cg_node_id(cg.get_node(outedge.target))
                if (uid, vid) not in rel2index:
                    continue
                tmp_pred_rels[rel2index[uid, vid]][outedge.predicate] = edge_probs[
                    outedge.source, outedge.target, outedge.predicate
                ]

        pred_rels = [
            [
                k
                for k, v in sorted(
                    tmp_pred_rels[idx].items(), key=itemgetter(1), reverse=True
                )
            ]
            for (uid, vid), idx in rel2index.items()
        ]

        return inkb_eval_table(Matrix([gold_rels]), Matrix([pred_rels]), k)

    def cta_at_k(
        self,
        example: Example[LinkedTable],
        cta_probs: dict[int, dict[str, float]],
        k: Optional[Union[int, Sequence[Optional[int]]]] = None,
    ):
        """Compute CTA performance @K. The classes of a column is aggregated from multiple semantic models so it's possible
        that perf@1 is different from the results of cta(). Also, it doesn't use approximate precision/recall so the results
        can also be different.
        """
        gold_sms = self.get_example_gold_sms(example)

        # re-use inkb_eval_table to compute CTA performance @K
        gold_types: list[set[str]] = []
        pred_types: list[list[str]] = []
        for c in example.table.table.columns:
            assert c.index == len(gold_types)
            gold_types.append(set())
            pred_types.append(
                [
                    k
                    for k, v in sorted(
                        cta_probs.get(c.index, {}).items(),
                        key=itemgetter(1),
                        reverse=True,
                    )
                ]
            )
        for sm in gold_sms:
            for c, type in _get_cta(sm.sm, self.sm_helper.ID_PROPS).items():
                gold_types[int(c)].add(self.wdns.get_entity_id(type))

        return inkb_eval_table(Matrix([gold_types]), Matrix([pred_types]), k)

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_example_gold_sms(
        self, example: Example[LinkedTable]
    ) -> list[WrappedSemanticModel]:
        return self.get_equiv_sms(example.sms)

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def norm_sm(self, sm: SemanticModel) -> WrappedSemanticModel:
        return WrappedSemanticModel(self.sm_helper.norm_sm(sm), is_normalized=True)

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def convert_sm_for_cpa(
        self, wrapped_sm: WrappedSemanticModel
    ) -> WrappedSemanticModel:
        """Convert a semantic model to another model for evaluating the CPA task:
        - SemModelTransformation.replace_class_nodes_by_subject_columns(sm, id_props)
        - SemModelTransformation.remove_isolated_nodes(sm)
        """
        assert wrapped_sm.is_normalized
        if wrapped_sm.is_cpa_transformed:
            return wrapped_sm

        sm = wrapped_sm.sm
        cpa_sm = sm.deep_copy()
        _cpa_transformation(cpa_sm, self.sm_helper.ID_PROPS)

        return WrappedSemanticModel(cpa_sm, is_normalized=True, is_cpa_transformed=True)

    def get_equiv_sms(self, sms: list[SemanticModel]) -> list[WrappedSemanticModel]:
        return [
            WrappedSemanticModel(equiv_sm, is_normalized=True)
            for sm in sms
            for equiv_sm in self.sm_helper.gen_equivalent_sm(
                sm, strict=False, incorrect_invertible_props={"P571", "P582"}
            )
        ]

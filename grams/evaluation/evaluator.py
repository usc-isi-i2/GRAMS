from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union
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

    def cta_at_k(
        self,
        example: Example[LinkedTable],
        cta_probs: dict[int, dict[str, float]],
        k: Optional[Union[int, Sequence[Optional[int]]]] = None,
    ):
        """Compute CTA performance @K."""
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

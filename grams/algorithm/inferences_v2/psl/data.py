from __future__ import annotations
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.context import AlgoContext
from grams.algorithm.inferences_v2.features.misc_feature import (
    BinaryPredicate,
    UnaryPredicate,
)
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models.wdproperty import WDProperty

import numpy as np
from dataclasses import dataclass
from typing import Any
from grams.algorithm.inferences_v2.features.helper import MISSING_VALUE, IDMap
from grams.algorithm.inferences_v2.features.inf_feature import InfFeature
from grams.algorithm.inferences_v2.psl.predicates import P
from sm.misc.fn_cache import CacheMethod


@dataclass
class PSLData:
    idmap: IDMap
    observations: dict[str, list]
    targets: dict[str, list]
    # features that are classified as relation features
    rel_feats: set[str]
    # features that are classified as type features
    type_feats: set[str]
    # features that are classified as structure features
    struct_feats: set[str]

    def filter_observations(self, predicates: set[str] | dict[str, Any]):
        observations = {k: v for k, v in self.observations.items() if k in predicates}
        return PSLData(
            self.idmap,
            observations,
            self.targets,
            self.rel_feats,
            self.type_feats,
            self.struct_feats,
        )

    @staticmethod
    def from_inf_features(feat: InfFeature) -> PSLData:
        observations = {}

        efeat = feat.edge_features
        ufeat = feat.node_features
        mfeat = feat.misc_features

        # fmt: off
        pred2feat = {
            P.RelFreqOverRow.name(): efeat.freq_over_row,
            P.RelFreqOverEntRow.name(): efeat.freq_over_ent_row,
            P.RelFreqOverPosRel.name(): efeat.freq_over_pos_rel,
            P.RelFreqUnmatchOverEntRow.name(): efeat.freq_unmatch_over_ent_row,
            P.RelFreqUnmatchOverPosRel.name(): efeat.freq_unmatch_over_pos_rel,
            P.RelNotFuncDependency.name(): efeat.not_func_dependency,

            P.TypeFreqOverRow.name(): ufeat.freq_over_row,
            P.TypeFreqOverEntRow.name(): ufeat.freq_over_ent_row,
            P.ExtendedTypeFreqOverRow.name(): ufeat.extended_freq_over_row,
            P.ExtendedTypeFreqOverEntRow.name(): ufeat.extended_freq_over_ent_row,
            P.TypeDiscoveredPropFreqOverRow.name(): ufeat.freq_discovered_prop_over_row,
            P.TypeDistance.name(): ufeat.type_distance,
        }

        for p in [P.RelFreqOverRow, P.RelFreqOverEntRow, P.RelFreqOverPosRel, P.RelFreqUnmatchOverEntRow, P.RelFreqUnmatchOverPosRel]:
            observations[p.name()] = np.stack(
                [efeat.source, efeat.target, efeat.statement, efeat.outprop, pred2feat[p.name()]],
                axis=1, dtype=np.object_
            )

        for p in [P.TypeFreqOverRow, P.TypeFreqOverEntRow, P.ExtendedTypeFreqOverRow, P.ExtendedTypeFreqOverEntRow, P.TypeDiscoveredPropFreqOverRow, P.TypeDistance]:
            observations[p.name()] = np.stack(
                [ufeat.node, ufeat.type, pred2feat[p.name()]],
                axis=1, dtype=np.object_
            )
        
        mask = efeat.not_func_dependency != MISSING_VALUE
        observations[P.RelNotFuncDependency.name()] = np.unique(np.stack([efeat.source[mask], efeat.target[mask], efeat.not_func_dependency[mask]], axis=1), axis=0)
        observations[P.Column.name()] = _stack_unnary_predicate(mfeat.column)
        observations[P.SubProp.name()] = _stack_binary_predicate(mfeat.subprop)
        observations[P.DataProperty.name()] = _stack_unnary_predicate(mfeat.dataproperty)
        observations[P.PropertyDomain.name()] = _stack_binary_predicate(mfeat.property_domain)
        observations[P.PropertyRange.name()] = _stack_binary_predicate(mfeat.property_range)
        observations[P.Rel.name()] = np.stack([efeat.source, efeat.target, efeat.statement, efeat.outprop], axis=1)
        observations[P.Type.name()] = np.stack([ufeat.node, ufeat.type], axis=1)
        observations[P.StatementProperty.name()] = np.unique(np.stack([efeat.statement, efeat.inprop], axis=1), axis=0)
        
        for p in [P.TypeHeaderSimilarity, P.RelHeaderSimilarity]:
            # we do not have feature for this predicate yet.
            observations[p.name()] = []
        
        targets = {} 
        targets[P.CorrectRel.name()] = observations[P.Rel.name()]
        targets[P.CorrectType.name()] = observations[P.Type.name()]
        # fmt: on

        return PSLData(
            idmap=feat.idmap,
            observations=observations,
            targets=targets,
            rel_feats={
                P.RelFreqOverRow.name(),
                P.RelFreqOverEntRow.name(),
                P.RelFreqOverPosRel.name(),
                P.RelFreqUnmatchOverEntRow.name(),
                P.RelFreqUnmatchOverPosRel.name(),
                P.RelNotFuncDependency.name(),
            },
            type_feats={
                P.TypeFreqOverRow.name(),
                P.TypeFreqOverEntRow.name(),
                P.ExtendedTypeFreqOverRow.name(),
                P.ExtendedTypeFreqOverEntRow.name(),
                P.TypeDiscoveredPropFreqOverRow.name(),
                P.TypeDistance.name(),
            },
            struct_feats=set(),
        )


def _stack_binary_predicate(pred: BinaryPredicate):
    return np.stack([pred.var1, pred.var2, pred.value], axis=1, dtype=np.object_)


def _stack_unnary_predicate(pred: UnaryPredicate):
    return np.stack([pred.var, pred.value], axis=1, dtype=np.object_)

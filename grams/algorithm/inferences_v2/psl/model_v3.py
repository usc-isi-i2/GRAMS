from __future__ import annotations
from pathlib import Path
from typing import (
    Mapping,
    Optional,
)
import numpy as np
from grams.algorithm.inferences.psl_lib import (
    PSLModel,
    RuleContainer,
)
from grams.algorithm.inferences_v2.features.inf_feature import InfFeature
from grams.algorithm.inferences_v2.psl.data import PSLData
from grams.algorithm.inferences.features.string_similarity import StringSimilarity
from pslpython.predicate import Predicate
from grams.algorithm.inferences_v2.psl.predicates import P
from grams.algorithm.inferences_v2.psl.config import PslConfig
from pslpython.rule import Rule
from loguru import logger


class PSLModelv3:
    VERSION = 100

    def __init__(
        self,
        config: PslConfig,
        temp_dir: Optional[Path],
    ):
        self.cfg = config
        self.temp_dir = temp_dir

        self.sim_fn = StringSimilarity.hybrid_jaccard_similarity
        self.disable_rules = set(config.disable_rules)
        self.model = self.get_model()

        default_weights = {
            P.Rel.name() + "_PRIOR_NEG": 1,
            P.Rel.name() + "_PRIOR_NEG_PARENT": 0.1,
            P.RelFreqOverRow.name(): 2,
            P.RelFreqOverEntRow.name(): 2,
            P.RelFreqOverPosRel.name(): 2,
            P.RelFreqUnmatchOverEntRow.name(): 2,
            P.RelFreqUnmatchOverPosRel.name(): 2,
            P.RelHeaderSimilarity.name(): 0.0,
            P.RelNotFuncDependency.name(): 100,
            P.Type.name() + "_PRIOR_NEG": 1,
            "TYPE_PRIOR_NEG_PARENT": 0.1,
            P.TypeFreqOverRow.name(): 2,
            P.TypeFreqOverEntRow.name(): 0,
            P.ExtendedTypeFreqOverRow.name(): 2,
            P.ExtendedTypeFreqOverEntRow.name(): 0,
            P.TypeHeaderSimilarity.name(): 0.0,
            P.TypeDiscoveredPropFreqOverRow.name(): 2,
            P.DataProperty.name(): 1,
            P.PropertyDomain.name(): 1,
            P.PropertyRange.name(): 1,
        }
        if len(config.rule_weights) > 0:
            default_weights.update(config.rule_weights)
        self.model.set_parameters(default_weights)

    def get_model(self):
        # ** DEFINE RULES **
        rules = RuleContainer()
        for feat in [
            P.RelFreqOverRow,
            P.RelFreqOverEntRow,
            P.RelFreqOverPosRel,
            P.RelHeaderSimilarity,
        ]:
            rules[feat.name()] = Rule(
                f"{P.Rel.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> {P.CorrectRel.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        for feat in [
            P.RelFreqUnmatchOverEntRow,
            P.RelFreqUnmatchOverPosRel,
        ]:
            rules[feat.name()] = Rule(
                f"{P.Rel.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> ~{P.CorrectRel.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        rules[P.RelNotFuncDependency.name()] = Rule(
            f"""
            {P.Rel.name()}(N1, N2, S1, P) &
            {P.Rel.name()}(N2, N3, S2, P2) &
            {P.CorrectRel.name()}(N1, N2, S1, P) &
            {P.RelNotFuncDependency.name()}(N2, N3) -> ~{P.CorrectRel.name()}(N2, N3, S2, P2)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )

        # ontology rules
        # the problem with this rule in the new setting (no statement node) is that the result of HasType
        # affects the prob. of the edge.
        # rules["HAS_TYPE_HAS_OUT_EDGE"] = Rule(
        #     f"""
        #     {P.RelProp.name()}(U, V, S, P) & ~{P.HasType.name()}(U) -> ~{P.CorrectRelProp.name()}(U, V, S, P)
        #     """,
        #     weighted=True,
        #     squared=True,
        #     weight=0.0,
        # )
        # target of a data property can't be an entity
        rules[P.DataProperty.name()] = Rule(
            f"""
            {P.Rel.name()}(U, V, S, P) & {P.DataProperty.name()}(P) & {P.CorrectRel.name()}(U, V, S, P) -> ~{P.CorrectType.name()}(V, T)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.PropertyDomain.name()] = Rule(
            # f"""
            # {P.Rel.name()}(U, V, S, P) & {P.Column.name()}(U) & {P.StatementProperty.name()}(S, P) &
            # {P.CorrectRel.name()}(U, V, S, P) & {P.PropertyDomain.name()}(P, T) & {P.Type.name()}(U, T) -> {P.CorrectType.name()}(U, T)
            # """,
            f"""
            {P.Rel.name()}(U, V, S, P) & {P.Column.name()}(U) & {P.StatementProperty.name()}(S, P) &
            {P.CorrectType.name()}(U, T) & ~{P.PropertyDomain.name()}(P, T) -> ~{P.CorrectRel.name()}(U, V, S, P)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.PropertyRange.name()] = Rule(
            # f"""
            # {P.Rel.name()}(U, V, S, P) & {P.Column.name()}(V) &
            # ~{P.DataProperty.name()}(P) & {P.CorrectRel.name()}(U, V, S, P) & {P.PropertyRange.name()}(P, T) & {P.Type.name()}(V, T) -> {P.CorrectType.name()}(V, T)
            # """,
            f"""
            {P.Rel.name()}(U, V, S, P) & {P.Column.name()}(V) &
            ~{P.DataProperty.name()}(P) & {P.CorrectType.name()}(V, T) & ~{P.PropertyRange.name()}(P, T) -> ~{P.CorrectRel.name()}(U, V, S, P)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )

        # prefer details prop/type
        rules[P.Rel.name() + "_PRIOR_NEG_PARENT"] = Rule(
            f"""
            {P.Rel.name()}(U, V, S, P) & {P.Rel.name()}(U, V, S2, PP) & (S != S2) &
            {P.SubProp.name()}(P, PP) -> ~{P.CorrectRel.name()}(U, V, S2, PP)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules["TYPE_PRIOR_NEG_PARENT"] = Rule(
            f"{P.Type.name()}(N, T) & {P.TypeDistance.name()}(N, T) -> ~{P.CorrectType.name()}(N, T)",
            weighted=True,
            weight=0.0,
            squared=True,
        )

        # default negative rel/types
        rules[P.Rel.name() + "_PRIOR_NEG"] = Rule(
            f"~{P.CorrectRel.name()}(U, V, S, P)",
            weight=0.0,
            weighted=True,
            squared=True,
        )
        rules["TYPE_PRIOR_NEG"] = Rule(
            f"~{P.CorrectType.name()}(N, T)", weight=0.0, weighted=True, squared=True
        )
        # disable as we do not need hastype
        # rules[P.HasType.name() + "_PRIOR_NEG"] = Rule(
        #     f"~{P.HasType.name()}(N)", weight=0.0, weighted=True, squared=True
        # )
        # rules["CORRECT_TYPE_IMPLY_HAS_TYPE"] = Rule(
        #     f"{P.CorrectType.name()}(N, T) -> {P.HasType.name()}(N)",
        #     weight=0.0,
        #     weighted=True,
        #     squared=True,
        # )
        # don't use this rule as the more candidate types we have, the more likely that they all have low probabilities.
        # Rule(f"{P.CorrectType.name()}(N, +T) <= 1", weighted=False)

        for feat in [
            P.TypeFreqOverRow,
            P.TypeFreqOverEntRow,
            P.ExtendedTypeFreqOverRow,
            P.ExtendedTypeFreqOverEntRow,
            P.TypeHeaderSimilarity,
            P.TypeDiscoveredPropFreqOverRow,
        ]:
            rules[feat.name()] = Rule(
                f"""
                {P.Type.name()}(N, T) & {feat.name()}(N, T) -> {P.CorrectType.name()}(N, T)
                """,
                weighted=True,
                weight=0.0,
                squared=True,
            )

        if self.disable_rules.difference(rules.keys()):
            raise Exception(
                f"Attempt to disable rules: {list(self.disable_rules.difference(rules.keys()))} that are not defined"
            )

        for rule in self.disable_rules:
            rules.pop(rule)

        return PSLModel(
            rules=rules,
            predicates=[x for x in vars(P).values() if isinstance(x, Predicate)],
            ignore_predicates_not_in_rules=True,
            temp_dir=str(self.temp_dir),
            required_predicates={P.StatementProperty.name()},
        )

    def predict(
        self,
        inffeat: InfFeature,
        verbose: bool = False,
    ):
        """Predict prob. of edges and prob. of columns' type"""
        data = PSLData.from_inf_features(inffeat)
        newdata = data.filter_observations(self.model.name2predicate)
        observations, targets = (
            newdata.observations,
            newdata.targets,
        )
        if len(newdata.observations) != len(data.observations):
            logger.warning(
                "The code is not efficient as it extracts more features than needed: {}",
                set(data.observations.keys()).difference(newdata.observations),
            )

        if verbose:
            for pname in observations:
                if len(observations[pname]) == 0:
                    logger.debug(f"No observations for predicate {pname}")

        output = self.model.predict(
            observations, targets, {}, force_setall=True, cleanup_tempdir=True
        )
        correct_rels = self.model.normalize_probs(
            output[P.CorrectRel.name()], eps=self.cfg.eps
        )
        correct_types = output[P.CorrectType.name()]

        edgefeat = inffeat.edge_features
        edge_probs = [
            correct_rels[
                edgefeat.source[i],
                edgefeat.target[i],
                edgefeat.statement[i],
                edgefeat.outprop[i],
            ]
            for i in range(len(edgefeat))
        ]
        nodefeat = inffeat.node_features
        node_probs = [
            correct_types[nodefeat.node[i], nodefeat.type[i]]
            for i in range(len(nodefeat))
        ]

        return np.array(edge_probs, dtype=np.float64), np.array(
            node_probs, dtype=np.float64
        )

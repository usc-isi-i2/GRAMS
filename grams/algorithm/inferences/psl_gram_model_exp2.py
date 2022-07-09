from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Literal
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdgeTriple,
    CGGraph,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.features.structure_feature2 import StructureFeature
from grams.algorithm.inferences.psl_lib import (
    IDMap,
    PSLModel,
    ReadableIDMap,
    RuleContainer,
)
from grams.algorithm.inferences.features.rel_feature2 import RelFeatures
from grams.algorithm.inferences.features.type_feature import (
    TypeFeatures,
)
from grams.inputs.linked_table import Link, LinkedTable
from kgdata.wikidata.db import get_wdprop_domain_db
from kgdata.wikidata.models import (
    WDEntity,
    WDEntityLabel,
    WDClass,
    WDProperty,
    WDQuantityPropertyStats,
)
from grams.algorithm.inferences.features.string_similarity import StringSimilarity
from kgdata.wikidata.models.wdproperty import WDPropertyDomains, WDPropertyRanges
from pslpython.predicate import Predicate
from pslpython.model import Model
from dataclasses import dataclass

from pslpython.rule import Rule
from loguru import logger


class P:
    """Holding list of predicates in the model."""

    # target predicates
    # CorrectRel = Predicate("CORRECT_REL", closed=False, size=3)
    CorrectRelProp = Predicate("CORRECT_REL_PROP", closed=False, size=4)
    CorrectRelQual = Predicate("CORRECT_REL_QUAL", closed=False, size=4)
    CorrectType = Predicate("CORRECT_TYPE", closed=False, size=2)
    # surrogated target predicates
    HasType = Predicate("HAS_TYPE", closed=False, size=1)

    # graph structure
    # Rel = Predicate("REL", closed=True, size=3)
    RelProp = Predicate("REL_PROP", closed=True, size=4)
    RelQual = Predicate("REL_QUAL", closed=True, size=4)
    Type = Predicate("TYPE", closed=True, size=2)
    # Statement = Predicate("STATEMENT", closed=True, size=1)
    Column = Predicate("COLUMN", closed=True, size=1)

    # ontology
    SubProp = Predicate("SUB_PROP", closed=True, size=2)
    # SubType = Predicate("SUB_TYPE", closed=True, size=2)
    # HasSubType = Predicate("HAS_SUB_TYPE", closed=True, size=2)
    TypeDistance = Predicate("TYPE_DISTANCE", closed=True, size=2)
    DataProperty = Predicate("DATA_PROPERTY", closed=True, size=1)
    PropertyDomain = Predicate("PROPERTY_DOMAIN", closed=True, size=2)  # (prop, domain)
    PropertyRange = Predicate("PROPERTY_RANGE", closed=True, size=2)

    # features
    RelFreqOverRow = Predicate("REL_FREQ_OVER_ROW", closed=True, size=4)
    RelFreqOverEntRow = Predicate("REL_FREQ_OVER_ENT_ROW", closed=True, size=4)
    RelFreqOverPosRel = Predicate("REL_FREQ_OVER_POS_REL", closed=True, size=4)
    RelFreqUnmatchOverEntRow = Predicate(
        "REL_FREQ_UNMATCH_OVER_ENT_ROW", closed=True, size=4
    )
    RelFreqUnmatchOverPosRel = Predicate(
        "REL_FREQ_UNMATCH_OVER_POS_REL", closed=True, size=4
    )
    RelNotFuncDependency = Predicate("REL_NOT_FUNC_DEPENDENCY", closed=True, size=2)
    RelHeaderSimilarity = Predicate("REL_HEADER_SIMILARITY", closed=True, size=4)

    TypeFreqOverRow = Predicate("TYPE_FREQ_OVER_ROW", closed=True, size=2)
    TypeFreqOverEntRow = Predicate("TYPE_FREQ_OVER_ENT_ROW", closed=True, size=2)
    ExtendedTypeFreqOverRow = Predicate(
        "EXTENDED_TYPE_FREQ_OVER_ROW", closed=True, size=2
    )
    ExtendedTypeFreqOverEntRow = Predicate(
        "EXTENDED_TYPE_FREQ_OVER_ENT_ROW", closed=True, size=2
    )
    TypeHeaderSimilarity = Predicate("TYPE_HEADER_SIMILARITY", closed=True, size=2)


class PSLGramModelExp2:
    def __init__(
        self,
        wdentities: Mapping[str, WDEntity],
        wdentity_labels: Mapping[str, WDEntityLabel],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wdprop_domains: Optional[Mapping[str, WDPropertyDomains]],
        wdprop_ranges: Optional[Mapping[str, WDPropertyRanges]],
        wd_numprop_stats: Mapping[str, WDQuantityPropertyStats],
        disable_rules: Optional[Iterable[str]] = None,
        example_id: Optional[str] = None,
    ):
        self.wdentities = wdentities
        self.wdentity_labels = wdentity_labels
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wdprop_domains = wdprop_domains
        self.wdprop_ranges = wdprop_ranges
        self.wd_numprop_stats = wd_numprop_stats
        self.sim_fn = StringSimilarity.hybrid_jaccard_similarity
        self.disable_rules = set(disable_rules or [])
        self.example_id = example_id
        self.model = self.get_model()

        self.model.set_parameters(
            {
                P.RelProp.name() + "_PRIOR_NEG": 1,
                P.RelQual.name() + "_PRIOR_NEG": 1,
                P.RelProp.name() + "_PRIOR_NEG_PARENT": 0.1,
                P.RelQual.name() + "_PRIOR_NEG_PARENT": 0.1,
                P.RelFreqOverRow.name() + "_PROP": 3,
                P.RelFreqOverRow.name() + "_QUAL": 3,
                P.RelFreqOverEntRow.name() + "_PROP": 3,
                P.RelFreqOverEntRow.name() + "_QUAL": 3,
                P.RelFreqOverPosRel.name() + "_PROP": 2,
                P.RelFreqOverPosRel.name() + "_QUAL": 2,
                P.RelFreqUnmatchOverEntRow.name() + "_PROP": 2,
                P.RelFreqUnmatchOverEntRow.name() + "_QUAL": 2,
                P.RelFreqUnmatchOverPosRel.name() + "_PROP": 2,
                P.RelFreqUnmatchOverPosRel.name() + "_QUAL": 2,
                P.RelHeaderSimilarity.name() + "_PROP": 0.0,
                P.RelHeaderSimilarity.name() + "_QUAL": 0.0,
                P.RelNotFuncDependency.name(): 100,
                P.Type.name() + "_PRIOR_NEG": 1,
                P.HasType.name() + "_PRIOR_NEG": 0.1,
                "TYPE_PRIOR_NEG_PARENT": 0.1,
                P.TypeFreqOverRow.name(): 2,
                P.TypeFreqOverEntRow.name(): 0,
                P.ExtendedTypeFreqOverRow.name(): 2,
                P.ExtendedTypeFreqOverEntRow.name(): 0,
                P.TypeHeaderSimilarity.name(): 0.0,
                "CASCADING_ERROR": 2,
                P.DataProperty.name(): 2,
                "HAS_TYPE_HAS_OUT_EDGE": 2,
                P.PropertyDomain.name(): 2,
                P.PropertyRange.name() + "_PROP": 2,
                P.PropertyRange.name() + "_QUAL": 2,
                "CORRECT_TYPE_IMPLY_HAS_TYPE": 6,
            }
        )

    def get_model(self):
        # ** DEFINE RULES **
        rules = RuleContainer()
        for feat in [
            P.RelFreqOverRow,
            P.RelFreqOverEntRow,
            P.RelFreqOverPosRel,
            P.RelHeaderSimilarity,
        ]:
            rules[feat.name() + "_PROP"] = Rule(
                f"{P.RelProp.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> {P.CorrectRelProp.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )
            rules[feat.name() + "_QUAL"] = Rule(
                f"{P.RelQual.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> {P.CorrectRelQual.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        for feat in [
            P.RelFreqUnmatchOverEntRow,
            P.RelFreqUnmatchOverPosRel,
        ]:
            rules[feat.name() + "_PROP"] = Rule(
                f"{P.RelProp.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> ~{P.CorrectRelProp.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )
            rules[feat.name() + "_QUAL"] = Rule(
                f"{P.RelQual.name()}(N1, N2, S, P) & {feat.name()}(N1, N2, S, P) -> ~{P.CorrectRelQual.name()}(N1, N2, S, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        rules[P.RelNotFuncDependency.name()] = Rule(
            f"""
            {P.RelProp.name()}(N1, N2, S1, P) &
            {P.RelProp.name()}(N2, N3, S2, P2) &
            {P.CorrectRelProp.name()}(N1, N2, S1, P) &
            {P.RelNotFuncDependency.name()}(N2, N3) -> ~{P.CorrectRelProp.name()}(N2, N3, S2, P2)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )

        # ontology rules
        # target of a data property can't be an entity
        rules[P.DataProperty.name()] = Rule(
            f"""
            {P.RelProp.name()}(U, V, S, P) & {P.DataProperty.name()}(P) & {P.CorrectRelProp.name()}(U, V, S, P) -> ~{P.CorrectType.name()}(V, T)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules["HAS_TYPE_HAS_OUT_EDGE"] = Rule(
            f"""
            {P.RelProp.name()}(U, V, S, P) & ~{P.HasType.name()}(U) -> ~{P.CorrectRelProp.name()}(U, V, S, P)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.PropertyDomain.name()] = Rule(
            f"""
            {P.RelProp.name()}(U, V, S, P) & {P.Column.name()}(U) &
            {P.CorrectType.name()}(U, T) & ~{P.PropertyDomain.name()}(P, T) -> ~{P.CorrectRelProp.name()}(U, V, S, P)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.PropertyRange.name() + "_PROP"] = Rule(
            f"""
            {P.RelProp.name()}(U, V, S, P) & {P.Column.name()}(V) &
            ~{P.DataProperty.name()}(P) & {P.CorrectType.name()}(V, T) & ~{P.PropertyRange.name()}(P, T) -> ~{P.CorrectRelProp.name()}(U, V, S, P)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.PropertyRange.name() + "_QUAL"] = Rule(
            f"""
            {P.RelQual.name()}(U, V, S, Q) & {P.Column.name()}(V) &
            ~{P.DataProperty.name()}(Q) & {P.CorrectType.name()}(V, T) & ~{P.PropertyRange.name()}(Q, T) -> ~{P.CorrectRelQual.name()}(U, V, S, Q)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )

        # prefer details prop/type
        rules[P.RelProp.name() + "_PRIOR_NEG_PARENT"] = Rule(
            f"""
            {P.RelProp.name()}(U, V, S, P) & {P.RelProp.name()}(U, V, S2, PP) & (S != S2) &
            {P.SubProp.name()}(P, PP) -> ~{P.CorrectRelProp.name()}(U, V, S2, PP)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules[P.RelQual.name() + "_PRIOR_NEG_PARENT"] = Rule(
            f"""
            {P.RelQual.name()}(U, V, S, P) & {P.RelQual.name()}(U, V, S2, PP) & (S != S2) &
            {P.SubProp.name()}(P, PP) -> ~{P.CorrectRelQual.name()}(U, V, S2, PP)
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
        rules[P.RelProp.name() + "_PRIOR_NEG"] = Rule(
            f"~{P.CorrectRelProp.name()}(U, V, S, P)",
            weight=0.0,
            weighted=True,
            squared=True,
        )
        rules[P.RelQual.name() + "_PRIOR_NEG"] = Rule(
            f"~{P.CorrectRelQual.name()}(U, V, S, P)",
            weight=0.0,
            weighted=True,
            squared=True,
        )
        rules["TYPE_PRIOR_NEG"] = Rule(
            f"~{P.CorrectType.name()}(N, T)", weight=0.0, weighted=True, squared=True
        )
        rules[P.HasType.name() + "_PRIOR_NEG"] = Rule(
            f"~{P.HasType.name()}(N)", weight=0.0, weighted=True, squared=True
        )
        rules["CORRECT_TYPE_IMPLY_HAS_TYPE"] = Rule(
            f"{P.CorrectType.name()}(N, T) -> {P.HasType.name()}(N)",
            weight=0.0,
            weighted=True,
            squared=True,
        )
        # don't use this rule as the more candidate types we have, the more likely that they all have low probabilities.
        # Rule(f"{P.CorrectType.name()}(N, +T) <= 1", weighted=False)

        rules["CASCADING_ERROR"] = Rule(
            f"""
            {P.RelProp.name()}(U, V1, S, P) & {P.RelQual.name()}(U, V2, S, Q) &
            ~{P.CorrectRelProp.name()}(U, V1, S, P) & (V1 != V2) -> ~{P.CorrectRelQual.name()}(U, V2, S, Q)
            """,
            weighted=True,
            weight=0.0,
            squared=True,
        )

        for feat in [
            P.TypeFreqOverRow,
            P.TypeFreqOverEntRow,
            P.ExtendedTypeFreqOverRow,
            P.ExtendedTypeFreqOverEntRow,
            P.TypeHeaderSimilarity,
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
            temp_dir=f"/tmp/pslpython/{self.example_id}"
            if self.example_id is not None
            else None,
        )

    def predict(
        self,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        verbose: bool = False,
        debug: bool = False,
    ) -> Tuple[Dict[CGEdgeTriple, float], Dict[int, Dict[str, float]]]:
        """Predict prob. of edges and prob. of columns' type"""
        idmap, observations, targets = self.extract_data(table, cg, dg)

        if verbose:
            for pname in observations:
                if len(observations[pname]) == 0:
                    logger.debug(f"No observations for predicate {pname}")

        output = self.model.predict(
            observations, targets, {}, force_setall=True, cleanup_tempdir=not debug
        )
        rel_probs = {}
        for terms, prob in output[P.CorrectRelProp.name()].items():
            uid, vid, sid, prop = [idmap.im(t) for t in terms]
            rel_probs[uid, sid, prop] = prob
            rel_probs[sid, vid, prop] = prob
        for terms, prob in output[P.CorrectRelQual.name()].items():
            uid, vid, sid, qual = [idmap.im(t) for t in terms]
            rel_probs[sid, vid, qual] = prob

        type_probs = {}
        for terms, prob in output[P.CorrectType.name()].items():
            u = cg.get_node(idmap.im(terms[0]))
            class_id = idmap.im(terms[1])
            assert isinstance(u, CGColumnNode)
            type_probs.setdefault(u.column, {})[class_id] = prob

        if debug:
            self.model.debug(idmap)

        # dict([x for x in rel_probs.items() if "".join(x[0]).find("column-4") != -1])

        return rel_probs, type_probs

    def extract_data(self, table: LinkedTable, cg: CGGraph, dg: DGGraph):
        """Extract data for our PSL model"""
        idmap = IDMap()
        # idmap = ReadableIDMap()

        rel_feats = RelFeatures(
            idmap,
            table,
            cg,
            dg,
            self.wdentities,
            self.wdentity_labels,
            self.wdclasses,
            self.wdprops,
            self.wd_numprop_stats,
            self.sim_fn,
        ).extract_features(
            [
                P.RelFreqOverRow.name(),
                P.RelFreqOverEntRow.name(),
                P.RelFreqOverPosRel.name(),
                P.RelFreqUnmatchOverEntRow.name(),
                P.RelFreqUnmatchOverPosRel.name(),
                P.RelNotFuncDependency.name(),
                P.RelHeaderSimilarity.name(),
            ]
        )

        type_feats = TypeFeatures(
            idmap,
            table,
            cg,
            dg,
            self.wdentities,
            self.wdclasses,
            self.wdprops,
            self.wd_numprop_stats,
            self.sim_fn,
        ).extract_features(
            [
                P.TypeFreqOverRow.name(),
                P.TypeFreqOverEntRow.name(),
                P.ExtendedTypeFreqOverRow.name(),
                P.ExtendedTypeFreqOverEntRow.name(),
                P.TypeDistance.name(),
                P.TypeHeaderSimilarity.name(),
            ]
        )

        candidate_types = {}
        for c, t, p in type_feats[P.TypeFreqOverRow.name()]:
            uid = idmap.im(c)
            if uid not in candidate_types:
                candidate_types[uid] = []
            candidate_types[uid].append(idmap.im(t))

        struct_feats = StructureFeature(
            idmap=idmap,
            table=table,
            cg=cg,
            dg=dg,
            wdentities=self.wdentities,
            wdentity_labels=self.wdentity_labels,
            wdclasses=self.wdclasses,
            wdprops=self.wdprops,
            wdprop_domains=self.wdprop_domains,
            wdprop_ranges=self.wdprop_ranges,
            wd_num_prop_stats=self.wd_numprop_stats,
            sim_fn=self.sim_fn,
            candidate_types=candidate_types,
        ).extract_features(
            [
                P.RelProp.name(),
                P.RelQual.name(),
                P.Type.name(),
                P.HasType.name(),
                P.Column.name(),
                P.SubProp.name(),
                P.DataProperty.name(),
                P.PropertyDomain.name(),
                P.PropertyRange.name(),
            ]
        )

        observations: Dict[str, list] = {}
        targets: Dict[str, list] = {}

        for p in self.model.model.get_predicates().values():
            p: Predicate
            if p.name() in rel_feats:
                observations[p.name()] = rel_feats[p.name()]
            if p.name() in type_feats:
                observations[p.name()] = type_feats[p.name()]
            if p.name() in struct_feats:
                observations[p.name()] = struct_feats[p.name()]

        targets[P.CorrectRelProp.name()] = observations[P.RelProp.name()].copy()
        targets[P.CorrectRelQual.name()] = observations[P.RelQual.name()].copy()
        targets[P.CorrectType.name()] = observations[P.Type.name()].copy()

        targets[P.HasType.name()] = [
            obs for obs in observations[P.HasType.name()] if len(obs) == 1
        ]
        observations[P.HasType.name()] = [
            obs for obs in observations[P.HasType.name()] if len(obs) == 2
        ]

        # fitlering predicates that are not in the model (e.g. because they are not used in any rule)
        filtered_observations = {
            k: v for k, v in observations.items() if k in self.model.name2predicate
        }

        if len(filtered_observations) != len(observations):
            logger.warning(
                "The code is not efficient as it extracts more features than needed: {}",
                set(observations.keys()).difference(filtered_observations),
            )

        return idmap, filtered_observations, targets

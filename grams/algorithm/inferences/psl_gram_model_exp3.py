from __future__ import annotations
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
)
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdgeTriple,
    CGGraph,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.features.structure_feature2 import StructureFeature
from grams.algorithm.inferences.psl_lib import (
    IDMap,
    PSLModel,
    ReadableIDMap,
    RuleContainer,
)
from grams.algorithm.inferences.features.rel_feature3 import RelFeatures3
from grams.algorithm.inferences.features.type_feature2 import (
    TypeFeatures2,
)
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.inferences.features.string_similarity import StringSimilarity
from pslpython.predicate import Predicate
from dataclasses import dataclass

from pslpython.rule import Rule
from loguru import logger


class P:
    """Holding list of predicates in the model."""

    # target predicates
    CorrectRel = Predicate("CORRECT_REL", closed=False, size=4)
    CorrectType = Predicate("CORRECT_TYPE", closed=False, size=2)

    # graph structure
    Rel = Predicate("REL", closed=True, size=4)
    Type = Predicate("TYPE", closed=True, size=2)
    Column = Predicate("COLUMN", closed=True, size=1)
    StatementProperty = Predicate("STATEMENT_PROPERTY", closed=True, size=2)

    # ontology
    SubProp = Predicate("SUB_PROP", closed=True, size=2)
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
    TypeDiscoveredPropFreqOverRow = Predicate(
        "TYPE_DISCOVERED_PROP_FREQ_OVER_ROW", closed=True, size=2
    )
    TypeHeaderSimilarity = Predicate("TYPE_HEADER_SIMILARITY", closed=True, size=2)


class PSLGramModelExp3:
    VERSION = 100

    def __init__(
        self,
        disable_rules: Optional[Iterable[str]] = None,
        rule_weights: Optional[Mapping[str, float]] = None,
        example_id: Optional[str] = None,
        use_readable_idmap: bool = False,
    ):
        self.sim_fn = StringSimilarity.hybrid_jaccard_similarity
        self.disable_rules = set(disable_rules or [])
        self.example_id = example_id
        self.use_readable_idmap = use_readable_idmap
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
        if rule_weights is not None:
            default_weights.update(rule_weights)
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
            temp_dir=f"/tmp/pslpython/{self.example_id}"
            if self.example_id is not None
            else None,
            required_predicates={P.StatementProperty.name()},
        )

    def predict(
        self,
        table: LinkedTable,
        cg: CGGraph,
        data: PSLData,
        verbose: bool = False,
        debug: bool = False,
    ) -> tuple[dict[CGEdgeTriple, float], dict[int, dict[str, float]]]:
        """Predict prob. of edges and prob. of columns' type"""
        newdata = data.filter_observations(self.model.name2predicate)
        idmap, observations, targets = (
            newdata.idmap,
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
            observations, targets, {}, force_setall=True, cleanup_tempdir=not debug
        )
        stmt2prop = dict(observations[P.StatementProperty.name()])
        rel_probs = {}
        for (u, v, s, p), prob in output[P.CorrectRel.name()].items():
            uid, vid, sid, prop = idmap.im(u), idmap.im(v), idmap.im(s), idmap.im(p)
            if stmt2prop[s] == p:
                rel_probs[uid, sid, prop] = prob
            rel_probs[sid, vid, prop] = prob

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


class PSLGramsModelData3:
    VERSION = 101

    def __init__(
        self,
        context: AlgoContext,
        sim_fn: Callable[
            [str, str], float
        ] = StringSimilarity.hybrid_jaccard_similarity,
        use_readable_idmap: bool = False,
    ):
        self.context = context
        self.sim_fn = sim_fn
        self.use_readable_idmap = use_readable_idmap

    def extract_data(self, table: LinkedTable, cg: CGGraph, dg: DGGraph):
        """Extract data for our PSL model"""
        idmap = ReadableIDMap() if self.use_readable_idmap else IDMap()

        rel_feats = RelFeatures3(idmap, table, cg, dg, self.context).extract_features(
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

        type_feats = TypeFeatures2(
            idmap,
            table,
            cg,
            dg,
            self.context,
            self.sim_fn,
        ).extract_features(
            [
                P.TypeFreqOverRow.name(),
                P.TypeFreqOverEntRow.name(),
                P.ExtendedTypeFreqOverRow.name(),
                P.ExtendedTypeFreqOverEntRow.name(),
                P.TypeDiscoveredPropFreqOverRow.name(),
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
            wdentities=self.context.wdentities,
            wdentity_labels=self.context.wdentity_labels,
            wdclasses=self.context.wdclasses,
            wdprops=self.context.wdprops,
            wdprop_domains=self.context.wdprop_domains,
            wdprop_ranges=self.context.wdprop_ranges,
            wd_num_prop_stats=self.context.wd_num_prop_stats,
            sim_fn=self.sim_fn,
            candidate_types=candidate_types,
        ).extract_features(
            [
                P.Rel.name(),
                P.Type.name(),
                P.StatementProperty.name(),
                P.Column.name(),
                P.SubProp.name(),
                P.DataProperty.name(),
                P.PropertyDomain.name(),
                P.PropertyRange.name(),
                "MIN_20_PERCENT_ENT_FROM_TYPE",
            ]
        )

        observations: dict[str, list] = {}
        observations.update(rel_feats)
        observations.update(type_feats)
        observations.update(struct_feats)

        targets: dict[str, list] = {}
        targets[P.CorrectRel.name()] = observations[P.Rel.name()].copy()
        targets[P.CorrectType.name()] = observations[P.Type.name()].copy()
        # targets[P.HasType.name()] = [
        #     obs for obs in observations[P.HasType.name()] if len(obs) == 1
        # ]
        # observations[P.HasType.name()] = [
        #     obs for obs in observations[P.HasType.name()] if len(obs) == 2
        # ]
        return PSLData(
            idmap,
            observations,
            targets,
            set(rel_feats.keys()),
            set(type_feats.keys()),
            set(struct_feats.keys()),
        )


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

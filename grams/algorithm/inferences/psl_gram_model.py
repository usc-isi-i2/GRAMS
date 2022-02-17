from __future__ import annotations
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Literal
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdgeTriple,
    CGGraph,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.psl_lib import IDMap, PSLModel, RuleContainer
from grams.algorithm.link_feature import LinkFeatureExtraction
from grams.algorithm.inferences.features.rel_feature import RelFeatures
from grams.algorithm.inferences.features.type_feature import (
    TypeFeatures,
)
from grams.inputs.linked_table import Link, LinkedTable
from kgdata.wikidata.models.qnode import QNode, QNodeLabel
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdproperty import WDProperty, WDQuantityPropertyStats
from pslpython.predicate import Predicate
from pslpython.model import Model
from dataclasses import dataclass

from pslpython.rule import Rule
from loguru import logger


class P:
    """Holding list of predicates in the model."""

    # target predicates
    CorrectRel = Predicate("CORRECT_REL", closed=False, size=3)
    CorrectType = Predicate("CORRECT_TYPE", closed=False, size=2)

    # graph structure
    Rel = Predicate("REL", closed=True, size=3)
    Type = Predicate("TYPE", closed=True, size=2)
    Statement = Predicate("STATEMENT", closed=True, size=1)

    # ontology
    SubProp = Predicate("SUB_PROP", closed=True, size=2)
    SubType = Predicate("SUB_TYPE", closed=True, size=2)
    HasSubType = Predicate("HAS_SUB_TYPE", closed=True, size=2)
    NotRange = Predicate("NOT_RANGE", closed=True, size=2)

    # features
    RelFreqOverRow = Predicate("REL_FREQ_OVER_ROW", closed=True, size=3)
    RelFreqOverEntRow = Predicate("REL_FREQ_OVER_ENT_ROW", closed=True, size=3)
    RelFreqOverPosRel = Predicate("REL_FREQ_OVER_POS_REL", closed=True, size=3)
    RelFreqUnmatchOverEntRow = Predicate(
        "REL_FREQ_UNMATCH_OVER_ENT_ROW", closed=True, size=3
    )
    RelFreqUnmatchOverPosRel = Predicate(
        "REL_FREQ_UNMATCH_OVER_POS_REL", closed=True, size=3
    )
    RelNotFuncDependency = Predicate("REL_NOT_FUNC_DEPENDENCY", closed=True, size=3)
    RelIncorrectDataType = Predicate("REL_INCORRECT_DATA_TYPE", closed=True, size=3)
    RelHeaderSimilarity = Predicate("REL_HEADER_SIMILARITY", closed=True, size=3)

    ExtendedTypeFreqOverRow = Predicate(
        "EXTENDED_TYPE_FREQ_OVER_ROW", closed=True, size=2
    )
    TypeHeaderSimilarity = Predicate("TYPE_HEADER_SIMILARITY", closed=True, size=2)


class PSLGramModel:
    def __init__(
        self,
        qnodes: Mapping[str, QNode],
        qnode_labels: Mapping[str, QNodeLabel],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_numprop_stats: Mapping[str, WDQuantityPropertyStats],
        sim_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.qnodes = qnodes
        self.qnode_labels = qnode_labels
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wd_numprop_stats = wd_numprop_stats
        self.sim_fn = sim_fn

        self.model = self.get_model()
        self.model.set_parameters(
            {
                "REL_PRIOR_NEG": 1,
                "REL_PRIOR_NEG_PARENT_PROP_1": 0.1,
                "REL_PRIOR_NEG_PARENT_PROP_2": 0.1,
                "REL_PRIOR_NEG_PARENT_QUALIFIER": 0.1,
                "CASCADING_ERROR_1": 2,
                "CASCADING_ERROR_2": 2,
                "CASCADING_ERROR_3": 2,
                P.RelFreqOverRow.name(): 2,
                P.RelFreqOverEntRow.name(): 2,
                P.RelFreqOverPosRel.name(): 2,
                P.RelFreqUnmatchOverEntRow.name(): 2,
                P.RelFreqUnmatchOverPosRel.name(): 2,
                P.RelIncorrectDataType.name(): 2,
                P.RelNotFuncDependency.name(): 100,
                "TYPE_PRIOR_NEG": 1,
                "TYPE_PRIOR_NEG_PARENT": 0.1,
                "TYPE_PROP_RANGE": 2,
                P.ExtendedTypeFreqOverRow.name(): 2,
                P.TypeHeaderSimilarity.name(): 0.1,
                P.RelHeaderSimilarity.name(): 2,
            }
        )

    def get_model(self):
        # ** DEFINE RULES **
        rules = RuleContainer()
        for feat in [
            P.RelFreqOverRow,
            P.RelFreqOverEntRow,
            P.RelFreqOverPosRel,
        ]:
            rules[feat.name()] = Rule(
                f"{P.Rel.name()}(N1, N2, P) & {feat.name()}(N1, N2, P) -> {P.CorrectRel.name()}(N1, N2, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        for feat in [
            P.RelFreqUnmatchOverEntRow,
            P.RelFreqUnmatchOverPosRel,
        ]:
            rules[feat.name()] = Rule(
                f"{P.Rel.name()}(N1, N2, P) & {feat.name()}(N1, N2, P) -> ~{P.CorrectRel.name()}(N1, N2, P)",
                weighted=True,
                squared=True,
                weight=0.0,
            )

        rules[P.RelNotFuncDependency.name()] = Rule(
            f"""
            {P.Rel.name()}(N1, N2, P) & 
            {P.Rel.name()}(N2, N3, P2) & 
            ~{P.Statement.name()}(N2) & 
            {P.CorrectRel.name()}(N1, N2, P) &
            {P.RelNotFuncDependency.name()}(N2, N3, P2) -> ~{P.CorrectRel.name()}(N2, N3, P2)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )

        # prefer details prop/type
        rules["REL_PRIOR_NEG_PARENT_PROP_1"] = Rule(
            f"""
            {P.Rel.name()}(U, S, P) & {P.Statement.name()}(S) &
            {P.Rel.name()}(U, S2, PP) & {P.Statement.name()}(S2) & 
            {P.SubProp.name()}(P, PP) -> ~{P.CorrectRel.name()}(U, S2, PP)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules["REL_PRIOR_NEG_PARENT_PROP_2"] = Rule(
            f"""
            {P.Rel.name()}(U, S, P) & {P.Statement.name()}(S) & {P.Rel.name()}(S, V, P) &
            {P.Rel.name()}(U, S2, PP) & {P.Statement.name()}(S2) & {P.Rel.name()}(S2, V, PP) &
            {P.SubProp.name()}(P, PP) -> ~{P.CorrectRel.name()}(S2, V, PP)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules["REL_PRIOR_NEG_PARENT_QUALIFIER"] = Rule(
            f"""
            {P.Rel.name()}(U, S, P) & {P.Statement.name()}(S) & {P.Rel.name()}(S, V, Q) & (P != Q) &
            {P.Rel.name()}(U, S2, PP) & {P.Statement.name()}(S2) & {P.Rel.name()}(S2, V, PQ) & (PQ != PP) &
            {P.SubProp.name()}(Q, PQ) -> ~{P.CorrectRel.name()}(S2, V, PQ)
            """,
            weighted=True,
            squared=True,
            weight=0.0,
        )
        rules["TYPE_PRIOR_NEG_PARENT"] = Rule(
            f"{P.Type.name()}(N, T) & {P.HasSubType.name()}(N, T) -> ~{P.CorrectType.name()}(N, T)",
            weighted=True,
            weight=0.0,
            squared=True,
        )

        # default negative rel/types
        rules["REL_PRIOR_NEG"] = Rule(
            f"~{P.CorrectRel.name()}(N1, N2, P)",
            weight=0.0,
            weighted=True,
            squared=True,
        )
        rules["TYPE_PRIOR_NEG"] = Rule(
            f"~{P.CorrectType.name()}(N, T)", weight=0.0, weighted=True, squared=True
        )

        rules["CASCADING_ERROR_1"] = Rule(
            f"""
            {P.Rel.name()}(N0, S, P) & {P.Statement.name()}(S) & {P.Rel.name()}(S, N1, P) & 
            {P.Rel.name()}(S, N2, Q) & (N1 != N2) & ~{P.CorrectRel.name()}(S, N1, P) -> ~{P.CorrectRel.name()}(S, N2, Q)
            """,
            weighted=True,
            weight=0.0,
            squared=True,
        )
        rules["CASCADING_ERROR_2"] = Rule(
            f"""
            {P.Rel.name()}(N0, S, P) & {P.Statement.name()}(S) & {P.Rel.name()}(S, N1, P) & 
            ~{P.CorrectRel.name()}(S, N1, P) -> ~{P.CorrectRel.name()}(N0, S, P)""",
            weighted=True,
            weight=0.0,
            squared=True,
        )
        rules["CASCADING_ERROR_3"] = Rule(
            f"""
            {P.Rel.name()}(N0, S, P) & {P.Statement.name()}(S) & {P.Rel.name()}(S, N1, P) & 
            ~{P.CorrectRel.name()}(N0, S, P) -> ~{P.CorrectRel.name()}(S, N1, P)
            """,
            weighted=True,
            weight=0.0,
            squared=True,
        )

        for feat in [P.ExtendedTypeFreqOverRow, P.TypeHeaderSimilarity]:
            rules[feat.name()] = Rule(
                f"""
                {P.Type.name()}(N, T) & {feat.name()}(N, T) -> {P.CorrectType.name()}(N, T)
                """,
                weighted=True,
                weight=0.0,
                squared=True,
            )

        return PSLModel(
            rules=rules,
            predicates=[x for x in vars(P).values() if isinstance(x, Predicate)],
            ignore_predicates_not_in_rules=True,
        )

    def predict(
        self, table: LinkedTable, cg: CGGraph, dg: DGGraph, verbose: bool = False
    ) -> Tuple[Dict[CGEdgeTriple, float], Dict[int, Dict[str, float]]]:
        """Predict prob. of edges and prob. of columns' type"""
        idmap, observations, targets = self.extract_data(table, cg, dg)

        if verbose:
            for pname in observations:
                if len(observations[pname]) == 0:
                    logger.debug(f"No observations for predicate {pname}")

        output = self.model.predict(observations, targets, {}, force_setall=True)
        rel_probs = {
            tuple(idmap.im(t) for t in terms): prob
            for terms, prob in output[P.CorrectRel.name()].items()
        }

        type_probs = {}
        for terms, prob in output[P.CorrectType.name()].items():
            u = cg.get_node(idmap.im(terms[0]))
            class_id = idmap.im(terms[1])
            assert isinstance(u, CGColumnNode)
            type_probs.setdefault(u.column, {})[class_id] = prob

        return rel_probs, type_probs

    def extract_data(self, table: LinkedTable, cg: CGGraph, dg: DGGraph):
        """Extract data for our PSL model"""
        cg_nodes = cg.nodes()
        cg_edges = cg.edges()

        idmap = IDMap()

        rel_feats = RelFeatures(
            idmap,
            table,
            cg,
            dg,
            self.qnodes,
            self.qnode_labels,
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
            ]
        )
        type_feats = TypeFeatures(
            idmap,
            table,
            cg,
            dg,
            self.qnodes,
            self.wdclasses,
            self.wdprops,
            self.wd_numprop_stats,
            self.sim_fn,
        ).extract_features([P.ExtendedTypeFreqOverRow.name(), P.HasSubType.name()])

        observations: Dict[str, list] = {}
        targets: Dict[str, list] = {}

        for p in self.model.model.get_predicates().values():
            p: Predicate
            if p.name() in rel_feats:
                observations[p.name()] = rel_feats[p.name()]
            if p.name() in type_feats:
                observations[p.name()] = type_feats[p.name()]

        observations[P.Rel.name()] = [
            (idmap.m(e.source), idmap.m(e.target), idmap.m(e.predicate))
            for e in cg_edges
        ]
        observations[P.Type.name()] = observations[
            P.ExtendedTypeFreqOverRow.name()
        ].copy()
        observations[P.Statement.name()] = [
            (idmap.m(u.id),) for u in cg_nodes if isinstance(u, CGStatementNode)
        ]
        observations[P.SubType.name()] = []
        observations[P.SubProp.name()] = []
        observations[P.RelHeaderSimilarity.name()] = []
        observations[P.TypeHeaderSimilarity.name()] = []

        targets[P.CorrectRel.name()] = observations[P.Rel.name()].copy()
        targets[P.CorrectType.name()] = observations[P.Type.name()].copy()

        # fitlering predicates that are not in the model (e.g. because they are not used in any rule)
        observations = {
            k: v for k, v in observations.items() if k in self.model.name2predicate
        }
        return idmap, observations, targets

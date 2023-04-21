from __future__ import annotations
from grams.core import AlgoContext, GramsDB
from grams.core.cangraph import CGEdge, CGNode
from grams.core.datagraph import DGNode
from grams.core.table import LinkedTable

class FeatureExtractorContext:
    def __init__(
        self,
        table: LinkedTable,
        nodes: list[DGNode],
        cg2dg: list[int],
        db: GramsDB,
        context: AlgoContext,
    ): ...
    def get_unmatch_discovered_links(
        self, cgu: CGNode, s: CGNode, cgv: CGNode, inedge: CGEdge, outedge: CGEdge
    ) -> int: ...
    def get_len_contradicted_information(
        self,
        cgu: CGNode,
        s: CGNode,
        cgv: CGNode,
        inedge: CGEdge,
        outedge: CGEdge,
        correct_entity_threshold: float,
    ) -> int: ...
    def get_contradicted_information(
        self,
        cgu: CGNode,
        s: CGNode,
        cgv: CGNode,
        inedge: CGEdge,
        outedge: CGEdge,
        correct_entity_threshold: float,
    ) -> list[ContradictedInformation]: ...

class ContradictedInformation:
    source: int
    target: int
    inedge: str
    outedge: str

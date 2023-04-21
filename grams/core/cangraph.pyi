from __future__ import annotations
from typing import Optional

from grams.core.datagraph import ContextSpan
from kgdata.core.models import Value

class CGNode:
    @staticmethod
    def column_node(id: int, label: str, column: int) -> CGNode: ...
    @staticmethod
    def entity_node(
        id: str, entity_id: str, context_span: Optional[ContextSpan]
    ) -> CGNode: ...
    @staticmethod
    def literal_node(
        id: str, value: Value, context_span: Optional[ContextSpan]
    ) -> CGNode: ...
    @staticmethod
    def statement_node(id: int, flow: list[CGStatementFlow]) -> CGNode: ...

class CGEdge:
    id: int
    source: int
    target: int
    predicate: str

    def __init__(self, id: int, source: int, target: int, predicate: str) -> None: ...

class CGStatementFlow:
    incoming: CGEdgeFlowSource
    outgoing: CGEdgeFlowTarget
    dg_stmts: list[str]

    def __init__(
        self,
        incoming: CGEdgeFlowSource,
        outgoing: CGEdgeFlowTarget,
        dg_stmts: list[str],
    ) -> None: ...

class CGEdgeFlowSource:
    dgsource: int
    cgsource: int
    edgeid: str

    def __init__(self, dgsource: int, cgsource: int, edgeid: str) -> None: ...

class CGEdgeFlowTarget:
    dgtarget: int
    cgtarget: int
    edgeid: str

    def __init__(self, dgtarget: int, cgtarget: int, edgeid: str) -> None: ...

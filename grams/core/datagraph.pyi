from __future__ import annotations
from typing import Any, Optional

from grams.core import Value

class Span:
    start: int
    end: int

    def __init__(self, start: int, end: int) -> None: ...

class ContextSpan:
    text: str
    span: Span

    def __init__(self, text: str, span: Span) -> None: ...

class CellNode:
    id: str
    value: str
    column: int
    row: int
    entity_ids: list[str]
    entity_spans: dict[str, list[Span]]
    entity_probs: dict[str, float]

    def __init__(
        self,
        id: str,
        value: str,
        column: int,
        row: int,
        entity_ids: list[str],
        entity_spans: list[tuple[str, list[Span]]],
        entity_probs: list[tuple[str, float]],
    ) -> None: ...

class EntityValueNode:
    id: str
    entity_id: str
    entity_prob: float
    context_span: Optional[ContextSpan]

    def __init__(
        self,
        id: str,
        entity_id: str,
        entity_prob: float,
        context_span: Optional[ContextSpan],
    ) -> None: ...

class LiteralValueNode:
    id: str
    value: Value
    context_span: Optional[ContextSpan]

    def __init__(
        self,
        id: str,
        value: Value,
        context_span: Optional[ContextSpan],
    ) -> None: ...

class EdgeFlowSource:
    source_id: str
    predicate: str

    def __init__(self, source_id: str, predicate: str) -> None: ...

class EdgeFlowTarget:
    target_id: str
    predicate: str

    def __init__(self, target_id: str, predicate: str) -> None: ...

# TODO: add FlowProvenance definition
FlowProvenance = Any

class StatementNode:
    id: str
    entity_id: str
    predicate: str
    is_in_kg: bool
    forward_flow: dict[EdgeFlowSource, set[EdgeFlowTarget]]
    reversed_flow: dict[EdgeFlowTarget, set[EdgeFlowSource]]
    flow: dict[tuple[EdgeFlowSource, EdgeFlowTarget], list[FlowProvenance]]

    def __init__(
        self,
        id: str,
        entity_id: str,
        predicate: str,
        is_in_kg: bool,
        forward_flow: dict[EdgeFlowSource, set[EdgeFlowTarget]],
        reversed_flow: dict[EdgeFlowTarget, set[EdgeFlowSource]],
        flow: dict[tuple[EdgeFlowSource, EdgeFlowTarget], list[FlowProvenance]],
    ) -> None: ...

class DGNode:
    @staticmethod
    def cell(cell: CellNode) -> DGNode: ...
    @staticmethod
    def entity(entity_value: EntityValueNode) -> DGNode: ...
    @staticmethod
    def literal(literal_value: LiteralValueNode) -> DGNode: ...
    @staticmethod
    def statement(statement: StatementNode) -> DGNode: ...

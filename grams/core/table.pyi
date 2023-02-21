from __future__ import annotations
from typing import Optional

class LinkedTable:
    id: str
    links: list[list[list[Link]]]
    columns: list[Column]
    context: Context

    def __init__(
        self,
        id: str,
        links: list[list[list[Link]]],
        columns: list[Column],
        context: Context,
    ) -> None: ...
    def get_links(self, row: int, col: int) -> list[Link]: ...

class Link:
    start: int
    end: int
    url: Optional[str]
    entities: list[EntityId]
    candidates: list[CandidateEntityId]

    def __init__(
        self,
        start: int,
        end: int,
        url: Optional[str],
        entities: list[EntityId],
        candidates: list[CandidateEntityId],
    ) -> None: ...

class Column:
    index: int
    name: Optional[str]
    values: list[str]

    def __init__(self, index: int, name: Optional[str], values: list[str]) -> None: ...

class Context:
    page_title: Optional[str]
    page_url: Optional[str]
    page_entities: Optional[list[EntityId]]

    def __init__(
        self,
        page_title: Optional[str],
        page_url: Optional[str],
        page_entities: Optional[list[EntityId]],
    ) -> None: ...

class CandidateEntityId:
    id: EntityId
    probability: float

    def __init__(self, id: EntityId, probability: float) -> None: ...

class EntityId(tuple[str]):
    def __init__(self, id: str) -> None: ...
    @property
    def id(self) -> str: ...

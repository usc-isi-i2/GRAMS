from __future__ import annotations

import copy
from functools import cached_property
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from graph.retworkx import BaseEdge, BaseNode, RetworkXStrDiGraph
from typing_extensions import TypeAlias

from kgdata.wikidata.models import WDValue
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdvalue import WDValueKind
from sm.misc.prelude import UnreachableError


# @dataclass(frozen=True)
@dataclass
class Span:
    __slots__ = ("start", "end")
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start

    def to_tuple(self):
        return self.start, self.end

    @staticmethod
    def from_tuple(tup):
        return Span(tup[0], tup[1])


@dataclass
class CellNode(BaseNode[str]):
    __slots__ = (
        "id",
        "value",
        "column",
        "row",
        "entity_ids",
        "entity_spans",
        "entity_probs",
    )
    id: str
    value: str
    column: int
    row: int
    entity_ids: List[str]
    entity_spans: Dict[str, List[Span]]
    entity_probs: Dict[str, float]

    def to_tuple(self):
        return (
            self.id,
            self.value,
            self.column,
            self.row,
            self.entity_ids,
            [(k, [v.to_tuple() for v in vs]) for k, vs in self.entity_spans.items()],
            self.entity_probs,
        )

    @staticmethod
    def from_tuple(tup):
        return CellNode(
            tup[0],
            tup[1],
            tup[2],
            tup[3],
            tup[4],
            {k: [Span.from_tuple(v) for v in vs] for k, vs in tup[5]},
            tup[6],
        )

    @staticmethod
    def get_id(row: int, col: int):
        return f"{row}-{col}"


@dataclass
class ContextSpan:
    __slots__ = ("text", "span")
    # the text of the context
    text: str
    span: Span

    def get_text_span(self):
        return self.text[self.span.start : self.span.end]

    def to_tuple(self):
        return self.text, self.span.to_tuple()

    @staticmethod
    def from_tuple(tup):
        return ContextSpan(tup[0], Span.from_tuple(tup[1]))


@dataclass
class LiteralValueNode(BaseNode[str]):
    __slots__ = ("id", "value", "context_span")
    id: str
    value: WDValue
    # not none if it is appear in the context
    context_span: Optional[ContextSpan]

    @property
    def is_context(self):
        return self.context_span is not None

    def to_tuple(self):
        return (
            self.id,
            self.value.to_tuple(),
            self.context_span.to_tuple() if self.context_span is not None else None,
        )

    @staticmethod
    def from_tuple(tup):
        wdvalue = tup[1]
        return LiteralValueNode(
            tup[0],
            WDValue(wdvalue[0], wdvalue[1]),
            ContextSpan.from_tuple(tup[2]) if tup[2] is not None else None,
        )


@dataclass
class EntityValueNode(BaseNode[str]):
    __slots__ = ("id", "qnode_id", "context_span", "qnode_prob")
    id: str
    qnode_id: str
    # not none if it is appear in the context
    context_span: Optional[ContextSpan]
    qnode_prob: float

    @property
    def is_context(self):
        return self.context_span is not None

    def to_tuple(self):
        return (
            self.id,
            self.qnode_id,
            self.context_span.to_tuple() if self.context_span is not None else None,
            self.qnode_prob,
        )

    @staticmethod
    def from_tuple(tup):
        return EntityValueNode(
            tup[0],
            tup[1],
            ContextSpan.from_tuple(tup[2]) if tup[2] is not None else None,
            tup[3],
        )


# edge id is actually key id
EdgeFlowSource = NamedTuple("EdgeFlowSource", [("source_id", str), ("edge_id", str)])
EdgeFlowTarget = NamedTuple("EdgeFlowTarget", [("target_id", str), ("edge_id", str)])


@dataclass
class StatementNode(BaseNode[str]):
    __slots__ = (
        "id",
        "qnode_id",
        "predicate",
        "is_in_kg",
        "forward_flow",
        "reversed_flow",
        "flow",
    )

    id: str
    # id of the qnode that contains the statement
    qnode_id: str
    # predicate of the statement
    predicate: str
    # whether this statement actually exist in KG
    is_in_kg: bool

    # recording which link in the source is connected to the target.
    forward_flow: Dict[EdgeFlowSource, Set[EdgeFlowTarget]]
    reversed_flow: Dict[EdgeFlowTarget, Set[EdgeFlowSource]]
    flow: Dict[Tuple[EdgeFlowSource, EdgeFlowTarget], List[FlowProvenance]]

    @property
    def value(self):
        return self.id

    @cached_property
    def statement_index(self):
        return DGStatementID.parse_id(self.id).statement_index

    def track_provenance(
        self,
        source_flow: EdgeFlowSource,
        target_flow: EdgeFlowTarget,
        provenances: List[FlowProvenance],
    ):
        ptr = self.forward_flow
        if source_flow not in ptr:
            ptr[source_flow] = set()
        ptr[source_flow].add(target_flow)

        ptr = self.reversed_flow
        if target_flow not in ptr:
            ptr[target_flow] = set()
        ptr[target_flow].add(source_flow)

        # TODO: merge the provenance if we can
        if (source_flow, target_flow) not in self.flow:
            self.flow[source_flow, target_flow] = provenances
        else:
            self.flow[source_flow, target_flow] = FlowProvenance.merge_lst(
                self.flow[source_flow, target_flow], provenances
            )

    def untrack_source_flow(self, source_flow: EdgeFlowSource):
        for target_flow in self.forward_flow.pop(source_flow):
            self.flow.pop((source_flow, target_flow))
            self.reversed_flow[target_flow].remove(source_flow)
            if len(self.reversed_flow[target_flow]) == 0:
                self.reversed_flow.pop(target_flow)

    def untrack_target_flow(self, target_flow: EdgeFlowTarget):
        for source_flow in self.reversed_flow.pop(target_flow):
            self.flow.pop((source_flow, target_flow))
            self.forward_flow[source_flow].remove(target_flow)
            if len(self.forward_flow[source_flow]) == 0:
                self.forward_flow.pop(source_flow)

    def untrack_flow(self, source_flow: EdgeFlowSource, target_flow: EdgeFlowTarget):
        self.flow.pop((source_flow, target_flow))
        self.forward_flow[source_flow].remove(target_flow)
        self.reversed_flow[target_flow].remove(source_flow)

        if len(self.forward_flow[source_flow]) == 0:
            self.forward_flow.pop(source_flow)
        if len(self.reversed_flow[target_flow]) == 0:
            self.reversed_flow.pop(target_flow)

    def has_source_flow(self, source_flow):
        return source_flow in self.forward_flow

    def has_target_flow(self, target_flow):
        return target_flow in self.reversed_flow

    def has_flow(self, source_flow, target_flow):
        return (source_flow, target_flow) in self.flow

    def iter_source_flow(self, target_flow: EdgeFlowTarget):
        for source_flow in self.reversed_flow[target_flow]:
            yield source_flow, self.flow[source_flow, target_flow]

    def iter_target_flow(self, source_flow: EdgeFlowSource):
        for target_flow in self.forward_flow[source_flow]:
            yield target_flow, self.flow[source_flow, target_flow]

    def get_provenance(self, source_flow: EdgeFlowSource, target_flow: EdgeFlowTarget):
        return self.flow[source_flow, target_flow]

    def get_provenance_by_edge(self, inedge: "DGEdge", outedge: "DGEdge"):
        return self.flow[
            EdgeFlowSource(inedge.source, inedge.predicate),
            EdgeFlowTarget(outedge.target, outedge.predicate),
        ]

    def is_same_flow(self, inedge: "DGEdge", outedge: "DGEdge") -> bool:
        return (
            EdgeFlowSource(inedge.source, inedge.predicate),
            EdgeFlowTarget(outedge.target, outedge.predicate),
        ) in self.flow

    def to_tuple(self):
        return (
            self.id,
            self.qnode_id,
            self.predicate,
            self.is_in_kg,
            self.forward_flow,
            self.reversed_flow,
            {k: [v.to_tuple() for v in vs] for k, vs in self.flow.items()},
            # [(tuple(k), [tuple(v) for v in vs]) for k, vs in self.forward_flow.items()],
            # [
            #     (tuple(k), [tuple(v) for v in vs])
            #     for k, vs in self.reversed_flow.items()
            # ],
            # [
            #     ((tuple(k[0]), tuple(k[1])), [v.to_tuple() for v in vs])
            #     for k, vs in self.flow.items()
            # ],
        )

    @staticmethod
    def from_tuple(tup):
        return StatementNode(
            tup[0],
            tup[1],
            tup[2],
            tup[3],
            tup[4],
            tup[5],
            {k: [FlowProvenance.from_tuple(v) for v in vs] for k, vs in tup[6].items()},
            # {
            #     EdgeFlowSource(k[0], k[1]): {EdgeFlowTarget(v[0], v[1]) for v in vs}
            #     for k, vs in tup[4]
            # },
            # {
            #     EdgeFlowTarget(k[0], k[1]): {EdgeFlowSource(v[0], v[1]) for v in vs}
            #     for k, vs in tup[5]
            # },
            # {
            #     (EdgeFlowSource(k[0][0], k[0][1]), EdgeFlowTarget(k[1][0], k[1][1])): [
            #         FlowProvenance.from_tuple(v) for v in vs
            #     ]
            #     for k, vs in tup[6]
            # },
        )


DGNode = Union[CellNode, LiteralValueNode, EntityValueNode, StatementNode]


class LinkGenMethod(Enum):
    """Methods to generate a link"""

    # this statement is generated by exact matching from the link
    FromWikidataLink = "from_wikidata_link"
    FromLiteralMatchingFunc = "from_literal_matching_function"
    FromInference = "from_inference"


# the index in which we discovered the matched value, for property, it is statement index, for qualifier, it is the index of the qualifier's value
FromLiteralMatchingFunc_GenArg = TypedDict(
    "FromLiteralMatchingFunc_GenArg",
    {"func": str, "value_index": int},
)
FromWikidataLink_GenArg = TypedDict("FromWikidataLink_GenArg", {"value_index": int})
FromInference_GenArg = TypedDict(
    "FromInference_GenArg",
    {"from_path": list[str], "method": Literal["subproperty", "transitive"]},
)


@dataclass
class FlowProvenance:
    __slots__ = ("gen_method", "gen_method_arg", "prob")
    """Contain information regarding how this relationship/flow has been generated (typically coming from the matching algorithm)"""

    # method that
    gen_method: LinkGenMethod
    gen_method_arg: Union[
        FromLiteralMatchingFunc_GenArg, FromWikidataLink_GenArg, FromInference_GenArg
    ]
    prob: float

    def merge(self, another: FlowProvenance) -> Union[FlowProvenance, None]:
        """Try to merge the provenance, if we cannot merge them, just return None"""
        if self.gen_method != another.gen_method:
            return None
        if self.gen_method == LinkGenMethod.FromWikidataLink:
            if self.gen_method_arg == another.gen_method_arg:
                return self
            return None
        if self.gen_method == LinkGenMethod.FromLiteralMatchingFunc:
            if self.gen_method_arg == another.gen_method_arg:
                if self.prob > another.prob:
                    return self
                return another
            return None
        if self.gen_method == LinkGenMethod.FromInference:
            if self.gen_method_arg == another.gen_method_arg:
                if self.prob > another.prob:
                    return self
                return another
            return None
        raise UnreachableError()

    @staticmethod
    def merge_lst(
        lst1: List[FlowProvenance], lst2: List[FlowProvenance]
    ) -> List[FlowProvenance]:
        """Assume that items within each list (lst1 & lst2) are not mergeable"""
        if len(lst1) == 0:
            return lst2
        elif len(lst2) == 0:
            return lst1

        lst = copy.copy(lst1)
        for item in lst2:
            for i in range(len(lst1)):
                resp = lst[i].merge(item)
                if resp is None:
                    # cannot merge them
                    lst.append(item)
                    break
                else:
                    # merge lst[i] & item
                    lst[i] = resp
                    break
        return lst

    def to_tuple(self):
        return self.gen_method.value, self.gen_method_arg, self.prob

    @staticmethod
    def from_tuple(tup):
        return FlowProvenance(LinkGenMethod(tup[0]), tup[1], tup[2])

    @staticmethod
    def from_inference(
        method: Literal["subproperty", "transitive"], from_path: list[str], prob: float
    ) -> FlowProvenance:
        return FlowProvenance(
            LinkGenMethod.FromInference,
            {"method": method, "from_path": from_path},
            prob,
        )


@dataclass
class DGStatementID:
    __slots__ = ("qnode_id", "predicate", "statement_index")
    qnode_id: str
    predicate: str
    statement_index: int

    def get_id(self):
        return f"stmt:{self.qnode_id}-{self.predicate}-{self.statement_index}"

    @staticmethod
    def parse_id(id: str) -> "DGStatementID":
        m = re.match(r"stmt:([^-]+)-([^-]+)-([^-]+)", id)
        assert m is not None
        qnode_id, predicate, stmt_index = [m.group(i) for i in range(1, 4)]
        stmt_index = int(stmt_index)
        return DGStatementID(qnode_id, predicate, stmt_index)


@dataclass
class DGPathNodeStatement:
    __slots__ = ("qnode_id", "predicate", "statement_index", "provenance")
    qnode_id: str
    predicate: str
    statement_index: int
    # how this DGPath statement has been matched
    provenance: FlowProvenance

    def get_id(self):
        return DGStatementID(
            self.qnode_id, self.predicate, self.statement_index
        ).get_id()

    # ############# METHODs to construct DGPathNodeStatement from provenance ##############
    @staticmethod
    def from_FromWikidataLink(
        qnode_id: str, predicate: str, stmt_index: int, qual_index: Optional[int]
    ):
        return DGPathNodeStatement(
            qnode_id,
            predicate,
            stmt_index,
            FlowProvenance(
                LinkGenMethod.FromWikidataLink,
                {"value_index": qual_index if qual_index is not None else stmt_index},
                1.0,
            ),
        )

    @staticmethod
    def from_FromLiteralMatchingFunc(
        qnode_id: str,
        predicate: str,
        stmt_index: int,
        qual_index: Optional[int],
        func: str,
        prob: float,
    ):
        return DGPathNodeStatement(
            qnode_id,
            predicate,
            stmt_index,
            FlowProvenance(
                LinkGenMethod.FromLiteralMatchingFunc,
                {
                    "func": func,
                    "value_index": qual_index if qual_index is not None else stmt_index,
                },
                prob,
            ),
        )


@dataclass
class DGPathNodeEntity:
    __slots__ = ("qnode_id",)
    qnode_id: str

    def get_id(self):
        return f"ent:{self.qnode_id}"


@dataclass
class DGPathNodeLiteralValue:
    __slots__ = ("value",)
    value: WDValue

    def get_id(self):
        return f"val:{self.value.to_string_repr()}"


@dataclass
class DGPathExistingNode:
    __slots__ = ("id",)
    id: str

    def get_id(self):
        return self.id


DGPathNode = Union[
    DGPathNodeStatement, DGPathNodeEntity, DGPathNodeLiteralValue, DGPathExistingNode
]


@dataclass
class DGPathEdge:
    __slots__ = ("value", "is_qualifier")
    value: str
    is_qualifier: bool

    @staticmethod
    def p(value: str):
        return DGPathEdge(value, is_qualifier=False)

    @staticmethod
    def q(value: str):
        return DGPathEdge(value, is_qualifier=True)


@dataclass
class DGPath:
    __slots__ = ("sequence",)
    # a sequence of path, is always
    sequence: List[Union[DGPathEdge, DGPathNode]]


@dataclass
class DGEdge(BaseEdge[str, str]):
    __slots__ = ("source", "target", "predicate", "is_qualifier", "is_inferred", "id")
    source: str
    target: str
    predicate: str
    is_qualifier: bool
    # is_inferred means that this edge is added based on dg inference (e.g., transitivity, etc.)
    is_inferred: bool
    # id can be initialized to be -1, and will be set by the graph
    id: int

    @property
    def key(self):
        return self.predicate

    @staticmethod
    def can_link(source: DGNode, target: DGNode):
        """Test if there can be two links between the two nodes. Basically, it just enforce the constraint that no link
        cross different rows
        """
        if not isinstance(source, CellNode) or not isinstance(target, CellNode):
            return True
        return source.row == target.row

    def to_tuple(self):
        return (
            self.source,
            self.target,
            self.predicate,
            self.is_qualifier,
            self.is_inferred,
            self.id,
        )

    @staticmethod
    def from_tuple(tup):
        return DGEdge(tup[0], tup[1], tup[2], tup[3], tup[4], tup[5])


class DGGraph(RetworkXStrDiGraph[str, DGNode, DGEdge]):
    def get_statement_node(self, nid: str) -> StatementNode:
        n = self.get_node(nid)
        assert isinstance(n, StatementNode)
        return n

    def get_cell_node(self, nid: str) -> CellNode:
        n = self.get_node(nid)
        assert isinstance(n, CellNode)
        return n

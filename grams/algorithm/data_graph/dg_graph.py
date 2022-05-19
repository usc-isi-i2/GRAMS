from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from functools import cmp_to_key
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
from graph.retworkx import (
    RetworkXStrDiGraph,
    BaseEdge,
    BaseNode,
)

import networkx as nx
import sm.misc as M
from grams.algorithm.kg_index import KGObjectIndex
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDValue, WDEntity, WDClass, WDProperty
from loguru import logger
from sm.misc.graph import viz_graph
from tqdm import tqdm
from typing_extensions import TypedDict


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


@dataclass
class CellNode(BaseNode[str]):
    id: str
    value: str
    column: int
    row: int
    entity_ids: List[str]
    entity_spans: Dict[str, List[Span]]


@dataclass
class ContextSpan:
    # the text of the context
    text: str
    span: Span

    def get_text_span(self):
        return self.text[self.span.start : self.span.end]


@dataclass
class LiteralValueNode(BaseNode[str]):
    id: str
    value: WDValue
    # not none if it is appear in the context
    context_span: Optional[ContextSpan]

    @property
    def is_context(self):
        return self.context_span is not None


@dataclass
class EntityValueNode(BaseNode[str]):
    id: str
    qnode_id: str
    # not none if it is appear in the context
    context_span: Optional[ContextSpan]

    @property
    def is_context(self):
        return self.context_span is not None


# edge id is actually key id
EdgeFlowSource = NamedTuple("EdgeFlowSource", [("source_id", str), ("edge_id", str)])
EdgeFlowTarget = NamedTuple("EdgeFlowTarget", [("target_id", str), ("edge_id", str)])


@dataclass
class StatementNode(BaseNode[str]):
    id: str
    # id of the qnode that contains the statement
    qnode_id: str
    # predicate of the statement
    predicate: str
    # whether this statement actually exist in KG
    is_in_kg: bool

    # recording which link in the source is connected to the target.
    forward_flow: Dict[EdgeFlowSource, Set[EdgeFlowTarget]] = field(
        default_factory=dict
    )
    reversed_flow: Dict[EdgeFlowTarget, Set[EdgeFlowSource]] = field(
        default_factory=dict
    )
    flow: Dict[Tuple[EdgeFlowSource, EdgeFlowTarget], List[FlowProvenance]] = field(
        default_factory=dict
    )

    @property
    def value(self):
        return self.id

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


DGNode = Union[CellNode, LiteralValueNode, EntityValueNode, StatementNode]


class LinkGenMethod(Enum):
    """Methods to generate a link"""

    # this statement is generated by exact matching from the link
    FromWikidataLink = "from_wikidata_link"
    FromLiteralMatchingFunc = "from_literal_matching_function"


@dataclass
class FlowProvenance:
    """Contain information regarding how this relationship/flow has been generated (typically coming from the matching algorithm)"""

    # method that
    gen_method: LinkGenMethod
    gen_method_arg: Any
    prob: float

    def merge(self, another: FlowProvenance) -> Union[FlowProvenance, None]:
        """Try to merge the provenance, if we cannot merge them, just return None"""
        if self.gen_method != another.gen_method:
            return None
        if self.gen_method == LinkGenMethod.FromWikidataLink:
            return self
        if self.gen_method == LinkGenMethod.FromLiteralMatchingFunc:
            if self.gen_method_arg == another.gen_method_arg:
                if self.prob > another.prob:
                    return self
                return another
            return None
        raise M.UnreachableError()

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


@dataclass
class DGStatementID:
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
    def from_FromWikidataLink(qnode_id, predicate, stmt_index):
        return DGPathNodeStatement(
            qnode_id,
            predicate,
            stmt_index,
            FlowProvenance(LinkGenMethod.FromWikidataLink, None, 1.0),
        )

    @staticmethod
    def from_FromLiteralMatchingFunc(qnode_id, predicate, stmt_index, fn_args, prob):
        return DGPathNodeStatement(
            qnode_id,
            predicate,
            stmt_index,
            FlowProvenance(LinkGenMethod.FromLiteralMatchingFunc, fn_args, prob),
        )


@dataclass
class DGPathNodeEntity:
    qnode_id: str

    def get_id(self):
        return f"ent:{self.qnode_id}"


@dataclass
class DGPathNodeLiteralValue:
    value: WDValue

    def get_id(self):
        return f"val:{self.value.to_string_repr()}"


@dataclass
class DGPathExistingNode:
    id: str

    def get_id(self):
        return self.id


DGPathNode = Union[
    DGPathNodeStatement, DGPathNodeEntity, DGPathNodeLiteralValue, DGPathExistingNode
]


@dataclass
class DGPathEdge:
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
    # a sequence of path, is always
    sequence: List[Union[DGPathEdge, DGPathNode]] = field(default_factory=list)


@dataclass
class DGEdge(BaseEdge[str, str]):
    source: str
    target: str
    predicate: str
    is_qualifier: bool
    # deprecated, will be replaced by the information stored directly in the statement edge flow
    # paths: List[DGPath]
    is_inferred: Optional[bool] = None
    id: int = -1  # set automatically by the graph

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


class DGGraph(RetworkXStrDiGraph[str, DGNode, DGEdge]):
    def get_statement_node(self, nid: str) -> StatementNode:
        n = self.get_node(nid)
        assert isinstance(n, StatementNode)
        return n

    def get_cell_node(self, nid: str) -> CellNode:
        n = self.get_node(nid)
        assert isinstance(n, CellNode)
        return n

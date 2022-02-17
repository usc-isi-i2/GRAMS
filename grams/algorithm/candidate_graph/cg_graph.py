import copy
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    EntityValueNode,
    LiteralValueNode,
)
from graph.retworkx import BaseNode, BaseEdge, RetworkXStrDiGraph
import numpy as np
from collections import Counter
from dataclasses import dataclass
from itertools import chain, combinations
from operator import itemgetter
from typing import (
    Dict,
    Mapping,
    Optional,
    Union,
    List,
    Set,
    Tuple,
    NamedTuple,
    Any,
    Callable,
)
from uuid import uuid4

import networkx as nx
from rdflib import RDFS
from typing_extensions import TypeGuard, TypedDict
from grams.algorithm.wdont import WDOnt
from kgdata.wikidata.models import QNode, DataValue, WDProperty, WDClass
from sm.misc.graph import viz_graph
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.data_graph import (
    DGNode,
    StatementNode,
    CellNode,
    ContextSpan,
    EdgeFlowSource,
    EdgeFlowTarget,
    FlowProvenance,
)
import sm.outputs as O
import sm.misc as M


@dataclass
class CGColumnNode(BaseNode[str]):
    id: str
    # column name
    label: str
    # column index
    column: int
    # list nodes' id in the data graph
    nodes: Set[str]

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class CGEntityValueNode(BaseNode[str]):
    id: str
    label: str
    qnode_id: str
    context_span: Optional[ContextSpan]

    @property
    def is_in_context(self):
        return self.context_span is not None

    @property
    def is_statement(self):
        return False

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class CGLiteralValueNode(BaseNode[str]):
    id: str
    label: str
    value: DataValue
    context_span: Optional[ContextSpan]

    @property
    def is_in_context(self):
        return self.context_span is not None

    def clone(self):
        return copy.deepcopy(self)


CGEdgeFlowTarget = NamedTuple(
    "CGEdgeFlowTarget", [("dg_target_id", str), ("sg_target_id", str), ("edge_id", str)]
)
# the source of the flow is not needed but still there for what reason? consistent?
CGEdgeFlowSource = NamedTuple(
    "CGEdgeFlowSource", [("dg_source_id", str), ("sg_source_id", str), ("edge_id", str)]
)


@dataclass
class CGStatementNode(BaseNode[str]):
    id: str
    # (source flow, target_flow) => dg statement id => provenance (we have multiple statement id because
    # for one qnode one property, we may have multiple matches statement
    forward_flow: Dict[CGEdgeFlowSource, Dict[CGEdgeFlowTarget, Set[str]]]
    reversed_flow: Dict[CGEdgeFlowTarget, Dict[CGEdgeFlowSource, Set[str]]]
    flow: Dict[
        Tuple[CGEdgeFlowSource, CGEdgeFlowTarget], Dict[str, List[FlowProvenance]]
    ]

    @staticmethod
    def get_id(source_id: str, source_predicate: str, target_id: str):
        return f"stmt:{source_id}-{source_predicate}-{target_id}"

    @staticmethod
    def new(id: str):
        return CGStatementNode(id, {}, {}, {})

    def clone(self):
        return copy.deepcopy(self)

    def track_provenance(
        self,
        source_flow: CGEdgeFlowSource,
        sid: str,
        target_flow: CGEdgeFlowTarget,
        provenances: List[FlowProvenance],
    ):
        """Track the provenance of which edges goes through the node"""
        if source_flow not in self.forward_flow:
            self.forward_flow[source_flow] = {}
        if target_flow not in self.forward_flow[source_flow]:
            self.forward_flow[source_flow][target_flow] = set()
        self.forward_flow[source_flow][target_flow].add(sid)

        if target_flow not in self.reversed_flow:
            self.reversed_flow[target_flow] = {}
        if source_flow not in self.reversed_flow[target_flow]:
            self.reversed_flow[target_flow][source_flow] = set()
        self.reversed_flow[target_flow][source_flow].add(sid)

        if (source_flow, target_flow) not in self.flow:
            self.flow[source_flow, target_flow] = {}
        assert (
            sid not in self.flow[source_flow, target_flow]
        ), "Gate to make sure the assumption (source_flow, target_flow, dg statement id) is unique correct"
        self.flow[source_flow, target_flow][sid] = provenances

    def compute_freq(
        self,
        _sg: nx.MultiDiGraph,
        dg: nx.MultiDiGraph,
        edge: "CGEdge",
        is_unique_freq: bool,
    ):
        """Compute the frequency of data edges between"""
        if not is_unique_freq:
            freq = 0
            if edge.target == self.id:
                # from node to statement, we only consider the target that is property of the statement
                # since we always create new statement based on the value
                for (source_flow, target_flow), stmt2prov in self.flow.items():
                    if (
                        source_flow.edge_id != edge.predicate
                        or source_flow.sg_source_id != edge.source
                    ):
                        continue
                    if target_flow.edge_id != edge.predicate:
                        continue
                    freq += 1
            else:
                for (source_flow, target_flow), stmt2prov in self.flow.items():
                    if (
                        target_flow.sg_target_id == edge.target
                        and target_flow.edge_id == edge.predicate
                    ):
                        freq += 1
            return freq

        unique_pairs = set()
        if edge.target == self.id:
            for (source_flow, target_flow), stmt2prov in self.flow.items():
                if (
                    source_flow.sg_source_id != edge.source
                    or source_flow.edge_id != edge.predicate
                ):
                    continue

                # the target is the one contain value of the statement property, which we should have
                # only one
                if target_flow.edge_id != edge.predicate:
                    continue

                dg_source = dg.nodes[source_flow.dg_source_id]["data"]
                if dg_source.is_cell:
                    source_val = dg_source.value
                else:
                    assert dg_source.is_entity_value
                    source_val = dg_source.qnode_id

                dg_target = dg.nodes[target_flow.dg_target_id]["data"]
                if dg_target.is_cell:
                    target_val = dg_target.value
                elif dg_target.is_literal_value:
                    target_val = dg_target.value.to_string_repr()
                else:
                    assert dg_target.is_entity_value
                    target_val = dg_target.qnode_id

                unique_pairs.add((source_val, target_val))
        else:
            for source_flow, target_flows in self.forward_flow.items():
                dg_source = dg.nodes[source_flow.dg_source_id]["data"]
                if dg_source.is_cell:
                    source_val = dg_source.value
                else:
                    assert dg_source.is_entity_value
                    source_val = dg_source.qnode_id
                for target_flow in target_flows.keys():
                    if (
                        target_flow.sg_target_id == edge.target
                        and target_flow.edge_id == edge.predicate
                    ):
                        dg_target = dg.nodes[target_flow.dg_target_id]["data"]
                        if dg_target.is_cell:
                            target_val = dg_target.value
                        elif dg_target.is_literal_value:
                            target_val = dg_target.value.to_string_repr()
                        else:
                            assert dg_target.is_entity_value
                            target_val = dg_target.qnode_id
                        unique_pairs.add((source_val, target_val))

        return len(unique_pairs)

    def get_edges_provenance(self, edges: List["CGEdge"]):
        """Retrieve all flow provenances that connects these edges.

        Note that in case we have more than one statement per flow, we merge the statements (i.e., merge
        the provenances) to return the best statements (since the provenance store how the link is generated,
        merged the provenance will select the best one).
        """
        select_target_flows = {(e.target, e.predicate) for e in edges}
        links = {}
        for source_flow, target_flows in self.forward_flow.items():
            # source_flow.predicate is always the same, just change to different data node
            # get list of statements that contains all edges
            stmts: Optional[
                Dict[str, Dict[CGEdgeFlowTarget, List[FlowProvenance]]]
            ] = None
            for target_flow in target_flows:
                if (
                    target_flow.sg_target_id,
                    target_flow.edge_id,
                ) not in select_target_flows:
                    continue
                if stmts is None:
                    stmts = {}
                    for stmt_id, provs in self.flow[source_flow, target_flow].items():
                        stmts[stmt_id] = {}
                        stmts[stmt_id][target_flow] = provs
                else:
                    # do intersection because we want to find the stmt contain all edges
                    stmt2provenances = self.flow[source_flow, target_flow]
                    remove_stmt_ids = set(stmts.keys()).difference(
                        stmt2provenances.keys()
                    )
                    for stmt_id in remove_stmt_ids:
                        stmts.pop(stmt_id)
                    for stmt_id, provenances in stmt2provenances.items():
                        if stmt_id not in stmts:
                            continue
                        # no merge since the target flow is unique
                        stmts[stmt_id][target_flow] = provenances

            if stmts is not None and len(stmts) > 0:
                # do a merge to get the provenances (this is the strategy to return the best statement)
                # note that all target flows will be appear, so the first step is to reverse target flow with statement id
                swap_stmts: Dict[CGEdgeFlowTarget, List[FlowProvenance]] = {
                    target_flow: [] for target_flow in next(iter(stmts.values()))
                }
                for stmt_id, _target_flows in stmts.items():
                    for target_flow, provenances in _target_flows.items():
                        swap_stmts[target_flow] = FlowProvenance.merge_lst(
                            swap_stmts[target_flow], provenances
                        )
                links[source_flow] = swap_stmts
        return links


@dataclass
class CGEdge(BaseEdge[str, str]):
    source: str
    target: str
    predicate: str
    features: Dict[str, Any]
    id: int = -1  # set automatically by cg graph

    @property
    def key(self):
        return self.predicate

    def clone(self):
        return copy.deepcopy(self)


CGNode = Union[CGColumnNode, CGEntityValueNode, CGLiteralValueNode, CGStatementNode]
CGEdgeTriple = Tuple[str, str, str]


class CGGraph(RetworkXStrDiGraph[str, CGNode, CGEdge]):
    def get_statement_node(self, uid: str) -> CGStatementNode:
        u = self.get_node(uid)
        assert isinstance(u, CGStatementNode)
        return u

    def remove_dangling_statement(self):
        """Remove statement nodes that have no incoming nodes or no outgoing edges"""
        for s in self.nodes():
            if isinstance(s, CGStatementNode) and (
                self.in_degree(s.id) == 0 or self.out_degree(s.id) == 0
            ):
                self.remove_node(s.id)

    def remove_standalone_nodes(self):
        """Remove nodes that do not any incoming / outgoing edges"""
        for u in self.nodes():
            if self.degree(u.id) == 0:
                self.remove_node(u.id)
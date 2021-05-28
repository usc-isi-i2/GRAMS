import copy
import numpy as np
from collections import Counter
from dataclasses import dataclass
from itertools import chain, combinations
from operator import itemgetter
from typing import Dict, Optional, Union, List, Set, Tuple, NamedTuple, Any, Callable
from uuid import uuid4

import networkx as nx
from rdflib import RDFS
from typing_extensions import TypedDict
from grams.algorithm.sm_wikidata import WDOnt
from kgdata.wikidata.models import QNode, DataValue, WDProperty, WDClass
from sm.misc.graph import viz_graph
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.data_graph import build_data_graph, DGNode, StatementNode, CellNode, \
    ContextSpan, EdgeFlowSource, EdgeFlowTarget, \
    FlowProvenance
import sm.outputs as O


@dataclass
class SGColumnNode:
    id: str
    # column name
    label: str
    # column index
    column: int
    # list nodes' id in the data graph
    nodes: Set[str]

    @property
    def is_column(self):
        return True

    @property
    def is_value(self):
        return False

    @property
    def is_statement(self):
        return False

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class SGEntityValueNode:
    id: str
    label: str
    qnode_id: str
    context_span: Optional[ContextSpan]

    @property
    def is_column(self):
        return False

    @property
    def is_value(self):
        return True

    @property
    def is_entity_value(self):
        return True

    @property
    def is_literal_value(self):
        return False

    @property
    def is_in_context(self):
        return self.context_span is not None

    @property
    def is_statement(self):
        return False

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class SGLiteralValueNode:
    id: str
    label: str
    value: DataValue
    context_span: Optional[ContextSpan]

    @property
    def is_column(self):
        return False

    @property
    def is_value(self):
        return True

    @property
    def is_entity_value(self):
        return False

    @property
    def is_literal_value(self):
        return True

    @property
    def is_in_context(self):
        return self.context_span is not None

    @property
    def is_statement(self):
        return False

    def clone(self):
        return copy.deepcopy(self)


SGEdgeFlowTarget = NamedTuple("SGEdgeFlowTarget", [("dg_target_id", str), ("sg_target_id", str), ("edge_id", str)])
# the source of the flow is not needed but still there for what reason? consistent?
SGEdgeFlowSource = NamedTuple("SGEdgeFlowSource", [("dg_source_id", str), ("sg_source_id", str), ("edge_id", str)])


@dataclass
class SGStatementNode:
    id: str
    # (source flow, target_flow) => dg statement id => provenance (we have multiple statement id because
    # for one qnode one property, we may have multiple matches statement
    forward_flow: Dict[SGEdgeFlowSource, Dict[SGEdgeFlowTarget, Set[str]]]
    reversed_flow: Dict[SGEdgeFlowTarget, Dict[SGEdgeFlowSource, Set[str]]]
    flow: Dict[Tuple[SGEdgeFlowSource, SGEdgeFlowTarget], Dict[str, List[FlowProvenance]]]

    @staticmethod
    def get_id(source_id: str, source_predicate: str, target_id: str):
        return f"stmt:{source_id}-{source_predicate}-{target_id}"

    @staticmethod
    def new(id: str):
        return SGStatementNode(id, {}, {}, {})

    @property
    def is_column(self):
        return False

    @property
    def is_value(self):
        return False

    @property
    def is_statement(self):
        return True

    def clone(self):
        return copy.deepcopy(self)

    def track_provenance(self, source_flow: SGEdgeFlowSource, sid: str, target_flow: SGEdgeFlowTarget, provenances: List[FlowProvenance]):
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
        assert sid not in self.flow[source_flow, target_flow], "Gate to make sure the assumption (source_flow, target_flow, dg statement id) is unique correct"
        self.flow[source_flow, target_flow][sid] = provenances

    def compute_freq(self, _sg: nx.MultiDiGraph, dg: nx.MultiDiGraph, edge: 'SGEdge', is_unique_freq: bool):
        """Compute the frequency of data edges between """
        if not is_unique_freq:
            freq = 0
            if edge.target_id == self.id:
                # from node to statement, we only consider the target that is property of the statement
                # since we always create new statement based on the value
                for (source_flow, target_flow), stmt2prov in self.flow.items():
                    if source_flow.edge_id != edge.predicate or source_flow.sg_source_id != edge.source_id:
                        continue
                    if target_flow.edge_id != edge.predicate:
                        continue
                    # same_stmt = False
                    # for stmt_id, prov in stmt2prov.items():
                    #     stmt: StatementNode = dg.nodes[stmt_id]['data']
                    #     if (EdgeFlowSource(source_flow.dg_source_id, source_flow.edge_id),
                    #         EdgeFlowTarget(target_flow.dg_target_id, target_flow.edge_id)) in stmt.flow:
                    #         same_stmt = True
                    #         break
                    #
                    # if same_stmt:
                    freq += 1
            else:
                for (source_flow, target_flow), stmt2prov in self.flow.items():
                    if target_flow.sg_target_id == edge.target_id and target_flow.edge_id == edge.predicate:
                        freq += 1
            return freq

        unique_pairs = set()
        if edge.target_id == self.id:
            for (source_flow, target_flow), stmt2prov in self.flow.items():
                if source_flow.sg_source_id != edge.source_id or source_flow.edge_id != edge.predicate:
                    continue

                # the target is the one contain value of the statement property, which we should have
                # only one
                if target_flow.edge_id != edge.predicate:
                    continue

                dg_source = dg.nodes[source_flow.dg_source_id]['data']
                if dg_source.is_cell:
                    source_val = dg_source.value
                else:
                    assert dg_source.is_entity_value
                    source_val = dg_source.qnode_id

                dg_target = dg.nodes[target_flow.dg_target_id]['data']
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
                dg_source = dg.nodes[source_flow.dg_source_id]['data']
                if dg_source.is_cell:
                    source_val = dg_source.value
                else:
                    assert dg_source.is_entity_value
                    source_val = dg_source.qnode_id
                for target_flow in target_flows.keys():
                    if target_flow.sg_target_id == edge.target_id and target_flow.edge_id == edge.predicate:
                        dg_target = dg.nodes[target_flow.dg_target_id]['data']
                        if dg_target.is_cell:
                            target_val = dg_target.value
                        elif dg_target.is_literal_value:
                            target_val = dg_target.value.to_string_repr()
                        else:
                            assert dg_target.is_entity_value
                            target_val = dg_target.qnode_id
                        unique_pairs.add((source_val, target_val))

        return len(unique_pairs)

    def get_edges_provenance(self, edges: List['SGEdge']):
        """Retrieve all flow provenances that connects these edges.

        Note that in case we have more than one statement per flow, we merge the statements (i.e., merge
        the provenances) to return the best statements (since the provenance store how the link is generated,
        merged the provenance will select the best one).
        """
        select_target_flows = {(e.target_id, e.predicate) for e in edges}
        links = {}
        for source_flow, target_flows in self.forward_flow.items():
            # source_flow.predicate is always the same, just change to different data node
            # get list of statements that contains all edges
            stmts: Dict[str, Dict[SGEdgeFlowTarget, List[FlowProvenance]]] = None
            for target_flow in target_flows:
                if (target_flow.sg_target_id, target_flow.edge_id) not in select_target_flows:
                    continue
                if stmts is None:
                    stmts = {}
                    for stmt_id, provs in self.flow[source_flow, target_flow].items():
                        stmts[stmt_id] = {}
                        stmts[stmt_id][target_flow] = provs
                else:
                    # do intersection because we want to find the stmt contain all edges
                    stmt2provenances = self.flow[source_flow, target_flow]
                    remove_stmt_ids = set(stmts.keys()).difference(stmt2provenances.keys())
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
                swap_stmts: Dict[SGEdgeFlowTarget, List[FlowProvenance]] = {
                    target_flow: []
                    for target_flow in next(iter(stmts.values()))
                }
                for stmt_id, _target_flows in stmts.items():
                    for target_flow, provenances in _target_flows.items():
                        swap_stmts[target_flow] = FlowProvenance.merge_lst(swap_stmts[target_flow], provenances)
                links[source_flow] = swap_stmts
        return links


@dataclass
class SGEdge:
    source_id: str
    target_id: str
    predicate: str
    features: Dict[str, Any]

    def clone(self):
        return copy.deepcopy(self)


SGNode = Union[SGColumnNode, SGEntityValueNode, SGLiteralValueNode, SGStatementNode]
NxSGNodeAttr = TypedDict('NxSGNodeAttr', data=SGNode)
NxSGEdgeAttr = TypedDict('NxSGEdgeAttr', data=SGEdge)
NxSGEdge = Tuple[str, str, str, NxSGEdgeAttr]


def get_label(porq: str, qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty]):
    if porq.startswith("Q"):
        if porq in wdclasses:
            item = wdclasses[porq]
        elif porq in qnodes:
            item = qnodes[porq]
    else:
        item = wdprops[porq]

    return f"{item.label} ({item.id})"


def merge_connected_components(graphs: List[nx.MultiDiGraph]):
    g = nx.MultiDiGraph()
    for gi in graphs:
        for uid, udata in gi.nodes.items():
            assert uid not in g.nodes
            g.add_node(uid, **udata)

        for uid, vid, eid, edge_data in gi.edges(data=True, keys=True):
            g.add_edge(uid, vid, key=eid, **edge_data)
    return g


@dataclass
class SemanticGraphConstructorArgs:
    table: LinkedTable
    dg: nx.MultiDiGraph
    sg: nx.MultiDiGraph
    # column type assignment, from column index (must stored as string) => QNode
    cta: Optional[Dict[str, str]] = None
    # semantic model
    sm: Optional[O.SemanticModel] = None


# noinspection PyMethodMayBeStatic
class SemanticGraphConstructor:
    STATEMENT_URI = WDOnt.STATEMENT_URI
    STATEMENT_REL_URI = WDOnt.STATEMENT_REL_URI

    def __init__(self, steps: List, qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty]):
        self.steps = steps
        self.qnodes = qnodes
        self.wdclasses = wdclasses
        self.wdprops = wdprops

    def run(self, table: LinkedTable, dg: nx.MultiDiGraph, debug=False):
        args = SemanticGraphConstructorArgs(table, dg, None)
        for i, step in enumerate(self.steps):
            step(self, args)
            if debug:
                viz_sg(args.sg, self.qnodes, self.wdclasses, self.wdprops, HOME_DIR / "graph_viz" / "debug", f"g_{i:02}")
        return args

    @staticmethod
    def get_sg_node_id(u: DGNode):
        if u.is_cell:
            return f"column-{u.column}"
        return u.id

    def get_sg_node_label(self, args: SemanticGraphConstructorArgs, u: DGNode):
        if u.is_cell:
            return args.table.table.columns[u.column].name
        if u.is_entity_value:
            qnode = self.qnodes[u.qnode_id]
            return f"{qnode.label} ({qnode.id})"
        if u.is_literal_value:
            return u.value.to_string_repr()
        raise M.UnreachableError("This function doesn't supposed to be called with statement node")

    @staticmethod
    def get_sg_subgraph(sg: nx.MultiDiGraph, edges: List[Tuple[str, str, str]]):
        """Get a subgraph of the semantic graph containing a set of edges"""
        sg_prime = nx.MultiDiGraph()
        for uid, vid, eid in edges:
            if uid not in sg_prime.nodes:
                sg_prime.add_node(uid, data=sg.nodes[uid]['data'])
            if vid not in sg_prime.nodes:
                sg_prime.add_node(vid, data=sg.nodes[vid]['data'])
            sg_prime.add_edge(uid, vid, key=eid, data=sg.edges[uid, vid, eid]['data'])
        return sg_prime
    
    @staticmethod
    def keep_one_simple_path_between_important_nodes(sg: nx.MultiDiGraph,
                                                     ranking_fn: Optional[Callable[[List[List[Tuple[str, str, str]]]], List[Tuple[str, str, str]]]] = None,
                                                     both_direction: bool = True):
        """
        Let important nodes be columns and context values. If there is more than one path between the important nodes,
        we only keep one. Note that this function doesn't remove a literal value if it is value of a property of a statement
        that appear in the chosen path.

        The default ranking function is select the path with shorter length. Supply your own function for better result.
        The both direction function will select the correct one
        """
        # figure out the important nodes
        important_nodes = set()
        for uid, udata in sg.nodes(data=True):
            u: SGNode = udata['data']
            if u.is_column or (u.is_value and u.is_in_context):
                important_nodes.add(uid)

        if ranking_fn is None:
            ranking_fn = lambda paths: sorted(paths, key=lambda path: len(path))[0]

        # select one path if there are multiple edges
        selected_edges = set()
        for uid, vid in combinations(important_nodes, 2):
            forward_paths = [
                path
                for path in nx.all_simple_edge_paths(sg, uid, vid, cutoff=6)
                if not any(edge[0] in important_nodes for edge in path[1:])
            ]
            backward_paths = [
                path
                for path in nx.all_simple_edge_paths(sg, vid, uid, cutoff=6)
                if not any(edge[0] in important_nodes for edge in path[1:])
            ]
            if both_direction:
                if len(forward_paths) + len(backward_paths) > 0:
                    selected_edges = selected_edges.union(ranking_fn(forward_paths + backward_paths))
            else:
                if len(forward_paths) > 0:
                    selected_edges = selected_edges.union(ranking_fn(forward_paths))
                if len(backward_paths) > 0:
                    selected_edges = selected_edges.union(ranking_fn(backward_paths))

        new_sg = SemanticGraphConstructor.get_sg_subgraph(sg, selected_edges)

        # add missing statement values
        for sid, sdata in list(new_sg.nodes(data=True)):
            stmt: SGStatementNode = sdata['data']
            if not stmt.is_statement:
                continue

            (uid, _, eid), = list(new_sg.in_edges(sid, keys=True))
            for vid, us_edges in new_sg[sid].items():
                if eid in us_edges:
                    # we have the statement property in the graph
                    break
            else:
                # we do not have the statement property in the graph. get it from the original graph
                for vid, us_edges in sg[sid].items():
                    if eid in us_edges:
                        if vid not in new_sg.nodes:
                            new_sg.add_node(vid, data=sg.nodes[vid]['data'])
                        new_sg.add_edge(sid, vid, key=eid, data=us_edges[eid]['data'])

        return new_sg

    @staticmethod
    def st_terminal_nodes(sg: nx.MultiDiGraph):
        terminal_nodes = set()
        for uid, udata in sg.nodes(data=True):
            u: SGNode = udata['data']
            if u.is_column:
                terminal_nodes.add(uid)
        return terminal_nodes

    # ##################################################################################################################
    # CODE FOR PIPELINE
    # ##################################################################################################################
    def init_sg(self, args: SemanticGraphConstructorArgs):
        sg = nx.MultiDiGraph()
        dg = args.dg

        # first step: add node to the graph
        for uid, udata in dg.nodes(data=True):
            u: DGNode = udata['data']
            if u.is_statement:
                # we can have more than one predicates per entity column, these are represented in the statement
                continue
            
            sgi = self.get_sg_node_id(u)
            if not sg.has_node(sgi):
                label = self.get_sg_node_label(args, u)
                if u.is_cell:
                    sgu = SGColumnNode(id=sgi, label=label, column=u.column, nodes=set())
                elif u.is_entity_value:
                    sgu = SGEntityValueNode(id=sgi, label=label, qnode_id=u.qnode_id, context_span=u.context_span)
                else:
                    assert u.is_literal_value
                    sgu = SGLiteralValueNode(id=sgi, label=label, value=u.value, context_span=u.context_span)
                sg.add_node(sgi, data=sgu)

            if sg.nodes[sgi]['data'].is_column:
                sg.nodes[sgi]['data'].nodes.add(uid)

        # second step: add link
        for uid, udata in dg.nodes(data=True):
            u: DGNode = udata['data']
            if u.is_statement:
                continue

            # add statement
            p2stmts: Dict[str, List[StatementNode]] = {}
            for _, sid, us_eid, us_edata in dg.out_edges(uid, data=True, keys=True):
                if us_eid not in p2stmts:
                    p2stmts[us_eid] = []
                p2stmts[us_eid].append(dg.nodes[sid]['data'])

            sgi = self.get_sg_node_id(u)
            for p, stmts in p2stmts.items():
                for stmt in stmts:
                    # we duplicate based on SG column node, not by DG node
                    children_values = {}
                    children_non_values = []

                    for _, vid, sv_eid, sv_edata in dg.out_edges(stmt.id, data=True, keys=True):
                        v: DGNode = dg.nodes[vid]['data']
                        if (vid, sv_eid) not in stmt.forward_flow[uid, p]:
                            # because of re-use statement, we need to only consider the children of this u node only
                            continue

                        if sv_eid == p:
                            tgi = self.get_sg_node_id(v)
                            if tgi not in children_values:
                                children_values[tgi] = []
                            children_values[tgi].append(v)
                        else:
                            children_non_values.append((sv_eid, v))

                    for tgi, vals in children_values.items():
                        stmt_id = SGStatementNode.get_id(sgi, p, tgi)
                        if stmt_id not in sg.nodes:
                            sg.add_node(stmt_id, data=SGStatementNode.new(stmt_id))

                        if (sgi, stmt_id, p) not in sg.edges:
                            sg.add_edge(sgi, stmt_id, key=p, data=SGEdge(sgi, stmt_id, p, {}))
                        if (stmt_id, tgi, p) not in sg.edges:
                            sg.add_edge(stmt_id, tgi, key=p, data=SGEdge(stmt_id, tgi, p, {}))

                        sg_stmt: SGStatementNode = sg.nodes[stmt_id]['data']
                        for v in vals:
                            sg_stmt.track_provenance(SGEdgeFlowSource(uid, sgi, p), stmt.id, SGEdgeFlowTarget(v.id, tgi, p), stmt.get_provenance(EdgeFlowSource(uid, p), EdgeFlowTarget(v.id, p)))

                        for qp, qv in children_non_values:
                            qtgi = self.get_sg_node_id(qv)
                            if (stmt_id, qtgi, qp) not in dg.edges:
                                sg.add_edge(stmt_id, qtgi, key=qp, data=SGEdge(stmt_id, qtgi, qp, {}))

                            sg_stmt.track_provenance(SGEdgeFlowSource(uid, sgi, p), stmt.id, SGEdgeFlowTarget(qv.id, qtgi, qp), stmt.get_provenance(EdgeFlowSource(uid, p), EdgeFlowTarget(qv.id, qp)))
        args.sg = sg

    def cta_majority_vote(self, args: SemanticGraphConstructorArgs):
        # temporary before we should refactor the whole s*
        """Predict the column types for each column in the semantic graph"""
        def add_merge_qnodes(cell: CellNode, cell2qnodes: Dict[str, List[QNode]]):
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            if len(cell.qnode_ids) > 1:
                # attempt to merge qnodes (spatial) if they are contained in each other
                # we should go even higher order
                ignore_qnodes = set()
                for q0_id in cell.qnode_ids:
                    q0 = self.qnodes[q0_id]
                    # location or located in the administrative entity
                    vals = {
                        stmt.value.as_qnode_id()
                        for stmt in chain(q0.props.get("P131", []), q0.props.get("P276", []))
                    }
                    for q1_id in cell.qnode_ids:
                        if q0_id == q1_id:
                            continue
                        if q1_id in vals:
                            # q0 is inside q1, ignore q1
                            ignore_qnodes.add(q1_id)
                cell2qnodes[cell.id] = [self.qnodes[q_id] for q_id in cell.qnode_ids if q_id not in ignore_qnodes]
            elif len(cell.qnode_ids) > 0:
                cell2qnodes[cell.id] = [self.qnodes[cell.qnode_ids[0]]]
            else:
                cell2qnodes[cell.id] = []

        args.cta = {}
        sg = args.sg
        dg = args.dg
        cell2qnodes = {}
        for uid, udata in sg.nodes(data=True):
            u: SGColumnNode = udata['data']
            if not u.is_column:
                continue

            # cells in this column
            cells: List[CellNode] = [dg.nodes[cid]['data'] for cid in u.nodes]
            covered_fractions = [
                sum(span.length for spans in cell.qnodes_span.values() for span in spans) / max(len(cell.value), 1)
                for cell in cells
                if len(cell.qnode_ids) >= 1
            ]
            if len(covered_fractions) == 0:
                continue
            avg_covered_fractions = np.mean(covered_fractions)
            if avg_covered_fractions < 0.8:
                continue

            for cell in cells:
                add_merge_qnodes(cell, cell2qnodes)
            merged_class_freq = dict(Counter([
                c
                for cell in cells
                for c in (stmt.value.as_qnode_id()
                    for qnode in cell2qnodes[cell.id]
                    for stmt in qnode.props.get("P31", [])
                )
            ]).items())
            if len(merged_class_freq) == 0:
                continue

            best_class, count = max(merged_class_freq.items(), key=itemgetter(1))
            args.cta[str(u.column)] = WDOnt.get_qnode_uri(best_class)
        
        return args

    def construct_sm(self, args: SemanticGraphConstructorArgs):
        """Get the final semantic model by combining relationships in SG and CTA"""
        sm = O.SemanticModel()
        readable_label_fn = lambda qnode_id: get_label(qnode_id, self.qnodes, self.wdclasses, self.wdprops)
        # a mapping from data node to class node for each data node that has a type
        dnode2cnode = {}
        for uid, udata in args.sg.nodes(data=True):
            u: SGNode = udata['data']
            if u.is_column:
                sm.add_node(O.DataNode(u.id, u.column, u.label))
                if u.column in args.cta:
                    qnode = self.wdclasses[args.cta[u.column]]
                    _cid = str(uuid4())
                    sm.add_node(O.ClassNode(_cid, qnode.get_uri(), f"wd:{qnode.id}", readable_label=readable_label_fn(qnode.id)))
                    dnode2cnode[u.id] = _cid
                    sm.add_edge(O.Edge(_cid, u.id, str(RDFS.label), "rdfs:label"))
            elif u.is_statement:
                sm.add_node(O.ClassNode(u.id, self.STATEMENT_URI, self.STATEMENT_REL_URI))
            elif u.is_value:
                if u.is_literal_value:
                    sm.add_node(O.LiteralNode(u.id, u.value.to_string_repr()))
                else:
                    sm.add_node(O.LiteralNode(u.id, u.value.as_qnode_id(), readable_label=readable_label_fn(u.value.as_qnode_id())))

        for uid, vid, eid, edata in args.sg.edges(data=True, keys=True):
            if uid in dnode2cnode:
                uid = dnode2cnode[uid]
            if vid in dnode2cnode:
                vid = dnode2cnode[vid]
            porq = self.wdprops[eid]
            sm.add_edge(O.Edge(uid, vid, porq.get_uri(), f"p:{porq.id}", readable_label=readable_label_fn(porq.id)))
        args.sm = sm

    def prune_sg_redundant_entity(self, args: SemanticGraphConstructorArgs):
        """Remove entity & context that is:
        - just an intermediate nodes that the parents only has one edge and the children has only one edge too
        - don't remove entity what is the main value of the statement
        """
        sm_g = args.sg

        # we are going to remove entity & context that is:
        # just a intermediate nodes that the parents only one edges and the children too
        stop = False
        while not stop:
            stop = True
            # for nid, ndata in list(sm_g.nodes(data=True)):
            #     n: SGNode = ndata['data']
            #     if n.is_statement:
            #         inedges: List[NxSGEdge] = list(sm_g.in_edges(nid, data=True, keys=True))
            #         outedges: List[NxSGEdge] = list(sm_g.out_edges(nid, data=True, keys=True))
            #
            #         if len(inedges) != 1:
            #             continue
            #         uid, _, seid, sedata = inedges[0]
            #         max_weight = -1
            #         for _, vid, teid, tedata in outedges:
            #             v: SGNode = sm_g.nodes[vid]['data']
            #             if teid != seid:
            #                 # we only want to consider value of the statement
            #                 continue
            #
            #             if v.is_column:
            #                 if tedata['data'].features['freq'] > max_weight:
            #                     max_weight = tedata['data'].features['freq']
            #         for _, vid, teid, tedata in outedges:
            #             v: SGNode = sm_g.nodes[vid]['data']
            #             if teid != seid:
            #                 # we only want to consider value of the statement
            #                 continue
            #
            #             if v.is_value:
            #                 if tedata['data'].features['freq'] < max_weight:
            #                     assert False
            #                     sm_g.remove_node(vid)
            #                     stop = False

            for nid, ndata in list(sm_g.nodes(data=True)):
                n: SGNode = ndata['data']
                if n.is_value and n.is_entity_value:
                    inedges: List[NxSGEdge] = list(sm_g.in_edges(nid, data=True, keys=True))
                    outedges: List[NxSGEdge] = list(sm_g.out_edges(nid, data=True, keys=True))

                    if len(inedges) == 0:
                        if all(sm_g.out_degree(vid) == 1 for _, vid, teid, tedata in outedges):
                            sm_g.remove_node(nid)
                            stop = False
                            continue

                    """
                    Remove the entity when it is not doing 
                    """
                    if len(inedges) > 1 or len(outedges) > 1:
                        continue

                    if len(inedges) > 0:
                        uid, _, seid, sedata = inedges[0]
                        if not all(sm_g.nodes[sib_nid]['data'].is_value for
                                   _, sib_nid in sm_g.out_edges(uid)):
                            continue

                    if len(outedges) > 0:
                        _, vid, teid, tedata = outedges[0]
                        if sm_g.out_degree(vid) != 1:
                            continue

                    # remove that node has only
                    sm_g.remove_node(nid)
                    stop = False

            for nid, ndata in list(sm_g.nodes(data=True)):
                n = ndata['data']
                if n.is_statement and (sm_g.in_degree(nid) == 0 or sm_g.out_degree(nid) == 0):
                    sm_g.remove_node(nid)
                    stop = False

    # ##################################################################################################################
    # CODE FOR COMPUTING FEATURES
    # ##################################################################################################################
    def calculate_link_frequency(self, args: SemanticGraphConstructorArgs):
        def get_in_edges(g, uid) -> List[SGEdge]:
            return [edata['data'] for _, _, eid, edata in g.in_edges(uid, data=True, keys=True)]

        sg = args.sg
        for sid, sdata in sg.nodes(data=True):
            s: SGStatementNode = sdata['data']
            if not s.is_statement:
                continue

            in_edge, = get_in_edges(sg, sid)
            in_edge.features['freq'] = s.compute_freq(sg, args.dg, in_edge, is_unique_freq=False)
            in_edge.features['unique_freq'] = s.compute_freq(sg, args.dg, in_edge, is_unique_freq=True)

            for _, vid, sv_eid, sv_edata in sg.out_edges(sid, data=True, keys=True):
                # count the frequency of the sv_eid by counting the number of times it is between specific instance
                sv_e: SGEdge = sv_edata['data']
                sv_e.features['freq'] = s.compute_freq(sg, args.dg, sv_e, is_unique_freq=False)
                sv_e.features['unique_freq'] = s.compute_freq(sg, args.dg, sv_e, is_unique_freq=True)

    # ##################################################################################################################
    # CODE FOR SOLVING THE STEINER TREE PROBLEM
    # ##################################################################################################################
    def st_maximum_arborescence_solver(self, args: SemanticGraphConstructorArgs):
        sm_g = nx.MultiDiGraph()
        for uid, vid, eid, edata in args.sg.edges(data=True, keys=True):
            if args.sg.nodes[vid]['data'].is_statement:
                # freq. for source -> statement
                # use a constant value as this link depends on the link coming out from the statement
                edata['weight'] = 0.1
            else:
                edata['weight'] = edata['data'].features['freq']

        # find the tree
        resp = nx.algorithms.tree.branchings.maximum_branching(args.sg, attr='weight', preserve_attrs=True)
        for s, t, e, edata in resp.edges(data=True, keys=True):
            edge = edata['data']
            if s not in sm_g:
                sm_g.add_node(s, **args.sg.nodes[s])
            if t not in sm_g:
                sm_g.add_node(t, **args.sg.nodes[t])
            sm_g.add_edge(s, t, key=edge.predicate, data=edge)

        # remove redundant nodes that is not relevant to the terminals
        # expand the statement with
        # for nid, ndata in list(sm_g.nodes(data=True)):
        #     n: SGNode = ndata['data']
        #     if n.is_statement:
        #         edges = list(sm_g.in_edges(nid, data=True, keys=True))
        #         if len(edges) == 0:
        #             continue
        #         assert len(edges) == 1
        #         uid, _, seid, edata = edges[0]
        #         source_e: SGEdge = edata['data']
        #
        #         # split when all props of statement is the same as the value of the statement
        #         need_split_stmt = True
        #         for _, vid, _, edata in sm_g.out_edges(nid, data=True, keys=True):
        #             if edata['data'].predicate != source_e.predicate:
        #                 need_split_stmt = False
        #                 break
        #
        #         if need_split_stmt:
        #             lst = list(sm_g.out_edges(nid, data=True, keys=True))
        #             for index, (_, vid, teid, edata) in enumerate(lst[1:]):
        #                 n2 = n.clone(without_provenance=True)
        #                 n2.id = n.id + f":{index}"
        #                 sm_g.add_node(n2.id, data=n2)
        #                 sm_g.add_edge(uid, n2.id, key=seid, data=source_e)
        #                 sm_g.add_edge(n2.id, vid, key=teid, data=edata['data'])
        #                 sm_g.remove_edge(n.id, vid, teid)

        # remove redundant nodes:
        # - a statement doesn't have any properties or incoming nodes
        # - an entity node that is a leaf node, and is the only child of the parent statement.
        while True:
            remove_nodes = set()
            for nid in sm_g:
                n: SGNode = sm_g.nodes[nid]['data']
                outdegree = sm_g.out_degree(nid)
                indegree = sm_g.in_degree(nid)

                if n.is_statement and (outdegree == 0 or indegree == 0):
                    # a statement doesn't have any properties or incoming nodes
                    remove_nodes.add(nid)
                elif n.is_value:
                    if outdegree == 0:
                        # an entity that doesn't have any outgoing edges, that can be used as something for the statement
                        # so we only remove it when the parent class has only one class
                        total_outdegree_parent = sum(
                            sm_g.out_degree(uid) for uid, vid, edata in sm_g.in_edges(nid, data=True))
                        if total_outdegree_parent == 1:
                            remove_nodes.add(nid)

                        # this extra entity will only be useful if it is not the qualifiers of the statement
                        if all(sm_g.in_degree(uid) == 1 and eid !=
                               next(iter(sm_g.in_edges(uid, data=True, keys=True)))[2] for uid, vid, eid, edata in
                               sm_g.in_edges(nid, data=True, keys=True)):
                            remove_nodes.add(nid)
                    if indegree == 0:
                        # an entity that doesn't have any incoming edges, but the only on predicate
                        total_outdegree_children = sum(
                            sm_g.out_degree(uid) for uid, vid, eid in sm_g.out_edges(nid, data=True))
                        if outdegree == 1 and total_outdegree_children == 1:
                            remove_nodes.add(nid)

            for nid in remove_nodes:
                sm_g.remove_node(nid)

            if len(remove_nodes) == 0:
                break

        # remove statement that has no qualifiers
        # for nid, ndata in list(sm_g.nodes(data=True)):
        #     n: SGNode = ndata['data']
        #     if n.is_statement and sm_g.in_degree(nid) == 1 and sm_g.out_degree(nid) == 1:
        #         uid, _, seid, sedata = list(sm_g.in_edges(nid, data=True, keys=True))[0]
        #         _, vid, teid, tedata = list(sm_g.out_edges(nid, data=True, keys=True))[0]
        #
        #         if sedata['data'].predicate == tedata['data'].predicate:
        #             # remove the statement that has no qualifiers
        #             sm_g.remove_node(nid)
        #             e: SGEdge = sedata['data'].clone(without_provenance=True)
        #             sm_g.add_edge(uid, vid, seid, data=e)
        args.sg = sm_g

    def st_networkx_st_solver(self, args: SemanticGraphConstructorArgs):
        sm_g = nx.MultiDiGraph()
        terminal_nodes = []
        for nid, ndata in args.sg.nodes(data=True):
            n: SGNode = ndata['data']
            if n.is_column or n.is_context:
                terminal_nodes.append(nid)

        return nx.algorithms.approximation.steinertree.steiner_tree(args.sg, terminal_nodes, weight='weight')


def viz_sg(sg, qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty], outdir: str, graph_id: str):
    get_label_fn = lambda x: get_label(x, qnodes, wdclasses, wdprops)
    colors = {
        "context": dict(fill="#C6E5FF", stroke="#5B8FF9"),
        "statement": dict(fill="#d9d9d9", stroke="#434343"),
        "kg": dict(fill="#b7eb8f", stroke="#135200"),
        "column": dict(fill='#ffd666', stroke='#874d00')
    }

    def node_fn(uid, udata):
        u: SGNode = udata['data']
        html = ""
        label = ""
        if u.is_value:
            nodetype = 'context' if u.is_in_context else 'kg'
            label = u.label
            if u.is_entity_value:
                qnodeid = u.qnode_id
                html = f"""<a href="http://www.wikidata.org/wiki/{qnodeid}" target="_blank">{get_label_fn(qnodeid)}</a>"""
        elif u.is_statement:
            nodetype = 'statement'
        else:
            assert u.is_column
            label = f"{u.label} ({u.column})"
            nodetype = 'column'

        return {
            "label": label,
            "style": colors[nodetype],
            "labelCfg": {
                "style": {
                    "fill": "black",
                    "background": {
                        "padding": [4, 4, 4, 4],
                        "radius": 3,
                        **colors[nodetype]
                    }
                }
            },
            "html": html
        }

    def edge_fn(eid, edata):
        edge: SGEdge = edata['data']
        label = get_label_fn(eid)
        if 'prob' in edge.features:
            label += f" p={edge.features['prob']:.6f}"

        html = f'<a href="http://www.wikidata.org/wiki/Property:{eid}" target="_blank">{get_label_fn(eid)}</a>'
        if len(edge.features) > 0:
            html_feat = []
            for feat, val in edge.features.items():
                html_feat.append(f"<li><span style='min-width: 100px'><b>{feat}:</b></span> {val:.2f}</li>")
            html += f"<ul>{''.join(html_feat)}</ul>"

        return {
            "label": label,
            "html": html
        }

    return viz_graph(sg, node_fn, edge_fn, outdir, graph_id)


if __name__ == '__main__':
    from sm_unk.config import HOME_DIR
    from sm_unk.dev.wikitable2wikidata.sxx_evaluation import get_input_data
    from sm_unk.dev.wikitable2wikidata.kg_index import KGObjectIndex

    table_index = 6

    max_n_hop = 2
    dataset_dir = HOME_DIR / "wikitable2wikidata/250tables"
    gold_models = get_input_data(dataset_dir, dataset_dir.name, only_curated=True)

    REDIS_CACHE_URL = 'redis://localhost:6379/8'
    qnodes = get_qnodes(dataset_dir, n_hop=max_n_hop + 1, test=True)
    wdclasses = WDClass.from_file(dataset_dir / "ontology", load_parent_closure=True)
    wdprops = WDProperty.from_file(load_parent_closure=True)

    kg_index_file = dataset_dir / "kg_index" / "object_index.2hop_transitive.pkl.gz"
    if kg_index_file.exists():
        kg_object_index = KGObjectIndex.deserialize(kg_index_file, verbose=True)
    else:
        index_qnode_ids = list(get_qnodes(dataset_dir, n_hop=1).keys())
        kg_object_index = KGObjectIndex.from_qnodes(index_qnode_ids, qnodes, wdprops,
                                                    n_hop=2,
                                                    verbose=True)
        kg_object_index.serialize(kg_index_file)

    table = gold_models[table_index][1]
    dg = build_data_graph(table, qnodes, wdprops, kg_object_index, max_n_hop=2)
    constructor = SemanticGraphConstructor([
        SemanticGraphConstructor.init_sg,
        # SemanticGraphConstructor.calculate_link_frequency,
        # SemanticGraphConstructor.prune_sg_redundant_entity,
    ], qnodes, wdclasses, wdprops)
    resp = constructor.run(table, dg, debug=False)
import operator
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Any, Optional

import networkx as nx


class PropPath:
    """A way to specify how to access property of a node/edge in networkx graph.
    Nested property is separated by a dot. For example: `data.column.name`.

    Id of a node or key of edge can be accessed using this path `@id`.
    """

    def __init__(self, path: str, allow_unknown_key: bool):
        self.origin_path = path
        self.allow_unknown_key = allow_unknown_key
        self.attrs = path.split(".")

        if self.attrs[0] == '@id':
            assert len(self.attrs) == 1


    def get_value(self, objectid: str, objectdata: dict):
        """Get value of the edge or node specified by the path

        Parameters
        ----------
        objectid: node id or edge key
        objectdata: node or edge data

        Returns
        -------

        """
        if self.attrs[0] == '@id':
            return objectid

        for attr in self.attrs:
            if isinstance(objectdata, dict):
                objectdata = objectdata[attr]
            elif hasattr(objectdata, attr):
                objectdata = getattr(objectdata, attr)
            else:
                if self.allow_unknown_key:
                    return None
                raise KeyError(attr)
        return objectdata

    def copy_to_dict(self, objectid: str, objectdata: dict, destination: dict):
        value = self.get_value(objectid, objectdata)
        assert not self.attrs[0] == '@id'

        for attr in self.attrs[:-1]:
            if attr not in destination:
                destination[attr] = {}
            destination = destination[attr]
        destination[self.attrs[-1]] = value


@dataclass
class Operator(Enum):
    Equal = "="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


class Condition(ABC):
    @abstractmethod
    def satisfy(self, objectid: str, object: dict):
        """Whether the condition is satisfied"""
        pass


class TrueCondition(Condition):

    def satisfy(self, objectid: str, object: dict):
        return True


class BasicCondition(Condition):

    def __init__(self, path: PropPath, ops: Operator, target_value: Any):
        if ops == Operator.Equal:
            self.ops = operator.eq
        elif ops == Operator.LT:
            self.ops = operator.lt
        elif ops == Operator.LTE:
            self.ops = operator.le
        elif ops == Operator.GT:
            self.ops = operator.gt
        elif ops == Operator.GTE:
            self.ops = operator.ge
        else:
            raise Exception("Unreachable!")

        self.path = path
        self.target_value = target_value

    def satisfy(self, objectid: str, object: dict):
        value = self.path.get_value(objectid, object)
        return self.ops(value, self.target_value)


class NotCondition(Condition):

    def __init__(self, condition: 'Condition'):
        self.condition = condition

    def satisfy(self, objectid: str, object: dict):
        return not self.condition.satisfy(objectid, object)


class AndCondition(Condition):

    def __init__(self, conditions: List['Condition']):
        self.conditions = conditions

    def satisfy(self, objectid: str, object: dict):
        return all(condition.satisfy(objectid, object) for condition in self.conditions)


class OrCondition(Condition):

    def __init__(self, conditions: List['Condition']):
        self.conditions = conditions

    def satisfy(self, objectid: str, object: dict):
        return any(condition.satisfy(objectid, object) for condition in self.conditions)


@dataclass
class NodeVariable:
    # id of the node in the query
    id: str
    condition: Condition


@dataclass
class EdgeVariable:
    source_id: str
    target_id: str
    condition: Condition
    is_optional: bool = False


class QueryBuilder:
    def __init__(self):
        self.node_ids = set()
        self.variables = []

    def n(self, id: Optional[str] = None, condition: Optional['QueryConditionBuilder'] = None):
        """Return a node variable builder"""
        if id is None:
            id = str(uuid.uuid4())
        assert id not in self.node_ids, "Re-defined node id"
        if condition is None:
            condition = TrueCondition()
        else:
            condition = condition.build()
        node = NodeVariable(id, condition)
        self.variables.append(node)
        return QueryNodeBuilder(self, node)

    def c(self, path: str, ops: str, value: Any):
        """Return a condition"""
        return QueryConditionBuilder(self, BasicCondition(PropPath(path), Operator(ops), value))

    def get_query(self):
        return self.variables


@dataclass
class QueryConditionBuilder:
    builder: QueryBuilder
    condition: Condition

    def build(self) -> Condition:
        return self.condition

    def __and__(self, other: 'QueryConditionBuilder'):
        return QueryConditionBuilder(self.builder, AndCondition([self.condition, other.condition]))

    def __or__(self, other: 'QueryConditionBuilder'):
        return QueryConditionBuilder(self.builder, OrCondition([self.condition, other.condition]))

    def __neg__(self):
        return QueryConditionBuilder(self.builder, NotCondition(self.condition))


@dataclass
class QueryNodeBuilder:
    builder: QueryBuilder
    node_variable: NodeVariable

    def to(self, other: 'QueryNodeBuilder', condition: Optional['QueryConditionBuilder'] = None):
        return self._make_edge(other, condition, False)

    def to_optional(self, other: 'QueryNodeBuilder', condition: Optional['QueryConditionBuilder'] = None):
        return self._make_edge(other, condition, True)

    def _make_edge(self, other: 'QueryNodeBuilder', condition: Optional['QueryConditionBuilder'] = None, optional: bool = False):
        """Add an edge to the query"""
        if condition is None:
            condition = TrueCondition()
        else:
            condition = condition.build()
        edge = EdgeVariable(self.node_variable.id, other.node_variable.id, condition)
        self.builder.variables.append(edge)
        return other


def query_graph(g: nx.MultiDiGraph, queries: List[Union[NodeVariable, EdgeVariable]]):
    """
    Query the networkx graph to select a subset of the graph.

    The algorithm works like this:
    We first build a mapping, where each node variable is associated to
    the list of matched nodes in the graph.
    Then, we loop through each edge variable in the query, and record matched edges in the graph

    Then, we loop through each edge variable in the query that connects two node variables,
    for each edge in the graph, if it does not satisfied the query condition, we remove
    its the source and target from the corresponding node variables.
    We do a while loop until there is no more nodes getting removed
    Then, we do another final loop to find all the edges.
    """
    node_vars = {}
    for query in queries:
        if isinstance(query, NodeVariable):
            matched_nodes = set()
            for uid in g:
                u = g.nodes[uid]
                if query.condition.satisfy(uid, u):
                    matched_nodes.add(uid)
            node_vars[query.id] = matched_nodes
        elif isinstance(query, EdgeVariable):
            pass
        else:
            raise NotImplementedError()

    stop = False
    while not stop:
        stop = True
        for query in queries:
            if isinstance(query, NodeVariable):
                pass
            elif isinstance(query, EdgeVariable) and not query.is_optional:
                source_ids = node_vars[query.source_id]
                target_ids = node_vars[query.target_id]

                satisfied_source_ids = set()
                satisfied_target_ids = set()

                for uid, vid, eid, edata in g.edges(data=True, keys=True):
                    if uid not in source_ids:
                        continue
                    if vid not in target_ids:
                        continue
                    if query.condition.satisfy(eid, edata):
                        satisfied_source_ids.add(uid)
                        satisfied_target_ids.add(vid)

                deleted_sources = source_ids.difference(satisfied_source_ids)
                deleted_targets = target_ids.difference(satisfied_target_ids)
                if len(deleted_sources) + len(deleted_targets) > 0:
                    stop = False
                    for uid in deleted_sources:
                        source_ids.remove(uid)
                    for vid in deleted_targets:
                        target_ids.remove(vid)
            else:
                raise NotImplementedError()

    matched_edges = set()
    for query in queries:
        if isinstance(query, NodeVariable):
            pass
        elif isinstance(query, EdgeVariable):
            source_ids = node_vars[query.source_id]
            target_ids = node_vars[query.target_id]

            for uid, vid, eid, edata in g.edges(data=True, keys=True):
                if uid not in source_ids:
                    continue
                if vid not in target_ids:
                    continue
                if query.condition.satisfy(eid, edata):
                    matched_edges.add((uid, vid, eid))
        else:
            raise NotImplementedError()

    # now create a new graph
    resp_g = nx.MultiDiGraph()
    for ids in node_vars.values():
        for id in ids:
            if id not in resp_g:
                resp_g.add_node(id, **g.nodes[id])

    for uid, vid, eid in matched_edges:
        resp_g.add_edge(uid, vid, key=eid, **g.edges[uid, vid, eid])
    return resp_g

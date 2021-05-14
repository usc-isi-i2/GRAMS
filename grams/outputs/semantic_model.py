import enum
import copy
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Iterable, List, Tuple, Set

import orjson
from IPython.core.display import display
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from IPython import get_ipython

import grams.misc
from grams.evaluation import sm_metrics
from grams.misc.misc import auto_wrap


@dataclass
class SemanticType:
    class_abs_uri: str
    predicate_abs_uri: str
    class_rel_uri: str
    predicate_rel_uri: str

    @property
    def label(self):
        return (self.class_rel_uri, self.predicate_rel_uri)

    def is_entity_type(self) -> bool:
        """Telling if this semantic type is for entity column
        """
        return self.predicate_abs_uri in {
            'http://www.w3.org/2000/01/rdf-schema#label',
            'http://schema.org/name'
        }

    def __hash__(self):
        return hash((self.class_abs_uri, self.predicate_abs_uri))

    def __eq__(self, other):
        if not isinstance(other, SemanticType):
            return False

        return self.class_abs_uri == other.class_abs_uri and self.predicate_abs_uri == other.predicate_abs_uri

    def __str__(self):
        return f"{self.class_rel_uri}--{self.predicate_rel_uri}"

    def __repr__(self):
        return f"SType({self})"


@dataclass
class ClassNode:
    id: str
    abs_uri: str
    rel_uri: str
    approximation: bool = False
    readable_label: Optional[str] = None

    @property
    def label(self):
        return self.readable_label or self.rel_uri

    @property
    def is_class_node(self):
        return True

    @property
    def is_data_node(self):
        return False

    @property
    def is_literal_node(self):
        return False


@dataclass
class DataNode:
    id: str
    col_index: int
    label: str

    @property
    def is_class_node(self):
        return False

    @property
    def is_data_node(self):
        return True

    @property
    def is_literal_node(self):
        return False


class LiteralNodeDataType(enum.Enum):
    String = "string"
    Entity = "entity-id"


@dataclass
class LiteralNode:
    id: str
    value: str
    # readable label of the literal node! should not confuse it with value
    readable_label: Optional[str] = None
    # whether the literal node is in the surround context of the dataset
    is_in_context: bool = False
    datatype: LiteralNodeDataType = LiteralNodeDataType.String

    @property
    def label(self):
        return self.readable_label or self.value

    @property
    def is_class_node(self):
        return False

    @property
    def is_data_node(self):
        return False

    @property
    def is_literal_node(self):
        return True


Node = Union[ClassNode, DataNode, LiteralNode]


@dataclass
class Edge:
    source: str
    target: str
    abs_uri: str
    rel_uri: str
    approximation: bool = False
    readable_label: Optional[str] = None
    # id of the edge, this is set automatically by the semantic model in case we have multiple edges between nodes
    # and we need to delete one of them
    id: Optional[str] = None

    @property
    def label(self):
        return self.readable_label or self.rel_uri


class SemanticModel:
    def __init__(self, graph: Optional[nx.Graph] = None, edge_id_counter: int = 0):
        if graph is not None:
            self.g = graph
        else:
            self.g = nx.MultiDiGraph()
        self.edge_id_counter = edge_id_counter

    def get_n_nodes(self):
        return len(self.g.nodes)

    def get_n_edges(self):
        return len(self.g.edges)

    def get_node(self, nid: str) -> Optional[Node]:
        if nid not in self.g.nodes:
            return None
        return self.g.nodes[nid]['data']

    def get_data_node(self, column_index: int) -> Optional[DataNode]:
        for uid, u in self.g.nodes.data("data"):
            if not u.is_class_node and u.col_index == column_index:
                return u
        return None

    def get_literal_node(self, value: str) -> Optional[LiteralNode]:
        for uid, u in self.g.nodes.data("data"):
            if not u.is_literal_node and u.value == value:
                return u
        return None

    def get_edges_between_nodes(self, source_id: str, target_id: str) -> List[Edge]:
        res = self.g.get_edge_data(source_id, target_id)
        if res is None:
            return []
        return [x['data'] for x in res.values()]

    def has_node(self, nid: str):
        return self.g.has_node(nid)

    def has_edge(self, edge: Edge):
        return self.g.has_edge(edge.source, edge.target)

    def add_node(self, node: Node):
        self.g.add_node(node.id, data=node)

    def add_edge(self, edge: Edge):
        self.edge_id_counter += 1
        edge.id = self.edge_id_counter
        self.g.add_edge(edge.source, edge.target, key=self.edge_id_counter, data=edge)

    def update_node(self, nid: str, node: Node):
        """Update the node content by id"""
        if nid == node.id:
            # just update the node content
            self.g.nodes[nid]['data'] = node
        else:
            # has to replace all the edges
            inedges = list(self.g.in_edges(nid, data=True, keys=True))
            outedges = list(self.g.out_edges(nid, data=True, keys=True))

            self.g.add_node(node.id, data=node)
            for uid, vid, eid, edata in inedges:
                e: Edge = edata['data']
                e.target = node.id
                self.g.remove_edge(uid, vid, eid)
                self.g.add_edge(uid, node.id, eid, data=e)

            for uid, vid, eid, edata in outedges:
                e: Edge = edata['data']
                e.source = node.id
                self.g.remove_edge(uid, vid, eid)
                self.g.add_edge(node.id, vid, eid, data=e)

            self.g.remove_node(nid)

    def remove_node(self, node_id: str):
        self.g.remove_node(node_id)

    def remove_edges_between_nodes(self, source_id: str, target_id: str, eid: Optional[str]=None):
        if eid is None:
            self.g.remove_edge(source_id, target_id)
        else:
            self.g.remove_edge(source_id, target_id, key=eid)

    def clone(self):
        return SemanticModel(copy.deepcopy(self.g), self.edge_id_counter)

    def iter_nodes(self) -> Iterable[Node]:
        return (u for uid, u in self.g.nodes.data("data"))

    def iter_edges(self) -> Iterable[Edge]:
        return (e for source, target, e in self.g.edges.data("data"))

    def incoming_edges(self, node_id: str) -> List[Edge]:
        """Get a list of incoming edges of a column"""
        lst = []
        for u, v, e in self.g.in_edges(node_id, data='data'):
            lst.append(e)
        return lst

    def outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get a list of outgoing edges"""
        lst = []
        for u, v, e in self.g.out_edges(node_id, data='data'):
            lst.append(e)
        return lst

    def children(self, node_id: str) -> List[Node]:
        lst = []
        for u, v, e in self.g.out_edges(node_id, data='data'):
            lst.append(self.g.nodes[v]['data'])
        return lst

    def get_semantic_types_of_column(self, col_index: int) -> List[SemanticType]:
        dnode = self.get_data_node(col_index)
        sem_types = set()
        for u, v, e in self.g.in_edges(dnode.id, data='data'):
            u = self.get_node(u)
            sem_types.add(SemanticType(u.abs_uri, e.abs_uri, u.rel_uri, e.rel_uri))
        return list(sem_types)

    def get_semantic_types(self) -> Set[SemanticType]:
        sem_types = set()
        for e in self.iter_edges():
            u = self.get_node(e.source)
            assert isinstance(u, ClassNode)
            if self.get_node(e.target).is_class_node:
                continue

            sem_types.add(SemanticType(u.abs_uri, e.abs_uri, u.rel_uri, e.rel_uri))
        return sem_types

    def to_json(self):
        return {
            "version": 1,
            "nodes": [asdict(u) for u in self.iter_nodes()],
            "edges": [asdict(e) for e in self.iter_edges()]
        }

    def to_json_file(self, outfile: Union[str, Path]):
        with open(outfile, "wb") as f:
            f.write(orjson.dumps(self.to_json(), option=orjson.OPT_INDENT_2))

    @staticmethod
    def from_json(record: dict):
        sm = SemanticModel()
        for u in record['nodes']:
            if 'col_index' in u:
                sm.add_node(DataNode(**u))
            elif 'abs_uri' in u:
                sm.add_node(ClassNode(**u))
            else:
                lnode = LiteralNode(**u)
                lnode.datatype = LiteralNodeDataType(lnode.datatype)
                sm.add_node(lnode)
        for e in record['edges']:
            assert sm.has_node(e['source']) and sm.has_node(e['target'])
            sm.add_edge(Edge(**e))
        return sm

    @staticmethod
    def from_json_file(infile: Union[str, Path]):
        with open(infile, "rb") as f:
            record = orjson.loads(f.read())
            return SemanticModel.from_json(record)

    def draw(self, filename=None, no_display: bool=False, max_char_per_line: int=20):
        """
        Parameters
        ----------
        filename : str | none
            output to a file or display immediately (inline if this is jupyter lab)

        no_display: bool
            if the code is running inside Jupyter, if enable, it returns the object and manually display (default is
            automatically display)

        max_char_per_line: int
            wrap the text if it's too long

        Returns
        -------
        """
        if filename is None:
            fobj = tempfile.NamedTemporaryFile()
            filename = fobj.name
        else:
            fobj = None

        dot_g = pydot.Dot(graph_type='digraph')
        for uid, u in self.g.nodes.data("data"):
            uid = uid.replace(":", "_")
            if u.is_class_node:
                label = auto_wrap(u.label.replace(":", "\:"), max_char_per_line)
                dot_g.add_node(
                    pydot.Node(name=uid, label=label, shape="ellipse", style="filled",
                               color="white",
                               fillcolor="lightgray"))
            elif u.is_data_node:
                label = auto_wrap(f"C{u.col_index}\:" + u.label.replace(":", "\:"), max_char_per_line)
                dot_g.add_node(
                    pydot.Node(name=uid, label=label, shape="plaintext", style="filled",
                               fillcolor="gold"))
            else:
                label = auto_wrap(u.value, max_char_per_line)
                dot_g.add_node(
                    pydot.Node(name=uid, label=label, shape="plaintext", style="filled",
                               fillcolor="purple"))

        for u, v, e in self.g.edges.data("data"):
            u = u.replace(":", "_")
            v = v.replace(":", "_")
            label = auto_wrap(e.label.replace(":", "\:"), max_char_per_line)
            dot_g.add_edge(pydot.Edge(u, v, label=label, color="brown", fontcolor="black"))

        # graphviz from anaconda does not support jpeg so use png instead
        dot_g.write(filename, prog='dot', format='png')

        if fobj is not None:
            img = Image.open(filename)
            try:
                if no_display:
                    return img
            finally:
                fobj.close()

            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    display(img)
                else:
                    plt.imshow(img, interpolation='antialiased')
                    plt.show()
            except NameError:
                plt.imshow(img, interpolation='antialiased')
                plt.show()
            finally:
                fobj.close()

    def draw_difference(self, gold_sm: 'SemanticModel', filename=None, no_display: bool=False, max_char_per_line: int=20):
        """
        Colors:
        * green, red for edges/nodes in the pred_sm that does not appear in the gold_sm
        * lightgray for edges/nodes that are in the gold_sm but not in the pred_sm

        Parameters
        ----------
        gold_sm : SemanticModel
            the correct semantic model that we are going to compare to
        filename : str | none
            output to a file or display immediately (inline if this is jupyter lab)

        no_display : bool
            if the code is running inside Jupyter, if enable, it returns the object and manually display (default is
            automatically display)

        max_char_per_line: int
            wrap the text if it's too long

        Returns
        -------
        """
        if filename is None:
            fobj = tempfile.NamedTemporaryFile()
            filename = fobj.name
        else:
            fobj = None

        bijection = sm_metrics.precision_recall_f1(gold_sm, self)['_bijection']
        dot_g = pydot.Dot(graph_type='digraph')
        data_nodes = set()
        for uid, u in self.g.nodes.data("data"):
            if u.is_class_node:
                if bijection.prime2x[uid] is None:
                    # this is a wrong node
                    fillcolor = 'tomato'
                else:
                    fillcolor = 'mediumseagreen'

                label = auto_wrap(u.label.replace(":", "\:"), max_char_per_line)
                dot_g.add_node(
                    pydot.Node(name=uid.replace(":", "_"), label=label, shape="ellipse",
                               style="filled",
                               color="white",
                               fillcolor=fillcolor))
            else:
                data_nodes.add(u.col_index)
                dot_uid = f"C{u.col_index:02d}_{u.label}"
                label = auto_wrap(f"C{u.col_index}: " + u.label.replace(":", "\:"), max_char_per_line)
                dot_g.add_node(
                    pydot.Node(name=dot_uid, label=label, shape="plaintext", style="filled",
                               fillcolor="gold"))

        # node in gold_sm doesn't appear in the pred_sm
        for uid, u in gold_sm.g.nodes.data("data"):
            if u.is_class_node:
                if bijection.x2prime[uid] is None:
                    # class node in gold model need to give a different namespace (`gold:`) to avoid collision
                    dot_uid = ("gold:" + uid).replace(":", "_")
                    dot_g.add_node(
                        pydot.Node(name=dot_uid, label=auto_wrap(u.label.replace(":", "\:"), max_char_per_line),
                                   shape="ellipse",
                                   style="filled",
                                   color="white",
                                   fillcolor='lightgray'))
            else:
                if u.col_index not in data_nodes:
                    dot_uid = f"C{u.col_index:02d}_{u.label}"
                    dot_g.add_node(
                        pydot.Node(name=dot_uid, label=auto_wrap(f"C{u.col_index}: " + u.label.replace(":", "\:"), max_char_per_line),
                                   shape="plaintext",
                                   style="filled",
                                   fillcolor="lightgray"))

        # add edges in pred_sm
        x_triples = {
            (uid, e.label, vid if gold_sm.get_node(vid).is_class_node else (gold_sm.get_node(vid).col_index, gold_sm.get_node(vid).label))
            for uid, vid, e in gold_sm.g.edges.data("data")
        }
        x_prime_triples = set()
        for uid, vid, e in self.g.edges.data("data"):
            v = self.get_node(vid)
            x_prime_triple = (bijection.prime2x[uid], e.label, bijection.prime2x[vid] if v.is_class_node else (v.col_index, v.label))
            x_prime_triples.add(x_prime_triple)
            if x_prime_triple in x_triples:
                color = 'darkgreen'
            else:
                color = 'red'

            dot_u = uid.replace(":", "_")
            dot_v = vid.replace(":", "_") if v.is_class_node else f"C{v.col_index:02d}_{v.label}"
            dot_g.add_edge(pydot.Edge(dot_u, dot_v,
                                      label=auto_wrap(e.label.replace(":", "\:"), max_char_per_line),
                                      color=color, fontcolor="black"))

        # add edges in gold_sm that is not in pred_sm
        for x_triple in x_triples:
            if x_triple not in x_prime_triples:
                # class node in gold model need to give a different namespace (`gold:`) to avoid collision
                dot_u = "gold:" + x_triple[0] if bijection.x2prime[x_triple[0]] is None else bijection.x2prime[x_triple[0]]
                dot_u = dot_u.replace(":", "_")

                if isinstance(x_triple[2], tuple):
                    dot_v = f"C{x_triple[2][0]:02d}_{x_triple[2][1]}"
                else:
                    dot_v = "gold:" + x_triple[2] if bijection.x2prime[x_triple[2]] is None else bijection.x2prime[x_triple[2]]
                    dot_v = dot_v.replace(":", "_")

                dot_g.add_edge(pydot.Edge(dot_u, dot_v,
                                          label=auto_wrap(x_triple[1].replace(":", "\:"), max_char_per_line), color='gray', fontcolor="black"))

        # graphviz from anaconda does not support jpeg so use png instead
        dot_g.write(filename, prog='dot', format='png')

        if fobj is not None:
            img = Image.open(filename)
            try:
                if no_display:
                    return img
            finally:
                fobj.close()

            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    display(img)
                else:
                    plt.imshow(img, interpolation='antialiased')
                    plt.show()
            except NameError:
                plt.imshow(img, interpolation='antialiased')
                plt.show()
            finally:
                fobj.close()

if __name__ == '__main__':
    record = grams.misc.deserialize_json("/workspace/sm-dev/grams/examples/ground-truth/table_01/version.01.json")['semantic_models'][0]
    sm = SemanticModel.from_json(record)
    print(list(sm.iter_nodes()))
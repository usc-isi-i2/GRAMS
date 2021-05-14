from pathlib import Path

import networkx as nx
from typing import List, Union, Optional

import ujson

from grams.misc.graph.query import PropPath
from grams.misc.deser import serialize_json, deserialize_json


def dump_json(g: nx.MultiDiGraph, outfile: Optional[str], node_props: List[str]=None, edge_props: List[str]=None, allow_unknown_key: bool=False):
    if node_props is not None:
        node_props = [PropPath(p, allow_unknown_key) for p in node_props]
    if edge_props is not None:
        edge_props = [PropPath(p, allow_unknown_key) for p in edge_props]

    nodes = []
    for nid, ndata in g.nodes(data=True):
        if node_props is not None:
            tmp = {}
            for p in node_props:
                p.copy_to_dict(nid, ndata, tmp)
        else:
            tmp = ndata
        nodes.append((nid, tmp))

    edges = []
    for uid, vid, eid, edata in g.edges(data=True, keys=True):
        if edge_props is not None:
            tmp = {}
            for p in edge_props:
                p.copy_to_dict(eid, edata, tmp)
        else:
            tmp = edata
        edges.append(((uid, vid, eid), tmp))

    if outfile is not None:
        serialize_json({
            "nodes": nodes,
            "edges": edges
        }, outfile, indent=4)
    else:
        return ujson.dumps({"nodes": nodes, "edges": edges})


def load_json(infile_or_data: Union[Path, str, dict, list]):
    if isinstance(infile_or_data, (list, dict)):
        data = infile_or_data
    else:
        data = deserialize_json(infile_or_data)
    g = nx.MultiDiGraph()
    for nid, ndata in data['nodes']:
        g.add_node(nid)

    for i in range(len(data['edges'])):
        data['edges'][i][0] = tuple(data['edges'][i][0])

    for (uid, vid, eid), edata in data['edges']:
        g.add_edge(uid, vid, eid)

    nx.set_node_attributes(g, dict(data['nodes']))
    nx.set_edge_attributes(g, dict(data['edges']))
    return g
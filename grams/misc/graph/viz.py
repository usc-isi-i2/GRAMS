import glob
import os, networkx as nx
import shutil
from pathlib import Path
from typing import Callable, Optional, Set, Tuple, List

import orjson


def viz_graph(g: nx.MultiDiGraph,
              node_fn: Callable[[str, dict], dict] = None,
              edge_fn: Callable[[str, dict], dict] = None,
              outdir: Optional[str] = None,
              graph_id: str = "default"):
    nodes = []
    edges = []

    if node_fn is None:
        node_fn = lambda uid, udata: {"label": uid}
    if edge_fn is None:
        edge_fn = lambda eid, edata: {"label": eid}

    for uid, vid, e, edata in g.edges(data=True, keys=True):
        edge = {
            "source": uid,
            "target": vid,
        }
        edge.update(edge_fn(e, edata))
        edges.append(edge)

    for uid in g:
        u = g.nodes[uid]
        node = {
            "id": uid,
        }
        node.update(node_fn(uid, u))
        nodes.append(node)

    return viz_raw_graph(nodes, edges, outdir, graph_id)
    

def viz_raw_graph(nodes: List[dict], edges: List[dict],
                  outdir: Optional[str], graph_id: str = "default"):
    if outdir is None:
        # directly return the result for rendering rather than plot them
        return nodes, edges

    Path(outdir).mkdir(exist_ok=True, parents=True)
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.html")
    shutil.copy(html_file, os.path.join(outdir, "graph.html"))

    with open(os.path.join(outdir, f"graph.{graph_id}.js"), "wb") as f:
        f.write(b"window.data = ")
        f.write(orjson.dumps({
            "nodes": nodes,
            "edges": edges
        }))

    graphs = sorted([
        Path(file).name[len("graph."):-3]
        for file in glob.glob(os.path.join(outdir, "graph.*.js"))
    ])
    with open(os.path.join(outdir, f"metadata.js"), "wb") as f:
        f.write(b"const metadata = ")
        f.write(orjson.dumps({
            "graphs": graphs,
            "currentGraph": [i for i, graph in enumerate(graphs) if graph == graph_id][0]
        }))

from typing import List, Dict, Tuple, Callable, Any, Optional

from sm_unk.config import HOME_DIR
from grams.misc.graph import *


def node_fn(nid, ndata):
    colors = {
        "context": dict(fill="#C6E5FF", stroke="#5B8FF9"),
        "statement": dict(fill="#d9d9d9", stroke="#434343"),
        "qnode": dict(fill="#b7eb8f", stroke="#135200"),
        "column": dict(fill='#ffd666', stroke='#874d00')
    }

    if ndata['data']['column'] is not None:
        nodetype = 'column'
    elif ndata['data']['is_context']:
        nodetype = 'context'
    else:
        nodetype = 'statement' if nid.startswith("stmt") else "qnode"

    return {
        "label": ndata['data']['label'],
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
    }


g = load_json(HOME_DIR / "graph_viz/graph.json")

# B = QueryBuilder()
#
# c1 = B.n("city", B.c("data.column", "=", 1))
# c2 = B.n("county", B.c("data.column", "=", 2))
# # c1 \
# #     .to(B.n(), B.c("data.predicate", "=", "P131")) \
# #     .to(c2, B.c("data.predicate", "=", "P131")) \
# #     .to(B.n(), B.c("data.predicate", "=", "P36")) \
# #     .to(c1, B.c("data.predicate", "=", "P36"))
# c1.to(B.n()).to(c2).to(B.n()).to(c1)
#
# g = query_graph(g, B.get_query())
# dump_json(g, HOME_DIR / "graph_viz/graph.2.json")
viz_graph(g, node_fn, lambda eid, e: {"label": e['data']['predicate'] + f" #{e['data']['weight']}"}, HOME_DIR / "graph_viz/test", "origin")


def maximum_spanning_tree_solver(origin_sm_g: nx.MultiDiGraph):
    sm_g = nx.MultiDiGraph()
    for uid, vid, eid, edata in origin_sm_g.edges(data=True, keys=True):
        edata['weight'] = edata['data']['weight']
    resp = nx.algorithms.tree.branchings.maximum_spanning_arborescence(origin_sm_g, attr='weight',
                                                                       preserve_attrs=True)
    count = 0
    for s, t, e, edata in resp.edges(data=True, keys=True):
        edge = edata['data']
        if s not in sm_g:
            sm_g.add_node(s, **origin_sm_g.nodes[s])
        if t not in sm_g:
            sm_g.add_node(t, **origin_sm_g.nodes[t])
        sm_g.add_edge(s, t, key=edge['predicate'], data=edge)
        count += edge['weight']

    while True:
        remove_nodes = []
        for nid in sm_g:
            n = sm_g.nodes[nid]['data']
            if not n['column'] is not None and not n['is_context']:
                if sm_g.out_degree(nid) == 0:
                    remove_nodes.append(nid)

        for nid in remove_nodes:
            sm_g.remove_node(nid)

        if len(remove_nodes) == 0:
            break
    return sm_g

g2 = maximum_spanning_tree_solver(g)
viz_graph(g2, node_fn, lambda eid, e: {"label": e['data']['predicate'] + f" #{e['data']['weight']}"}, HOME_DIR / "graph_viz/test", "pruning")
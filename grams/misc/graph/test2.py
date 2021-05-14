from operator import itemgetter
from typing import List, Dict, Tuple, Callable, Any, Optional

from loguru import logger

from sm_unk.config import HOME_DIR
from sm_unk.misc.graph import *

g1 = load_json(HOME_DIR / "graph_viz/graph.json")
g2 = load_json(HOME_DIR / "graph_viz/graph.json")

print(len(g1.nodes), len(g2.nodes))
print(len(g1.edges), len(g2.edges))

for nid, ndata in g1.nodes(data=True):
    assert nid in g2.nodes
for uid, vid, eid, edata in g1.edges(data=True, keys=True):
    assert (uid, vid, eid) in g2.edges, (uid, vid, eid)
    assert edata['data']['weight'] == g2.edges[(uid, vid, eid)]['data']['weight']


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
    print(count)


# @logger.catch
def solver2(g: nx.MultiDiGraph):
    new_g = {}
    roots = []
    for uid in g:
        new_g[uid] = {}
        if not uid.startswith("stmt:"):
            roots.append(uid)

        for vid in g[uid]:
            max_edge = max([(eid, edata['data']['weight']) for eid, edata in g[uid][vid].items()], key=itemgetter(1))
            new_g[uid][vid] = {
                "eid": max_edge[0],
                "weight": max_edge[1]
            }

    from sm_unk.misc.graph.edmonds import mst

    input_g = {
        uid: {vid: 1 / edata['weight'] for vid, edata in u.items()}
        for uid, u in new_g.items()
    }

    sols = []
    sol_weights = float('inf')
    for root in roots:
        resp = mst(root, input_g)
        resp_weight = sum(weight for uid, u in resp.items() for vid, weight in u.items())
        if resp_weight < sol_weights:
            if abs(resp_weight - sol_weights) > 1e-7:
                # not number error
                sols = []
                sol_weights = resp_weight
            sols.append(resp)

    for sol in sols:
        resp_g = nx.MultiDiGraph()
        for uid, u in sol.items():
            resp_g.add_node(uid, **g.nodes[uid])
            for vid, weight in u.items():
                if vid not in resp_g:
                    resp_g.add_node(vid, **g.nodes[vid])

                assert new_g[uid][vid]['weight'] == 1.0 / weight
                resp_g.add_edge(uid, vid, key=new_g[uid][vid]['eid'], **g.edges[uid, vid, new_g[uid][vid]['eid']])

        weight = sum(edata['data']['weight'] for uid, vid, eid, edata in resp_g.edges(data=True, keys=True))
        print(weight)
        return resp_g
    print(sol_weights)
    print(sols)


maximum_spanning_tree_solver(g1)
# maximum_spanning_tree_solver(g2)
g1 = solver2(g1)
# g2 = solver2(g2)
viz_graph(g1, lambda nid, n: {"label": n['data']['label']}, lambda eid, e: {"label": e['data']['predicate'] + f" #{e['data']['weight']}"}, HOME_DIR / "graph_viz", "g1")
# viz_graph(g2, lambda nid, n: {"label": n['data']['label']}, lambda eid, e: {"label": e['data']['predicate'] + f" #{e['data']['weight']}"}, HOME_DIR / "graph_viz", "g2")
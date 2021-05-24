from collections import defaultdict
from operator import itemgetter
from grams.algorithm.semantic_graph import SGEdge, SemanticGraphConstructor
from typing import *
import sm.misc as M
import networkx as nx


class SemTab2020PostProcessing:
    instance = None

    def __init__(self):
        from sm_unk.config import HOME_DIR
        dataset_dir = HOME_DIR / "wikitable2wikidata/semtab2020"
        self.CPA_targets = defaultdict(list)
        self.CTA_targets = defaultdict(list)
        for r in M.deserialize_csv(dataset_dir / "CPA_Round4_targets.csv"):
            self.CPA_targets[r[0]].append((int(r[1]), int(r[2])))
        for r in M.deserialize_csv(dataset_dir / "CTA_Round4_targets.csv"):
            self.CTA_targets[r[0]].append(int(r[1]))
    
    @staticmethod
    def get_instance():
        if SemTab2020PostProcessing.instance is None:
            SemTab2020PostProcessing.instance = SemTab2020PostProcessing()
        return SemTab2020PostProcessing.instance

    def solve_post_process(self, table, sg, dg, pred_with_probs: Dict[Tuple[str, str, str], float], threshold):
        # get the steiner tree first
        ci2id = {
            unode.column: uid
            for uid, unode in sg.nodes(data='data')
            if unode.is_column
        }
        selected_edges = set()
        for sci, tci in self.CPA_targets[table.id]:
            uid = ci2id[sci]
            vid = ci2id[tci]

            if not sg.has_node(uid) or not sg.has_node(vid):
                continue
            # among those paths, select the one with best prob
            paths = list(nx.all_simple_edge_paths(sg, uid, vid, cutoff=2))
            if len(paths) == 0:
                continue
            path_probs = []
            for path in paths:
                path_prob = 0
                for uid, vid, eid in path:
                    edge: SGEdge = sg.edges[uid, vid, eid]['data']
                    path_prob += pred_with_probs[uid, vid, eid]
                path_probs.append(path_prob)
            
            select_path, select_path_prob = max(zip(paths, path_probs), key=itemgetter(1))
            selected_edges = selected_edges.union(select_path)
        
        new_sg = SemanticGraphConstructor.get_sg_subgraph(sg, selected_edges)
        return new_sg
        
    def remove_dangling_statement(self, sg: nx.MultiDiGraph):
        ids = set()
        for sid, s in list(sg.nodes(data='data')):
            if s.is_statement and (sg.in_degree(sid) == 0 or sg.out_degree(sid) == 0):
                ids.add(sid)
        for id in ids:
            sg.remove_node(id)
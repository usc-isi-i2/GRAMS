from dataclasses import dataclass
import os
from operator import itemgetter
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any, Optional, Set, Iterable, Union

import networkx as nx
import requests
from omegaconf import OmegaConf
from rdflib import RDFS

import sm.misc as M
import sm.outputs as O
import grams.inputs as I

from grams.algorithm.data_graph import build_data_graph, BuildDGOption
from grams.algorithm.kg_index import TraversalOption, KGObjectIndex
from grams.algorithm.semantic_graph import SemanticGraphConstructor, SemanticGraphConstructorArgs, viz_sg
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG, ROOT_DIR

from grams.algorithm.psl_solver import PSLSteinerTreeSolver

from kgdata.wikidata.models import QNode, WDProperty, WDClass, WDQuantityPropertyStats
from kgdata.wikidata.db import get_qnode_db, get_wdprop_db, get_wdclass_db, query_wikidata_entities


@dataclass
class Annotation:
    sm: O.SemanticModel
    dg: nx.MultiDiGraph
    sg: nx.MultiDiGraph
    pred_sg: nx.MultiDiGraph
    pred_cta: Dict[int, Dict[str, float]]


class GRAMS:
    def __init__(self, data_dir: str, cfg=None, proxy: bool=True):
        self.timer = M.Timer()
        self.cfg = cfg if cfg is None else DEFAULT_CONFIG
        self.is_proxy_db = proxy

        with self.timer.watch('init grams db'):
            read_only = not proxy
            self.qnodes = get_qnode_db(os.path.join(data_dir, "qnodes.db"), compression=True, read_only=read_only, proxy=proxy, is_singleton=True)
            self.wdclasses = get_wdclass_db(os.path.join(data_dir, "wdclasses.db"), compression=False, read_only=read_only, proxy=proxy, is_singleton=True)
            self.wdprops = get_wdprop_db(os.path.join(data_dir, "wdprops.db"), compression=False, read_only=read_only, proxy=proxy, is_singleton=True)
            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(os.path.join(data_dir, "quantity_prop_stats"))

        self.build_dg_option = getattr(BuildDGOption, cfg.data_graph.options[0])
        for op in cfg.data_graph.options[1:]:
            self.build_dg_option = self.build_dg_option | getattr(BuildDGOption, op)

    def annotate(self, table: I.LinkedTable, verbose: bool=False):
        qnode_ids = {link.qnode_id
                     for rlinks in table.links for links in rlinks
                     for link in links if link.qnode_id is not None}
        if table.context.page_qnode is not None:
            qnode_ids.add(table.context.page_qnode)

        with self.timer.watch('retrieving qnodes'):
            qnodes = self.get_entities(qnode_ids, n_hop=2, verbose=verbose)
        wdclasses = self.wdclasses.cache_dict()
        wdprops = self.wdprops.cache_dict()

        with self.timer.watch('build kg object index'):
            kg_object_index = KGObjectIndex.from_qnodes(
                qnode_ids, qnodes, wdprops,
                n_hop=self.cfg.data_graph.max_n_hop, traversal_option=TraversalOption.TransitiveOnly)

        with self.timer.watch("build dg & sg"):
            dg = build_data_graph(table, qnodes, wdprops,
                                  kg_object_index, max_n_hop=self.cfg.data_graph.max_n_hop,
                                  options=self.build_dg_option)
            constructor = SemanticGraphConstructor([
                SemanticGraphConstructor.init_sg,
            ], qnodes, wdclasses, wdprops)
            sg = constructor.run(table, dg, debug=False).sg
        
        with self.timer.watch('run inference'):
            psl_solver = PSLSteinerTreeSolver(
                qnodes, wdclasses, wdprops, self.wd_numprop_stats,
                disable_rules=set(self.cfg.psl.disable_rules), sim_fn=None,
                # cache_dir=outdir / f"{override_psl_cachedir}cache/psl",
                postprocessing_method=self.cfg.psl.postprocessing, enable_logging=self.cfg.psl.enable_logging)
            pred_sg, pred_cta = psl_solver.run(dict(table=table, semanticgraph=sg, datagraph=dg))
            pred_cta = {int(ci.replace("column-", "")): classes for ci, classes in pred_cta.items()}
            cta = {ci: max(classes.items(), key=itemgetter(1))[0] for ci, classes in pred_cta.items()}

        sm = self.create_sm_from_cta_cpa(table, pred_sg, cta, qnodes, wdclasses, wdprops)
        return Annotation(sm=sm, dg=dg, sg=sg, pred_sg=pred_sg, pred_cta=cta)

    def create_sm_from_cta_cpa(self, table: I.LinkedTable, sg: nx.MultiDiGraph, cta: Dict[int, str], qnodes: Dict[str, QNode], wdclasses: Dict[str, WDClass], wdprops: Dict[str, WDProperty]):
        sm = O.SemanticModel()
        sm_helper = WikidataSemanticModelHelper(qnodes, wdclasses, wdprops)
        # create class nodes first
        classcount = {}
        classmap = {}
        for cid, qnode_id in cta.items():
            dnode = O.DataNode(id=f'col-{cid}', col_index=cid, label=table.table.columns[cid].name)

            # somehow, they may end-up predict multiple classes, we need to select one
            if qnode_id.find(" ") != -1:
                qnode_id = qnode_id.split(" ")[0]
            curl = sm_helper.get_qnode_uri(qnode_id)
            cnode_id = f"{curl}:{classcount.get(qnode_id, 0)}"
            classcount[qnode_id] = classcount.get(qnode_id, 0) + 1

            try:
                cnode_label = sm_helper.get_qnode_label(curl)
            except KeyError:
                cnode_label = f"wd:{qnode_id}"
            cnode = O.ClassNode(id=cnode_id, abs_uri=curl, rel_uri=f"wd:{qnode_id}", readable_label=cnode_label)
            classmap[dnode.id] = cnode.id

            sm.add_node(dnode)
            sm.add_node(cnode)
            sm.add_edge(O.Edge(source=cnode.id, target=dnode.id,
                               abs_uri=str(RDFS.label), rel_uri="rdfs:label"))

        for uid, vid, edge in sg.edges(data='data'):
            unode = sg.nodes[uid]['data']
            vnode = sg.nodes[vid]['data']

            if unode.is_column:
                suid = f"col-{unode.column}"
            else:
                suid = unode.id
            if vnode.is_column:
                svid = f"col-{vnode.column}"
            else:
                svid = vnode.id

            if sm.has_node(suid):
                if suid in classmap:
                    suid = classmap[suid]
                source = sm.get_node(suid)
            elif unode.is_column:
                # create a data node
                source = O.DataNode(id=suid, col_index=unode.column, label=table.table.get_column_by_index(unode.column).name)
                sm.add_node(source)
            else:
                # create a statement node
                source = O.ClassNode(id=suid, abs_uri='http://wikiba.se/ontology#Statement', rel_uri='wikibase:Statement')
                sm.add_node(source)
            if sm.has_node(svid):
                if svid in classmap:
                    svid = classmap[svid]
                target = sm.get_node(svid)
            elif vnode.is_column:
                target = O.DataNode(id=svid, col_index=vnode.column, label=table.table.get_column_by_index(vnode.column).name)
                sm.add_node(target)
            else:
                target = O.ClassNode(id=svid, abs_uri='http://wikiba.se/ontology#Statement',
                                     rel_uri='wikibase:Statement')
                sm.add_node(target)

            prop_uri = sm_helper.get_prop_uri(edge.predicate)
            sm.add_edge(O.Edge(source=source.id, target=target.id, abs_uri=prop_uri, rel_uri=f"p:{edge.predicate}",
                               readable_label=sm_helper.get_pnode_label(prop_uri)))

        return sm

    def get_entities(self, qnode_ids: Set[str], n_hop: int = 1, verbose: bool = False) -> Dict[str, QNode]:
        assert n_hop <= 2
        batch_size = 30
        qnodes: Dict[str, QNode] = {}
        for qnode_id in qnode_ids:
            qnode = self.qnodes.get(qnode_id, None)
            if qnode is not None:
                qnodes[qnode_id] = qnode
        
        if self.is_proxy_db:
            qnode_ids = [qnode_id for qnode_id in qnode_ids if qnode_id not in qnodes]
            if len(qnode_ids) > 0:
                resp = M.parallel_map(
                    query_wikidata_entities,
                    [qnode_ids[i:i+batch_size] for i in range(0, len(qnode_ids), batch_size)],
                    show_progress=verbose,
                    progress_desc=f'query wikidata for get entities in hop: {n_hop}',
                    is_parallel=True)
                for odict in resp:
                    for k, v in odict.items():
                        qnodes[k] = v
                        self.qnodes[k] = v

        if n_hop > 1:
            next_qnode_ids = set()
            for qnode in qnodes.values():
                for p, stmts in qnode.props.items():
                    for stmt in stmts:
                        if stmt.value.is_qnode():
                            next_qnode_ids.add(stmt.value.as_qnode_id())
                        for qvals in stmt.qualifiers.values():
                            next_qnode_ids = next_qnode_ids.union(qval.as_qnode_id() for qval in qvals if qval.is_qnode())
            next_qnode_ids = list(next_qnode_ids.difference(qnodes.keys()))
            for qnode_id in next_qnode_ids:
                qnode = self.qnodes.get(qnode_id, None)
                if qnode is not None:
                    qnodes[qnode_id] = qnode
            
            if self.is_proxy_db:
                next_qnode_ids = [qnode_id for qnode_id in next_qnode_ids if qnode_id not in qnodes]
                if len(next_qnode_ids) > 0:
                    resp = M.parallel_map(
                        query_wikidata_entities,
                        [next_qnode_ids[i:i+batch_size] for i in range(0, len(next_qnode_ids), batch_size)],
                        show_progress=verbose,
                        progress_desc=f'query wikidata for get entities in hop: {n_hop}',
                        is_parallel=True)
                    for odict in resp:
                        for k, v in odict.items():
                            qnodes[k] = v
                            self.qnodes[k] = v
        return qnodes


if __name__ == '__main__':
    cfg = OmegaConf.load(ROOT_DIR / "grams.yaml")

    # tbl = I.LinkedTable.from_csv_file(ROOT_DIR / "examples/novartis/tables/table_03.csv")
    # data = M.deserialize_json(ROOT_DIR / "examples/novartis/ground-truth/table_03/version.01.json")
    # sm = O.SemanticModel.from_dict(data['semantic_models'][0])
    # sm.draw()
    # exit(0)
    tbl = I.LinkedTable.from_dict(M.deserialize_json(ROOT_DIR / "examples/misc/tables/president_of_the_national_council_austria_10_0f1733248af445ee5a7d360a648bf9b1.json"))
    # tbl = I.W2WTable.from_csv_file(ROOT_DIR / "examples/t2dv2/tables/29414811_2_4773219892816395776.csv")
    # for ri in range(tbl.size()):
    #     for ci in range(len(tbl.table.columns)):
    #         tbl.links[ri][ci] = []
    # cea = M.deserialize_csv(ROOT_DIR / "examples/t2dv2/tables/29414811_2_4773219892816395776.candidates.tsv", delimiter="\t")
    # gold_cea = {}
    # for r in cea:
    #     ri, ci = int(r[0]), int(r[1])
    #     gold_cea[ri, ci] = r[2]
    #     tbl.links[ri][ci] = [
    #         I.Link(0, len(tbl.table.columns[ci].values[ri]), "", x.split(":")[0])
    #         for x in r[3:]
    #     ]
    grams = GRAMS(ROOT_DIR / "data", cfg)
    annotation = grams.annotate(tbl)
    annotation.sm.draw()
    print(annotation.sm.to_dict())
    grams.timer.report()

    # # %%
    # # ent column
    # from grams.algorithm.sm_wikidata import WDOnt
    # wdont = WDOnt(grams.qnodes, grams.wdclasses, grams.wdprops)
    # from collections import defaultdict
    # from grams.algorithm.data_graph import *
    # # %%
    # ci, classid = list(annotation.pred_cta.items())[0]
    # pred_ents = []
    # props = {'P136', 'P178', 'P400', 'P577'}
    # final_prediction = []
    # original_prediction = []
    # for ri in range(tbl.size()):
    #     can_ents = []
    #     for link in tbl.links[ri][ci]:
    #         if link.qnode_id is not None:
    #             qnode = grams.qnodes[link.qnode_id]
    #             if any(stmt.value.as_qnode_id() == classid for stmt in qnode.props.get("P31", [])):
    #                 can_ents.append(qnode.id)
    #
    #     qnode2score = defaultdict(set)
    #     for stmt, edges in annotation.dg[f"{ri}-{ci}"].items():
    #         assert len(edges) == 1
    #         edgeid = list(edges.keys())[0]
    #         if edgeid not in props:
    #             continue
    #         stmt = annotation.dg.nodes[stmt]['data']
    #         for target_flow in stmt.forward_flow[EdgeFlowSource(source_id=f"{ri}-{ci}", edge_id=edgeid)]:
    #             target = annotation.dg.nodes[target_flow.target_id]['data']
    #             if target.is_cell and target.column != ci:
    #                 qnode2score[stmt.qnode_id].add(target.column)
    #
    #     best_can, best_can_score = None, -1
    #     for can_ent in can_ents:
    #         can_ent_score = len(qnode2score.get(can_ent, []))
    #         if can_ent_score > best_can_score:
    #             best_can = can_ent
    #             best_can_score = can_ent_score
    #
    #     # print(ri, gold_cea[ri, ci], best_can == gold_cea[ri, ci], can_ents[0] == gold_cea[ri, ci], can_ents)
    #     print(ri, wdont.get_qnode_label(gold_cea[ri, ci]), best_can == gold_cea[ri, ci], can_ents[0] == gold_cea[ri, ci], [wdont.get_qnode_label(x) for x in can_ents])
    #     final_prediction.append((gold_cea[ri, ci], best_can))
    #     original_prediction.append((gold_cea[ri, ci], tbl.links[ri][ci][0].qnode_id))
    # #%%
    # print("TOP1", sum(x[0] == x[1] for x in final_prediction) / len(final_prediction))
    # print("TOP1 origin", sum(x[0] == x[1] for x in original_prediction) / len(final_prediction))
    #
    # count = 0
    # for ri in range(tbl.size()):
    #     if any(link.qnode_id == gold_cea[ri, ci] for link in tbl.links[ri][ci][:5]):
    #         count += 1
    # print("TOP5", count / len(final_prediction))
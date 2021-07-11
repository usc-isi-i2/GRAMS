import copy
import shutil
from operator import itemgetter
from pathlib import Path

from typing import Tuple, List

import networkx as nx
from redis import Redis

"""
Setup the input & output for testing the Steiner Tree solver
"""


class Env:
    def __init__(self, cfg=None, dataset_dir=None, dirname="default", n_tables: int=None,
                 load_qnode_test: bool=True, auto_path_increment: bool = False):
        if dataset_dir is None:
            dataset_dir = HOME_DIR / "wikitable2wikidata/250tables"
        self.dataset_dir = dataset_dir
        self.cfg = cfg
        # whether we enforce inverse constraint apply on object column.. only happen on the semtab2020 dataset
        self.strict = not self.dataset_dir.name.startswith("semtab2020")
        self.force_inversion = self.dataset_dir.name.startswith("semtab2020")

        self.gold_models = get_input_data(self.dataset_dir, self.dataset_dir.name, only_curated=True, n_tables=n_tables)
        self.qnodes = get_qnodes(self.dataset_dir, test=load_qnode_test, n_hop=self.cfg.data_graph.max_n_hop + 1)
        self.wdclasses = WDClass.from_file(self.dataset_dir / "ontology", load_parent_closure=True)
        self.wdprops = WDProperty.from_file(load_parent_closure=True)
        self.wd_numprop_stats = WDQuantityPropertyStats.from_dir()

        kg_index_file = dataset_dir / "kg_index" / "object_index.2hop_transitive.pkl.gz"
        if kg_index_file.exists():
            self.kg_object_index = KGObjectIndex.deserialize(kg_index_file, verbose=True)
        else:
            index_qnode_ids = list(get_qnodes(dataset_dir, n_hop=1, no_wdclass=True).keys())
            self.kg_object_index = KGObjectIndex.from_qnodes(index_qnode_ids, self.qnodes, self.wdprops,
                                                        n_hop=2, traversal_option=TraversalOption.TransitiveOnly,
                                                        verbose=True)
            self.kg_object_index.serialize(kg_index_file)

        self.evaluator = Evaluation(self.qnodes, self.wdclasses, self.wdprops)
        self.workdir = self.dataset_dir / "dev" / dirname
        if auto_path_increment:
            self.workdir = Path(M.get_incremental_path(self.workdir))
        self.workdir.mkdir(exist_ok=True, parents=True)

        cache_fn = lambda fn: M.redis_cache_func(REDIS_CACHE_URL, namespace=f"{self.dataset_dir.name}.{fn.__name__}", instance_method=False)(fn)

        # self.get_semantic_models = cache_fn(self.get_semantic_models)
        self.get_best_semantic_model = cache_fn(self.get_best_semantic_model)
        self.get_data_graph = cache_fn(self.get_data_graph)
        self.get_init_sg = cache_fn(self.get_init_sg)
        self.get_oracle_sgs = cache_fn(self.get_oracle_sgs)
        self.cache_fn = cache_fn

    def get_semantic_models(self, table_index: int):
        sms, table = self.gold_models[table_index]
        return [equiv_sm for sm in sms for equiv_sm in self.evaluator.wd_smhelper.gen_equivalent_sm(sm, self.strict, self.force_inversion)]

    def get_best_semantic_model(self, table_index: int):
        """Get the best semantic model that yield the maximum recall with the built data graph"""
        resp = self.get_init_sg(table_index)
        sms = self.get_semantic_models(table_index)

        perf = [
            (i, self.evaluator.oracle_pruning_cpa(resp.table, sm, copy.deepcopy(resp.sg)))
            for i, sm in enumerate(sms)
        ]
        i, perf = max(perf, key=lambda x: x[1]['recall'])
        return sms[i]

    def get_data_graph(self, table_index: int):
        sms, table = self.gold_models[table_index]

        build_data_graph_option = getattr(BuildDGOption, self.cfg.data_graph.options[0])
        for op in self.cfg.data_graph.options[1:]:
            build_data_graph_option = build_data_graph_option | getattr(BuildDGOption, op)

        dg = build_data_graph(table, self.qnodes, self.wdprops, self.kg_object_index, max_n_hop=self.cfg.data_graph.max_n_hop, options=build_data_graph_option)
        return dg

    def get_init_sg(self, table_index: int):
        sms, table = self.gold_models[table_index]
        dg = self.get_data_graph(table_index)
        constructor = SemanticGraphConstructor([
            SemanticGraphConstructor.init_sg,
            SemanticGraphConstructor.calculate_link_frequency,
            SemanticGraphConstructor.prune_sg_redundant_entity,
        ], self.qnodes, self.wdclasses, self.wdprops)
        resp = constructor.run(table, dg, debug=False)
        return resp

    def get_oracle_sgs(self, table_index: int):
        resp = self.get_init_sg(table_index)
        sms = self.get_semantic_models(table_index)
        perf = [
            self.evaluator.oracle_pruning_cpa(resp.table, sm, copy.deepcopy(resp.sg))
            for sm in sms
        ]
        best_recall = max([x['recall'] for x in perf])
        perf = [p for p in perf if p['recall'] == best_recall]

        oracle_sgs = []
        for p in perf:
            cpa_pred_sm: O.SemanticModel = p['cpa_pred_sm']

            # TODO: with new update from oracle_pruning_cpa, it's now also returned the semantic graph so we shouldn't to this. but needs to test before removing this code
            # have to exploit the fact that oracle_pruning_cpa function create a semantic model that reuse
            # the id in the original SG
            edges = [
                # convert uri back to predicate id to get the eid
                (edge.source, edge.target, self.evaluator.wd_smhelper.get_prop_id(edge.abs_uri))
                for edge in cpa_pred_sm.iter_edges()
            ]
            oracle_sg = SemanticGraphConstructor.get_sg_subgraph(resp.sg, edges)
            oracle_sgs.append(oracle_sg)
        return oracle_sgs

    def eval_cpa(self, table_index: int, st: nx.MultiDiGraph):
        """Evaluate the CPA performance"""
        table = self.gold_models[table_index][1]
        equiv_sms = self.get_semantic_models(table_index)
        cpa_sm, cpa_resp = max([
            (equiv_sm, self.evaluator.cpa(table, equiv_sm, st, False))
            for equiv_sm in equiv_sms
        ], key=lambda x: x[1]['f1'] if x[1] is not None else 0.0)
        if cpa_resp is None:
            result = (-1, -1, -1)
        else:
            result = (cpa_resp['precision'], cpa_resp['recall'], cpa_resp['f1'])
        return result, cpa_sm

    def eval_steiner_tree(self, table_index: int, st: nx.MultiDiGraph, oracle_sgs: List[nx.MultiDiGraph]=None):
        """Evaluate the Steiner Tree algorithm. Note this function is completely different
        with evaluate the correctness of the final semantic model (simply because it doesn't include
        all possible relationships).
        """
        if oracle_sgs is None:
            oracle_sgs = self.get_oracle_sgs(table_index)
        oracle_sms = [self.evaluator.convert_sg_for_cpa(sg) for sg in oracle_sgs]
        predict_sm = self.evaluator.convert_sg_for_cpa(st)
        best_resp = (-1, -1, -1)
        best_oracle_sg = None

        for (oracle_sm, oracle_sg) in zip(oracle_sms, oracle_sgs):
            resp = sm_metrics.precision_recall_f1(oracle_sm, predict_sm, self.evaluator.prop_score_fn)
            precision, recall, f1 = [resp[k] for k in ['precision', 'recall', 'f1']]

            if f1 > best_resp[-1]:
                best_resp = (precision, recall, f1)
                best_oracle_sg = oracle_sg

        return best_resp, best_oracle_sg

    def clear_viz_dir(self):
        if (self.workdir / "viz").exists():
            shutil.rmtree(self.workdir / "viz")
        (self.workdir / "viz").mkdir(exist_ok=True)

    def viz_dg(self, dg, id):
        viz_dg(dg, self.qnodes, self.wdprops, self.workdir / "viz", id)

    def viz_sg(self, sg, id):
        viz_sg(sg, self.qnodes, self.wdclasses, self.wdprops, self.workdir / "viz", id)

    def viz_sm(self, sm, id):
        self.evaluator.wd_smhelper.viz_sm(sm, self.workdir / "viz", id)

    def sg_to_sm(self, table_index, sg):
        """Convert semantic graph into semantic model for visualization purpose"""
        sm = O.SemanticModel()
        for uid, udata in sg.nodes(data=True):
            u: SGNode = udata['data']
            if u.is_statement:
                sm.add_node(
                    O.ClassNode(u.id, SemanticGraphConstructor.STATEMENT_URI,
                                SemanticGraphConstructor.STATEMENT_REL_URI, False, "Statement"))
            elif u.is_value:
                if u.is_entity_value:
                    sm.add_node(O.LiteralNode(u.id, self.evaluator.wd_smhelper.get_qnode_uri(u.qnode_id), readable_label=u.label))
                elif u.is_literal_value:
                    sm.add_node(O.LiteralNode(u.id, u.label, u.label))
            else:
                assert u.is_column
                sm.add_node(O.DataNode(u.id, u.column, self.gold_models[table_index][1].table.get_column_by_index(u.column)))

        for uid, vid, eid, edata in sg.edges(data=True, keys=True):
            e: SGEdge = edata['data']
            sm.add_edge(
                O.Edge(uid, vid, self.evaluator.wd_smhelper.get_prop_uri(e.predicate), f"p:{e.predicate}", False))

        return sm
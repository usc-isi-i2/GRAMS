from gc import disable
import os
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Set, Tuple, Union
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.literal_matchers import TextParserConfigs, LiteralMatch
from grams.algorithm.postprocessing.simple_path import PostProcessingSimplePath
from grams.algorithm.postprocessing.steiner_tree import PostProcessingSteinerTree
from hugedict.parallel.parallel import Parallel
from kgdata.wikidata.models.qnode import QNodeLabel
from loguru import logger
import rltk
from tqdm import tqdm
import networkx as nx
import sm.misc as M
import sm.outputs as O
from kgdata.wikidata.db import (
    WDProxyDB,
    get_qnode_db,
    get_qnode_label_db,
    get_wdclass_db,
    get_wdprop_db,
    query_wikidata_entities,
)
from kgdata.wikidata.models import QNode, WDClass, WDProperty, WDQuantityPropertyStats
from rdflib import RDFS

import grams.inputs as I
from grams.algorithm.data_graph import DGConfigs, DGFactory
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption
from grams.algorithm.psl_solver import PSLInference

# from grams.algorithm.semantic_graph import SemanticGraphConstructor
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG


@dataclass
class Annotation:
    sm: O.SemanticModel
    # data graph
    dg: DGGraph
    # candidate graph
    cg: CGGraph
    # probabilities of each edge in sg (uid, vid, eid)
    cg_edge_probs: Dict[Tuple[str, str, str], float]
    # probabilities of types of each column: column index -> type -> probability
    cta_probs: Dict[int, Dict[str, float]]
    # predicted candidate graph where incorrect relations are removed by threshold & post-processing algorithm
    pred_cpa: CGGraph
    # predicted column types
    pred_cta: Dict[int, str]


class GRAMS:
    """Implementation of GRAMS. The main method is `annotate`"""

    def __init__(
        self,
        data_dir: Union[Path, str],
        cfg=None,
        proxy: bool = True,
    ):
        self.timer = M.Timer()
        self.cfg = cfg if cfg is not None else DEFAULT_CONFIG

        with self.timer.watch("init grams db"):
            read_only = not proxy
            self.qnodes = get_qnode_db(
                os.path.join(data_dir, "qnodes.db"),
                read_only=read_only,
                proxy=proxy,
            )
            if proxy:
                assert isinstance(self.qnodes, WDProxyDB)
            if os.path.exists(os.path.join(data_dir, "qnode_labels.db")):
                self.qnode_labels = get_qnode_label_db(
                    os.path.join(data_dir, "qnode_labels.db"),
                )
            else:
                self.qnode_labels: MutableMapping[str, QNodeLabel] = {}
            self.wdclasses = get_wdclass_db(
                os.path.join(data_dir, "wdclasses.db"),
                read_only=read_only,
                proxy=proxy,
            )
            self.wdprops = get_wdprop_db(
                os.path.join(data_dir, "wdprops.db"),
                read_only=read_only,
                proxy=proxy,
            )
            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(
                os.path.join(data_dir, "quantity_prop_stats")
            )

        self.update_config()

    def update_config(self):
        """Update the current configuration of the algorithm based on the current configuration stored in this object"""
        for name, value in self.cfg.data_graph.configs.items():
            if not hasattr(DGConfigs, name):
                raise Exception(f"Invalid configuration for data_graph: {name}")
            setattr(DGConfigs, name, value)

        for name, value in self.cfg.literal_matcher.text_parser.items():
            if not hasattr(TextParserConfigs, name):
                raise Exception(
                    f"Invalid configuration for literal_matcher.text_parser: {name}"
                )
            setattr(TextParserConfigs, name, value)

        for name, value in self.cfg.literal_matcher.matchers.items():
            if not hasattr(LiteralMatch, name):
                raise Exception(
                    f"Invalid configuration for literal_matcher.matchers: {name}"
                )
            setattr(LiteralMatch, name, value)

    def annotate(self, table: I.LinkedTable, verbose: bool = False) -> Annotation:
        """Annotate a linked table"""
        qnode_ids = {
            link.entity_id
            for rlinks in table.links
            for links in rlinks
            for link in links
            if link.entity_id is not None
        }
        qnode_ids.update(
            (
                candidate.entity_id
                for rlinks in table.links
                for links in rlinks
                for link in links
                for candidate in link.candidates
            )
        )
        if table.context.page_entity_id is not None:
            qnode_ids.add(table.context.page_entity_id)

        with self.timer.watch("retrieving qnodes"):
            qnodes = self.get_entities(
                qnode_ids, n_hop=self.cfg.data_graph.max_n_hop, verbose=verbose
            )
        wdclasses = self.wdclasses.cache_dict()
        wdprops = self.wdprops.cache_dict()

        nonexistent_qnode_ids = qnode_ids.difference(qnodes.keys())
        if len(nonexistent_qnode_ids) > 0:
            logger.info("Removing non-existent qnodes: {}", list(nonexistent_qnode_ids))
            table.remove_nonexistent_entities(nonexistent_qnode_ids)

        with self.timer.watch("retrieving qnodes label"):
            qnode_labels = self.get_entity_labels(qnodes, verbose=verbose)

        with self.timer.watch("build kg object index"):
            kg_object_index = KGObjectIndex.from_qnodes(
                list(qnode_ids.intersection(qnodes.keys())),
                qnodes,
                wdprops,
                n_hop=self.cfg.data_graph.max_n_hop,
                traversal_option=TraversalOption.TransitiveOnly,
            )

        with self.timer.watch("build dg & sg"):
            dg_factory = DGFactory(qnodes, wdprops)
            dg = dg_factory.create_dg(
                table, kg_object_index, max_n_hop=self.cfg.data_graph.max_n_hop
            )
            cg_factory = CGFactory(qnodes, qnode_labels, wdclasses, wdprops)
            cg = cg_factory.create_cg(table, dg)

        with self.timer.watch("run inference"):

            # def sim_fn(x, y):
            #     return 1 - rltk.levenshtein_distance(x.lower(), y.lower()) / max(
            #         len(x), len(y)
            #     )
            sim_fn = None

            psl_solver = PSLInference(
                qnodes,
                wdclasses,
                wdprops,
                self.wd_numprop_stats,
                disable_rules=set(self.cfg.psl.disable_rules),
                sim_fn=sim_fn,
                enable_logging=self.cfg.psl.enable_logging,
            )
            edge_probs, cta_probs = psl_solver.run(table, dg, cg)
            if self.cfg.psl.postprocessing == "select_simplepath":
                pp = PostProcessingSimplePath(
                    table, cg, dg, edge_probs, self.cfg.psl.threshold
                )
            elif self.cfg.psl.postprocessing == "steiner_tree":
                pp = PostProcessingSteinerTree(
                    table, cg, dg, edge_probs, self.cfg.psl.threshold
                )
            else:
                raise NotImplementedError(self.cfg.psl.postprocessing)

            pred_cpa = pp.get_result()
            pred_cta = {
                ci: max(classes.items(), key=itemgetter(1))[0]
                for ci, classes in cta_probs.items()
            }

        sm_helper = WikidataSemanticModelHelper(
            qnodes, qnode_labels, wdclasses, wdprops
        )
        sm = sm_helper.create_sm(table, pred_cpa, pred_cta)
        sm = sm_helper.minify_sm(sm)
        return Annotation(
            sm=sm,
            dg=dg,
            cg=cg,
            cg_edge_probs=edge_probs,
            cta_probs=cta_probs,
            pred_cpa=pred_cpa,
            pred_cta=pred_cta,
        )

    def get_entities(
        self, qnode_ids: Set[str], n_hop: int = 1, verbose: bool = False
    ) -> Dict[str, QNode]:
        assert n_hop <= 2
        batch_size = 30
        qnodes: Dict[str, QNode] = {}
        pp = Parallel()
        for qnode_id in qnode_ids:
            qnode = self.qnodes.get(qnode_id, None)
            if qnode is not None:
                qnodes[qnode_id] = qnode

        if isinstance(self.qnodes, WDProxyDB):
            missing_qnode_ids = [
                qnode_id
                for qnode_id in qnode_ids
                if qnode_id not in qnodes
                and not self.qnodes.does_not_exist_locally(qnode_id)
            ]
            if len(missing_qnode_ids) > 0:
                resp = pp.map(
                    query_wikidata_entities,
                    [
                        missing_qnode_ids[i : i + batch_size]
                        for i in range(0, len(missing_qnode_ids), batch_size)
                    ],
                    show_progress=verbose,
                    progress_desc=f"query wikidata for get missing entities in hop 1",
                    is_parallel=True,
                )
                for odict in resp:
                    for k, v in odict.items():
                        qnodes[k] = v
                        self.qnodes[k] = v

        if n_hop > 1:
            next_qnode_ids = set()
            # for qnode in tqdm(
            #     qnodes.values(), desc="gather entities in 2nd hop", disable=not verbose
            # ):
            for qnode in qnodes.values():
                for p, stmts in qnode.props.items():
                    for stmt in stmts:
                        if stmt.value.is_qnode():
                            next_qnode_ids.add(stmt.value.as_entity_id())
                        for qvals in stmt.qualifiers.values():
                            next_qnode_ids = next_qnode_ids.union(
                                qval.as_entity_id() for qval in qvals if qval.is_qnode()
                            )
            next_qnode_ids = list(next_qnode_ids.difference(qnodes.keys()))
            for qnode_id in tqdm(
                next_qnode_ids,
                desc="load entities in 2nd hop from db",
                disable=not verbose,
            ):
                qnode = self.qnodes.get(qnode_id, None)
                if qnode is not None:
                    qnodes[qnode_id] = qnode

            if isinstance(self.qnodes, WDProxyDB):
                next_qnode_ids = [
                    qnode_id
                    for qnode_id in next_qnode_ids
                    if qnode_id not in qnodes
                    and not self.qnodes.does_not_exist_locally(qnode_id)
                ]
                if len(next_qnode_ids) > 0:
                    resp = pp.map(
                        query_wikidata_entities,
                        [
                            next_qnode_ids[i : i + batch_size]
                            for i in range(0, len(next_qnode_ids), batch_size)
                        ],
                        show_progress=verbose,
                        progress_desc=f"query wikidata for get missing entities in hop {n_hop}",
                        is_parallel=True,
                    )
                    for odict in resp:
                        for k, v in odict.items():
                            qnodes[k] = v
                            self.qnodes[k] = v
        return qnodes

    def get_entity_labels(
        self, qnodes: Dict[str, QNode], verbose: bool = False
    ) -> Dict[str, str]:
        id2label = {}
        for qnode in tqdm(qnodes.values(), disable=not verbose, desc=""):
            qnode: QNode
            id2label[qnode.id] = str(qnode.label)
            for stmts in qnode.props.values():
                for stmt in stmts:
                    if stmt.value.is_qnode():
                        qnode_id = stmt.value.as_entity_id()
                        if qnode_id in self.qnode_labels:
                            label = self.qnode_labels[qnode_id]
                        else:
                            label = qnode_id
                        id2label[qnode_id] = label
                    for qvals in stmt.qualifiers.values():
                        for qval in qvals:
                            if qval.is_qnode():
                                qnode_id = qval.as_entity_id()
                                if qnode_id in self.qnode_labels:
                                    label = self.qnode_labels[qnode_id]
                                else:
                                    label = qnode_id
                                id2label[qnode_id] = label
        return id2label

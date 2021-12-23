import os
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union
from grams.algorithm.literal_match import TextParser

import networkx as nx
import sm.misc as M
import sm.outputs as O
from kgdata.wikidata.db import (
    WDProxyDB,
    get_qnode_db,
    get_wdclass_db,
    get_wdprop_db,
    query_wikidata_entities,
)
from kgdata.wikidata.models import QNode, WDClass, WDProperty, WDQuantityPropertyStats
from rdflib import RDFS

import grams.inputs as I
from grams.algorithm.data_graph import DGConfigs, DGFactory
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption
from grams.algorithm.psl_solver import PSLSteinerTreeSolver
from grams.algorithm.semantic_graph import SemanticGraphConstructor
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG


@dataclass
class Annotation:
    sm: O.SemanticModel
    # data graph
    dg: nx.MultiDiGraph
    # semantic graph
    sg: nx.MultiDiGraph
    # predicted semantic graph where incorrect relations are removed
    pred_sg: nx.MultiDiGraph
    # predicted column types
    pred_cta: Dict[int, Dict[str, float]]


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
                compression=True,
                read_only=read_only,
                proxy=proxy,
                is_singleton=True,
            )
            if proxy:
                assert isinstance(self.qnodes, WDProxyDB)
            self.wdclasses = get_wdclass_db(
                os.path.join(data_dir, "wdclasses.db"),
                compression=False,
                read_only=read_only,
                proxy=proxy,
                is_singleton=True,
            )
            self.wdprops = get_wdprop_db(
                os.path.join(data_dir, "wdprops.db"),
                compression=False,
                read_only=read_only,
                proxy=proxy,
                is_singleton=True,
            )
            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(
                os.path.join(data_dir, "quantity_prop_stats")
            )

        self.update_config()

    def update_config(self):
        """Update the current configuration of the algorithm based on the current configuration stored in this object"""
        for name, value in self.cfg.data_graph.configs.items():
            if not hasattr(DGConfigs, name):
                raise Exception(f"Invalid configuration for data gram: {name}")
            setattr(DGConfigs, name, value)

        TextParser.NUM_COVER_FRACTION_THRESHOLD = (
            self.cfg.literal_matcher.text_parser.NUM_COVER_FRACTION_THRESHOLD
        )

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
            qnodes = self.get_entities(qnode_ids, n_hop=2, verbose=verbose)
        wdclasses = self.wdclasses.cache_dict()
        wdprops = self.wdprops.cache_dict()

        if len(qnode_ids) != len(qnodes):
            nonexistent_qnode_ids = qnode_ids.difference(qnodes.keys())
            table.remove_nonexistent_entities(nonexistent_qnode_ids)

        with self.timer.watch("build kg object index"):
            kg_object_index = KGObjectIndex.from_qnodes(
                list(qnodes.keys()),
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
            constructor = SemanticGraphConstructor(
                [
                    SemanticGraphConstructor.init_sg,
                ],
                qnodes,
                wdclasses,
                wdprops,
            )
            sg = constructor.run(table, dg, debug=False).sg

        with self.timer.watch("run inference"):
            psl_solver = PSLSteinerTreeSolver(
                qnodes,
                wdclasses,
                wdprops,
                self.wd_numprop_stats,
                disable_rules=set(self.cfg.psl.disable_rules),
                sim_fn=None,
                # cache_dir=outdir / f"{override_psl_cachedir}cache/psl",
                postprocessing_method=self.cfg.psl.postprocessing,
                enable_logging=self.cfg.psl.enable_logging,
            )
            pred_sg, pred_cta = psl_solver.run(
                {"table": table, "semanticgraph": sg, "datagraph": dg}
            )
            pred_cta = {
                int(ci.replace("column-", "")): classes
                for ci, classes in pred_cta.items()
            }
            cta = {
                ci: max(classes.items(), key=itemgetter(1))[0]
                for ci, classes in pred_cta.items()
            }

        sm_helper = WikidataSemanticModelHelper(qnodes, wdclasses, wdprops)
        sm = sm_helper.create_sm(table, pred_sg, cta)
        sm = sm_helper.minify_sm(sm)
        return Annotation(sm=sm, dg=dg, sg=sg, pred_sg=pred_sg, pred_cta=cta)

    def get_entities(
        self, qnode_ids: Set[str], n_hop: int = 1, verbose: bool = False
    ) -> Dict[str, QNode]:
        assert n_hop <= 2
        batch_size = 30
        qnodes: Dict[str, QNode] = {}
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
                resp = M.parallel_map(
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
            for qnode_id in next_qnode_ids:
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
                    resp = M.parallel_map(
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

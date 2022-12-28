import os
from dataclasses import dataclass
from operator import itemgetter
from functools import partial
from pathlib import Path
from typing import Dict, MutableMapping, Set, Tuple, Union
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences.psl_gram_model import PSLGramModel
from grams.algorithm.inferences.psl_gram_model_exp import PSLGramModelExp
from grams.algorithm.inferences.psl_gram_model_exp2 import PSLGramModelExp2
from grams.algorithm.inferences.psl_lib import PSLModel
from grams.algorithm.postprocessing import (
    MinimumArborescence,
    PairwiseSelection,
    PostProcessingSimplePath,
    SteinerTree,
)
from hugedict.prelude import CacheDict, Parallel
from kgdata.wikidata.models import WDEntityLabel
from loguru import logger
from tqdm import tqdm
from sm.outputs.semantic_model import SemanticModel
from timer import Timer
import serde.prelude as serde
from kgdata.wikidata.db import (
    WDProxyDB,
    get_entity_db,
    get_entity_label_db,
    get_wdclass_db,
    get_wdprop_db,
    get_wdprop_domain_db,
    get_wdprop_range_db,
    query_wikidata_entities,
)
from kgdata.wikidata.models import (
    WDEntity,
    WDClass,
    WDQuantityPropertyStats,
)

import grams.inputs as I
from grams.algorithm.data_graph import DGFactory
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption

from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG


@dataclass
class Annotation:
    sm: SemanticModel
    # data graph
    dg: DGGraph
    # candidate graph
    cg: CGGraph
    # probabilities of each edge in cg (uid, vid, edge key)
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
        self.timer = Timer()
        self.cfg = cfg if cfg is not None else DEFAULT_CONFIG

        with self.timer.watch("init grams db"):
            read_only = not proxy
            self.wdentities = get_entity_db(
                os.path.join(data_dir, "wdentities.db"),
                read_only=read_only,
                proxy=proxy,
            )
            if proxy:
                assert isinstance(self.wdentities, WDProxyDB)
            if os.path.exists(os.path.join(data_dir, "wdentity_labels.db")):
                self.wdentity_labels = get_entity_label_db(
                    os.path.join(data_dir, "wdentity_labels.db"),
                )
            else:
                self.wdentity_labels: MutableMapping[str, WDEntityLabel] = {}
            self.wdclasses = get_wdclass_db(
                os.path.join(data_dir, "wdclasses.db"),
                read_only=read_only,
                proxy=proxy,
            )
            if os.path.exists(os.path.join(data_dir, "wdclasses.fixed.jl")):
                self.wdclasses = self.wdclasses.cache()
                assert isinstance(self.wdclasses, CacheDict)
                for record in serde.jl.deser(
                    os.path.join(data_dir, "wdclasses.fixed.jl")
                ):
                    cls = WDClass.from_dict(record)
                    self.wdclasses._cache[cls.id] = cls
            self.wdprops = get_wdprop_db(
                os.path.join(data_dir, "wdprops.db"),
                read_only=read_only,
                proxy=proxy,
            )
            if os.path.exists(os.path.join(data_dir, "wdprop_domains.db")):
                self.wdprop_domains = get_wdprop_domain_db(
                    os.path.join(data_dir, "wdprop_domains.db"),
                    read_only=True,
                )
            else:
                self.wdprop_domains = None

            if os.path.exists(os.path.join(data_dir, "wdprop_ranges.db")):
                self.wdprop_ranges = get_wdprop_range_db(
                    os.path.join(data_dir, "wdprop_ranges.db"),
                    read_only=True,
                )
            else:
                self.wdprop_ranges = None

            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(
                os.path.join(data_dir, "quantity_prop_stats")
            )

        self.update_config()

    def update_config(self):
        """Update the current configuration of the algorithm based on the current configuration stored in this object"""
        # TODO: switch to new configuration
        # for name, value in self.cfg.data_graph.configs.items():
        #     if not hasattr(DGConfigs, name):
        #         raise Exception(f"Invalid configuration for data_graph: {name}")
        #     setattr(DGConfigs, name, value)

        # for name, value in self.cfg.literal_matcher.text_parser.items():
        #     if not hasattr(TextParserConfigs, name):
        #         raise Exception(
        #             f"Invalid configuration for literal_matcher.text_parser: {name}"
        #         )
        #     setattr(TextParserConfigs, name, value)

        # for name, value in self.cfg.literal_matcher.matchers.items():
        #     if not hasattr(LiteralMatch, name):
        #         raise Exception(
        #             f"Invalid configuration for literal_matcher.matchers: {name}"
        #         )
        #     setattr(LiteralMatch, name, value)

    def annotate(self, table: I.LinkedTable, verbose: bool = False) -> Annotation:
        """Annotate a linked table"""
        wdentity_ids = {
            link.entity_id
            for rlinks in table.links
            for links in rlinks
            for link in links
            if link.entity_id is not None
        }
        wdentity_ids.update(
            (
                candidate.entity_id
                for rlinks in table.links
                for links in rlinks
                for link in links
                for candidate in link.candidates
            )
        )
        if table.context.page_entity_id is not None:
            wdentity_ids.add(table.context.page_entity_id)

        with self.timer.watch("retrieving entities"):
            wdentities = self.get_entities(
                wdentity_ids, n_hop=self.cfg.data_graph.max_n_hop, verbose=verbose
            )
        wdclasses = self.wdclasses.cache()
        wdprops = self.wdprops.cache()

        nonexistent_wdentity_ids = wdentity_ids.difference(wdentities.keys())
        if len(nonexistent_wdentity_ids) > 0:
            logger.info(
                "Removing non-existent entities: {}", list(nonexistent_wdentity_ids)
            )
            table.remove_nonexistent_entities(nonexistent_wdentity_ids)

        with self.timer.watch("retrieving entities label"):
            wdentity_labels = self.get_entity_labels(wdentities, verbose=verbose)

        with self.timer.watch("build kg object index"):
            kg_object_index = KGObjectIndex.from_entities(
                list(wdentity_ids.intersection(wdentities.keys())),
                wdentities,
                wdprops,
                n_hop=self.cfg.data_graph.max_n_hop,
                traversal_option=TraversalOption.TransitiveOnly,
            )

        with self.timer.watch("build dg & sg"):
            dg_factory = DGFactory(wdentities, wdprops)
            dg = dg_factory.create_dg(
                table, kg_object_index, max_n_hop=self.cfg.data_graph.max_n_hop
            )
            cg_factory = CGFactory(wdentities, wdentity_labels, wdclasses, wdprops)
            cg = cg_factory.create_cg(table, dg)

        with self.timer.watch("run inference"):
            if self.cfg.psl.experiment_model:
                logger.debug("Using experiment PSL model")
                cls = partial(
                    {
                        "exp": PSLGramModelExp,
                        "exp2": PSLGramModelExp2,
                    }[self.cfg.psl.experiment_model],
                    wdprop_domains=self.wdprop_domains,
                    wdprop_ranges=self.wdprop_ranges,
                )
            else:
                cls = PSLGramModel

            edge_probs, cta_probs = cls(
                wdentities=wdentities,
                wdentity_labels=self.wdentity_labels,
                wdclasses=wdclasses,
                wdprops=wdprops,
                wd_numprop_stats=self.wd_numprop_stats,
                disable_rules=self.cfg.psl.disable_rules,
            ).predict(table, cg, dg, verbose=verbose, debug=False)

            edge_probs = PSLModel.normalize_probs(edge_probs, eps=self.cfg.psl.eps)

            if self.cfg.psl.postprocessing == "steiner_tree":
                pp = SteinerTree(table, cg, dg, edge_probs, self.cfg.psl.threshold)
            elif self.cfg.psl.postprocessing == "arborescence":
                pp = MinimumArborescence(
                    table, cg, dg, edge_probs, self.cfg.psl.threshold
                )
            elif self.cfg.psl.postprocessing == "simplepath":
                pp = PostProcessingSimplePath(
                    table, cg, dg, edge_probs, self.cfg.psl.threshold
                )
            elif self.cfg.psl.postprocessing == "pairwise":
                pp = PairwiseSelection(
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
            wdentities, wdentity_labels, wdclasses, wdprops
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
        self, wdentity_ids: Set[str], n_hop: int = 1, verbose: bool = False
    ) -> Dict[str, WDEntity]:
        assert n_hop <= 2
        batch_size = 30
        wdentities: Dict[str, WDEntity] = {}
        pp = Parallel()
        for wdentity_id in wdentity_ids:
            wdentity = self.wdentities.get(wdentity_id, None)
            if wdentity is not None:
                wdentities[wdentity_id] = wdentity

        if isinstance(self.wdentities, WDProxyDB):
            missing_qnode_ids = [
                wdentity_id
                for wdentity_id in wdentity_ids
                if wdentity_id not in wdentities
                and not self.wdentities.does_not_exist_locally(wdentity_id)
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
                        wdentities[k] = v
                        self.wdentities[k] = v

        if n_hop > 1:
            next_wdentity_ids = set()
            for wdentity in wdentities.values():
                for p, stmts in wdentity.props.items():
                    for stmt in stmts:
                        if stmt.value.is_qnode(stmt.value):
                            next_wdentity_ids.add(stmt.value.as_entity_id())
                        for qvals in stmt.qualifiers.values():
                            next_wdentity_ids = next_wdentity_ids.union(
                                qval.as_entity_id()
                                for qval in qvals
                                if qval.is_qnode(qval)
                            )
            next_wdentity_ids = list(next_wdentity_ids.difference(wdentities.keys()))
            for wdentity_id in tqdm(
                next_wdentity_ids,
                desc="load entities in 2nd hop from db",
                disable=not verbose,
            ):
                wdentity = self.wdentities.get(wdentity_id, None)
                if wdentity is not None:
                    wdentities[wdentity_id] = wdentity

            if isinstance(self.wdentities, WDProxyDB):
                next_wdentity_ids = [
                    qnode_id
                    for qnode_id in next_wdentity_ids
                    if qnode_id not in wdentities
                    and not self.wdentities.does_not_exist_locally(qnode_id)
                ]
                if len(next_wdentity_ids) > 0:
                    resp = pp.map(
                        query_wikidata_entities,
                        [
                            next_wdentity_ids[i : i + batch_size]
                            for i in range(0, len(next_wdentity_ids), batch_size)
                        ],
                        show_progress=verbose,
                        progress_desc=f"query wikidata for get missing entities in hop {n_hop}",
                        is_parallel=True,
                    )
                    for odict in resp:
                        for k, v in odict.items():
                            wdentities[k] = v
                            self.wdentities[k] = v
        return wdentities

    def get_entity_labels(
        self, wdentities: Dict[str, WDEntity], verbose: bool = False
    ) -> Dict[str, str]:
        id2label: Dict[str, str] = {}
        for qnode in tqdm(wdentities.values(), disable=not verbose, desc=""):
            qnode: WDEntity
            id2label[qnode.id] = str(qnode.label)
            for stmts in qnode.props.values():
                for stmt in stmts:
                    if stmt.value.is_qnode(stmt.value):
                        qnode_id = stmt.value.as_entity_id()
                        if qnode_id in self.wdentity_labels:
                            label = self.wdentity_labels[qnode_id].label
                        else:
                            label = qnode_id
                        id2label[qnode_id] = label
                    for qvals in stmt.qualifiers.values():
                        for qval in qvals:
                            if qval.is_qnode(qval):
                                qnode_id = qval.as_entity_id()
                                if qnode_id in self.wdentity_labels:
                                    label = self.wdentity_labels[qnode_id].label
                                else:
                                    label = qnode_id
                                id2label[qnode_id] = label
        return id2label

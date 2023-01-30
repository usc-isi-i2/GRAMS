from __future__ import annotations
from collections.abc import Mapping

from dataclasses import dataclass, field
from functools import partial, reduce
from operator import itemgetter
from pathlib import Path
from typing import Optional, Union
from grams.actors.augcan_actor import AugCanActor, AugCanParams
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import (
    CellNode,
    DGEdge,
    DGGraph,
    EntityValueNode,
    LiteralValueNode,
    StatementNode,
)
from grams.algorithm.inferences.psl_gram_model_exp3 import PSLGramModelExp3
from hugedict.prelude import HugeMutableMapping
from kgdata.wikidata.models import WDProperty
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdentitylabel import WDEntityLabel

import numpy as np
from osin.apis.remote_exp import RemoteExpRun
from osin.types.pyobject import OTable
import ray
from loguru import logger
from osin.integrations.ream import OsinActor
from ream.actors.base import BaseActor
from ream.dataset_helper import DatasetDict, DatasetQuery
from ream.helper import orjson_dumps
from sm.outputs.semantic_model import SemanticModel
from timer import Timer

import grams.inputs as I
from grams.actors.dataset_actor import GramsELDatasetActor
from grams.actors.db_actor import GramsDB
from grams.evaluator import Evaluator
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.data_graph import DGFactory
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.inferences.psl_config import PslConfig
from grams.algorithm.inferences.psl_gram_model import PSLGramModel
from grams.algorithm.inferences.psl_gram_model_exp import PSLGramModelExp
from grams.algorithm.inferences.psl_gram_model_exp2 import PSLGramModelExp2
from grams.algorithm.inferences.psl_lib import PSLModel
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption
from grams.algorithm.literal_matchers import (
    LiteralMatch,
    LiteralMatchConfigs,
    TextParser,
    TextParserConfigs,
)
from grams.algorithm.postprocessing import (
    MinimumArborescence,
    PairwiseSelection,
    PostProcessingSimplePath,
    SteinerTree,
)
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG
from grams.inputs.linked_table import LinkedTable
from grams.main import Annotation
from ned.actors.evaluate_helper import EvalArgs
from sm.dataset import Example
from sm.misc.ray_helper import ray_put, ray_map
from grams.actors.actor_helpers import to_grams_db, eval_dataset
from ream.cache_helper import (
    Cache,
    Cacheable,
    unwrap_cache_decorators,
)


@dataclass
class GramsParams:
    data_dir: Path = field(
        metadata={"help": "Path to a directory containing databases"},
    )
    proxy_db: bool = field(
        default=True,
        metadata={"help": "Whether to use a proxy database for the semantic model"},
    )
    augcan: AugCanParams = field(
        default_factory=AugCanParams,
        metadata={"help": "Configuration for the Augmented Candidate algorithm"},
    )
    data_graph: DGConfigs = field(
        default_factory=DGConfigs,
        metadata={"help": "Configuration for the data graph"},
    )
    text_parser: TextParserConfigs = field(
        default_factory=TextParserConfigs,
        metadata={"help": "Configuration for the text parser"},
    )
    literal_matchers: LiteralMatchConfigs = field(
        default_factory=LiteralMatchConfigs,
        metadata={"help": "Configuration for the literal matchers"},
    )
    psl: PslConfig = field(
        default_factory=PslConfig,
        metadata={"help": "Configuration for the PSL model"},
    )


class GramsActor(OsinActor[I.LinkedTable, GramsParams]):
    """GRAMS for Semantic Modeling"""

    NAME = "Semantic Modeling"
    VERSION = 105
    EXP_VERSION = 3

    def __init__(self, params: GramsParams, dataset_actor: GramsELDatasetActor):
        db = GramsDB(params.data_dir, params.proxy_db)
        augcan_actor = AugCanActor(params.augcan, db, dataset_actor)

        if params.augcan.threshold <= 1.0:
            dep_actors: list[BaseActor] = [dataset_actor, augcan_actor]
        else:
            dep_actors: list[BaseActor] = [dataset_actor]
        super().__init__(params, dep_actors)

        self.timer = Timer()
        self.cfg = DEFAULT_CONFIG
        self.db = db
        self.dataset_actor = dataset_actor
        self.augcan_actor = augcan_actor

    def run(self, table: LinkedTable, verbose: bool = False):
        return annotate(self.db, self.params, table, verbose)

    def run_dataset(self, dsquery: str):
        dsdict = self.get_dataset(dsquery)
        output: DatasetDict[list[Annotation]] = DatasetDict(
            dsdict.name, {}, dsdict.provenance
        )
        for name, ds in dsdict.items():
            cachedir = self.get_working_fs().root
            if len(ds) > 1:
                dbref = ray_put(self.db.data_dir)
                cfgref = ray_put(self.params)
                cachedirref = ray_put(cachedir)
                lst = ray_map(
                    ray_annotate.remote,
                    [
                        (dbref, cfgref, example.table, cachedirref, False)
                        for example in ds
                    ],
                    desc="Annotating tables",
                    verbose=True,
                )
                output[name] = [x[0] for x in lst]
                for x in lst:
                    self.timer.merge(x[1])
            else:
                # output[name] = [annotate(self.db, self.params, ds[0].table, True)[0]]
                lst = [
                    cacheable_annotate(cachedir, self.db, self.params, ex.table, True)
                    for ex in ds
                ]
                output[name] = [x[0] for x in lst]
                for x in lst:
                    self.timer.merge(x[1])

        return output

    def evaluate(self, eval_args: EvalArgs):
        evalout = {}
        for dsquery in eval_args.dsqueries:
            dsquery_p = DatasetQuery.from_string(dsquery)
            dsdict = self.get_dataset(dsquery)
            ann_dsdict = self.run_dataset(dsquery)

            for name, examples in dsdict.items():
                with self.new_exp_run(
                    dataset=dsquery_p.get_query(name),
                ) as exprun:
                    primitive_output, primitive_ex_output = eval_dataset(
                        self.db,
                        examples,
                        [ann.sm for ann in ann_dsdict[name]],
                        anns=ann_dsdict[name],
                        exprun=exprun,
                    )
                    self.logger.info(
                        "Dataset: {}\n{}",
                        dsquery_p.get_query(name),
                        orjson_dumps(primitive_output).decode(),
                    )
                    if exprun is not None:
                        exprun.update_output(
                            primitive=dict(
                                workdir=str(self.get_working_fs().root),
                                **primitive_output,
                            )
                        )
                    evalout[dsquery_p.get_query(name)] = (
                        primitive_output,
                        primitive_ex_output,
                    )

        self.timer.report(self.logger.debug)
        return evalout

    def get_dataset(self, dsquery: str):
        if self.params.augcan.threshold <= 1.0:
            dsdict = self.augcan_actor.run_dataset(dsquery)
        else:
            dsdict = self.dataset_actor.run_dataset(dsquery)
        return dsdict


@ray.remote
def ray_annotate(
    db: Union[GramsDB, Path],
    cfg: GramsParams,
    table: LinkedTable,
    cachedir: Path,
    verbose: bool,
):
    try:
        # return annotate(to_grams_db(db), cfg, table, verbose)
        return cacheable_annotate(cachedir, to_grams_db(db), cfg, table, verbose)
    except Exception as e:
        raise Exception("Failed to annotate table: " + table.id) from e


class CacheableAnnotator(Cacheable):
    def __init__(
        self, workdir: Path, timer: Timer, db: GramsDB, cfg: GramsParams, verbose: bool
    ):
        super().__init__(workdir)
        self.timer = timer
        self.db = db
        self.cfg = cfg
        self.verbose = verbose

        self.wdclasses = self.db.wdclasses.cache()
        self.wdprops = self.db.wdprops.cache()

    def dg_serialize(self, dg: DGGraph):
        # return {"nodes": dg.nodes(), "edges": dg.edges()}
        # return {"nodes": dg.nodes(), "edges": [e.to_tuple() for e in dg.iter_edges()]}
        return {
            "nodes": [n.to_tuple() for n in dg.nodes()],
            "edges": [e.to_tuple() for e in dg.iter_edges()],
        }

    def dg_deserialize(self, obj: dict):
        g = DGGraph()
        for node in obj["nodes"]:
            size = len(node)
            if size == 3:
                denode = LiteralValueNode.from_tuple(node)
            elif size == 4:
                denode = EntityValueNode.from_tuple(node)
            elif isinstance(node[3], bool):
                denode = StatementNode.from_tuple(node)
            else:
                denode = CellNode.from_tuple(node)

            # g.add_node(node)
            g.add_node(denode)
        for edge in obj["edges"]:
            g.add_edge(DGEdge.from_tuple(edge))
            # g.add_edge(edge)
        return g

    def annotate(self, table: LinkedTable):
        table = self.preprocess_table(table)
        dg, cg = self.build_graphs(table)

        import pickle, orjson

        timer = Timer()
        with timer.watch_and_report("serialize dg 2"):
            sdg2 = pickle.dumps(self.dg_serialize(dg))
        with timer.watch_and_report("serialize dg"):
            sdg = pickle.dumps(dg)
        # with timer.watch_and_report("serialize dg 3"):
        #     sdg3 = orjson.dumps(
        #         self.dg_serialize(dg),
        #         option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        #     )

        # with timer.watch_and_report("deserialize dg 2 (part)"):
        #     abc = pickle.loads(sdg2)
        with timer.watch_and_report("deserialize dg"):
            odg = pickle.loads(sdg)
        with timer.watch_and_report("deserialize dg 2"):
            odg2 = self.dg_deserialize(pickle.loads(sdg2))
        # with timer.watch_and_report("deserialize dg 3"):
        #     odg3 = self.dg_deserialize(orjson.loads(sdg3))
        with timer.watch_and_report("deserialize dg"):
            odg = pickle.loads(sdg)

        with timer.watch_and_report("serialize cg"):
            scg = pickle.dumps(cg)
        with timer.watch_and_report("deserialize cg"):
            ocg = pickle.loads(scg)

        exit(0)

        edge_probs, cta_probs, pred_cpa, pred_cta = self.run_inference(table)
        # wdentity_labels = self.get_entity_labels(table)
        sm_helper = WikidataSemanticModelHelper(
            self.db.wdentities,
            # wdentity_labels,
            {},
            self.db.wdclasses,
            self.db.wdprops,
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

    @Cache.pickle.sqlite(
        cache_key=lambda self, table: table.id.encode(),
        compression="lz4",
        log_serde_time=False,
    )
    def preprocess_table(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        nonexistent_wdentity_ids = wdentity_ids.difference(wdentities.keys())
        if len(nonexistent_wdentity_ids) > 0:
            logger.info(
                "Removing non-existent entities: {}", list(nonexistent_wdentity_ids)
            )
            table.remove_nonexistent_entities(nonexistent_wdentity_ids)
        return table

    @Cache.pickle.sqlite(
        cache_key=lambda self, table: table.id.encode(),
        compression="lz4",
        log_serde_time=True,
    )
    def build_graphs(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        wdentity_labels = self.get_entity_labels(table)

        with self.timer.watch("build kg object index"):
            kg_object_index = KGObjectIndex.from_entities(
                list(wdentity_ids.intersection(wdentities.keys())),
                wdentities,
                self.wdprops,
                n_hop=self.cfg.data_graph.max_n_hop,
                traversal_option=TraversalOption.TransitiveOnly,
            )

        with self.timer.watch("build dg & sg"):
            text_parser = TextParser(self.cfg.text_parser)
            literal_match = LiteralMatch(wdentities, self.cfg.literal_matchers)

            dg_factory = DGFactory(
                wdentities,
                self.wdprops,
                text_parser,
                literal_match,
                self.cfg.data_graph,
            )
            dg = dg_factory.create_dg(
                table, kg_object_index, max_n_hop=self.cfg.data_graph.max_n_hop
            )
            cg_factory = CGFactory(
                wdentities,
                wdentity_labels,
                self.wdclasses,
                self.wdprops,
            )
            cg = cg_factory.create_cg(table, dg)

        return dg, cg

    @Cache.pickle.sqlite(
        cache_key=lambda self, table: table.id.encode(),
        compression="lz4",
        log_serde_time=False,
    )
    def run_inference(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        dg, cg = self.build_graphs(table)

        with self.timer.watch("run inference"):
            if self.cfg.psl.experiment_model:
                logger.debug(
                    "Using experiment PSL model: {}", self.cfg.psl.experiment_model
                )
                if int(self.cfg.psl.experiment_model[3:]) >= 3:
                    context = AlgoContext(
                        data_dir=self.db.data_dir,
                        wdprop_domains=self.db.wdprop_domains,
                        wdprop_ranges=self.db.wdprop_ranges,
                        wdentities=wdentities,
                        wdentity_labels=self.db.wdentity_labels,
                        wdclasses=self.wdclasses,
                        wdprops=self.wdprops,
                        wd_num_prop_stats=self.db.wd_numprop_stats,
                    )
                    pslmodel = PSLGramModelExp3(
                        context=context, disable_rules=self.cfg.psl.disable_rules
                    )
                else:
                    pslmodel = PSLGramModelExp2(
                        wdprop_domains=self.db.wdprop_domains,
                        wdprop_ranges=self.db.wdprop_ranges,
                        wdentities=wdentities,
                        wdentity_labels=self.db.wdentity_labels,
                        wdclasses=self.wdclasses,
                        wdprops=self.wdprops,
                        wd_numprop_stats=self.db.wd_numprop_stats,
                        disable_rules=self.cfg.psl.disable_rules,
                        rule_weights=dict(self.cfg.psl.rule_weights),
                    )
            else:
                pslmodel = PSLGramModel(
                    wdentities=wdentities,
                    wdentity_labels=self.db.wdentity_labels,
                    wdclasses=self.wdclasses,
                    wdprops=self.wdprops,
                    wd_numprop_stats=self.db.wd_numprop_stats,
                    disable_rules=self.cfg.psl.disable_rules,
                )

            edge_probs, cta_probs = pslmodel.predict(
                table, cg, dg, verbose=self.verbose, debug=False
            )

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

        return edge_probs, cta_probs, pred_cpa, pred_cta

    @Cache.mem(cache_key=lambda self, table: table.id)
    def get_entity_labels(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        with self.timer.watch("retrieve entity labels"):
            wdentity_labels = self.db.get_entity_labels(wdentities, self.verbose)
        return wdentity_labels

    @Cache.mem(cache_key=lambda self, table: table.id)
    def retrieving_entities(self, table: LinkedTable):
        with self.timer.watch("retrieve entities"):
            wdentity_ids: set[str] = {
                entid
                for links in table.links.flat_iter()
                for link in links
                for entid in link.entities
            }
            wdentity_ids.update(
                (
                    candidate.entity_id
                    for links in table.links.flat_iter()
                    for link in links
                    for candidate in link.candidates
                )
            )
            wdentity_ids.update(table.context.page_entities)
            wdentities = self.db.get_entities(
                wdentity_ids, n_hop=self.cfg.data_graph.max_n_hop, verbose=self.verbose
            )

        return wdentity_ids, wdentities


def cacheable_annotate(
    cachedir: Path, db: GramsDB, cfg: GramsParams, table: LinkedTable, verbose: bool
):
    timer = Timer()
    annotator = CacheableAnnotator(cachedir, timer, db, cfg, verbose)
    return annotator.annotate(table), timer


def annotate(
    db: GramsDB, cfg: GramsParams, table: LinkedTable, verbose: bool
) -> tuple[Annotation, Timer]:
    timer = Timer()

    with timer.watch("retrieve entities"):
        wdentity_ids: set[str] = {
            entid
            for links in table.links.flat_iter()
            for link in links
            for entid in link.entities
        }
        wdentity_ids.update(
            (
                candidate.entity_id
                for links in table.links.flat_iter()
                for link in links
                for candidate in link.candidates
            )
        )
        wdentity_ids.update(table.context.page_entities)
        wdentities = db.get_entities(
            wdentity_ids, n_hop=cfg.data_graph.max_n_hop, verbose=verbose
        )
        wdclasses = db.wdclasses.cache()
        wdprops = db.wdprops.cache()

    nonexistent_wdentity_ids = wdentity_ids.difference(wdentities.keys())
    if len(nonexistent_wdentity_ids) > 0:
        logger.info(
            "Removing non-existent entities: {}", list(nonexistent_wdentity_ids)
        )
        table.remove_nonexistent_entities(nonexistent_wdentity_ids)

    with timer.watch("retrieve entity labels"):
        wdentity_labels = db.get_entity_labels(wdentities, verbose)

    with timer.watch("build kg object index"):
        kg_object_index = KGObjectIndex.from_entities(
            list(wdentity_ids.intersection(wdentities.keys())),
            wdentities,
            wdprops,
            n_hop=cfg.data_graph.max_n_hop,
            traversal_option=TraversalOption.TransitiveOnly,
        )

    with timer.watch("build dg & sg"):
        text_parser = TextParser(cfg.text_parser)
        literal_match = LiteralMatch(wdentities, cfg.literal_matchers)

        dg_factory = DGFactory(
            wdentities, wdprops, text_parser, literal_match, cfg.data_graph
        )
        dg = dg_factory.create_dg(
            table, kg_object_index, max_n_hop=cfg.data_graph.max_n_hop
        )
        cg_factory = CGFactory(
            wdentities,
            wdentity_labels,
            wdclasses,
            wdprops,
        )
        cg = cg_factory.create_cg(table, dg)

    with timer.watch("run inference"):
        if cfg.psl.experiment_model:
            logger.debug("Using experiment PSL model: {}", cfg.psl.experiment_model)
            if int(cfg.psl.experiment_model[3:]) >= 3:
                context = AlgoContext(
                    data_dir=db.data_dir,
                    wdprop_domains=db.wdprop_domains,
                    wdprop_ranges=db.wdprop_ranges,
                    wdentities=wdentities,
                    wdentity_labels=db.wdentity_labels,
                    wdclasses=wdclasses,
                    wdprops=wdprops,
                    wd_num_prop_stats=db.wd_numprop_stats,
                )
                pslmodel = PSLGramModelExp3(
                    context=context, disable_rules=cfg.psl.disable_rules
                )
            else:
                pslmodel = PSLGramModelExp2(
                    wdprop_domains=db.wdprop_domains,
                    wdprop_ranges=db.wdprop_ranges,
                    wdentities=wdentities,
                    wdentity_labels=db.wdentity_labels,
                    wdclasses=wdclasses,
                    wdprops=wdprops,
                    wd_numprop_stats=db.wd_numprop_stats,
                    disable_rules=cfg.psl.disable_rules,
                    rule_weights=dict(cfg.psl.rule_weights),
                )
        else:
            pslmodel = PSLGramModel(
                wdentities=wdentities,
                wdentity_labels=db.wdentity_labels,
                wdclasses=wdclasses,
                wdprops=wdprops,
                wd_numprop_stats=db.wd_numprop_stats,
                disable_rules=cfg.psl.disable_rules,
            )

        edge_probs, cta_probs = pslmodel.predict(
            table, cg, dg, verbose=verbose, debug=False
        )

        edge_probs = PSLModel.normalize_probs(edge_probs, eps=cfg.psl.eps)

        if cfg.psl.postprocessing == "steiner_tree":
            pp = SteinerTree(table, cg, dg, edge_probs, cfg.psl.threshold)
        elif cfg.psl.postprocessing == "arborescence":
            pp = MinimumArborescence(table, cg, dg, edge_probs, cfg.psl.threshold)
        elif cfg.psl.postprocessing == "simplepath":
            pp = PostProcessingSimplePath(table, cg, dg, edge_probs, cfg.psl.threshold)
        elif cfg.psl.postprocessing == "pairwise":
            pp = PairwiseSelection(table, cg, dg, edge_probs, cfg.psl.threshold)
        else:
            raise NotImplementedError(cfg.psl.postprocessing)

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
    return (
        Annotation(
            sm=sm,
            dg=dg,
            cg=cg,
            cg_edge_probs=edge_probs,
            cta_probs=cta_probs,
            pred_cpa=pred_cpa,
            pred_cta=pred_cta,
        ),
        timer,
    )

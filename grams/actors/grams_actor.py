from __future__ import annotations

from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Union
from grams.actors.augcan_actor import AugCanActor, AugCanParams
from grams.algorithm.context import AlgoContext
from grams.algorithm.inferences.psl_gram_model_exp3 import (
    PSLData,
    PSLGramModelExp3,
    PSLGramsModelData3,
)

import orjson
import ray
from loguru import logger
from osin.integrations.ream import OsinActor
from ream.actors.base import BaseActor
from ream.dataset_helper import DatasetDict, DatasetQuery
from ream.helper import orjson_dumps
from sm.outputs.semantic_model import SemanticModel
from timer import Timer
from grams.algorithm.candidate_graph.cg_graph import CGGraph

import grams.inputs as I
from grams.actors.dataset_actor import GramsELDatasetActor
from grams.actors.db_actor import GramsDB
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.data_graph import DGFactory
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.inferences.psl_config import PslConfig
from grams.algorithm.inferences.psl_gram_model import PSLGramModel
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
from sm.misc.ray_helper import ray_put, ray_map
from grams.actors.actor_helpers import to_grams_db, eval_dataset
from ream.cache_helper import (
    Cache,
    Cacheable,
)


@dataclass
class AnnotationV2:
    # predicted semantic model
    sm: SemanticModel
    # original candidate graph
    cg: CGGraph
    # inference's features
    features: PSLData
    # probabilities of each edge in cg (uid, vid, edge key)
    cg_edge_probs: dict[tuple[str, str, str], float]
    # probabilities of types of each column: column index -> type -> probability
    cta_probs: dict[int, dict[str, float]]
    # predicted candidate graph where incorrect relations are removed by threshold & post-processing algorithm
    pred_cpa: CGGraph
    # predicted column types
    pred_cta: dict[int, str]


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
    VERSION = 107
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
        cachedir = Path("/tmp")  # since we disable cache, this won't be used
        return cacheable_annotate(
            cachedir, self.db, self.params, table, verbose, enable_cache=False
        )[0]

    def run_dataset(self, dsquery: str):
        dsdict = self.get_dataset(dsquery)
        output: DatasetDict[list[AnnotationV2]] = DatasetDict(
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
                lst = [
                    cacheable_annotate(
                        cachedir,
                        self.db,
                        self.params,
                        ex.table,
                        verbose=True,
                        enable_cache=True,
                    )
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
        return cacheable_annotate(
            cachedir, to_grams_db(db), cfg, table, verbose, enable_cache=True
        )
    except Exception as e:
        raise Exception("Failed to annotate table: " + table.id) from e


def cacheable_annotate(
    cachedir: Path,
    db: GramsDB,
    cfg: GramsParams,
    table: LinkedTable,
    verbose: bool,
    enable_cache: bool,
):
    timer = Timer()
    annotator = CacheableAnnotator(cachedir, timer, db, cfg, verbose, enable_cache)
    return annotator.annotate(table), timer


def get_cache_key(*args: str):
    def key(self, table):
        return orjson.dumps(dict([(k, getattr(self, k)) for k in args], table=table.id))

    if len(args) == 0:
        return lambda self, table: table.id.encode()
    return key


class CacheableAnnotator(Cacheable):
    def __init__(
        self,
        workdir: Path,
        timer: Timer,
        db: GramsDB,
        cfg: GramsParams,
        verbose: bool,
        enable_cache: bool = True,
    ):
        super().__init__(workdir)
        self.timer = timer
        self.db = db
        self.cfg = cfg
        self.verbose = verbose
        self.disable_cache = not enable_cache

        self.wdclasses = self.db.wdclasses.cache()
        self.wdprops = self.db.wdprops.cache()

        get_prov = lambda *x: ":".join([str(xi) for xi in x])

        if self.cfg.psl.experiment_model == "exp3":
            INFER_MODEL = get_prov(
                self.cfg.psl.experiment_model, PSLGramModelExp3.VERSION
            )
            INFER_MODEL_DATA = get_prov(
                self.cfg.psl.experiment_model, PSLGramsModelData3.VERSION
            )
        else:
            raise NotImplementedError()

        self.GRAPH_PROV = 101
        self.INFER_MODEL_DATA_PROV = get_prov(self.GRAPH_PROV, INFER_MODEL_DATA)
        self.INFER_MODEL_PROV = get_prov(self.INFER_MODEL_DATA_PROV, INFER_MODEL)

    def annotate(self, table: LinkedTable):
        table = self.preprocess_table(table)
        cg = self.get_candidategraph(table)
        psldata = self.get_inference_data(table)

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
        return AnnotationV2(
            sm=sm,
            cg=cg,
            features=psldata,
            cg_edge_probs=edge_probs,
            cta_probs=cta_probs,
            pred_cpa=pred_cpa,
            pred_cta=pred_cta,
        )

    @Cache.pickle.sqlite(
        cache_key=get_cache_key(),
        compression="lz4",
        log_serde_time=False,
        disable="disable_cache",
        mem_persist=True,
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

    @Cache.mem(cache_key=get_cache_key("GRAPH_PROV"))
    def get_datagraph(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)

        with self.timer.watch("build kg object index"):
            kg_object_index = KGObjectIndex.from_entities(
                list(wdentity_ids.intersection(wdentities.keys())),
                wdentities,
                self.wdprops,
                n_hop=self.cfg.data_graph.max_n_hop,
                traversal_option=TraversalOption.TransitiveOnly,
            )

        with self.timer.watch("build dg"):
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
        return dg

    @Cache.pickle.sqlite(
        cache_key=get_cache_key("GRAPH_PROV"),
        compression="lz4",
        log_serde_time=True,
        disable="disable_cache",
        mem_persist=True,
    )
    def get_candidategraph(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        wdentity_labels = self.get_entity_labels(table)
        dg = self.get_datagraph(table)
        with self.timer.watch("build cg"):
            cg_factory = CGFactory(
                wdentities,
                wdentity_labels,
                self.wdclasses,
                self.wdprops,
            )
            cg = cg_factory.create_cg(table, dg)
        return cg

    @Cache.pickle.sqlite(
        cache_key=get_cache_key("INFER_MODEL_DATA_PROV"),
        compression="lz4",
        log_serde_time=True,
        disable="disable_cache",
        mem_persist=True,
    )
    def get_inference_data(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        dg = self.get_datagraph(table)
        cg = self.get_candidategraph(table)

        with self.timer.watch("extract inference's data"):
            assert self.cfg.psl.experiment_model == "exp3"
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

            psldata = PSLGramsModelData3(
                context, use_readable_idmap=self.cfg.psl.use_readable_idmap
            ).extract_data(table, cg, dg)
        return psldata

    @Cache.pickle.sqlite(
        cache_key=get_cache_key("INFER_MODEL_PROV"),
        compression="lz4",
        log_serde_time=True,
        disable="disable_cache",
        mem_persist=True,
    )
    def run_inference(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        cg = self.get_candidategraph(table)
        psldata = self.get_inference_data(table)

        with self.timer.watch("run inference"):
            logger.debug(
                "Using experiment PSL model: {}", self.cfg.psl.experiment_model
            )
            assert self.cfg.psl.experiment_model == "exp3"
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
                context=context,
                disable_rules=self.cfg.psl.disable_rules,
                rule_weights=dict(self.cfg.psl.rule_weights),
            )
            edge_probs, cta_probs = pslmodel.predict(
                table, cg, data=psldata, verbose=self.verbose, debug=False
            )
            edge_probs = PSLModel.normalize_probs(edge_probs, eps=self.cfg.psl.eps)
            if self.cfg.psl.postprocessing == "steiner_tree":
                pp = SteinerTree(table, cg, edge_probs, self.cfg.psl.threshold)
            elif self.cfg.psl.postprocessing == "arborescence":
                pp = MinimumArborescence(table, cg, edge_probs, self.cfg.psl.threshold)
            elif self.cfg.psl.postprocessing == "simplepath":
                pp = PostProcessingSimplePath(
                    table, cg, edge_probs, self.cfg.psl.threshold
                )
            elif self.cfg.psl.postprocessing == "pairwise":
                pp = PairwiseSelection(table, cg, edge_probs, self.cfg.psl.threshold)
            else:
                raise NotImplementedError(self.cfg.psl.postprocessing)

            pred_cpa = pp.get_result()
            pred_cta = {
                ci: max(classes.items(), key=itemgetter(1))[0]
                for ci, classes in cta_probs.items()
            }

        return edge_probs, cta_probs, pred_cpa, pred_cta

    @Cache.mem(cache_key=get_cache_key())
    def get_entity_labels(self, table: LinkedTable):
        wdentity_ids, wdentities = self.retrieving_entities(table)
        with self.timer.watch("retrieve entity labels"):
            wdentity_labels = self.db.get_entity_labels(wdentities, self.verbose)
        return wdentity_labels

    @Cache.mem(cache_key=get_cache_key())
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

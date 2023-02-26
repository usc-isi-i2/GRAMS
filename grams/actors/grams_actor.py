from __future__ import annotations

from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Union
from grams.actors.augcan_actor import AugCanActor, AugCanParams
from grams.actors.grams_infdata_actor import GramsInfDataActor, InfData
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from grams.algorithm.context import AlgoContext
from grams.algorithm.inferences.psl_gram_model_exp3 import (
    PSLData,
    PSLGramModelExp3,
    PSLGramsModelData3,
)
from grams.algorithm.inferences_v2.features.inf_feature import InfFeature

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
from grams.actors.db_actor import GramsDB, GramsDBActor
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
    features: InfFeature
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
    psl: PslConfig = field(
        default_factory=PslConfig,
        metadata={"help": "Configuration for the PSL model"},
    )


class GramsActor(OsinActor[I.LinkedTable, GramsParams]):
    """GRAMS for Semantic Modeling"""

    VERSION = 107
    EXP_NAME = "Semantic Modeling"
    EXP_VERSION = 3

    def __init__(
        self,
        params: GramsParams,
        db_actor: GramsDBActor,
        preprocess_actor: GramsPreprocessActor,
        infdata_actor: GramsInfDataActor,
    ):
        super().__init__(params, [db_actor, preprocess_actor, infdata_actor])

        self.timer = Timer()
        self.cfg = DEFAULT_CONFIG
        self.db_actor = db_actor
        self.preprocess_actor = preprocess_actor
        self.infdata_actor = infdata_actor

    def run_dataset(self, dsquery: str):
        with self.timer.watch("Preprocess dataset"):
            dsdict = self.preprocess_actor.run_dataset(
                dsquery, self.infdata_actor.params.data_graph.max_n_hop
            )
        with self.timer.watch("Prepare inference data"):
            infdata_dsdict = self.infdata_actor.run_dataset(dsquery)

        output: DatasetDict[list[AnnotationV2]] = DatasetDict(
            dsdict.name, {}, dsdict.provenance
        )
        ref = None
        for name, ds in dsdict.items():
            if len(ds) > 1:
                if ref is None:
                    ref = (ray_put(self.db_actor.db.data_dir), ray_put(self.params))
                lst = ray_map(
                    ray_annotate.remote,
                    [
                        (ref[0], ref[1], ex.table, infdata, False)
                        for ex, infdata in zip(ds, infdata_dsdict[name])
                    ],
                    desc="Annotating tables",
                    verbose=True,
                )
                output[name] = [x[0] for x in lst]
                for x in lst:
                    self.timer.merge(x[1])
            else:
                lst = [
                    annotate(
                        self.db_actor.db,
                        self.params,
                        ex.table,
                        infdata,
                        verbose=True,
                    )
                    for ex, infdata in zip(ds, infdata_dsdict[name])
                ]
                output[name] = [x[0] for x in lst]
                for x in lst:
                    self.timer.merge(x[1])

        return output

    def evaluate(self, eval_args: EvalArgs):
        evalout = {}
        for dsquery in eval_args.dsqueries:
            dsquery_p = DatasetQuery.from_string(dsquery)
            dsdict = self.preprocess_actor.run_dataset(
                dsquery, self.infdata_actor.params.data_graph.max_n_hop
            )
            ann_dsdict = self.run_dataset(dsquery)

            for name, examples in dsdict.items():
                with self.new_exp_run(
                    dataset=dsquery_p.get_query(name),
                ) as exprun:
                    primitive_output, primitive_ex_output = eval_dataset(
                        self.db_actor.db,
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


@ray.remote
def ray_annotate(
    db: Union[GramsDB, Path],
    cfg: GramsParams,
    table: LinkedTable,
    infdata: InfData,
    verbose: bool,
):
    try:
        return annotate(to_grams_db(db), cfg, table, infdata, verbose)
    except Exception as e:
        raise Exception("Failed to annotate table: " + table.id) from e


def annotate(
    db: GramsDB,
    cfg: GramsParams,
    table: LinkedTable,
    infdata: InfData,
    verbose: bool,
):
    timer = Timer()

    assert cfg.psl.experiment_model == "exp3"

    with timer.watch(f"Inference ({cfg.psl.experiment_model})"):
        pslmodel = PSLGramModelExp3(
            disable_rules=cfg.psl.disable_rules,
            rule_weights=dict(cfg.psl.rule_weights),
        )
        edge_probs, cta_probs = pslmodel.predict(
            table, infdata.cg, data=infdata.features, verbose=verbose, debug=False
        )
        edge_probs = PSLModel.normalize_probs(edge_probs, eps=cfg.psl.eps)

    with timer.watch(f"Postprocessing ({cfg.psl.postprocessing})"):
        if cfg.psl.postprocessing == "steiner_tree":
            pp = SteinerTree(table, infdata.cg, edge_probs, cfg.psl.threshold)
        elif cfg.psl.postprocessing == "arborescence":
            pp = MinimumArborescence(table, infdata.cg, edge_probs, cfg.psl.threshold)
        elif cfg.psl.postprocessing == "simplepath":
            pp = PostProcessingSimplePath(
                table, infdata.cg, edge_probs, cfg.psl.threshold
            )
        elif cfg.psl.postprocessing == "pairwise":
            pp = PairwiseSelection(table, infdata.cg, edge_probs, cfg.psl.threshold)
        else:
            raise NotImplementedError(cfg.psl.postprocessing)

        pred_cpa = pp.get_result()
        pred_cta = {
            ci: max(classes.items(), key=itemgetter(1))[0]
            for ci, classes in cta_probs.items()
        }

    with timer.watch(f"Convert results to semantic model"):
        sm_helper = WikidataSemanticModelHelper(
            db.get_auto_cached_entities(table),
            # wdentity_labels,
            {},
            db.wdclasses.cache(),
            db.wdprops.cache(),
        )
        sm = sm_helper.create_sm(table, pred_cpa, pred_cta)
        sm = sm_helper.minify_sm(sm)

    return (
        AnnotationV2(
            sm=sm,
            cg=infdata.cg,
            features=infdata.features,
            cg_edge_probs=edge_probs,
            cta_probs=cta_probs,
            pred_cpa=pred_cpa,
            pred_cta=pred_cta,
        ),
        timer,
    )

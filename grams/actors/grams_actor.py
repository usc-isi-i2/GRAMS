from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from grams.actors.db_actor import GramsDB
from grams.actors.evaluator import Evaluator
from grams.algorithm.literal_matchers import TextParser, TextParserConfigs
from grams.inputs.linked_table import LinkedTable
from ned.actors.evaluate_helper import EvalArgs
import numpy as np

from osin.integrations.ream import OsinActor
from dataclasses import dataclass
from operator import itemgetter
from functools import partial
from pathlib import Path
from typing import Union
from grams.algorithm.candidate_graph.cg_factory import CGFactory
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
from loguru import logger
import ray
from sm.dataset import Example
from sm.misc.ray_helper import get_instance, ray_init, ray_map
from timer import Timer

import grams.inputs as I
from grams.algorithm.data_graph import DGFactory
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption

from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG
from grams.main import Annotation
from grams.algorithm.inferences.psl_config import PslConfig
from grams.actors.dataset_actor import GramsDatasetActor, GramsELDatasetActor
from ream.dataset_helper import DatasetDict, DatasetQuery


@dataclass
class GramsParams:
    data_dir: Path = field(
        metadata={"help": "Path to a directory containing databases"},
    )
    proxy_db: bool = field(
        default=True,
        metadata={"help": "Whether to use a proxy database for the semantic model"},
    )
    data_graph: DGConfigs = field(
        default_factory=DGConfigs,
        metadata={"help": "Configuration for the data graph"},
    )
    text_parser: TextParserConfigs = field(
        default_factory=TextParserConfigs,
        metadata={"help": "Configuration for the text parser"},
    )
    psl: PslConfig = field(
        default_factory=PslConfig,
        metadata={"help": "Configuration for the PSL model"},
    )


class GramsActor(OsinActor[I.LinkedTable, GramsParams]):
    """GRAMS for Semantic Modeling"""

    NAME = "Semantic Modeling"
    VERSION = 100

    def __init__(self, params: GramsParams, linked_dataset_actor: GramsELDatasetActor):
        super().__init__(params, [linked_dataset_actor])
        self.timer = Timer()
        self.cfg = DEFAULT_CONFIG
        self.db = GramsDB(params.data_dir, params.proxy_db)
        self.dataset_actor = linked_dataset_actor

    def run(self, table: LinkedTable, verbose: bool = False):
        return annotate(self.db, self.params, table, verbose)

    def run_dataset(self, dsquery: str):
        ray_init(log_to_driver=False)

        dbref = ray.put(self.db.data_dir)
        cfgref = ray.put(self.params)

        dsdict = self.dataset_actor.run_dataset(dsquery)
        output: DatasetDict[list[Annotation]] = DatasetDict(
            dsdict.name, {}, dsdict.provenance
        )
        for name, ds in dsdict.items():
            args = []
            for example in ds:
                args.append((dbref, cfgref, example.table, False))
            output[name] = ray_map(
                ray_annotate.remote, args, desc="Annotating tables", verbose=True
            )

        return output

    def evaluate(self, eval_args: EvalArgs):
        evaluator = Evaluator(
            self.db.wdentities,
            self.db.wdentity_labels,
            self.db.wdclasses,
            self.db.wdprops,
        )
        sms = []

        predictions: list[
            tuple[
                DatasetDict[list[Example[LinkedTable]]], DatasetDict[list[Annotation]]
            ]
        ] = []
        for dsquery in eval_args.dsqueries:
            dsdict = self.dataset_actor.run_dataset(dsquery)
            ann_dsdict = self.run_dataset(dsquery)
            predictions.append((dsdict, ann_dsdict))

            for name, examples in dsdict.items():
                for e in examples:
                    sms.extend(evaluator.get_equiv_sms(e))
                sms.extend([ann.sm for ann in ann_dsdict[name]])
        evaluator.update_score_fns(sms)

        for di, dsquery in enumerate(eval_args.dsqueries):
            dsquery_p = DatasetQuery.from_string(dsquery)
            dsdict, ann_dsdict = predictions[di]
            for name, examples in dsdict.items():
                eval_outputs = []

                for i, (example, ann) in enumerate(zip(examples, ann_dsdict[name])):
                    try:
                        evalout = evaluator.cpa_cta(example, ann.sm)
                    except:
                        logger.error(
                            "Failed to evaluate example: {} - {}", i, example.table.id
                        )
                        raise

                    eval_outputs.append(
                        (example.table.id, evalout["cpa"], evalout["cta"])
                    )

                with self.new_exp_run(
                    dataset=dsquery_p.get_query(name),
                ) as exprun:
                    if exprun is not None:
                        cpa_precision, cpa_recall = np.mean(
                            [y.precision for x, y, z in eval_outputs]
                        ), np.mean([y.recall for x, y, z in eval_outputs])
                        cpa_f1 = (
                            2
                            * cpa_precision
                            * cpa_recall
                            / max(cpa_precision + cpa_recall, 1e-9)  # type: ignore
                        )

                        cta_precision, cta_recall = np.mean(
                            [z.precision for x, y, z in eval_outputs]
                        ), np.mean([z.recall for x, y, z in eval_outputs])
                        cta_f1 = (
                            2
                            * cta_precision
                            * cta_recall
                            / max(cta_precision + cta_recall, 1e-9)  # type: ignore
                        )

                        exprun.update_output(
                            primitive={
                                "workdir": str(self.get_working_fs().root),
                                "cpa": {
                                    "precision": float(cpa_precision),
                                    "recall": float(cpa_recall),
                                    "f1": float(cpa_f1),
                                },
                                "cta": {
                                    "precision": float(cta_precision),
                                    "recall": float(cta_recall),
                                    "f1": float(cta_f1),
                                },
                            }
                        )
                        print(
                            {
                                "workdir": str(self.get_working_fs().root),
                                "cpa": {
                                    "precision": float(cpa_precision),
                                    "recall": float(cpa_recall),
                                    "f1": float(cpa_f1),
                                },
                                "cta": {
                                    "precision": float(cta_precision),
                                    "recall": float(cta_recall),
                                    "f1": float(cta_f1),
                                },
                            }
                        )


@ray.remote
def ray_annotate(
    db: Union[GramsDB, Path], cfg: GramsParams, table: LinkedTable, verbose: bool
):
    return annotate(db, cfg, table, verbose)


def annotate(
    db: Union[GramsDB, Path], cfg: GramsParams, table: LinkedTable, verbose: bool
) -> Annotation:
    if isinstance(db, Path):
        datadir = db
        db = get_instance(
            lambda: GramsDB(datadir, False),
            "GramsDB",
        )

    timer = Timer()

    with timer.watch("retrieving entities"):
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

    with timer.watch("build kg object index"):
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
        dg_factory = DGFactory(wdentities, wdprops, text_parser, cfg.data_graph)
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
            logger.debug("Using experiment PSL model")
            cls = partial(
                {
                    "exp": PSLGramModelExp,
                    "exp2": PSLGramModelExp2,
                }[cfg.psl.experiment_model],
                wdprop_domains=db.wdprop_domains,
                wdprop_ranges=db.wdprop_ranges,
            )
        else:
            cls = PSLGramModel

        edge_probs, cta_probs = cls(
            wdentities=wdentities,
            wdentity_labels=db.wdentity_labels,
            wdclasses=wdclasses,
            wdprops=wdprops,
            wd_numprop_stats=db.wd_numprop_stats,
            disable_rules=cfg.psl.disable_rules,
        ).predict(table, cg, dg, verbose=verbose, debug=False)

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
    return Annotation(
        sm=sm,
        dg=dg,
        cg=cg,
        cg_edge_probs=edge_probs,
        cta_probs=cta_probs,
        pred_cpa=pred_cpa,
        pred_cta=pred_cta,
    )

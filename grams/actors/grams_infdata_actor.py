from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union
from loguru import logger

from osin.integrations.ream import OsinActor
from ream.data_model_helper import NumpyDataModelContainer
from ream.prelude import ActorVersion, Cache, DatasetDict, DatasetQuery

from grams.actors.actor_helpers import EvalArgs
from grams.actors.db_actor import GramsDB, GramsDBActor, to_grams_db
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.data_graph.dg_factory import DGFactory
from grams.algorithm.inferences_v2.features.inf_feature import (
    InfFeature,
    InfFeatureExtractor,
)
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption
from grams.algorithm.literal_matchers.literal_match import (
    LiteralMatch,
    LiteralMatchConfigs,
)
from grams.algorithm.literal_matchers.text_parser import TextParser, TextParserConfigs
from grams.inputs.linked_table import LinkedTable
from sm.misc.ray_helper import enhance_error_info, ray_map, ray_put, track_runtime
from timer import watch_and_report


logger = logger.bind(name=__name__)


@dataclass
class GramsInfDataParams:
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


@dataclass
class InfData(NumpyDataModelContainer):
    cg: CGGraph
    feat: InfFeature


class InfDataDatasetDict(DatasetDict[list[InfData]]):
    serde = (InfData.batch_save, InfData.batch_load, None)


class GramsInfDataActor(OsinActor[LinkedTable, GramsInfDataParams]):
    VERSION = ActorVersion.create(109, [InfFeatureExtractor])

    def __init__(
        self,
        params: GramsInfDataParams,
        dbactor: GramsDBActor,
        preprocess_actor: GramsPreprocessActor,
    ):
        super().__init__(params, [dbactor, preprocess_actor])
        self.dbactor = dbactor
        self.preprocess_actor = preprocess_actor

    @Cache.cls.dir(
        cls=InfDataDatasetDict,
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str) -> InfDataDatasetDict:
        dsdict = self.preprocess_actor.run_dataset(
            dsquery, self.params.data_graph.max_n_hop
        )
        newdsdict: InfDataDatasetDict = InfDataDatasetDict(
            dsdict.name,
            {},
            dsdict.provenance,
        )

        using_ray = sum(len(ds) for ds in dsdict.values()) > 1
        dbref = ray_put(self.dbactor.db.data_dir) if using_ray else self.dbactor.db
        paramref = ray_put(self.params) if using_ray else self.params

        for name, ds in dsdict.items():
            newds = ray_map(
                create_inference_data,
                [(dbref, paramref, ex.table) for ex in ds],
                desc=f"Creating inference data",
                verbose=True,
                using_ray=using_ray,
                is_func_remote=False,
            )

            id2cg = {}
            for ex, (cg, feat) in zip(ds, newds):
                id2cg[ex.table.id] = cg

            newdsdict[name] = [InfData(cg, feat) for cg, feat in newds]
        return newdsdict

    def evaluate(self, eval_args: EvalArgs):
        evalout = {}
        for dsquery in eval_args.dsqueries:
            dsquery_p = DatasetQuery.from_string(dsquery)
            dsdict = self.preprocess_actor.run_dataset(
                dsquery, self.params.data_graph.max_n_hop
            )
            infdata_dsdict = self.run_dataset(dsquery)
            # for name, examples in dsdict.items():
            #     with self.new_exp_run(
            #         dataset=dsquery_p.get_query(name),
            #     ) as exprun:
            #         pass

        return evalout


# @track_runtime("2.id", "/data/binhvu/sm-dev/data/home/timetrack.sqlite")
@enhance_error_info("2.id")
def create_inference_data(
    db: Union[GramsDB, Path],
    params: GramsInfDataParams,
    table: LinkedTable,
    verbose: bool = True,
):
    db = to_grams_db(db)

    with watch_and_report(
        "load data", preprint=True, print_fn=logger.debug, disable=not verbose
    ):
        wdentity_ids, wdentities = db.get_table_entities(
            table, params.data_graph.max_n_hop, verbose
        )
        wdentity_labels = db.get_table_entity_labels(
            table, params.data_graph.max_n_hop, verbose
        )
    wdclasses = db.wdclasses.cache()
    wdprops = db.wdprops.cache()

    with watch_and_report(
        "build index", preprint=True, print_fn=logger.debug, disable=not verbose
    ):
        kg_object_index = KGObjectIndex.from_entities(
            list(wdentity_ids.intersection(wdentities.keys())),
            wdentities,
            wdprops,
            n_hop=params.data_graph.max_n_hop,
            traversal_option=TraversalOption.TransitiveOnly,
        )
    text_parser = TextParser(params.text_parser)
    literal_match = LiteralMatch(wdentities, params.literal_matchers)

    with watch_and_report(
        "build data graph", preprint=True, print_fn=logger.debug, disable=not verbose
    ):
        dg_factory = DGFactory(
            wdentities,
            wdprops,
            text_parser,
            literal_match,
            params.data_graph,
        )
        dg = dg_factory.create_dg(
            table, kg_object_index, max_n_hop=params.data_graph.max_n_hop
        )

    with watch_and_report(
        "build candidate graph",
        preprint=True,
        print_fn=logger.debug,
        disable=not verbose,
    ):
        cg_factory = CGFactory(
            wdentities,
            wdentity_labels,
            wdclasses,
            wdprops,
        )
        cg = cg_factory.create_cg(table, dg)

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

    return cg, InfFeatureExtractor(context).extract(table, dg, cg, verbose=verbose)

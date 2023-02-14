from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from grams.actors.augcan_actor import AugCanActor
from grams.actors.db_actor import GramsDB, GramsDBActor, to_grams_db
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from grams.algorithm.candidate_graph.cg_factory import CGFactory
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.data_graph.dg_factory import DGFactory
from grams.algorithm.inferences.psl_gram_model_exp3 import PSLData, PSLGramsModelData3
from grams.algorithm.kg_index import KGObjectIndex, TraversalOption
from grams.algorithm.literal_matchers.literal_match import (
    LiteralMatch,
    LiteralMatchConfigs,
)
from grams.algorithm.literal_matchers.text_parser import TextParser, TextParserConfigs
from grams.inputs.linked_table import LinkedTable
from ned.actors.evaluate_helper import EvalArgs
from osin.integrations.ream import OsinActor
import ray
from ream.actors.base import BaseActor
from ream.cache_helper import Cache, Cacheable
from ream.dataset_helper import DatasetDict, DatasetQuery
from sm.misc.ray_helper import ray_map
from grams.algorithm.inferences_v2.features.feature import (
    InfFeatureExtractor,
    InfFeature,
    InfBatchFeature,
)


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
class InfData(InfBatchFeature):
    cgs: dict[str, CGGraph]

    @staticmethod
    def from_inf_batch_feature(cgs: dict[str, CGGraph], feat: InfBatchFeature):
        return InfData(
            cgs=cgs,
            idmap=feat.idmap,
            edge_features=feat.edge_features,
            node_features=feat.node_features,
            index=feat.index,
        )


class InfDatasetDict(DatasetDict[InfData]):
    serde = (InfData.save, InfData.load, None)


class GramsInfDataActor(OsinActor[LinkedTable, GramsInfDataParams]):
    VERSION = 101

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
        cls=InfDatasetDict,
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str) -> InfDatasetDict:
        dsdict = self.preprocess_actor.run_dataset(
            dsquery, self.params.data_graph.max_n_hop
        )
        newdsdict: InfDatasetDict = InfDatasetDict(
            dsdict.name, {}, dsdict.provenance + ";infdata"
        )

        dbref = None
        paramref = None

        for name, ds in dsdict.items():
            if len(ds) > 1:
                if dbref is None:
                    dbref = ray.put(self.dbactor.db.data_dir)
                    paramref = ray.put(self.params)
                newds = ray_map(
                    ray_create_inference_data.remote,
                    [(dbref, paramref, ex.table) for ex in ds],
                    desc=f"Creating inference data",
                    verbose=True,
                )
            else:
                newds = [
                    create_inference_data(self.dbactor.db, self.params, ex.table)
                    for ex in ds
                ]

            id2cg = {}
            for ex, (cg, feat) in zip(ds, newds):
                id2cg[ex.table.id] = cg

            newdsdict[name] = InfData.from_inf_batch_feature(
                id2cg,
                InfBatchFeature.merge(
                    [(ex.table.id, cg_feat[1]) for ex, cg_feat in zip(ds, newds)]
                ),
            )
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


@ray.remote
def ray_create_inference_data(
    db: Union[GramsDB, Path], params: GramsInfDataParams, table: LinkedTable
):
    db = to_grams_db(db)
    try:
        return create_inference_data(db, params, table)
    except Exception as e:
        raise Exception("Failed to extract inference data: " + table.id) from e


def create_inference_data(
    db: GramsDB, params: GramsInfDataParams, table: LinkedTable, verbose: bool = True
):
    wdentity_ids, wdentities = db.get_table_entities(
        table, params.data_graph.max_n_hop, verbose
    )
    wdentity_labels = db.get_table_entity_labels(
        table, params.data_graph.max_n_hop, verbose
    )
    wdclasses = db.wdclasses.cache()
    wdprops = db.wdprops.cache()

    kg_object_index = KGObjectIndex.from_entities(
        list(wdentity_ids.intersection(wdentities.keys())),
        wdentities,
        wdprops,
        n_hop=params.data_graph.max_n_hop,
        traversal_option=TraversalOption.TransitiveOnly,
    )
    text_parser = TextParser(params.text_parser)
    literal_match = LiteralMatch(wdentities, params.literal_matchers)

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

    return cg, InfFeatureExtractor(context).extract(table, dg, cg)

    # psldata = PSLGramsModelData3(context, use_readable_idmap=True).extract_data(
    #     table, cg, dg
    # )

    # return InfData(cg, psldata)

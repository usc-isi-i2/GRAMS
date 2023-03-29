from __future__ import annotations

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING, Literal

from nptyping import Float32, Float64, NDArray, Shape
from osin.integrations.ream import OsinActor
from ream.cache_helper import Cache, CacheArgsHelper
from ream.data_model_helper import NumpyDataModel, NumpyDataModelContainer
from ream.dataset_helper import DatasetQuery
from ream.prelude import DatasetDict, EnumParams
from serde.helper import orjson_dumps
from timer import Timer
from tqdm import tqdm

from grams.actors.actor_helpers import EvalArgs, eval_dataset
from grams.actors.db_actor import GramsDBActor
from grams.actors.grams_actor import AnnotationV2
from grams.actors.grams_infdata_actor import GramsInfDataActor, InfData
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from grams.algorithm.inferences_v2.ind.ind_model import IndConfig, IndModel
from grams.algorithm.inferences_v2.psl.config import PslConfig
from grams.algorithm.inferences_v2.psl.model_v3 import PSLModelv3
from grams.algorithm.postprocessing.steiner_tree import SteinerTree
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.evaluation.evaluator import Evaluator
from grams.inputs.linked_table import LinkedTable
from sm.dataset import Example
from sm.misc.ray_helper import enhance_error_info, ray_map, ray_put
from sm.outputs.semantic_model import SemanticModel


@dataclass
class GramsInfParams(EnumParams):
    method: Literal["psl", "ind"] = field(
        default="psl",
        metadata={
            "help": "The inference method to use",
            "variants": {
                "psl": PSLModelv3,
                "ind": IndModel,
            },
        },
    )
    psl: PslConfig = field(
        default_factory=PslConfig,
        metadata={"help": "Configuration for the PSL model"},
    )
    ind: IndConfig = field(
        default_factory=IndConfig,
        metadata={"help": "Configuration for the Ind model"},
    )


class InfProb(NumpyDataModel):
    __slots__ = ["prob"]
    prob: NDArray[Shape["*"], Float64]


InfProb.init()


@dataclass
class InfResult(NumpyDataModelContainer):
    edge_prob: InfProb
    node_prob: InfProb


class InfResDatasetDict(DatasetDict[list[InfResult]]):
    serde = (InfResult.batch_save, InfResult.batch_load, None)


class GramsInfActor(OsinActor[LinkedTable, GramsInfParams]):
    """GRAMS' Inference Step"""

    VERSION = 101
    EXP_NAME = "Semantic Modeling Inference"
    EXP_VERSION = 4

    def __init__(
        self,
        params: GramsInfParams,
        db_actor: GramsDBActor,
        preprocess_actor: GramsPreprocessActor,
        infdata_actor: GramsInfDataActor,
    ):
        super().__init__(params, [db_actor, preprocess_actor, infdata_actor])

        self.timer = Timer()
        self.db_actor = db_actor
        self.preprocess_actor = preprocess_actor
        self.infdata_actor = infdata_actor

        self.db = db_actor.db

    def get_provenance(self):
        if self.params.method == "psl":
            return f"inf:psl={PSLModelv3.VERSION}"
        if self.params.method == "ind":
            return f"inf:ind={IndModel.VERSION}"
        raise NotImplementedError()

    @Cache.cls.dir(
        cls=InfResDatasetDict,
        cache_self_args=CacheArgsHelper.gen_cache_self_args(get_provenance),
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str) -> InfResDatasetDict:
        with self.timer.watch("Preprocess dataset"):
            dsdict = self.preprocess_actor.run_dataset(
                dsquery, self.infdata_actor.params.data_graph.max_n_hop
            )

        with self.timer.watch("Prepare inference data"):
            infdata_dsdict = self.infdata_actor.run_dataset(dsquery)

        with self.timer.watch(f"Inference ({self.params.method})"):
            out_dsdict = InfResDatasetDict(
                infdata_dsdict.name,
                {},
                self._fmt_prov(infdata_dsdict.provenance, self.get_provenance()),
            )

            method = self.get_trained_method()
            use_ray = sum(len(infdatas) for infdatas in infdata_dsdict.values()) > 1
            methodref = ray_put(method) if use_ray else method

            for name, infdatas in infdata_dsdict.items():
                exds = dsdict[name]
                out_dsdict[name] = ray_map(
                    predict,
                    [(methodref, exds[i].table.id, x) for i, x in enumerate(infdatas)],
                    desc=f"Run inference ({name})",
                    verbose=True,
                    using_ray=use_ray,
                    is_func_remote=False,
                )
        return out_dsdict

    def get_trained_method(self):
        if self.params.method == "psl":
            return PSLModelv3(self.params.psl, temp_dir=None)
        if self.params.method == "ind":

            def dsquery(query: str):
                with self.timer.watch("Preprocess dataset"):
                    dsdict = self.preprocess_actor.run_dataset(
                        query, self.infdata_actor.params.data_graph.max_n_hop
                    )

                with self.timer.watch("Prepare inference data"):
                    infdata_dsdict = self.infdata_actor.run_dataset(query).map(
                        lambda lst: [x.feat for x in lst]
                    )
                return infdata_dsdict, dsdict

            evaluator = Evaluator(
                self.db.wdentities.cache(),
                self.db.wdentity_labels.cache(),
                self.db.wdclasses.cache(),
                self.db.wdprops.cache(),
                self.db.data_dir,
            )
            model = IndModel(self.params.ind)
            model.train(self.params.ind.train_args, dsquery, evaluator)
            return model
        raise NotImplementedError()

    def evaluate(self, eval_args: EvalArgs):
        evalout = {}
        for dsquery in eval_args.dsqueries:
            dsquery_p = DatasetQuery.from_string(dsquery)

            infres_dsdict = self.run_dataset(dsquery)
            dsdict = self.preprocess_actor.run_dataset(
                dsquery, self.infdata_actor.params.data_graph.max_n_hop
            )
            infdata_dsdict = self.infdata_actor.run_dataset(dsquery)

            for name, examples in dsdict.items():
                with self.new_exp_run(
                    dataset=dsquery_p.get_query(name),
                ) as exprun:
                    anns = self._normalize_results(
                        examples,
                        infdata_dsdict[name],
                        infres_dsdict[name],
                        threshold=eval_args.threshold,
                    )
                    primitive_output, primitive_ex_output = eval_dataset(
                        self.db,
                        examples,
                        pred_sms=[ann.sm for ann in anns],
                        anns=anns,
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

    def _normalize_results(
        self,
        examples: list[Example[LinkedTable]],
        infdatas: list[InfData],
        infress: list[InfResult],
        threshold: float,
    ) -> list[AnnotationV2]:
        out = []

        wdclasses = self.db.wdclasses.cache()
        wdprops = self.db.wdprops.cache()

        for ex, infdata, infres in tqdm(zip(examples, infdatas, infress)):
            ex_id = ex.table.id

            with self.timer.watch(f"Postprocessing (SteinerTree)"):
                cg = infdata.cg
                idmap = infdata.feat.idmap
                estart, eend = 0, len(infdata.feat.edge_features)
                eprobs = infres.edge_prob.prob
                edge_feats = infdata.feat.edge_features
                output_edge_probs: dict[tuple[str, str, str], float] = {}

                for i in range(estart, eend):
                    u = idmap.im(edge_feats.source[i])
                    v = idmap.im(edge_feats.target[i])
                    s = idmap.im(edge_feats.statement[i])
                    outedge = idmap.im(edge_feats.outprop[i])
                    if edge_feats.inprop[i] == edge_feats.outprop[i]:
                        output_edge_probs[u, s, outedge] = eprobs[i]
                    output_edge_probs[s, v, outedge] = eprobs[i]

                ustart, uend = 0, len(infdata.feat.node_features)
                uprobs = infres.node_prob.prob
                node_feats = infdata.feat.node_features
                output_node_probs: dict[int, dict[str, float]] = defaultdict(dict)

                for i in range(ustart, uend):
                    u = cg.get_column_node(idmap.im(node_feats.node[i])).column
                    t = idmap.im(node_feats.type[i])
                    output_node_probs[u][t] = uprobs[i]

                pp = SteinerTree(ex.table, infdata.cg, output_edge_probs, threshold)
                pred_cpa = pp.get_result()
                pred_cta = {
                    ci: max(classes.items(), key=itemgetter(1))[0]
                    for ci, classes in output_node_probs.items()
                }

            with self.timer.watch(f"Convert results to semantic model"):
                sm_helper = WikidataSemanticModelHelper(
                    self.db.get_auto_cached_entities(ex.table),
                    # wdentity_labels,
                    {},
                    wdclasses,
                    wdprops,
                )
                sm = sm_helper.create_sm(ex.table, pred_cpa, pred_cta)
                sm = sm_helper.minify_sm(sm)

            out.append(
                AnnotationV2(
                    sm=sm,
                    cg=cg,
                    features=infdata.feat,
                    cg_edge_probs=output_edge_probs,
                    cta_probs=output_node_probs,
                    pred_cpa=pred_cpa,
                    pred_cta=pred_cta,
                )
            )

        return out


@enhance_error_info(lambda method, id, input: id)
def predict(method: PSLModelv3, id, input: InfData) -> InfResult:
    if isinstance(method, PSLModelv3):
        # set different temp dir for each example
        method.model.temp_dir = f"/tmp/pslpython/{id}"
        if os.path.exists(method.model.temp_dir):
            shutil.rmtree(method.model.temp_dir)

    edge_probs, node_probs = method.predict(input.feat)
    return InfResult(InfProb(edge_probs), InfProb(node_probs))

from __future__ import annotations
from collections import defaultdict

from dataclasses import dataclass, field
from operator import itemgetter
from typing import Literal, TYPE_CHECKING
from grams.actors.actor_helpers import eval_dataset
from grams.actors.db_actor import GramsDBActor
from grams.actors.grams_actor import AnnotationV2
from grams.actors.grams_infdata_actor import GramsInfDataActor, InfData
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from grams.algorithm.inferences_v2.psl.model_v3 import PSLModelv3
from grams.algorithm.inferences_v2.psl.config import PslConfig
from grams.algorithm.postprocessing.steiner_tree import SteinerTree
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.inputs.linked_table import LinkedTable
from nptyping import Float32, Float64, NDArray, Shape
from osin.integrations.ream import OsinActor
from ream.cache_helper import Cache, CacheArgsHelper
from ream.data_model_helper import NumpyDataModel, NumpyDataModelContainer
from ream.dataset_helper import DatasetQuery
from ream.prelude import DatasetDict, EnumParams
from serde.helper import orjson_dumps
from sm.dataset import Example
from sm.outputs.semantic_model import SemanticModel
from timer import Timer
from tqdm import tqdm


if TYPE_CHECKING:
    from grams.actors.__main__ import EvalArgs


@dataclass
class GramsInfParams(EnumParams):
    method: Literal["psl"] = field(
        default="psl",
        metadata={
            "help": "The inference method to use",
            "variants": {
                "psl": PSLModelv3,
            },
        },
    )
    psl: PslConfig = field(
        default_factory=PslConfig,
        metadata={"help": "Configuration for the PSL model"},
    )


class InfProb(NumpyDataModel):
    __slots__ = ["prob"]
    prob: NDArray[Shape["*"], Float64]


InfProb.init()


@dataclass
class InfResult(NumpyDataModelContainer):
    edge_prob: InfProb
    node_prob: InfProb


class InfResDatasetDict(DatasetDict[InfResult]):
    serde = (InfResult.save, InfResult.load, None)


class GramsInfActor(OsinActor[LinkedTable, GramsInfParams]):
    """GRAMS' Inference Step"""

    VERSION = 100
    EXP_NAME = "Semantic Modeling Inference"
    EXP_VERSION = 3

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

            for name, infdata in infdata_dsdict.items():
                edge_probs, node_probs = method.predict(infdata)
                out_dsdict[name] = InfResult(InfProb(edge_probs), InfProb(node_probs))

        return out_dsdict

    def get_trained_method(self):
        if self.params.method == "psl":
            return PSLModelv3(self.params.psl, self.get_working_fs().root / "psl")
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
        infdata: InfData,
        infres: InfResult,
        threshold: float,
    ) -> list[AnnotationV2]:
        out = []

        wdclasses = self.db.wdclasses.cache()
        wdprops = self.db.wdprops.cache()

        for ex in tqdm(examples):
            ex_id = ex.table.id

            with self.timer.watch(f"Postprocessing (SteinerTree)"):
                cg = infdata.cgs[ex_id]
                idmap = infdata.idmap[ex_id]
                estart, eend = infdata.edge_index[ex_id]
                eprobs = infres.edge_prob.prob
                edge_feats = infdata.edge_features
                output_edge_probs: dict[tuple[str, str, str], float] = {}

                for i in range(estart, eend):
                    u = idmap.im(edge_feats.source[i])
                    v = idmap.im(edge_feats.target[i])
                    s = idmap.im(edge_feats.statement[i])
                    outedge = idmap.im(edge_feats.outprop[i])
                    if edge_feats.inprop[i] == edge_feats.outprop[i]:
                        output_edge_probs[u, s, outedge] = eprobs[i]
                    output_edge_probs[s, v, outedge] = eprobs[i]

                ustart, uend = infdata.node_index[ex_id]
                uprobs = infres.node_prob.prob
                node_feats = infdata.node_features
                output_node_probs: dict[int, dict[str, float]] = defaultdict(dict)

                for i in range(ustart, uend):
                    u = cg.get_column_node(idmap.im(node_feats.node[i])).column
                    t = idmap.im(node_feats.type[i])
                    output_node_probs[u][t] = uprobs[i]

                pp = SteinerTree(
                    ex.table, infdata.cgs[ex_id], output_edge_probs, threshold
                )
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
                    features=infdata,
                    cg_edge_probs=output_edge_probs,
                    cta_probs=output_node_probs,
                    pred_cpa=pred_cpa,
                    pred_cta=pred_cta,
                )
            )

        return out

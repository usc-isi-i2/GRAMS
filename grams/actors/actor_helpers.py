from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field

from pathlib import Path
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode
from typing import List, Optional, Union, TYPE_CHECKING
from grams.evaluation.autolabeler import AutoLabeler
from grams.algorithm.context import AlgoContext
from ned.metrics import PrecisionRecallF1
import numpy as np
from loguru import logger
from osin.apis.remote_exp import RemoteExpRun
from osin.types.pyobject import OHTML, OTable
from osin.types.pyobject.html import OListHTML

from grams.actors.db_actor import GramsDB, to_grams_db
from grams.evaluation.evaluator import CPAEvalRes, Evaluator
from grams.inputs.linked_table import LinkedTable
import ray
from sm.dataset import Example
from sm.evaluation.cpa_cta_metrics import CTAEvalOutput
from sm.misc.fn_cache import CacheMethod
from sm.misc.ray_helper import get_instance, ray_map, enhance_error_info
from sm.namespaces.namespace import OutOfNamespace
from sm.namespaces.wikidata import WikidataNamespace
from sm.outputs.semantic_model import (
    ClassNode,
    Edge,
    LiteralNode,
    LiteralNodeDataType,
    SemanticModel,
)
from grams.algorithm.inferences.psl_gram_model_exp3 import P
from tqdm import tqdm


if TYPE_CHECKING:
    from grams.actors.grams_actor import AnnotationV2


@dataclass
class EvalArgs:
    dsqueries: List[str] = field(
        metadata={"help": "List of dataset queries to evaluate"}
    )
    threshold: float = 0.5


def eval_dataset(
    db: GramsDB,
    examples: list[Example[LinkedTable]],
    pred_sms: list[SemanticModel],
    anns: Optional[list[AnnotationV2]] = None,
    exprun: Optional[RemoteExpRun] = None,
):
    wdentities = db.get_auto_cached_entities(None)
    wdentity_labels = db.wdentity_labels.cache()
    wdclasses = db.wdclasses.cache()
    wdprops = db.wdprops.cache()

    evaluator = Evaluator(wdentities, wdentity_labels, wdclasses, wdprops, db.data_dir)

    hits = [3, 5, 10]

    cpas: list[CPAEvalRes] = []
    ctas: list[CTAEvalOutput] = []
    cpa_at_k: dict[str, list[PrecisionRecallF1]] = defaultdict(list)
    cta_at_k: dict[str, list[PrecisionRecallF1]] = defaultdict(list)
    ceas = []
    for i, (example, sm) in enumerate(zip(examples, pred_sms)):
        try:
            cpas.append(evaluator.cpa(example, sm))
        except:
            logger.error("Failed to evaluate example: {} - {}", i, example.table.id)
            raise

        if anns is not None:
            for k, v in evaluator.cta_at_k(example, anns[i].cta_probs, k=hits)[
                0
            ].items():
                cta_at_k[k].append(v)
            for k, v in evaluator.cpa_at_k(
                example, anns[i].cg, anns[i].cg_edge_probs, k=hits
            )[0].items():
                cpa_at_k[k].append(v)

        ctas.append(evaluator.cta(example, sm))
        ceas.append(evaluator.cea(example, k=[1, 5]))

    avg_cpa = PrecisionRecallF1.avg([cpa.result for cpa in cpas])
    avg_cta = PrecisionRecallF1.avg(ctas)
    avg_cta_k = {k: PrecisionRecallF1.avg(v) for k, v in cta_at_k.items()}
    avg_cpa_k = {k: PrecisionRecallF1.avg(v) for k, v in cpa_at_k.items()}

    # calculate cea macro scores
    cea_macro = {}
    cea_micro = {}

    for t in ceas:
        for name, value in t["value"].items():
            if name not in cea_macro:
                cea_macro[name] = {k: [v] for k, v in value.iter_prf()}
                cea_micro[name] = t["confusion_matrix"][name]
                continue

            for k, v in value.iter_prf():
                cea_macro[name][k].append(v)
            cea_micro[name] = cea_micro[name] + t["confusion_matrix"][name]
    for name, value in cea_macro.items():
        cea_macro[name] = {k: float(np.mean(v)) for k, v in value.items()}
        cea_micro[name] = {
            "precision": cea_micro[name].precision(),
            "recall": cea_micro[name].recall(),
            "f1": cea_micro[name].f1(),
        }

    ex_details = []
    for i, e in enumerate(examples):
        ex_details.append(
            {
                "cpa": {
                    "normal": {
                        "p": cpas[i].result.precision,
                        "r": cpas[i].result.recall,
                        "f1": cpas[i].result.f1,
                    },
                    **{
                        k: {
                            "p": v[i].precision,
                            "r": v[i].recall,
                            "f1": v[i].f1,
                        }
                        for k, v in cpa_at_k.items()
                    },
                },
                "cta": {
                    "normal": {
                        "p": ctas[i].precision,
                        "r": ctas[i].recall,
                        "f1": ctas[i].f1,
                    },
                    **{
                        k: {
                            "p": v[i].precision,
                            "r": v[i].recall,
                            "f1": v[i].f1,
                        }
                        for k, v in cta_at_k.items()
                    },
                },
                "cea": {
                    name: {
                        "p": value.precision,
                        "r": value.recall,
                        "f1": value.f1,
                    }
                    for name, value in ceas[i]["value"].items()
                },
            }
        )

    if exprun is not None:
        # log the results of each example
        # if len(examples) > 1:
        #     complex_objs = ray_map(
        #         ray_extract_complex_objects.remote,
        #         [
        #             (db.data_dir, e, pred_sms[i], anns[i] if anns is not None else None)
        #             for i, e in enumerate(examples)
        #         ],
        #         desc="creating annotation debug info.",
        #         verbose=True,
        #     )
        # else:
        complex_objs = [
            extract_complex_objects(
                db, e, pred_sms[i], anns[i] if anns is not None else None
            )
            for i, e in enumerate(tqdm(examples))
        ]
        for i, e in enumerate(examples):
            exprun.update_example_output(
                example_id=str(i),
                example_name=e.table.id,
                primitive=ex_details[i],
                complex=complex_objs[i],
            )

    # fmt: off
    logger.info(
        "for copying...\nrun-id\tcpa-p\tcpa-r\tcpa-f1\tcta-p\tcta-r\tcta-f1\n{}",
        ",".join(
            [str(0 if exprun is None else exprun.id)] +
            ["%.2f" % (round(float(x) * 100, 2)) for x in [avg_cpa.precision, avg_cpa.recall, avg_cpa.f1, avg_cta.precision, avg_cta.recall, avg_cta.f1]]
    ))
    # fmt: on

    return {
        "cpa": {
            "normal": {
                "precision": float(avg_cpa.precision),
                "recall": float(avg_cpa.recall),
                "f1": float(avg_cpa.f1),
            },
            **{
                k: {
                    "precision": float(avg_cpa_k[k].precision),
                    "recall": float(avg_cpa_k[k].recall),
                    "f1": float(avg_cpa_k[k].f1),
                }
                for k in [f"perf@{kp}" for kp in hits] + ["perf@all"]
                if k in avg_cpa_k
            },
        },
        "cta": {
            "normal": {
                "precision": float(avg_cta.precision),
                "recall": float(avg_cta.recall),
                "f1": float(avg_cta.f1),
            },
            **{
                k: {
                    "precision": float(avg_cta_k[k].precision),
                    "recall": float(avg_cta_k[k].recall),
                    "f1": float(avg_cta_k[k].f1),
                }
                for k in [f"perf@{kp}" for kp in hits] + ["perf@all"]
                if k in avg_cta_k
            },
        },
        "cea": {
            "macro": cea_macro,
            "micro": cea_micro,
        },
    }, ex_details


class AuxComplexObjectBase:
    def __init__(self, context: AlgoContext) -> None:
        self.context = context
        self.wdns = WikidataNamespace.create()
        self.wdentities = context.wdentities
        self.wdprops = context.wdprops

    def get_ent_label(self, entid: str):
        if entid not in self.wdentities:
            return entid
        return f"{self.wdentities[entid].label} ({entid})"

    def get_prop_label(self, propid: str):
        if propid not in self.wdprops:
            return propid
        return f"{self.wdprops[propid].label} ({propid})"

    def get_ent_label_with_description(self, entid: str):
        if entid not in self.wdentities:
            return entid
        ent = self.wdentities[entid]
        return f"{ent.label} ({entid}): {ent.description}"


class AuxComplexTableObject(AuxComplexObjectBase):
    def get_table(self, example: Example[LinkedTable]) -> OTable:
        nrows, ncols = example.table.table.shape()
        columns = [
            f"{ci}. " + (c.clean_name or "")
            for ci, c in enumerate(example.table.table.columns)
        ]
        return OTable(
            [
                {columns[ci]: self._get_cell(example, ri, ci) for ci in range(ncols)}
                for ri in range(nrows)
            ]
        )

    def _get_cell(self, example: Example[LinkedTable], row: int, col: int):
        from htbuilder import a, b, code, div, li, pre, span, ul

        links = example.table.links[row, col]
        if len(links) == 0:
            return str(example.table.table[row, col])

        cell = str(example.table.table[row, col])
        ent_el = []
        can_el = []
        for link in links:
            x = div(style="margin-left: 8px")(
                "-&nbsp;",
                a(href=link.url)(f"{cell[link.start:link.end]}: "),
                *[
                    span(
                        ", " if i > 0 else "",
                        a(href=self.wdns.get_entity_abs_uri(entid))(entid),
                    )
                    for i, entid in enumerate(link.entities)
                ],
            )
            popx = div(
                self.get_ent_label_with_description(entid) for entid in link.entities
            )
            ent_el.append(OHTML(str(x), str(popx)))

            gold_ents = set(link.entities)

            # showing the top 5 candidates, as longer won't give more info.
            discans = [(i, can) for i, can in enumerate(link.candidates)][:5]
            if all(can.entity_id not in gold_ents for _, can in discans):
                # add missing pred gold
                discans.extend(
                    [
                        (i, can)
                        for i, can in enumerate(link.candidates)
                        if can.entity_id in gold_ents
                    ]
                )

            for i, canid in discans:
                x = div(style="margin-left: 8px")(
                    f"- {i}.&nbsp;",
                    span(
                        "(",
                        code(style="color: green; font-weight: bold")("C"),
                        ")&nbsp;",
                    )
                    if canid.entity_id in gold_ents
                    else "",
                    a(href=self.wdns.get_entity_abs_uri(canid.entity_id))(
                        canid.entity_id
                    ),
                    ": ",
                    span(title=canid.probability)(round(canid.probability, 4)),
                )
                popx = self.get_ent_label_with_description(canid.entity_id)
                can_el.append(OHTML(str(x), str(popx)))

        items = [
            OHTML(
                str(
                    span(
                        b("Cell: "),
                        cell,
                    )
                ),
            ),
            OHTML(str(b("Ground-truth: "))),
            *ent_el,
            OHTML(str(b("Candidates: "))),
            *can_el,
            # OHTML(
            #     str(
            #         div(
            #             *[b("Candidates: "), ul(can_el)] if len(can_el) > 0 else [],
            #         )
            #     )
            # ),
        ]
        return OListHTML(items)


class AuxComplexSmObject(AuxComplexObjectBase):
    def get_sm(self, example: Example[LinkedTable], sm: SemanticModel) -> OHTML:
        html = (
            sm.deep_copy()
            .add_readable_label(self.update_readable_label)
            .print(env="browser")
        )
        assert html is not None
        return OHTML(html)

    def get_sms(self, example: Example[LinkedTable], sms: list[SemanticModel]) -> OHTML:
        from htbuilder import b, div

        htmls = []
        for i, sm in enumerate(sms):
            html = (
                sm.deep_copy()
                .add_readable_label(self.update_readable_label)
                .print(env="browser")
            )
            assert html is not None
            htmls.append(b(f"Semantic Model {i}:"))
            htmls.append(html)

        return OHTML(str(div(*htmls)))

    def update_readable_label(self, item: ClassNode | LiteralNode | Edge):
        if isinstance(item, (ClassNode, Edge)):
            abs_uri = item.abs_uri
        elif (
            isinstance(item, LiteralNode)
            and item.datatype == LiteralNodeDataType.Entity
        ):
            abs_uri = item.value
        else:
            return

        try:
            entid = self.wdns.get_entity_id(abs_uri)
        except OutOfNamespace:
            try:
                entid = self.wdns.get_prop_id(abs_uri)
            except OutOfNamespace:
                return

        if entid in self.wdentities:
            item.readable_label = self.get_ent_label(entid)


class AuxComplexFeatures(AuxComplexObjectBase):
    def __init__(self, context: AlgoContext, datadir: Path):
        super().__init__(context)
        self.autolabeler = AutoLabeler(
            Evaluator(
                context.wdentities,
                context.wdentity_labels,
                context.wdclasses,
                context.wdprops,
                datadir,
            )
        )

    def get_structure_features(
        self,
        example: Example[LinkedTable],
        ann: AnnotationV2,
    ):
        feats = {
            feat: ann.features.observations[feat]
            for feat in [P.PropertyDomain.name(), "MIN_20_PERCENT_ENT_FROM_TYPE"]
        }

        records = {"Property-Domain": [], "Min-20-Percent": []}
        for prop, type in feats[P.PropertyDomain.name()]:
            records["Property-Domain"].append(
                {
                    "prop": self.get_prop_label(prop),
                    "type": self.get_ent_label(type),
                }
            )
        for col, prop, type in feats["MIN_20_PERCENT_ENT_FROM_TYPE"]:
            records["Min-20-Percent"].append(
                {
                    "column": col,
                    "prop": self.get_prop_label(prop),
                    "type": self.get_ent_label(type),
                }
            )
        return {k: OTable(v) for k, v in records.items()}

    def get_type_features(
        self,
        example: Example[LinkedTable],
        ann: AnnotationV2,
    ) -> OTable:
        sms = self.autolabeler.evaluator.get_example_gold_sms(example)
        pred_types = [
            (ci, ctype) for ci, ctypes in ann.cta_probs.items() for ctype in ctypes
        ]
        labeled_pred_types = self.autolabeler.label_types(
            pred_types,
            sms,
        )
        labeled_types = dict(zip(pred_types, labeled_pred_types))

        def get_column_index(uid: str):
            node = ann.cg.get_node(uid)
            assert isinstance(node, CGColumnNode)
            return node.column

        ex_id = example.table.id
        idmap = ann.features.idmap
        ufeat = ann.features.node_features

        key2row = {
            (get_column_index(idmap.im(ufeat.node[i])), idmap.im(ufeat.type[i])): i
            for i in range(len(ann.features.node_features))
        }

        records = []
        for ci, ctypes in ann.cta_probs.items():
            for ctype, prob in ctypes.items():
                record = {
                    "index": ci,
                    "name": example.table.table.columns[ci].clean_name,
                    "type": self.get_ent_label(ctype),
                    "prob": float(prob),
                    "label": labeled_types[ci, ctype],
                }
                for name, value in [
                    ("freq_over_row", ufeat.freq_over_row),
                    ("freq_over_ent_row", ufeat.freq_over_ent_row),
                    ("extended_freq_over_row", ufeat.extended_freq_over_row),
                    ("extended_freq_over_ent_row", ufeat.extended_freq_over_ent_row),
                    ("type_distance", ufeat.type_distance),
                    (
                        "freq_discovered_prop_over_row",
                        ufeat.freq_discovered_prop_over_row,
                    ),
                ]:
                    record[name] = float(value[key2row[ci, ctype]])

                records.append(record)
        return OTable(records)

    def get_rel_features(
        self,
        example: Example[LinkedTable],
        ann: AnnotationV2,
    ) -> OTable:
        idmap = ann.features.idmap
        efeat = ann.features.edge_features

        sms = self.autolabeler.evaluator.get_example_gold_sms(example)
        rel2label = self.autolabeler.label_relationships(ann.cg, sms, return_edge=False)
        rel2key = {
            (
                idmap.im(efeat.source[i]),
                idmap.im(efeat.target[i]),
                idmap.im(efeat.statement[i]),
                idmap.im(efeat.outprop[i]),
            ): i
            for i in range(len(ann.features.edge_features))
        }
        objects = {
            rel: {
                "source": rel[0],
                "target": rel[1],
                "stmt": rel[2],
                "predicate": self.get_prop_label(rel[3]),
                "label": label,
                "prob": float(ann.cg_edge_probs[rel[2], rel[1], rel[3]]),
            }
            for rel, label in rel2label.items()
        }

        for name, feat in [
            ("freq_over_row", efeat.freq_over_row),
            ("freq_over_ent_row", efeat.freq_over_ent_row),
            ("freq_over_pos_rel", efeat.freq_over_pos_rel),
            ("freq_unmatch_over_ent_row", efeat.freq_unmatch_over_ent_row),
            ("freq_unmatch_over_pos_rel", efeat.freq_unmatch_over_pos_rel),
        ]:
            for rel, i in rel2key.items():
                objects[rel][name] = float(feat[i])

        return OTable(list(objects.values()))


@enhance_error_info(lambda db, example, pred_sm, ann: example.table.id)
def extract_complex_objects(
    db: Union[GramsDB, Path],
    example: Example[LinkedTable],
    pred_sm: SemanticModel,
    ann: Optional[AnnotationV2],
):
    db = to_grams_db(db)
    context = AlgoContext.from_grams_db(db, cache=True)
    aux_sm_extractor = AuxComplexSmObject(context)
    objs = {
        "table": AuxComplexTableObject(context).get_table(example),
        "gold-sm": aux_sm_extractor.get_sms(example, example.sms),
        "pred-sm": aux_sm_extractor.get_sm(example, pred_sm),
    }
    if ann is not None:
        aux_complex_extractor = AuxComplexFeatures(context, db.data_dir)
        # objs["rel-feat-v2"] = aux_complex_extractor.get_rel_features(
        #     example, ann, "v2"
        # )
        objs["rel-feat"] = aux_complex_extractor.get_rel_features(example, ann)
        objs["type-feat"] = aux_complex_extractor.get_type_features(example, ann)
        # for name, tbl in aux_complex_extractor.get_structure_features(
        #     example, ann
        # ).items():
        #     objs[f"structure-feat:{name}"] = tbl
    return objs

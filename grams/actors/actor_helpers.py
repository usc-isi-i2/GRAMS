from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Mapping, Optional, Union
from grams.algorithm.autolabeler import AutoLabeler
from grams.algorithm.context import AlgoContext
from grams.algorithm.inferences.psl_lib import IdentityIDMap, ReadableIDMap
from grams.main import Annotation
import numpy as np
from hugedict.prelude import HugeMutableMapping
from loguru import logger
from osin.apis.remote_exp import RemoteExpRun
from osin.types.pyobject import OHTML, OTable
from osin.types.pyobject.html import OListHTML

from grams.actors.db_actor import GramsDB
from grams.evaluator import Evaluator
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDEntity, WDProperty
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdentitylabel import WDEntityLabel
import ray
from sm.dataset import Example
from sm.misc.ray_helper import get_instance, ray_map
from sm.namespaces.namespace import OutOfNamespace
from sm.namespaces.wikidata import WikidataNamespace
from sm.outputs.semantic_model import (
    ClassNode,
    Edge,
    LiteralNode,
    LiteralNodeDataType,
    SemanticModel,
)
from grams.algorithm.inferences.psl_gram_model_exp2 import P
from grams.algorithm.inferences.features.rel_feature2 import RelFeatures as RelFeatures2
from grams.algorithm.inferences.features.rel_feature3 import RelFeatures3


def eval_dataset(
    db: GramsDB,
    examples: list[Example[LinkedTable]],
    pred_sms: list[SemanticModel],
    anns: Optional[list[Annotation]] = None,
    exprun: Optional[RemoteExpRun] = None,
):
    wdentities = db.wdentities.cache()
    wdentity_labels = db.wdentity_labels.cache()
    wdclasses = db.wdclasses.cache()
    wdprops = db.wdprops.cache()

    evaluator = Evaluator(wdentities, wdentity_labels, wdclasses, wdprops)
    sms = []
    for e in examples:
        sms.extend(evaluator.get_equiv_sms(e))
    sms.extend(pred_sms)
    evaluator.update_score_fns(sms)

    eval_outputs = []
    for i, (example, sm) in enumerate(zip(examples, pred_sms)):
        try:
            evalout = evaluator.cpa_cta(example, sm)
        except:
            logger.error("Failed to evaluate example: {} - {}", i, example.table.id)
            raise

        cea = evaluator.cea(example, k=[1, 5])
        eval_outputs.append((example.table.id, evalout["cpa"], evalout["cta"], cea))

    cpa_precision, cpa_recall, cpa_f1 = (
        np.mean([y.precision for x, y, z, t in eval_outputs]),
        np.mean([y.recall for x, y, z, t in eval_outputs]),
        np.mean([y.f1 for x, y, z, t in eval_outputs]),
    )

    cta_precision, cta_recall, cta_f1 = (
        np.mean([z.precision for x, y, z, t in eval_outputs]),
        np.mean([z.recall for x, y, z, t in eval_outputs]),
        np.mean([z.f1 for x, y, z, t in eval_outputs]),
    )

    # calculate cea macro scores
    cea_macro = {}
    cea_micro = {}

    for x, y, z, t in eval_outputs:
        for name, value in t["value"].items():
            if name not in cea_macro:
                cea_macro[name] = {k: [v] for k, v in value.items()}
                cea_micro[name] = t["confusion_matrix"][name]
                continue

            for k, v in value.items():
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
                    "precision": eval_outputs[i][1].precision,
                    "recall": eval_outputs[i][1].recall,
                    "f1": eval_outputs[i][1].f1,
                },
                "cta": {
                    "precision": eval_outputs[i][2].precision,
                    "recall": eval_outputs[i][2].recall,
                    "f1": eval_outputs[i][2].f1,
                },
                "cea": {
                    name: {
                        "p": value["precision"],
                        "r": value["recall"],
                        "f1": value["f1"],
                    }
                    for name, value in eval_outputs[i][3]["value"].items()
                },
            }
        )

    if exprun is not None:
        # log the results of each example
        complex_objs = ray_map(
            ray_extract_complex_objects.remote,
            [
                (db.data_dir, e, pred_sms[i], anns[i] if anns is not None else None)
                for i, e in enumerate(examples)
            ],
            desc="creating annotation debug info.",
            verbose=True,
        )
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
            ["%.2f" % (round(float(x) * 100, 2)) for x in [cpa_precision, cpa_recall, cpa_f1, cta_precision, cta_recall, cta_f1]]
    ))
    # fmt: on

    return {
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
        "cea": {
            "macro": cea_macro,
            "micro": cea_micro,
        },
    }, ex_details


class AuxComplexObjectBase:
    def __init__(self, wdentities: Mapping[str, WDEntity]) -> None:
        self.wdns = WikidataNamespace.create()
        self.wdentities = wdentities

    def get_ent_label(self, entid: str):
        if entid not in self.wdentities:
            return entid
        return f"{self.wdentities[entid].label} ({entid})"

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
    def __init__(self, db: GramsDB):
        context = AlgoContext.from_grams_db(db, cache=True)
        super().__init__(context.wdentities)

        self.db = db
        self.context = context
        self.autolabeler = AutoLabeler(context)

    def get_features(
        self,
        example: Example[LinkedTable],
        ann: Annotation,
        version: Literal["v2", "v3"],
    ) -> OTable:
        sms = self.autolabeler.get_equiv_sms(example.sms)
        rel2label = self.autolabeler.label_relationships(ann.cg, sms, return_edge=False)
        if version == "v2":
            rel2feat = self.extract_rel_features2(example, ann)
        else:
            assert version == "v3"
            rel2feat = self.extract_rel_features3(example, ann)

        objects = {
            rel: {
                "source": rel[0],
                "target": rel[1],
                "stmt": rel[2],
                "predicate": rel[3],
                "label": label,
                "prob": ann.cg_edge_probs[rel[2], rel[1], rel[3]],
            }
            for rel, label in rel2label.items()
        }

        for name, feats in rel2feat.items():
            for rel, feat in feats.items():
                objects[rel][name] = feat

        return OTable(list(objects.values()))

    def extract_rel_features2(self, example: Example[LinkedTable], ann: Annotation):
        # fmt: off
        feats = {
            "RelFreqOverRow": P.RelFreqOverRow.name(),
            "RelFreqOverEntRow": P.RelFreqOverEntRow.name(),
            "RelFreqOverPosRel": P.RelFreqOverPosRel.name(),
            "RelFreqUnmatchOverEntRow": P.RelFreqUnmatchOverEntRow.name(),
            "RelFreqUnmatchOverPosRel": P.RelFreqUnmatchOverPosRel.name(),
            "RelNotFuncDependency": P.RelNotFuncDependency.name(),
        }
        idmap = IdentityIDMap()
        featvalues = RelFeatures2(idmap, example.table, ann.cg, ann.dg, self.context.wdentities, self.context.wdentity_labels,
            self.context.wdclasses, self.context.wdprops, self.context.wd_num_prop_stats, None).extract_features(list(feats.values()))
        RelNotFuncDependency = {v[:2]: v[2] for v in featvalues[P.RelNotFuncDependency.name()]}
        featvalues[P.RelNotFuncDependency.name()] = []
        for v in featvalues[P.RelFreqOverRow.name()]:
            if (v[0], v[1]) not in RelNotFuncDependency:
                continue
            nv = list(v[:4])
            nv.append(RelNotFuncDependency[(v[0], v[1])])
            featvalues[P.RelNotFuncDependency.name()].append(nv)
        # fmt: on
        return {
            k: {tuple(v[:4]): v[4] for v in featvalues[nk]} for k, nk in feats.items()
        }

    def extract_rel_features3(self, example: Example[LinkedTable], ann: Annotation):
        # fmt: off
        feats = {
            "RelFreqOverRow": P.RelFreqOverRow.name(),
            "RelFreqOverEntRow": P.RelFreqOverEntRow.name(),
            "RelFreqOverPosRel": P.RelFreqOverPosRel.name(),
            "RelFreqUnmatchOverEntRow": P.RelFreqUnmatchOverEntRow.name(),
            "RelFreqUnmatchOverPosRel": P.RelFreqUnmatchOverPosRel.name(),
            "RelNotFuncDependency": P.RelNotFuncDependency.name(),
        }
        idmap = IdentityIDMap()
        featvalues = RelFeatures3(idmap, example.table, ann.cg, ann.dg, self.context).extract_features(list(feats.values()))
        RelNotFuncDependency = {v[:2]: v[2] for v in featvalues[P.RelNotFuncDependency.name()]}
        featvalues[P.RelNotFuncDependency.name()] = []
        for v in featvalues[P.RelFreqOverRow.name()]:
            if (v[0], v[1]) not in RelNotFuncDependency:
                continue
            nv = list(v[:4])
            nv.append(RelNotFuncDependency[(v[0], v[1])])
            featvalues[P.RelNotFuncDependency.name()].append(nv)
        # fmt: on
        return {
            k: {tuple(v[:4]): v[4] for v in featvalues[nk]} for k, nk in feats.items()
        }


@ray.remote
def ray_extract_complex_objects(
    db: Union[GramsDB, Path],
    example: Example[LinkedTable],
    pred_sm: SemanticModel,
    ann: Optional[Annotation],
):
    db = to_grams_db(db)
    wdentities = db.wdentities.cache()
    try:
        objs = extract_complex_objects(wdentities, example, pred_sm)
        if ann is not None:
            objs["rel-feat-v2"] = AuxComplexFeatures(db).get_features(
                example, ann, "v2"
            )
            objs["rel-feat-v3"] = AuxComplexFeatures(db).get_features(
                example, ann, "v3"
            )
    except Exception as e:
        raise Exception(
            "Cannot extract complex objects for example: %s" % example.table.id
        ) from e
    return objs


def extract_complex_objects(
    wdentities: Mapping[str, WDEntity],
    example: Example[LinkedTable],
    pred_sm: SemanticModel,
):
    objs = {
        "table": AuxComplexTableObject(wdentities).get_table(example),
        "gold-sm": AuxComplexSmObject(wdentities).get_sms(example, example.sms),
        "pred-sm": AuxComplexSmObject(wdentities).get_sm(example, pred_sm),
    }
    return objs


def to_grams_db(db: Union[GramsDB, Path]) -> GramsDB:
    if isinstance(db, Path):
        datadir = db
        db = get_instance(
            lambda: GramsDB(datadir, False),
            "GramsDB",
        )
    return db

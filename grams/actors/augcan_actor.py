from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import ray
from osin.integrations.ream import OsinActor
from ream.cache_helper import Cache
from ream.dataset_helper import DatasetDict
from sm_datasets.datasets import Datasets
from timer import watch_and_report

import grams.core as gcore
import grams.core.steps as gcoresteps
from grams.actors.actor_helpers import EvalArgs, to_grams_db
from grams.actors.dataset_actor import GramsELDatasetActor
from grams.actors.db_actor import GramsDB, GramsDBActor
from grams.algorithm.literal_matchers.string_match import StrSim
from grams.inputs.linked_table import CandidateEntityId, ExtendedLink, LinkedTable
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdvalue import WDValue
from sm.dataset import Example
from sm.inputs.link import WIKIDATA, EntityId
from sm.misc.ray_helper import enhance_error_info, ray_map, ray_put


@dataclass
class AugCanParams:
    similarity: str = field(
        default="levenshtein",
        metadata={
            "help": "augment candidate entities discovered from matching entity properties with values in the same row. "
            "Using the similarity function and threshold to filter out irrelevant candidates. "
        },
    )
    threshold: float = field(
        default=2.0,
        metadata={
            "help": "add candidate entities discovered from matching entity properties with values in the same row. "
            "Any value greater than 1.0 mean we do not apply the similarity function"
        },
    )
    use_column_name: bool = field(
        default=False,
        metadata={
            "help": "add column name to the search text. This is useful for value such as X town, Y town, Z town. "
        },
    )
    search_all_columns: bool = field(
        default=False,
        metadata={"help": "search all columns instead of just the entity columns."},
    )


class AugCanActor(OsinActor[str, AugCanParams]):
    VERSION = 105

    def __init__(
        self,
        params: AugCanParams,
        db_actor: GramsDBActor,
        dataset_actor: GramsELDatasetActor,
    ):
        super().__init__(params, [db_actor, dataset_actor])
        self.dataset_actor = dataset_actor
        self.db = db_actor.db
        self.strsim: Callable[[str, str], float] = getattr(
            StrSim, self.params.similarity
        )

        assert round(self.params.threshold, 3) == self.params.threshold
        self.provenance = f"augcan:{self.params.similarity}:{self.params.threshold}"

    @Cache.cls.dir(
        cls=DatasetDict,
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str):
        dsdict = self.dataset_actor.run_dataset(dsquery)
        newdsdict: DatasetDict[list[Example[LinkedTable]]] = DatasetDict(
            dsdict.name, {}, dsdict.provenance + ";" + self.provenance
        )

        using_ray = sum(len(x) for x in dsdict.values()) > 1
        dbref = ray_put(self.db.data_dir, using_ray)
        paramref = ray_put(self.params, using_ray)

        for name, ds in dsdict.items():
            newdsdict[name] = ray_map(
                rust_augment_candidates,
                [(dbref, ex, paramref) for ex in ds],
                desc="augmenting candidates",
                verbose=True,
                using_ray=using_ray,
                is_func_remote=False,
            )
        return newdsdict

    def evaluate(self, eval_args: EvalArgs):
        for dsquery in eval_args.dsqueries:
            with watch_and_report(f"augment candidates {dsquery}"):
                # self.run_dataset(dsquery)
                self.check_rust_implementation(dsquery)

    def check_rust_implementation(self, dsquery: str):
        dsdict = self.dataset_actor.run_dataset(dsquery)

        using_ray = sum(len(x) for x in dsdict.values()) > 1
        dbref = ray_put(self.db.data_dir, using_ray)
        paramref = ray_put(self.params, using_ray)

        out = {}
        for name, ds in dsdict.items():
            out[name] = ray_map(
                check_rust_implementation,
                [(dbref, ex, paramref) for ex in ds],
                desc="checking rust implementation",
                verbose=True,
                using_ray=using_ray,
                is_func_remote=False,
            )
        return out


@enhance_error_info(lambda data_dir, example, params: example.table.id)
def rust_augment_candidates(
    data_dir: Path,
    example: Example[LinkedTable],
    params: AugCanParams,
) -> Example[LinkedTable]:
    cfg = gcoresteps.CandidateLocalSearchConfig(
        params.similarity,
        params.threshold,
        params.use_column_name,
        None,
        params.search_all_columns,
    )
    gcore.GramsDB.init(str(data_dir))
    cdb = gcore.GramsDB.get_instance()

    newtable = example.table.to_rust()
    context = cdb.get_algo_context(newtable, n_hop=2)
    newtable = gcoresteps.candidate_local_search(newtable, context, cfg)

    # copy results back to python
    newlinks = example.table.links.shallow_copy()
    nrows, ncols = example.table.shape()
    for ri in range(nrows):
        for ci in range(ncols):
            tmp_links = newtable.get_links(ri, ci)
            if len(tmp_links) == 0:
                newlinks[ri, ci] = []
            else:
                newlinks[ri, ci] = [
                    ExtendedLink(
                        start=tmp_links[0].start,
                        end=tmp_links[0].end,
                        url=tmp_links[0].url,
                        entities=[
                            EntityId(e.id, WIKIDATA) for e in tmp_links[0].entities
                        ],
                        candidates=[
                            CandidateEntityId(
                                EntityId(c.id.id, WIKIDATA), c.probability
                            )
                            for c in tmp_links[0].candidates
                        ],
                    )
                ]

    return Example(
        sms=example.sms,
        table=LinkedTable(
            table=example.table.table,
            context=example.table.context,
            links=newlinks,
        ),
    )


@enhance_error_info(lambda data_dir, example, params: example.table.id)
def check_rust_implementation(
    db: Union[GramsDB, Path],
    example: Example[LinkedTable],
    params: AugCanParams,
):
    db = to_grams_db(db) if not isinstance(db, GramsDB) else db

    with watch_and_report("rust"):
        table1 = rust_augment_candidates(db.data_dir, example, params).table
    with watch_and_report("python"):
        table2 = augment_candidates(
            db, example, getattr(StrSim, params.similarity), params
        ).table

    for ri, ci, links1 in table1.links.enumerate_flat_iter():
        links2 = table2.links[ri, ci]

        assert len(links1) == len(links2)
        for link1, link2 in zip(links1, links2):
            c1 = {c.entity_id: c.probability for c in link1.candidates}
            c2 = {c.entity_id: c.probability for c in link2.candidates}
            diff_keys = set(c1.keys()).symmetric_difference(c2.keys())
            assert len(diff_keys) == 0, f"Found: {diff_keys} at {ri}, {ci}"

            for eid in c1.keys():
                assert round(c1[eid], 7) == round(c2[eid], 7), f"{c1[eid]} != {c2[eid]}"

    return 1


@enhance_error_info(lambda db, example, strsim, params: example.table.id)
def augment_candidates(
    db: Union[GramsDB, Path],
    example: Example[LinkedTable],
    strsim: Callable[[str, str], float],
    params: AugCanParams,
):
    db = to_grams_db(db) if not isinstance(db, GramsDB) else db
    wdentities = db.get_auto_cached_entities(example.table)

    nrows, ncols = example.table.shape()
    next_entity_cache: dict[str, set[str]] = {}
    entity_columns: list[int] = []
    for ci in range(ncols):
        if any(
            any(len(link.candidates) > 0 for link in example.table.links[ri, ci])
            for ri in range(nrows)
        ):
            entity_columns.append(ci)

    newlinks = example.table.links.shallow_copy()

    for ci in entity_columns:
        other_ent_columns = [oci for oci in entity_columns if oci != ci]

        for ri in range(nrows):
            links = example.table.links[ri, ci]
            if len(links) == 0:
                continue

            row: list[tuple[int, str]] = [
                (oci, example.table.table[ri, oci]) for oci in other_ent_columns
            ]
            next_ent_ids = gather_next_entities(
                {can.entity_id for link in links for can in link.candidates},
                wdentities,
                next_entity_cache,
            )
            next_ents = [wdentities[eid] for eid in next_ent_ids]

            for oci, value in row:
                value = value.strip()
                if value == "":
                    continue

                if params.use_column_name:
                    header = example.table.table.columns[oci].clean_name or ""
                    values = {
                        value,
                        (value + " " + header).strip(),
                        (header + " " + value).strip(),
                    }
                else:
                    values = [value]

                # search for value in next_ents
                match_ents = search_value(
                    values,
                    next_ents,
                    strsim,
                    params.threshold,
                )
                if len(match_ents) > 0:
                    olinks = newlinks[ri, oci]
                    for olink in olinks:
                        for can in olink.candidates:
                            if can.entity_id in match_ents:
                                match_ents.pop(can.entity_id)

                    candidates = [
                        CandidateEntityId(eid, score)
                        for eid, score in match_ents.items()
                    ]
                    if len(olinks) == 0:
                        newlinks[ri, oci] = [
                            ExtendedLink(
                                start=0,
                                end=len(example.table.table[ri, oci]),
                                url=None,
                                entities=[],
                                candidates=candidates,
                            )
                        ]
                    else:
                        (
                            olink,
                        ) = olinks  # the previous actor ensure we only have one link
                        newlinks[ri, oci] = [
                            ExtendedLink(
                                start=olink.start,
                                end=olink.end,
                                url=olink.url,
                                entities=olink.entities,
                                candidates=sorted(
                                    olink.candidates + candidates,
                                    key=lambda c: c.probability,
                                    reverse=True,
                                ),
                            )
                        ]
    return Example(
        example.sms,
        LinkedTable(
            table=example.table.table,
            context=example.table.context,
            links=newlinks,
        ),
    )


def search_value(
    values: list[str] | set[str],
    ents: list[WDEntity],
    strsim: Callable[[str, str], float],
    threshold: float,
) -> dict[EntityId, float]:
    match_ents = {}
    for ent in ents:
        if len(ent.aliases) > 0:
            ent_score = max(
                max(strsim(value, ent.label) for value in values),
                max(
                    max(strsim(value, alias) for value in values)
                    for alias in ent.aliases
                ),
            )
        else:
            ent_score = max(strsim(value, ent.label) for value in values)
        if ent_score >= threshold:
            match_ents[EntityId(ent.id, WIKIDATA)] = ent_score
    return match_ents


def gather_next_entities(
    ent_ids: set[str],
    wdentities: Mapping[str, WDEntity],
    cache: dict[str, set[str]],
):
    next_entities = set()
    for ent_id in ent_ids:
        if ent_id not in cache:
            next_tmp = set()
            ent = wdentities[ent_id]
            for stmts in ent.props.values():
                for stmt in stmts:
                    if WDValue.is_entity_id(stmt.value):
                        next_tmp.add(stmt.value.value["id"])
                    next_tmp.update(
                        qval.value["id"]
                        for qvals in stmt.qualifiers.values()
                        for qval in qvals
                        if WDValue.is_entity_id(qval)
                    )
            cache[ent_id] = next_tmp

        next_entities.update(cache[ent_id])

    return next_entities

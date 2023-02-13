from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union
from grams.actors.dataset_actor import GramsELDatasetActor
from grams.actors.db_actor import GramsDB, GramsDBActor
from grams.actors.actor_helpers import to_grams_db
from grams.algorithm.literal_matchers.string_match import StrSim
from grams.inputs.linked_table import CandidateEntityId, ExtendedLink, LinkedTable
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdvalue import WDValue
from osin.integrations.ream import OsinActor
from ream.cache_helper import Cache
from ream.dataset_helper import DatasetDict
from sm.dataset import Example
from sm.inputs.link import WIKIDATA, EntityId
from sm.misc.ray_helper import ray_put, ray_map
import ray


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


class AugCanActor(OsinActor[str, AugCanParams]):
    VERSION = 100

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

    @Cache.cls.file(
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

        dbref = None

        for name, ds in dsdict.items():
            if len(ds) > 1:
                if dbref is None:
                    dbref = ray_put(self.db.data_dir)
                newdsdict[name] = ray_map(
                    ray_augmented_candidates.remote,
                    [
                        (dbref, ex, self.params.similarity, self.params.threshold)
                        for ex in ds
                    ],
                    desc="augmenting candidates",
                    verbose=True,
                )
            else:
                strsim = getattr(StrSim, self.params.similarity)
                newdsdict[name] = [
                    augmented_candidates(
                        ex,
                        self.db.get_auto_cached_entities(ex.table),
                        strsim,
                        self.params.threshold,
                    )
                    for ex in ds
                ]
        return newdsdict


@ray.remote
def ray_augmented_candidates(
    db: Union[GramsDB, Path],
    example: Example[LinkedTable],
    strsim: str,
    threshold: float,
):
    db = to_grams_db(db)
    wdentities = db.get_auto_cached_entities(example.table)

    try:
        return augmented_candidates(
            example, wdentities, getattr(StrSim, strsim), threshold
        )
    except Exception as e:
        raise Exception("Failed to augment table: " + example.table.id) from e


def augmented_candidates(
    example: Example[LinkedTable],
    wdentities: Mapping[str, WDEntity],
    strsim: Callable[[str, str], float],
    threshold: float,
):
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

            row = [(oci, example.table.table[ri, oci]) for oci in other_ent_columns]
            next_ent_ids = gather_next_entities(
                {can.entity_id for link in links for can in link.candidates},
                wdentities,
                next_entity_cache,
            )
            next_ents = [wdentities[eid] for eid in next_ent_ids]

            for oci, value in row:
                # search for value in next_ents
                match_ents = search_value(value, next_ents, strsim, threshold)
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
                        olinks.append(
                            ExtendedLink(
                                start=0,
                                end=len(example.table.table[ri, oci]),
                                url=None,
                                entities=[],
                                candidates=candidates,
                            )
                        )
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
    value: str,
    ents: list[WDEntity],
    strsim: Callable[[str, str], float],
    threshold: float,
) -> dict[EntityId, float]:
    match_ents = {}
    for ent in ents:
        if len(ent.aliases) > 0:
            ent_score = max(
                strsim(value, ent.label),
                max(strsim(value, alias) for alias in ent.aliases),
            )
        else:
            ent_score = strsim(value, ent.label)
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
                        next_entities.add(stmt.value.value["id"])
                    next_tmp.update(
                        qval.value["id"]
                        for qvals in stmt.qualifiers.values()
                        for qval in qvals
                        if WDValue.is_entity_id(qval)
                    )
            cache[ent_id] = next_tmp

        next_entities.update(cache[ent_id])

    return next_entities

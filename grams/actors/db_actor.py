from __future__ import annotations

from collections.abc import Mapping
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, MutableMapping, Optional, Set, Union
from grams.inputs.linked_table import LinkedTable
from hugedict.types import HugeMutableMapping
from kgdata.wikidata.models.wdproperty import WDProperty
from ream.actor_version import ActorVersion
from ream.cache_helper import Cache
from ream.helper import orjson_dumps

import serde.prelude as serde
from hugedict.prelude import CacheDict, Parallel
from ream.actors.base import BaseActor
from sm.misc.ray_helper import get_instance
from timer import watch_and_report
from tqdm import tqdm

from kgdata.wikidata.db import (
    WDProxyDB,
    WikidataDB,
    get_entity_db,
    get_entity_label_db,
    get_wdclass_db,
    get_wdprop_db,
    get_wdprop_domain_db,
    get_wdprop_range_db,
    query_wikidata_entities,
)
from kgdata.wikidata.models import (
    WDClass,
    WDEntity,
    WDEntityLabel,
    WDQuantityPropertyStats,
)


class GramsDB:
    VERSION = 102

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

        # cache entities that are used per table
        # this option should be cleared after running the algorithm
        # to avoid memory overflow
        self.autocached_entities: dict[Optional[str], Mapping[str, WDEntity]] = {}

        with watch_and_report("init grams db"):
            self.db = WikidataDB(data_dir)
            self.wdentities = self.db.wdentities
            self.wdentity_labels = self.db.wdentity_labels
            self.wdclasses = self.db.wdclasses
            self.wdprops = self.db.wdprops
            self.wdprop_domains = self.db.wdprop_domains
            self.wdprop_ranges = self.db.wdprop_ranges
            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(
                os.path.join(data_dir, "quantity_prop_stats")
            )

    @Cache.mem(
        cache_key=lambda self, table, max_n_hop, verbose: (table.id.encode(), max_n_hop)
    )
    def get_table_entity_labels(
        self, table: LinkedTable, max_n_hop: int, verbose: bool
    ):
        wdentity_ids, wdentities = self.get_table_entities(table, max_n_hop, verbose)
        wdentity_labels = self.get_entity_labels(wdentities, verbose)
        return wdentity_labels

    @Cache.mem(
        cache_key=lambda self, table, max_n_hop, verbose: (table.id.encode(), max_n_hop)
    )
    def get_table_entities(self, table: LinkedTable, max_n_hop: int, verbose: bool):
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
        wdentities = self.get_entities(wdentity_ids, n_hop=max_n_hop, verbose=verbose)

        return wdentity_ids, wdentities

    def get_auto_cached_entities(
        self, table: Optional[LinkedTable]
    ) -> Mapping[str, WDEntity]:
        """Get the cached entities for the given table."""
        key = None if table is None else table.id
        if key not in self.autocached_entities:
            self.autocached_entities[key] = self.wdentities.cache()
        return self.autocached_entities[key]

    def clear_auto_cached_entities(self, table: Optional[LinkedTable]):
        """Clear the cached entities for the given table."""
        key = None if table is None else table.id
        if key in self.autocached_entities:
            del self.autocached_entities[key]

    def get_entities(
        self, wdentity_ids: Set[str], n_hop: int = 1, verbose: bool = False
    ) -> Dict[str, WDEntity]:
        assert n_hop <= 2
        batch_size = 30
        wdentities: Dict[str, WDEntity] = {}
        pp = Parallel()
        for wdentity_id in wdentity_ids:
            wdentity = self.wdentities.get(wdentity_id, None)
            if wdentity is not None:
                wdentities[wdentity_id] = wdentity

        if isinstance(self.wdentities, WDProxyDB):
            missing_qnode_ids = [
                wdentity_id
                for wdentity_id in wdentity_ids
                if wdentity_id not in wdentities
                and not self.wdentities.does_not_exist_locally(wdentity_id)
            ]
            if len(missing_qnode_ids) > 0:
                resp = pp.map(
                    query_wikidata_entities,
                    [
                        missing_qnode_ids[i : i + batch_size]
                        for i in range(0, len(missing_qnode_ids), batch_size)
                    ],
                    show_progress=verbose,
                    progress_desc=f"query wikidata for get missing entities in hop 1",
                    is_parallel=True,
                )
                for odict in resp:
                    for k, v in odict.items():
                        wdentities[k] = v
                        self.wdentities[k] = v

        if n_hop > 1:
            next_wdentity_ids = set()
            for wdentity in wdentities.values():
                for p, stmts in wdentity.props.items():
                    for stmt in stmts:
                        if stmt.value.is_qnode(stmt.value):
                            next_wdentity_ids.add(stmt.value.as_entity_id())
                        for qvals in stmt.qualifiers.values():
                            next_wdentity_ids = next_wdentity_ids.union(
                                qval.as_entity_id()
                                for qval in qvals
                                if qval.is_qnode(qval)
                            )
            next_wdentity_ids = list(next_wdentity_ids.difference(wdentities.keys()))
            for wdentity_id in tqdm(
                next_wdentity_ids,
                desc="load entities in 2nd hop from db",
                disable=not verbose,
            ):
                wdentity = self.wdentities.get(wdentity_id, None)
                if wdentity is not None:
                    wdentities[wdentity_id] = wdentity

            if isinstance(self.wdentities, WDProxyDB):
                next_wdentity_ids = [
                    qnode_id
                    for qnode_id in next_wdentity_ids
                    if qnode_id not in wdentities
                    and not self.wdentities.does_not_exist_locally(qnode_id)
                ]
                if len(next_wdentity_ids) > 0:
                    resp = pp.map(
                        query_wikidata_entities,
                        [
                            next_wdentity_ids[i : i + batch_size]
                            for i in range(0, len(next_wdentity_ids), batch_size)
                        ],
                        show_progress=verbose,
                        progress_desc=f"query wikidata for get missing entities in hop {n_hop}",
                        is_parallel=True,
                    )
                    for odict in resp:
                        for k, v in odict.items():
                            wdentities[k] = v
                            self.wdentities[k] = v
        return wdentities

    def get_entity_labels(
        self, wdentities: Mapping[str, WDEntity], verbose: bool = False
    ) -> Dict[str, str]:
        id2label: Dict[str, str] = {}
        for qnode in tqdm(wdentities.values(), disable=not verbose, desc=""):
            qnode: WDEntity
            id2label[qnode.id] = str(qnode.label)
            for stmts in qnode.props.values():
                for stmt in stmts:
                    if stmt.value.is_qnode(stmt.value):
                        qnode_id = stmt.value.as_entity_id()
                        if qnode_id in self.wdentity_labels:
                            label = self.wdentity_labels[qnode_id].label
                        else:
                            label = qnode_id
                        id2label[qnode_id] = label
                    for qvals in stmt.qualifiers.values():
                        for qval in qvals:
                            if qval.is_qnode(qval):
                                qnode_id = qval.as_entity_id()
                                if qnode_id in self.wdentity_labels:
                                    label = self.wdentity_labels[qnode_id].label
                                else:
                                    label = qnode_id
                                id2label[qnode_id] = label
        return id2label


@dataclass
class GramsDBParams:
    data_dir: Path = field(
        metadata={"help": "Path to a directory containing databases"},
    )


class GramsDBActor(BaseActor[str, GramsDBParams]):
    VERSION = ActorVersion.create(102, [GramsDB])

    def __init__(self, params: GramsDBParams):
        super().__init__(params, [])
        self.db = GramsDB(params.data_dir)


def to_grams_db(db: Union[GramsDB, Path]) -> GramsDB:
    if isinstance(db, Path):
        datadir = db
        db = get_instance(
            lambda: GramsDB(datadir),
            f"GramsDB[{datadir}]",
        )
    return db

from __future__ import annotations

from collections.abc import Mapping
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, MutableMapping, Set
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models.wdproperty import WDProperty
from ream.cache_helper import Cache
from ream.helper import orjson_dumps

import serde.prelude as serde
from hugedict.prelude import CacheDict, Parallel
from ream.actors.base import BaseActor
from timer import Timer
from tqdm import tqdm

from kgdata.wikidata.db import (
    WDProxyDB,
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
    VERSION = 100

    def __init__(self, data_dir: Path, proxy_db: bool):
        self.data_dir = data_dir
        self.proxy_db = proxy_db

        with Timer().watch_and_report("init grams db"):
            read_only = not proxy_db
            self.wdentities = get_entity_db(
                os.path.join(data_dir, "wdentities.db"),
                read_only=read_only,
                proxy=proxy_db,
            )
            if proxy_db:
                assert isinstance(self.wdentities, WDProxyDB)
            if os.path.exists(os.path.join(data_dir, "wdentity_labels.db")):
                self.wdentity_labels = get_entity_label_db(
                    os.path.join(data_dir, "wdentity_labels.db"),
                )
            else:
                self.wdentity_labels: MutableMapping[str, WDEntityLabel] = {}
            self.wdclasses = get_wdclass_db(
                os.path.join(data_dir, "wdclasses.db"),
                read_only=read_only,
                proxy=proxy_db,
            )
            if os.path.exists(os.path.join(data_dir, "wdclasses.fixed.jl")):
                self.wdclasses = self.wdclasses.cache()
                assert isinstance(self.wdclasses, CacheDict)
                for record in serde.jl.deser(
                    os.path.join(data_dir, "wdclasses.fixed.jl")
                ):
                    cls = WDClass.from_dict(record)
                    self.wdclasses._cache[cls.id] = cls
            self.wdprops = get_wdprop_db(
                os.path.join(data_dir, "wdprops.db"),
                read_only=read_only,
                proxy=proxy_db,
            )
            if os.path.exists(os.path.join(data_dir, "wdprop_domains.db")):
                self.wdprop_domains = get_wdprop_domain_db(
                    os.path.join(data_dir, "wdprop_domains.db"),
                    read_only=True,
                )
            else:
                self.wdprop_domains = None

            if os.path.exists(os.path.join(data_dir, "wdprop_ranges.db")):
                self.wdprop_ranges = get_wdprop_range_db(
                    os.path.join(data_dir, "wdprop_ranges.db"),
                    read_only=True,
                )
            else:
                self.wdprop_ranges = None

            self.wd_numprop_stats = WDQuantityPropertyStats.from_dir(
                os.path.join(data_dir, "quantity_prop_stats")
            )

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

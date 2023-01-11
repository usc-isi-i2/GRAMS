from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from typing import Optional
from grams.actors.db_actor import GramsDB
from kgdata.wikidata.models import (
    WDEntity,
    WDProperty,
    WDQuantityPropertyStats,
    WDEntityLabel,
    WDClass,
    WDPropertyDomains,
    WDPropertyRanges,
)
from kgdata.wikidata.db import WikidataDB


@dataclass
class AlgoContext:
    """A context object that contains information needed for the algorithm to run.
    This object is used to pass information deep into the algorithm, without having to modify
    the function signatures of all the functions that need this information.
    """

    data_dir: Path

    wdentities: Mapping[str, WDEntity]
    wdentity_labels: Mapping[str, WDEntityLabel]
    wdclasses: Mapping[str, WDClass]
    wdprops: Mapping[str, WDProperty]
    wd_num_prop_stats: Mapping[str, WDQuantityPropertyStats]
    wdprop_domains: Optional[Mapping[str, WDPropertyDomains]]
    wdprop_ranges: Optional[Mapping[str, WDPropertyRanges]]

    @cached_property
    def wdentity_wikilinks(self):
        return WikidataDB(self.data_dir).wdentity_wikilinks

    @staticmethod
    def from_grams_db(db: GramsDB, cache: bool = True):
        assert cache
        return AlgoContext(
            data_dir=db.data_dir,
            wdentities=db.wdentities.cache(),
            wdentity_labels=db.wdentity_labels.cache(),
            wdclasses=db.wdclasses.cache(),
            wdprops=db.wdprops.cache(),
            wd_num_prop_stats=db.wd_numprop_stats,
            wdprop_domains=db.wdprop_domains.cache()
            if db.wdprop_domains is not None
            else None,
            wdprop_ranges=db.wdprop_ranges.cache()
            if db.wdprop_ranges is not None
            else None,
        )

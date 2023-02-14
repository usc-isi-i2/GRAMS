from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from grams.actors.actor_helpers import to_grams_db
from grams.actors.augcan_actor import AugCanActor
from grams.actors.dataset_actor import GramsELDatasetActor
from grams.actors.db_actor import GramsDB, GramsDBActor
from grams.inputs.linked_table import LinkedTable
from loguru import logger
from osin.integrations.ream import OsinActor
import ray
from ream.cache_helper import Cache
from ream.dataset_helper import DatasetDict
from ream.params_helper import NoParams
from sm.dataset import Example
from sm.misc.ray_helper import ray_map, ray_put


@dataclass
class GramsPreprocessParams:
    remove_unk_entities: bool = field(
        default=True,
        metadata={"help": "remove candidate entities that are not in the database"},
    )


class GramsPreprocessActor(OsinActor[str, GramsPreprocessParams]):
    VERSION = 100

    def __init__(
        self,
        params: GramsPreprocessParams,
        dbactor: GramsDBActor,
        dataset_actor: GramsELDatasetActor,
        augcan_actor: AugCanActor,
    ):
        super().__init__(params, [dbactor, dataset_actor, augcan_actor])
        self.dbactor = dbactor
        self.dataset_actor = dataset_actor
        self.augcan_actor = augcan_actor

        self.provenance = f"preprocess:remove_unk={self.params.remove_unk_entities}"

    @Cache.cls.dir(
        cls=DatasetDict,
        mem_persist=True,
        compression="lz4",
        log_serde_time=True,
    )
    def run_dataset(self, dsquery: str, max_n_hop: int):
        if self.augcan_actor.params.threshold <= 1.0:
            dsdict = self.augcan_actor.run_dataset(dsquery)
        else:
            dsdict = self.dataset_actor.run_dataset(dsquery)

        if not self.params.remove_unk_entities:
            return dsdict

        newdsdict: DatasetDict[list[Example[LinkedTable]]] = DatasetDict(
            dsdict.name,
            {},
            dsdict.provenance + ";" + (self.provenance + f"&n_hop={max_n_hop}"),
        )
        dbref = None

        for name, ds in dsdict.items():
            if len(ds) > 1:
                if dbref is None:
                    dbref = ray_put(self.dbactor.db.data_dir)
                new_tables = ray_map(
                    ray_preprocess_table.remote,
                    [(dbref, ex.table, max_n_hop) for ex in ds],
                    desc="preprocess table",
                    verbose=True,
                )
            else:
                new_tables = [
                    preprocess_table(self.dbactor.db, ex.table, max_n_hop) for ex in ds
                ]
            newdsdict[name] = [Example(ex.sms, tbl) for ex, tbl in zip(ds, new_tables)]
        return dsdict


@ray.remote
def ray_preprocess_table(db: Union[GramsDB, Path], table: LinkedTable, max_n_hop: int):
    db = to_grams_db(db)
    return preprocess_table(db, table, max_n_hop)


def preprocess_table(db: GramsDB, table: LinkedTable, max_n_hop: int):
    wdentity_ids, wdentities = db.get_table_entities(table, max_n_hop, verbose=True)
    nonexistent_wdentity_ids = wdentity_ids.difference(wdentities.keys())
    if len(nonexistent_wdentity_ids) > 0:
        logger.info(
            "Removing non-existent entities: {}", list(nonexistent_wdentity_ids)
        )
        table.remove_nonexistent_entities(nonexistent_wdentity_ids)
    return table

from inspect import signature
import os
from dataclasses import dataclass, field, fields, make_dataclass
from pathlib import Path
from typing import Generic, List, Optional, Tuple, get_type_hints
from ned.actors.db import DBActor
from functools import partial
import yada
from loguru import logger
from osin.apis.osin import Osin
from osin.integrations.ream import OsinActor
from ream.prelude import ActorGraph, ReamWorkspace, configure_loguru

from grams.actors.augcan_actor import AugCanActor
from grams.actors.dataset_actor import GramsDatasetActor, GramsELDatasetActor
from grams.actors.db_actor import GramsDBActor
from grams.actors.grams_actor import GramsActor
from grams.actors.grams_inf_actor import GramsInfActor
from grams.actors.grams_infdata_actor import GramsInfDataActor
from grams.actors.grams_preprocess_actor import GramsPreprocessActor
from kgdata.wikidata.db import WikidataDB
from ned.actors.candidate_generation import CanGenActor
from ned.actors.candidate_ranking import CanRankActor
from ned.actors.dataset.prelude import NEDDatasetActor, NEDAutoDatasetActor
from ned.actors.entity_recognition import EntityRecognitionActor
from sm.misc.ray_helper import set_ray_init_args
from ream.cli_helper import CLI

########################################################
# CONFIG REAM AND DEFINE ACTOR GRAPH
HOME_DIR = Path(os.environ["HOME_DIR"]).absolute()
REMOTE_OSIN = os.environ.get("REMOTE_OSIN", None)
ENABLE_OSIN = os.environ.get("ENABLE_OSIN", None)
DATABASE_DIR = HOME_DIR / "databases"

set_ray_init_args(log_to_driver=False)
configure_loguru()
if REMOTE_OSIN is not None:
    OsinActor._osin = Osin.remote(REMOTE_OSIN, "/tmp/osin")
elif ENABLE_OSIN in {"1", "true"}:
    OsinActor._osin = Osin.local(HOME_DIR / "osin")

ReamWorkspace.init(HOME_DIR / "ream")
WikidataDB.init(DATABASE_DIR)
# fmt: off
graph: ActorGraph = ActorGraph.auto({
    "db": DBActor, "eda": NEDAutoDatasetActor, "ed": NEDDatasetActor, 
    "er": EntityRecognitionActor, "cg": CanGenActor, "cr": CanRankActor,
    "gd": GramsDatasetActor, "gde": GramsELDatasetActor, "gau": AugCanActor, 
    "gp": GramsPreprocessActor, "gid": GramsInfDataActor, "gdb": GramsDBActor,
    "gia": GramsInfActor, "ga": GramsActor
})
# fmt: on
########################################################


@dataclass
class EvalArgs:
    dsqueries: List[str] = field(
        metadata={"help": "List of dataset queries to evaluate"}
    )
    threshold: float = 0.5


main = partial(CLI.main, graph)

if __name__ == "__main__":
    CLI.main(graph)

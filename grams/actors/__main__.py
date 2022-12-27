import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yada
from loguru import logger
from osin.apis.osin import Osin
from osin.integrations.ream import OsinActor
from ream.prelude import ActorGraph, ReamWorkspace, configure_loguru

from kgdata.wikidata.db import WikidataDB
from ned.actors.candidate_generation import CanGenActor
from ned.actors.candidate_ranking import CanRankActor
from ned.actors.entity_recognition import EntityRecognitionActor
from ned.actors.dataset import NEDDatasetActor
from ned.actors.evaluate_helper import EvalArgs
from grams.actors.dataset_actor import GramsDatasetActor, GramsELDatasetActor
from grams.actors.grams_actor import GramsActor

########################################################
# CONFIG REAM AND DEFINE ACTOR GRAPH
HOME_DIR = Path(os.environ["HOME_DIR"]).absolute()
REMOTE_OSIN = os.environ.get("REMOTE_OSIN", None)
DATABASE_DIR = HOME_DIR / "databases"

configure_loguru()
if os.environ.get("REAM_ENV", "") == "prod":
    if REMOTE_OSIN is None:
        OsinActor._osin = Osin.local(HOME_DIR / "osin")
    else:
        OsinActor._osin = Osin.remote(REMOTE_OSIN, "/tmp/osin")
else:
    if REMOTE_OSIN is None:
        OsinActor._osin = Osin.local(HOME_DIR / "osin-dev")
    else:
        OsinActor._osin = Osin.remote(REMOTE_OSIN, "/tmp/osin")
ReamWorkspace.init(HOME_DIR / "ream")
WikidataDB.init(DATABASE_DIR)
# fmt: off
graph: ActorGraph = ActorGraph.auto({
    "ed": NEDDatasetActor, "er": EntityRecognitionActor, "cg": CanGenActor, "cr": CanRankActor,
    "gd": GramsDatasetActor, "gde": GramsELDatasetActor, "ga": GramsActor
})
# fmt: on
########################################################


@dataclass
class MainArgs:
    actor: str
    eval_args: EvalArgs
    logfile: Optional[str] = "logs/run_{time}.log"
    allow_unknown_args: bool = False


def main(sysargs=None):
    logger.debug("Started!")
    args, remain_args = yada.Parser1(MainArgs).parse_known_args(sysargs)
    graph.run(
        actor_class=args.actor,
        actor_method="evaluate",
        run_args=args.eval_args,
        args=remain_args,
        log_file=args.logfile,
        allow_unknown_args=args.allow_unknown_args,
    )
    logger.debug("Finished!")


def get_actor(sysargs=None):
    args, remain_args = yada.Parser1(MainArgs).parse_known_args(sysargs)
    return graph.create_actor(
        actor_class=args.actor,
        args=remain_args,
        log_file=args.logfile,
    )


if __name__ == "__main__":
    main()
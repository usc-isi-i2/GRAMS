from dataclasses import dataclass
from typing import Mapping

from kgdata.wikidata.models.qnode import QNode


@dataclass
class LiteralMatchKit:
    """Storing objects that may be needed for matching"""

    qnodes: Mapping[str, QNode]

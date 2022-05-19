from dataclasses import dataclass
from typing import Mapping

from kgdata.wikidata.models import WDEntity


@dataclass
class LiteralMatchKit:
    """Storing objects that may be needed for matching"""

    wdentities: Mapping[str, WDEntity]

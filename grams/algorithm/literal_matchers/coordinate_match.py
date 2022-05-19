from typing import Tuple
from grams.algorithm.literal_matchers.text_parser import ParsedTextRepr
from kgdata.wikidata.models import WDValue
from grams.algorithm.literal_matchers.types import LiteralMatchKit


def globecoordinate_test(
    kgvalue: WDValue, value: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgvalue: the value in KG
        value: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    return False, 0.0

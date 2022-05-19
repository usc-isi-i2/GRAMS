import re
from typing import Tuple, cast
from grams.algorithm.literal_matchers.text_parser import (
    ParsedTextRepr,
    ParsedDatetimeRepr,
)
from kgdata.wikidata.models import WDValueQuantity
from grams.algorithm.literal_matchers.types import LiteralMatchKit


def quantity_test(
    kgval: WDValueQuantity, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    if val.number is None:
        return False, 0.0

    kgnum = float(kgval.value["amount"])
    num = val.number.number

    if abs(kgnum - num) < 1e-5:
        return True, 1.0
    if kgnum == 0:
        diff_percentage = abs(kgnum - num) / 1e-7
    else:
        diff_percentage = abs((kgnum - num) / kgnum)

    if diff_percentage < 0.05:
        # within 5%
        return True, 0.95 - diff_percentage
    # if diff_percentage < 0.1:
    #     return True, 0.9 - diff_percentage
    return False, 0.0

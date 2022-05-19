from typing import Tuple
from grams.algorithm.literal_matchers.string_match import string_match_similarity
from grams.algorithm.literal_matchers.text_parser import ParsedTextRepr
from kgdata.wikidata.models import WDValue
from grams.algorithm.literal_matchers.types import LiteralMatchKit


def entity_similarity_test(
    kgval: WDValue, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching
    Returns:
        a tuple of (whether it's matched, confidence)
    """

    if not kgval.is_qnode(kgval):
        # not handle lexical or property yet
        return False, 0.0

    qnode = kit.wdentities[kgval.as_entity_id()]
    lst = [x.strip() for x in val.normed_string.split(",")]
    for label in [str(qnode.label)] + [str(x) for x in qnode.aliases]:
        for item in lst:
            result = string_match_similarity(label, item)
            if result[0] is True:
                return result
    return False, 0.0

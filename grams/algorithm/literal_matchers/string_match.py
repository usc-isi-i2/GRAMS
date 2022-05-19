from typing import Tuple, cast
from grams.algorithm.literal_matchers.text_parser import ParsedTextRepr
from kgdata.wikidata.models import WDValueMonolingualText, WDValueString
import rltk
from grams.algorithm.literal_matchers.types import LiteralMatchKit


def monolingual_exact_test(
    kgval: WDValueMonolingualText, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    kgtext = kgval.value
    if val.normed_string == kgtext["text"].strip():
        return True, 1.0
    return False, 0.0


def string_exact_test(
    kgval: WDValueString, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    kgtext = kgval.value
    if val.normed_string == kgtext.strip():
        return True, 1.0
    return False, 0.0


def monolingual_similarity_test(
    kgval: WDValueMonolingualText, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    kgtext = kgval.value
    return string_match_similarity(kgtext["text"].strip(), val.normed_string)


def string_similarity_test(
    kgval: WDValueString, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    kgtext = kgval.value
    return string_match_similarity(kgtext.strip(), val.normed_string)


def string_match_similarity(s1: str, s2: str) -> Tuple[bool, float]:
    """Compare if two strings are similar

    Args:
        s1: the value in KG
        s2: the value in the cell

    Returns:
        a tuple of (whether it's matched, confidence)
    """
    if s1 == s2:
        return True, 1.0
    # calculate edit distance
    distance = rltk.levenshtein_distance(s1, s2)
    if distance <= 1:
        return True, 0.95
    elif distance > 1 and distance / max(1, min(len(s1), len(s2))) < 0.03:
        return True, 0.85
    return False, 0.0

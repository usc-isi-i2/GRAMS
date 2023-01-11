from typing import Tuple, cast
from grams.algorithm.literal_matchers.text_parser import ParsedTextRepr
from kgdata.wikidata.models import WDValueMonolingualText, WDValueString
import rltk
from grams.algorithm.literal_matchers.types import LiteralMatchKit
import rltk.similarity as sim
from sm.misc.funcs import filter_duplication
from scipy.optimize import linear_sum_assignment


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


class StrSim:
    @staticmethod
    def levenshtein(entity_mention: str, entity_label: str):
        return sim.levenshtein_similarity(entity_mention, entity_label)

    @staticmethod
    def jaro_winkler(entity_mention: str, entity_label: str):
        return sim.jaro_winkler_similarity(
            entity_mention,
            entity_label,
            threshold=0.7,
            scaling_factor=0.1,
            prefix_len=4,
        )

    @staticmethod
    def hybird_jaccard(
        entity_mention: str,
        entity_label: str,
        threshold=0.5,
        function=sim.jaro_winkler_similarity,
        parameters=None,
        lower_bound=None,
    ):
        """
        Generalized Jaccard Measure.

        NOTE:
        Copy from rltk.hybrid module to fix unnecessary type check, so that set1 and set2 can be a sequence. 
        Since using slightly changing order of tokens can change the similarity score. Is it a bug?

        Args:
            set1 (set): Set 1.
            set2 (set): Set 2.
            threshold (float, optional): The threshold to keep the score of similarity function. \
                Defaults to 0.5.
            function (function, optional): The reference of a similarity measure function. \
                It should return the value in range [0,1]. If it is set to None, \
                `jaro_winlker_similarity` will be used.
            parameters (dict, optional): Other parameters of function. Defaults to None.
            lower_bound (float): This is for early exit. If the similarity is not possible to satisfy this value, \
                the function returns immediately with the return value 0.0. Defaults to None.

        Returns:
            float: Hybrid Jaccard similarity.

        Examples:
            >>> def hybrid_test_similarity(m ,n):
            ...     ...
            >>> rltk.hybrid_jaccard_similarity(set(['a','b','c']), set(['p', 'q']), function=hybrid_test_similarity)
            0.533333333333
        """
        mention_tokens = entity_mention.split(" ")
        entity_label_tokens = entity_label.split(" ")

        set1 = filter_duplication(mention_tokens)
        set2 = filter_duplication(entity_label_tokens)

        parameters = parameters if isinstance(parameters, dict) else {}

        if len(set1) > len(set2):
            set1, set2 = set2, set1
        total_num_matches = len(set1)

        matching_score = [[1.0] * len(set2) for _ in range(len(set1))]
        row_max = [0.0] * len(set1)
        for i, s1 in enumerate(set1):
            for j, s2 in enumerate(set2):
                score = function(s1, s2, **parameters)
                if score < threshold:
                    score = 0.0
                row_max[i] = max(row_max[i], score)
                matching_score[i][j] = (
                    1.0 - score
                )  # munkres finds out the smallest element

            if lower_bound:
                max_possible_score_sum = sum(
                    row_max[: i + 1] + [1] * (total_num_matches - i - 1)
                )
                max_possible = (
                    1.0
                    * max_possible_score_sum
                    / float(len(set1) + len(set2) - total_num_matches)
                )
                if max_possible < lower_bound:
                    return 0.0

        # run munkres, finds the min score (max similarity) for each row
        row_idx, col_idx = linear_sum_assignment(matching_score)

        # recover scores
        score_sum = 0.0
        for r, c in zip(row_idx, col_idx):
            score_sum += 1.0 - matching_score[r][c]

        if len(set1) + len(set2) - total_num_matches == 0:
            return 1.0
        sim = float(score_sum) / float(len(set1) + len(set2) - total_num_matches)
        if lower_bound and sim < lower_bound:
            return 0.0
        return sim

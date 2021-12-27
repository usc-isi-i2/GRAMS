from typing import Tuple
from grams.algorithm.literal_matchers.string_match import string_match_similarity
from grams.algorithm.literal_matchers.text_parser import ParsedTextRepr
from kgdata.wikidata.models.qnode import DataValue, DataValueWikibaseEntityId

from kgdata.wikidata.models.qnode import QNode
from grams.algorithm.literal_matchers.types import LiteralMatchKit


def entity_similarity_test(
    kgval: DataValue, val: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgval: the value in KG
        val: the value in the cell
        kit: objects that may be needed for matching
    Returns:
        a tuple of (whether it's matched, confidence)
    """

    if not kgval.is_qnode():
        # not handle lexical or property yet
        return False, 0.0

    ent_id = kgval.as_entity_id()
    if ent_id not in kit.qnodes:
        # NOTE: original code to check for missing qnodes in KG subset
        # if not stmt.value.is_qnode():
        #     # lexical
        #     continue
        # if stmt.value.as_entity_id() not in self.qnodes:
        #     # this can happen due to some of the qnodes is in the link, but is missing in the KG
        #     # this is very rare so we can employ some check to make sure this is not due to
        #     # our wikidata subset
        #     is_error_in_kg = any(
        #         any(
        #             _s.value.is_qnode()
        #             and _s.value.as_entity_id() in self.qnodes
        #             for _s in _stmts
        #         )
        #         for _p, _stmts in source.props.items()
        #     ) or stmt.value.as_entity_id().startswith("L")
        #     if not is_error_in_kg:
        #         raise Exception(
        #             f"Missing qnodes in your KG subset: {stmt.value.as_entity_id()}"
        #         )
        #     continue
        return False, 0.0
    qnode = kit.qnodes[ent_id]
    lst = [x.strip() for x in val.normed_string.split(",")]
    for label in [str(qnode.label)] + [str(x) for x in qnode.aliases]:
        for item in lst:
            result = string_match_similarity(label, item)
            if result[0] is True:
                return result
    return False, 0.0

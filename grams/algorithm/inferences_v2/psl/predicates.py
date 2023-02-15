from __future__ import annotations
from pslpython.predicate import Predicate


class P:
    """Holding list of predicates in the model."""

    # target predicates
    CorrectRel = Predicate("CORRECT_REL", closed=False, size=4)
    CorrectType = Predicate("CORRECT_TYPE", closed=False, size=2)

    # graph structure
    Rel = Predicate("REL", closed=True, size=4)
    Type = Predicate("TYPE", closed=True, size=2)
    Column = Predicate("COLUMN", closed=True, size=1)
    StatementProperty = Predicate("STATEMENT_PROPERTY", closed=True, size=2)

    # ontology
    SubProp = Predicate("SUB_PROP", closed=True, size=2)
    TypeDistance = Predicate("TYPE_DISTANCE", closed=True, size=2)
    DataProperty = Predicate("DATA_PROPERTY", closed=True, size=1)
    PropertyDomain = Predicate("PROPERTY_DOMAIN", closed=True, size=2)  # (prop, domain)
    PropertyRange = Predicate("PROPERTY_RANGE", closed=True, size=2)

    # features
    RelFreqOverRow = Predicate("REL_FREQ_OVER_ROW", closed=True, size=4)
    RelFreqOverEntRow = Predicate("REL_FREQ_OVER_ENT_ROW", closed=True, size=4)
    RelFreqOverPosRel = Predicate("REL_FREQ_OVER_POS_REL", closed=True, size=4)
    RelFreqUnmatchOverEntRow = Predicate(
        "REL_FREQ_UNMATCH_OVER_ENT_ROW", closed=True, size=4
    )
    RelFreqUnmatchOverPosRel = Predicate(
        "REL_FREQ_UNMATCH_OVER_POS_REL", closed=True, size=4
    )
    RelNotFuncDependency = Predicate("REL_NOT_FUNC_DEPENDENCY", closed=True, size=2)
    RelHeaderSimilarity = Predicate("REL_HEADER_SIMILARITY", closed=True, size=4)

    TypeFreqOverRow = Predicate("TYPE_FREQ_OVER_ROW", closed=True, size=2)
    TypeFreqOverEntRow = Predicate("TYPE_FREQ_OVER_ENT_ROW", closed=True, size=2)
    ExtendedTypeFreqOverRow = Predicate(
        "EXTENDED_TYPE_FREQ_OVER_ROW", closed=True, size=2
    )
    ExtendedTypeFreqOverEntRow = Predicate(
        "EXTENDED_TYPE_FREQ_OVER_ENT_ROW", closed=True, size=2
    )
    TypeDiscoveredPropFreqOverRow = Predicate(
        "TYPE_DISCOVERED_PROP_FREQ_OVER_ROW", closed=True, size=2
    )
    TypeHeaderSimilarity = Predicate("TYPE_HEADER_SIMILARITY", closed=True, size=2)

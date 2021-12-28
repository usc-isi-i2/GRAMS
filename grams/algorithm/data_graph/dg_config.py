class DGConfigs:
    """Holding configuration for data graph construction"""

    # allow kg search for the same entity - whether we try to discover relationships between the same entity in same row but different columns
    # recommend to disable.
    ALLOW_SAME_ENT_SEARCH = False
    # removing unnecessary entities to keep the graph size reasonable
    PRUNE_REDUNDANT_ENT = True
    # remove leaf entities that is property of other nodes but the node doesn't have any other qualifiers
    PRUNE_SINGLE_LEAF_ENT = False
    # path discovering using index.
    # recommend to turn this on
    USE_KG_INDEX = True
    # enable context usage
    USE_CONTEXT = True

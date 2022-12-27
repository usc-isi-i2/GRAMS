from dataclasses import dataclass, field


@dataclass
class DGConfigs:
    """Holding configuration for data graph construction"""

    ALLOW_SAME_ENT_SEARCH: bool = field(
        default=False,
        metadata={
            "help": "allow kg search for the same entity - whether we try to discover relationships "
            "between the same entity in same row but different columns"
        },
    )
    PRUNE_REDUNDANT_ENT: bool = field(
        default=True,
        metadata={
            "help": "removing unnecessary entities to keep the graph size reasonable"
        },
    )
    PRUNE_SINGLE_LEAF_ENT: bool = field(
        default=False,
        metadata={
            "help": "remove leaf entities that is property of other nodes but "
            "the node doesn't have any other qualifiers"
        },
    )
    USE_KG_INDEX: bool = field(
        default=True,
        metadata={"help": "discover path using index. recommend to turn this on"},
    )
    USE_CONTEXT: bool = field(default=True, metadata={"help": "enable context usage"})

    max_n_hop: int = field(
        default=1, metadata={"help": "retriving entities up to this number of hops"}
    )

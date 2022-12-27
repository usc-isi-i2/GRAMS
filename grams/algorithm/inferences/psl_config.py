from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PslConfig:
    enable_logging: bool = field(
        default=False, metadata={"help": "whether to enable PSL logging"}
    )
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "values less than this threshold will be considered as false"
        },
    )
    eps: float = field(
        default=0.000,
        metadata={
            "help": "the threshold for the difference between two probabilities to be considered as the same"
        },
    )
    postprocessing: Literal[
        "steiner_tree", "arborescence", "simplepath", "pairwise"
    ] = field(
        default="steiner_tree",
        metadata={"help": "postprocessing method to use to get the final result"},
    )
    disable_rules: list[str] = field(
        default_factory=list, metadata={"help": "list of PSL rules to disable"}
    )
    experiment_model: Literal["exp", "exp2"] = field(
        default="exp2",
        metadata={"help": "PSL model to use"},
    )

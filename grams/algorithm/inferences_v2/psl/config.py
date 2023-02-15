from dataclasses import dataclass, field
from typing import Literal


def parse_rule_weight(s: str):
    name, weight = s.split("=")
    return name, float(weight)


@dataclass
class PslConfig:
    enable_logging: bool = field(
        default=False, metadata={"help": "whether to enable PSL logging"}
    )
    eps: float = field(
        default=0.000,
        metadata={
            "help": "the threshold for the difference between two probabilities to be considered as the same"
        },
    )
    disable_rules: list[str] = field(
        default_factory=list, metadata={"help": "list of PSL rules to disable"}
    )
    rule_weights: list[tuple[str, float]] = field(
        default_factory=list,
        metadata={"help": "weights of PSL rules", "parser": parse_rule_weight},
    )

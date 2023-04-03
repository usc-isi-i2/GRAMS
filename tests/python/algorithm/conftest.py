from typing import Dict
import pytest
from pytest_mock import MockerFixture
from grams.algorithm.inferences.psl_lib import PSLModel, RuleContainer
from pslpython.predicate import Predicate
from pslpython.model import ModelError
from pslpython.rule import Rule


@pytest.fixture()
def acquaintance_model():
    return PSLModel(
        predicates=[
            Predicate("KNOWS", closed=True, size=2),
            Predicate("LIKES", closed=True, size=2),
            Predicate("FRIEND", closed=False, size=2),
        ],
        rules=RuleContainer(
            {
                "NEG_PRIOR": Rule(
                    "~FRIEND(A, B)",
                    weighted=True,
                    weight=0.1,
                    squared=True,
                ),
                "LIKE": Rule(
                    "KNOWS(A, B) & LIKES(A, B) -> FRIEND(A, B)",
                    weighted=True,
                    weight=0.8,
                    squared=True,
                ),
            }
        ),
    )


def norm_psl_output(output: Dict[str, Dict[tuple, float]], digits: int):
    return {
        k: {k2: round(v2, digits) for k2, v2 in v.items()} for k, v in output.items()
    }

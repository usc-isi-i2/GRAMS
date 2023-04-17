import pytest
from pytest_mock import MockerFixture
from grams.algorithm.inferences.psl_lib import PSLModel, RuleContainer
from pslpython.predicate import Predicate
from pslpython.model import ModelError
from pslpython.rule import Rule
from tests.python.algorithm.conftest import norm_psl_output


def test_parameters(acquaintance_model: PSLModel):
    m = acquaintance_model
    params = m.parameters()
    assert len(params) == 2
    assert params["NEG_PRIOR"] == 0.1
    assert params["LIKE"] == 0.8

    observations = {
        "KNOWS": [
            ("A", "B"),
            ("A", "C"),
        ],
        "LIKES": [("A", "B")],
    }
    targets = {
        "FRIEND": [
            ("A", "B"),
            ("A", "C"),
        ]
    }

    resp = m.predict(observations, targets)
    assert norm_psl_output(resp, 3) == {
        "FRIEND": {
            ("A", "B"): 0.889,
            ("A", "C"): 0.001,
        }
    }

    m.set_parameters(
        {
            "NEG_PRIOR": 0.2,
            "LIKE": 0.5,
        }
    )

    params = m.parameters()
    assert len(params) == 2
    assert params["NEG_PRIOR"] == 0.2
    assert params["LIKE"] == 0.5

    resp = m.predict(observations, targets)
    assert norm_psl_output(resp, 3) == {
        "FRIEND": {
            ("A", "B"): 0.714,
            ("A", "C"): 0.001,
        }
    }


def test_log_errors(acquaintance_model: PSLModel, mocker: MockerFixture):
    m = acquaintance_model

    stub = mocker.stub(name="log_errors")
    mocker.patch.object(m, "log_errors", stub)
    stub.assert_not_called()

    with pytest.raises(ModelError):
        m.predict(
            {
                "KNOWS": [
                    ("A", "B"),
                    ("A", "C"),
                ],
                "LIKES": [("A", "B")],
            },
            {
                "FRIEND": [
                    ("A", "B"),
                    ("A", "C", "E"),
                ]
            },
        )
    stub.assert_called()
    assert len(m.logs) > 0


class TestRuleContainer:
    def test_no_duplicated_rule(self):
        container = RuleContainer()
        container["ABC"] = Rule("~No(A, B)", weighted=False)
        with pytest.raises(AssertionError):
            container["ABC"] = Rule("~No(A, B)", weighted=False)

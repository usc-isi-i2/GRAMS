from pathlib import Path
from grams.algorithm.literal_matchers.text_parser import (
    TextParser,
    TextParserConfigs,
)
from grams.algorithm.literal_matchers.literal_match import LiteralMatchConfigs

import grams.core as gcore
import grams.core.steps as gcore_steps
import grams.core.literal_matchers as gcore_matcher
from grams.inputs import LinkedTable
import pytest
from sm.dataset import Example
from sm_datasets.datasets import Datasets

from scripts.config import HOME_DIR, DATA_DIR


@pytest.fixture(scope="session")
def wt250():
    return Datasets().wt250()


@pytest.fixture()
def tbl(resource_dir: Path):
    return LinkedTable.from_csv_file(
        resource_dir / "data_matching/list_of_highest_mountains.csv"
    )


def test_data_matching(tbl: LinkedTable):
    gcore.GramsDB.init(str(HOME_DIR / "databases"))
    cdb = gcore.GramsDB.get_instance()

    rtable = tbl.to_rust()
    nrows, ncols = tbl.shape()
    parser = TextParser(TextParserConfigs())
    rcells = [
        [parser.parse(tbl.table[ri, ci]).to_rust() for ci in range(ncols)]
        for ri in range(nrows)
    ]
    context = cdb.get_algo_context(rtable, n_hop=1)
    literal_matcher = gcore_matcher.LiteralMatcher(LiteralMatchConfigs().to_rust())
    g = gcore_steps.data_matching.matching(
        rtable,
        rcells,
        context,
        literal_matcher,
        [],
        ["P31"],
        allow_same_ent_search=False,
        use_context=True,
    )

    it = g.iter_rels()
    assert len(it) > 0

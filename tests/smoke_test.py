from pathlib import Path
import tempfile, pytest

from grams.cli import cli
from grams.prelude import ROOT_DIR, DATA_DIR


@pytest.mark.skipif(
    not (DATA_DIR / "wdprop_domains.db").exists(), reason="no local databases"
)
def test_semtab2020_novartis():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        cli(
            # fmt: off
            [
                "-i", ROOT_DIR / "examples/semtab2020_novartis/tables/3MQ7IT3G.csv",
                "-o", tempdir / "3MQ7IT3G.json",
                "-pv",
            ],
            # fmt: on
            standalone_mode=False,
        )

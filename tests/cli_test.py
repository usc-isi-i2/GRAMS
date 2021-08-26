import tempfile
from pathlib import Path

from grams.cli import io_parser, IOFile


def test_io_parser():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        for path in [
            "website1/table_01.csv",
            "website2/gdp/uk.csv",
            "website2/population/france.csv",
        ]:
            (tempdir / path).parent.mkdir(exist_ok=True, parents=True)
            with open(str(tempdir / path), "w") as f:
                f.write("c1\tc2\tc3\nc11\tc12\tc13")

        output = io_parser(str(tempdir / "website2/{group}/{filename}.csv"), str(tempdir / "output/{group}_{filename}.csv"))
        assert output == [
            IOFile(tempdir / "website2/gdp/uk.csv", tempdir / "output/gdp_uk.csv"),
            IOFile(tempdir / "website2/population/france.csv", tempdir / "output/population_france.csv"),
        ]

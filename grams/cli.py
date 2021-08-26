import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import click
from omegaconf import OmegaConf

import grams.inputs as I
import sm.misc as M
from grams.config import ROOT_DIR
from grams.main import GRAMS


@dataclass
class IOFile:
    infile: Path
    outfile: Path


@click.command()
@click.option(
    "-i",
    "--infiles",
    required=True,
    help="""path to input tables. You can use a combination of wildcard (`*`) or named wildcard wrapped by curly
    brackets (e.g., `{name}` or `{group1}`), which behaves as wildcard. The named wildcard can be used in
    `outfiles` to name the output file. The list of input tables is retrieved by glob.glob (unix style pathname
    pattern expansion). Examples:
    - `/tables/{filename}.json`
    - `/tables/{filename}.csv`
""",
)
@click.option(
    "-o",
    "--outfiles",
    help="""path to the output annotations. You can only use named wildcard defined in `infiles`. Examples:
    - `/outputs/{filename}/version.01.json`""",
)
@click.option(
    "-d",
    "--data_dir",
    default=ROOT_DIR / "data",
    help="pass through option to GRAMS.data_dir",
)
@click.option(
    "-p",
    "--proxy",
    is_flag=True,
    default=False,
    help="pass through option to GRAMS.proxy",
)
@click.option(
    "--cfg_file",
    default=str(ROOT_DIR / "grams.yaml"),
    help="cfg_file contains configuration of GRAMS",
)
@click.option(
    "-r", "--viz", is_flag=True, default=False, help="visualize the annotated models"
)
def cli(
    infiles: str, outfiles: str, data_dir: str, proxy: bool, cfg_file: str, viz: bool
):
    """Annotate tables using GRAMS

    Args:
        infiles: path to input tables. You can use a combination of wildcard (`*`) or named wildcard wrapped by curly
            brackets (e.g., `{name}` or `{group1}`), which behaves as wildcard. The named wildcard can be used in
            `outfiles` to name the output file. The list of input tables is retrieved by glob.glob (unix style pathname
            pattern expansion). Examples:
            - `/tables/{filename}.json`
            - `/tables/{filename}.csv`
        outfiles: path to the output annotations. You can only use named wildcard defined in `infiles`. Examples:
            - `/outputs/{filename}/version.01.json`
        data_dir: pass through option to GRAMS.data_dir
        proxy: pass through option to GRAMS.proxy
        cfg_file: cfg_file contains configuration of GRAMS
    """
    cfg = OmegaConf.load(cfg_file)
    io_files = io_parser(infiles=infiles, outfiles=outfiles)
    tables = []

    for io_file in io_files:
        if io_file.infile.name.endswith(".csv"):
            tbl = I.LinkedTable.from_csv_file(io_file.infile)
        elif io_file.infile.name.endswith(".json"):
            tbl = I.LinkedTable.from_dict(M.deserialize_json(io_file.infile))
        else:
            raise NotImplementedError("%s" % io_file.infile.suffix)
        io_file.outfile.parent.mkdir(exist_ok=True, parents=True)
        tables.append(tbl)

    grams = GRAMS(data_dir=data_dir, cfg=cfg, proxy=proxy)

    for io_file, tbl in zip(io_files, tables):
        annotation = grams.annotate(tbl)
        M.serialize_json(
            {"semantic_models": [annotation.sm.to_dict()]}, io_file.outfile
        )
        if viz:
            annotation.sm.draw(io_file.outfile.parent / (io_file.outfile.stem + ".png"))


def io_parser(infiles: str, outfiles: str) -> List[IOFile]:
    # convert infile pattern torn
    infile_glob_ptn = re.sub("{([a-zA-Z0-9_]+)}", "*", infiles)
    # get a regex pattern that extract groups from input path to format the outfile pattern
    infile_re_ptn = re.sub("{([a-zA-Z0-9_]+)}", r"(?P<\g<1>>[a-zA-Z0-9_]+)", infiles)

    outputs = []
    for infile in sorted(glob.glob(infile_glob_ptn)):
        m = re.match(infile_re_ptn, infile)
        outfile = outfiles.format(**m.groupdict())
        outputs.append(IOFile(infile=Path(infile), outfile=Path(outfile)))
    return outputs


if __name__ == "__main__":
    cli()

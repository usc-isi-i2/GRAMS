import glob

from omegaconf import OmegaConf

from grams.prelude import *

cwd = ROOT_DIR / "examples/semtab2020_novartis"
cfg = OmegaConf.load(ROOT_DIR / "grams.yaml")
grams = GRAMS(DATA_DIR, cfg)

gt = [
    ([O.SemanticModel.from_json(sm) for sm in r['semantic_models']], I.LinkedTable.from_json(r['table']))
    for r in [M.deserialize_json(infile) for infile in glob.glob(str(cwd / "tables/*.json"))]
]

def annotate(table):
    global grams
    return grams.annotate(table)

annotations = M.parallel_map(annotate, [tbl for sms, tbl in gt], show_progress=True, is_parallel=True)

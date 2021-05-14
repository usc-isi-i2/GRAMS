import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
if 'DATA_DIR' not in os.environ:
    DATA_DIR = ROOT_DIR / "data"
else:
    DATA_DIR = Path(os.environ['DATA_DIR'])

assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"
DEFAULT_CONFIG = OmegaConf.load(ROOT_DIR / "grams.yaml")

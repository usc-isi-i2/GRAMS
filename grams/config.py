import os
from pathlib import Path

from omegaconf import OmegaConf

PKG_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR = PKG_DIR.parent
if "DATA_DIR" not in os.environ:
    DATA_DIR = ROOT_DIR / "data"
else:
    DATA_DIR = Path(os.environ["DATA_DIR"])
DEFAULT_CONFIG = OmegaConf.load(PKG_DIR / "grams.yaml")

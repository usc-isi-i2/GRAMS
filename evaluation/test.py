from dataclasses import dataclass
import os, time
from operator import itemgetter
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any, Optional, Set, Iterable, Union

import networkx as nx
import requests
from omegaconf import OmegaConf
from rdflib import RDFS
from tqdm.auto import tqdm

import sm.misc as M
import sm.outputs as O
import grams.inputs as I

from grams.algorithm.data_graph import build_data_graph, BuildDGOption
from grams.algorithm.kg_index import TraversalOption, KGObjectIndex
from grams.algorithm.semantic_graph import SemanticGraphConstructor, SemanticGraphConstructorArgs, viz_sg
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper
from grams.config import DEFAULT_CONFIG, ROOT_DIR

from grams.main import GRAMS, get_qnode_db
from kgdata.wikidata.models import QNode, WDProperty, WDClass, WDQuantityPropertyStats
from kgdata.wikidata.db import get_qnode_db, get_wdprop_db, get_wdclass_db, query_wikidata_entities

from sm_unk.dev.wikitable2wikidata.sxx_evaluation import get_input_data
from memory_profiler import profile
"""
Evaluate performance of GRAMS on datasets
"""

@profile
def main():
    cfg = OmegaConf.load(ROOT_DIR / "grams.yaml")
    HOME_DIR = Path("/workspace/sm-dev/data/home")
    # dataset_dir = HOME_DIR / "wikitable2wikidata/250tables"
    dataset_dir = HOME_DIR / "wikitable2wikidata/semtab2020"
    gold_models = get_input_data(dataset_dir, dataset_dir.name, n_tables=30, only_curated = True, complete_missing_links = True)
    grams = GRAMS(data_dir=HOME_DIR / "databases", cfg=cfg, proxy=False)

    lst = [x[1] for x in gold_models]
    grams.annotate(lst[0])
    grams.annotate(lst[1])
    grams.annotate(lst[2])
    grams.annotate(lst[3])
    grams.annotate(lst[4])
    grams.annotate(lst[5])
    grams.annotate(lst[6])
    grams.annotate(lst[7])
    grams.annotate(lst[8])
    grams.annotate(lst[9])
    grams.annotate(lst[10])
    grams.annotate(lst[11])
    grams.annotate(lst[12])
    grams.annotate(lst[13])
    grams.annotate(lst[14])
    grams.annotate(lst[15])
    grams.annotate(lst[16])
    grams.annotate(lst[17])
    grams.annotate(lst[18])
    grams.annotate(lst[19])
    grams.annotate(lst[20])


if __name__ == '__main__':
    main()

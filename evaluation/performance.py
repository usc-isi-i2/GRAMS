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
"""
Evaluate performance of GRAMS on datasets
"""

cfg = OmegaConf.load(ROOT_DIR / "grams.yaml")
HOME_DIR = Path("/workspace/sm-dev/data/home")
dataset_dir = HOME_DIR / "wikitable2wikidata/250tables"
# dataset_dir = HOME_DIR / "wikitable2wikidata/semtab2020"
gold_models = get_input_data(dataset_dir, dataset_dir.name, only_curated = True, complete_missing_links = True)
grams = GRAMS(data_dir=HOME_DIR / "databases", cfg=cfg, proxy=False)


def run_one_table(tbl):
    global grams
    start = time.time()
    res = grams.annotate(tbl)
    return tbl.id, time.time() - start


# for i, x in tqdm(enumerate(gold_models)):
#     grams.annotate(x[1])

with M.Timer().watch_and_report('execution time'):
    results = M.parallel_map(
        run_one_table,
        [x[1] for x in gold_models],
        show_progress=True,
        progress_desc='annotating tables',
        is_parallel=True,
        # n_processes=16,
    )
# M.serialize_json(results, "/data/binhvu/workspace/sm-dev/grams/evaluation/data.json", indent=4)
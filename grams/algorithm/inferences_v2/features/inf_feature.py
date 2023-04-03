from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, TypeVar

from grams.algorithm.candidate_graph.cg_graph import CGGraph
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences_v2.features.edge_feature import (
    EdgeFeature,
    EdgeFeatureExtractor,
)
from grams.algorithm.inferences_v2.features.helper import IDMap, K, OffsetIDMap
from grams.algorithm.inferences_v2.features.misc_feature import (
    MiscFeatureExtractor,
    MiscInfFeature,
)
from grams.algorithm.inferences_v2.features.node_feature import (
    NodeFeature,
    NodeFeatureExtractor,
)
from grams.inputs.linked_table import LinkedTable
from loguru import logger
from nptyping import Bool, Float32, Float64, Int32, NDArray, Shape
from ream.data_model_helper import (
    NumpyDataModel,
    NumpyDataModelContainer,
    NumpyDataModelHelper,
)
import serde.pickle
from timer import watch_and_report

logger = logger.bind(name=__name__)


@dataclass
class InfFeature(NumpyDataModelContainer):
    idmap: IDMap[str]
    edge_features: EdgeFeature
    node_features: NodeFeature
    misc_features: MiscInfFeature


class InfFeatureExtractor:
    """Extracting features in a candidate graph."""

    VERSION = 107

    def __init__(self, context: AlgoContext):
        self.context = context

    def extract(
        self,
        table: LinkedTable,
        dg: DGGraph,
        cg: CGGraph,
        verbose: bool = False,
    ) -> InfFeature:
        idmap = IDMap()

        edge_feat_extractor = EdgeFeatureExtractor(idmap, table, cg, dg, self.context)
        node_feat_extractor = NodeFeatureExtractor(idmap, table, cg, dg, self.context)
        misc_feat_extractor = MiscFeatureExtractor(
            idmap,
            table,
            cg,
            dg,
            self.context,
            candidate_types={
                u.id: cantypes
                for u, cantypes in node_feat_extractor.get_column_cantypes()
            },
        )

        # build idmap first
        for u in cg.iter_nodes():
            idmap.add(u.id)
        for pred in sorted({e.predicate for e in cg.iter_edges()}):
            idmap.add(pred)
        for c in node_feat_extractor.get_cantypes():
            idmap.add(c)

        with watch_and_report(
            "extract edge feature",
            print_fn=logger.debug,
            preprint=True,
            disable=not verbose,
        ):
            edge_features = edge_feat_extractor.extract()

        with watch_and_report(
            "extract node feature",
            print_fn=logger.debug,
            preprint=True,
            disable=not verbose,
        ):

            node_features = node_feat_extractor.extract()
        with watch_and_report(
            "extract misc feature",
            print_fn=logger.debug,
            preprint=True,
            disable=not verbose,
        ):
            misc_features = misc_feat_extractor.extract()

        return InfFeature(
            idmap=idmap,
            edge_features=edge_features,
            node_features=node_features,
            misc_features=misc_features,
        )

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
from nptyping import Bool, Float32, Float64, Int32, NDArray, Shape
from ream.data_model_helper import (
    NumpyDataModel,
    NumpyDataModelContainer,
    NumpyDataModelHelper,
)
import serde.pickle


@dataclass
class InfFeature(NumpyDataModelContainer):
    idmap: IDMap[str]
    edge_features: EdgeFeature
    node_features: NodeFeature
    misc_features: MiscInfFeature


@dataclass
class InfBatchFeature(InfFeature):
    edge_index: dict[str, tuple[int, int]]
    node_index: dict[str, tuple[int, int]]
    idmap: dict[str, IDMap[str]]
    edge_features: EdgeFeature
    node_features: NodeFeature
    misc_features: MiscInfFeature

    @staticmethod
    def merge(feats: list[tuple[str, InfFeature]]) -> InfBatchFeature:
        id_offset = 0
        idmap = {}

        edge_feats = []
        node_feats = []
        misc_feats = []

        for name, feat in feats:
            idmap[name] = OffsetIDMap.from_idmap(id_offset, feat.idmap)
            edge_feats.append(feat.edge_features.shift_id(id_offset))
            node_feats.append(feat.node_features.shift_id(id_offset))
            misc_feats.append(feat.misc_features.shift_id(id_offset))

            id_offset += len(feat.idmap.invert_map)

        names = [name for name, feat in feats]
        edge_index = NumpyDataModelHelper.create_simple_index(names, edge_feats)
        node_index = NumpyDataModelHelper.create_simple_index(names, node_feats)

        return InfBatchFeature(
            edge_index=edge_index,
            node_index=node_index,
            idmap=idmap,
            edge_features=EdgeFeature.concatenate(edge_feats),
            node_features=NodeFeature.concatenate(node_feats),
            misc_features=MiscInfFeature.concatenate(misc_feats),
        )


class InfFeatureExtractor:
    """Extracting features in a candidate graph."""

    VERSION = 104

    def __init__(self, context: AlgoContext):
        self.context = context

    def extract(
        self,
        table: LinkedTable,
        dg: DGGraph,
        cg: CGGraph,
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

        return InfFeature(
            idmap=idmap,
            edge_features=edge_feat_extractor.extract(),
            node_features=node_feat_extractor.extract(),
            misc_features=misc_feat_extractor.extract(),
        )

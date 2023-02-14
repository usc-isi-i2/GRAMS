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
from grams.algorithm.inferences_v2.features.helper import IDMap, K
from grams.algorithm.inferences_v2.features.node_feature import (
    NodeFeature,
    NodeFeatureExtractor,
)
from grams.inputs.linked_table import LinkedTable
from nptyping import Bool, Float32, Float64, Int32, NDArray, Shape
from ream.data_model_helper import NumpyDataModel, NumpyDataModelContainer
import serde.pickle


@dataclass
class InfFeature(NumpyDataModelContainer):
    idmap: IDMap[str]
    edge_features: EdgeFeature
    node_features: NodeFeature

    # def save(self, file: Path, compression: Optional[str] = None):
    #     compression = f".{compression}" if compression is not None else ""
    #     file.mkdir(parents=True, exist_ok=True)
    #     serde.pickle.ser(self.idmap, file / f"idmap.pkl{compression}")
    #     self.edge_features.save(file / f"edge_features.dat{compression}")
    #     self.node_features.save(file / f"node_features.dat{compression}")
    #     for k, v in self.predicates.items():
    #         v.save(file / f"predicates/{k}.dat{compression}")

    # @classmethod
    # def load(cls, file: Path, compression: Optional[str] = None):
    #     compression = f".{compression}" if compression is not None else ""

    #     idmapfile = file / f"idmap.pkl{compression}"
    #     if idmapfile.exists():
    #         raise FileNotFoundError(f"File {idmapfile} does not exist")

    #     idmap = serde.pickle.deser(idmapfile)
    #     edge_features = EdgeFeature.load(file / f"edge_features.dat{compression}")
    #     node_features = NodeFeature.load(file / f"node_features.dat{compression}")
    #     predicates = {}
    #     for predfile in (file / "predicates").iterdir():
    #         if predfile.suffixes[0] == ".dat":
    #             k = predfile.name[: sum(len(s) for s in predfile.suffixes)]
    #             predicates[k] = BinaryPredicates.load(predfile)

    #     return cls(idmap, edge_features, node_features, predicates)


@dataclass
class InfBatchFeature(InfFeature):
    index: dict[str, tuple[int, int]]
    idmap: IDMap[tuple[str, str]]
    edge_features: EdgeFeature
    node_features: NodeFeature

    @staticmethod
    def merge(feats: list[tuple[str, InfFeature]]) -> InfBatchFeature:
        idmap = IDMap()
        index = {}

        edge_feats = []
        node_feats = []

        for name, feat in feats:
            offset = len(idmap.invert_map)
            for k, nk in feat.idmap.map.items():
                nk2 = idmap.add((name, k))
                assert nk2 == nk + offset

            edge_feats.append(feat.edge_features.shift_id(offset))
            node_feats.append(feat.node_features.shift_id(offset))
            index[name] = (offset, len(idmap.invert_map))

        return InfBatchFeature(
            index=index,
            idmap=idmap,
            edge_features=EdgeFeature.concatenate(edge_feats),
            node_features=NodeFeature.concatenate(node_feats),
        )


class InfFeatureExtractor:
    """Extracting features in a candidate graph."""

    VERSION = 100

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
        )


class BinaryPredicates(NumpyDataModel):
    var1: NDArray[Shape["*"], Int32]
    var2: NDArray[Shape["*"], Int32]
    value: NDArray[Shape["*"], Float64]

    def shift_id(self, offset: int):
        return BinaryPredicates(
            var1=self.var1 + offset,
            var2=self.var2 + offset,
            value=self.value,
        )


class TenaryPredicates(NumpyDataModel):
    var1: NDArray[Shape["*"], Int32]
    var2: NDArray[Shape["*"], Int32]
    var3: NDArray[Shape["*"], Int32]
    value: NDArray[Shape["*"], Float64]

    def shift_id(self, offset: int):
        return TenaryPredicates(
            var1=self.var1 + offset,
            var2=self.var2 + offset,
            var3=self.var3 + offset,
            value=self.value,
        )


BinaryPredicates.init()
TenaryPredicates.init()

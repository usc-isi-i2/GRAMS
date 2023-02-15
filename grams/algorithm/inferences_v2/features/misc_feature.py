from __future__ import annotations
from dataclasses import dataclass
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEntityValueNode,
    CGGraph,
    CGStatementNode,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import DGGraph
from grams.algorithm.inferences_v2.features.helper import IDMap
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdproperty import (
    WDProperty,
    WDPropertyDomains,
    WDPropertyRanges,
)
import numpy as np
from nptyping import Float64, Int32, NDArray, Shape
from ream.data_model_helper import NumpyDataModel, NumpyDataModelContainer
from sm.misc.fn_cache import CacheMethod


class UnaryPredicate(NumpyDataModel):
    __slots__ = ("var", "value")

    var: NDArray[Shape["*"], Int32]
    value: NDArray[Shape["*"], Float64]

    def shift_id(self, offset: int):
        return UnaryPredicate(
            var=self.var + offset,
            value=self.value,
        )


class BinaryPredicate(NumpyDataModel):
    __slots__ = ("var1", "var2", "value")

    var1: NDArray[Shape["*"], Int32]
    var2: NDArray[Shape["*"], Int32]
    value: NDArray[Shape["*"], Float64]

    def shift_id(self, offset: int):
        return BinaryPredicate(
            var1=self.var1 + offset,
            var2=self.var2 + offset,
            value=self.value,
        )


class TenaryPredicate(NumpyDataModel):
    __slots__ = ("var1", "var2", "var3", "value")

    var1: NDArray[Shape["*"], Int32]
    var2: NDArray[Shape["*"], Int32]
    var3: NDArray[Shape["*"], Int32]
    value: NDArray[Shape["*"], Float64]

    def shift_id(self, offset: int):
        return TenaryPredicate(
            var1=self.var1 + offset,
            var2=self.var2 + offset,
            var3=self.var3 + offset,
            value=self.value,
        )


UnaryPredicate.init()
BinaryPredicate.init()
TenaryPredicate.init()


@dataclass
class MiscInfFeature(NumpyDataModelContainer):
    column: UnaryPredicate
    subprop: BinaryPredicate
    dataproperty: UnaryPredicate
    property_domain: BinaryPredicate
    property_range: BinaryPredicate

    def shift_id(self, offset: int):
        return MiscInfFeature(
            column=self.column.shift_id(offset),
            subprop=self.subprop.shift_id(offset),
            dataproperty=self.dataproperty.shift_id(offset),
            property_domain=self.property_domain.shift_id(offset),
            property_range=self.property_range.shift_id(offset),
        )

    @staticmethod
    def concatenate(feats: list[MiscInfFeature]) -> MiscInfFeature:
        return MiscInfFeature(
            column=UnaryPredicate.concatenate([feat.column for feat in feats]),
            subprop=BinaryPredicate.concatenate([feat.subprop for feat in feats]),
            dataproperty=UnaryPredicate.concatenate(
                [feat.dataproperty for feat in feats]
            ),
            property_domain=BinaryPredicate.concatenate(
                [feat.property_domain for feat in feats]
            ),
            property_range=BinaryPredicate.concatenate(
                [feat.property_range for feat in feats]
            ),
        )


class MiscFeatureExtractor:
    """Extracting misc features in a candidate graph."""

    VERSION = 101

    NO_DOMAIN_PROPS = {"P31", "P279", "P1647"}
    NO_RANGE_PROPS = {"P31", "P279", "P1647"}

    def __init__(
        self,
        idmap: IDMap,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        context: AlgoContext,
        candidate_types: dict[str, list[str]],
    ):
        self.idmap = idmap
        self.table = table
        self.cg = cg
        self.dg = dg
        self.context = context

        self.wdprops = context.wdprops
        self.wdclasses = context.wdclasses
        self.wdentities = context.wdentities
        self.wdprop_domains = context.wdprop_domains
        self.wdprop_ranges = context.wdprop_ranges

        self.cg_nodes = self.cg.nodes()
        self.cg_edges = self.cg.edges()
        self.candidate_types = candidate_types

    def extract(self) -> MiscInfFeature:
        return MiscInfFeature(
            column=self.COLUMN(),
            subprop=self.SUB_PROP(),
            dataproperty=self.DATA_PROPERTY(),
            property_domain=self.PROPERTY_DOMAIN(),
            property_range=self.PROPERTY_RANGE(),
        )

    def COLUMN(self) -> UnaryPredicate:
        """Extract column of a table for properties used in the candidate graph"""
        idmap = self.idmap
        var = np.array(
            [idmap.m(u.id) for u in self.cg_nodes if isinstance(u, CGColumnNode)],
            dtype=np.int32,
        )
        return UnaryPredicate(var, np.ones(len(var), dtype=np.float64))

    def SUB_PROP(self) -> BinaryPredicate:
        """Extract subproperty of relationship for properties used in the candidate graph"""
        idmap = self.idmap
        props = self.get_props()

        var1 = []
        var2 = []
        value = []

        for prop in props.values():
            for sub in props.values():
                if sub.id in prop.ancestors:
                    var1.append(idmap.m(prop.id))
                    var2.append(idmap.m(sub.id))
                    value.append(1.0)

        return BinaryPredicate(
            var1=np.array(var1, dtype=np.int32),
            var2=np.array(var2, dtype=np.int32),
            value=np.array(value, dtype=np.float64),
        )

    def PROPERTY_DOMAIN(self) -> BinaryPredicate:
        prop_domains = self.get_prop_domains()
        u2props = {}
        out = set()
        idmap = self.idmap

        for e in self.cg_edges:
            u = self.cg.get_node(e.source)
            if isinstance(u, CGStatementNode):
                # u is the statement node, so we don't need to care as we are only interested in domains
                continue

            if isinstance(u, CGEntityValueNode):
                # always satisfied
                continue

            if u.id not in u2props:
                u2props[u.id] = []
            u2props[u.id].append(e.predicate)

        for uid, props in u2props.items():
            for prop in props:
                if prop in self.NO_DOMAIN_PROPS:
                    for type in self.candidate_types.get(uid, []):
                        out.add((idmap.m(prop), idmap.m(type)))
                else:
                    domains = prop_domains[prop]
                    for type in self.candidate_types.get(uid, []):
                        if type in domains:
                            out.add((idmap.m(prop), idmap.m(type)))
        out2 = list(out)
        var1 = [x[0] for x in out2]
        var2 = [x[1] for x in out2]
        return BinaryPredicate(
            var1=np.array(var1, dtype=np.int32),
            var2=np.array(var2, dtype=np.int32),
            value=np.ones(len(var1), dtype=np.float64),
        )

    def PROPERTY_RANGE(self) -> BinaryPredicate:
        prop_ranges = self.get_prop_ranges()
        props = self.get_props()

        v2props = {}
        out = set()
        idmap = self.idmap

        for e in self.cg_edges:
            v = self.cg.get_node(e.target)
            if (
                isinstance(v, CGStatementNode)
                or isinstance(v, CGEntityValueNode)
                or not props[e.predicate].is_object_property()
            ):
                # v is the statement node, so we don't need to care as we are only interested in ranges
                # v is an entity, it is always satisfied
                # e is not an object property, we don't have range
                continue

            if v.id not in v2props:
                v2props[v.id] = []
            v2props[v.id].append(e.predicate)

        for vid, props in v2props.items():
            for prop in props:
                if prop in self.NO_RANGE_PROPS:
                    for type in self.candidate_types.get(vid, []):
                        out.add((idmap.m(prop), idmap.m(type)))
                else:
                    ranges = prop_ranges[prop]
                    for type in self.candidate_types.get(vid, []):
                        if type in ranges:
                            out.add((idmap.m(prop), idmap.m(type)))
        out2 = list(out)
        var1 = [x[0] for x in out2]
        var2 = [x[1] for x in out2]
        return BinaryPredicate(
            var1=np.array(var1, dtype=np.int32),
            var2=np.array(var2, dtype=np.int32),
            value=np.ones(len(var1), dtype=np.float64),
        )

    def DATA_PROPERTY(self) -> UnaryPredicate:
        "Extract data properties used in CG"
        props = self.get_props()
        idmap = self.idmap

        var = np.array(
            [idmap.m(p.id) for p in props.values() if p.is_data_property()],
            dtype=np.int32,
        )
        return UnaryPredicate(var, np.ones(len(var), dtype=np.float64))

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_props(self) -> dict[str, WDProperty]:
        """Get properties used in the candidate graph"""
        prop_ids = {edge.predicate for edge in self.cg_edges}
        return {prop_id: self.wdprops[prop_id] for prop_id in prop_ids}

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_classes(self) -> dict[str, WDClass]:
        """Get classes used in the candidate graph"""
        class_ids = {type for types in self.candidate_types.values() for type in types}
        return {class_id: self.wdclasses[class_id] for class_id in class_ids}

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_prop_domains(self) -> dict[str, WDPropertyDomains]:
        prop_ids = {edge.predicate for edge in self.cg_edges}
        assert self.wdprop_domains is not None, "Property domains not provided"
        return {
            prop_id: self.wdprop_domains[prop_id]
            for prop_id in prop_ids
            if prop_id not in self.NO_DOMAIN_PROPS
        }

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_prop_ranges(self) -> dict[str, WDPropertyRanges]:
        props = self.get_props()
        assert self.wdprop_ranges is not None, "Property ranges not provided"
        return {
            prop_id: self.wdprop_ranges[prop_id]
            for prop_id, prop in props.items()
            if prop_id not in self.NO_RANGE_PROPS and prop.is_object_property()
        }

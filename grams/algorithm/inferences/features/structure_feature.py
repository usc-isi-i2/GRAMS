from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)
from operator import attrgetter
from grams.algorithm.data_graph.dg_graph import DGGraph, DGNode, EntityValueNode
from grams.algorithm.inferences.psl_lib import IDMap

from grams.algorithm.data_graph import CellNode
from grams.algorithm.literal_matchers import TextParser
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGNode,
    CGStatementNode,
)
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import (
    WDEntity,
    WDProperty,
    WDQuantityPropertyStats,
    WDEntityLabel,
    WDClass,
)
from sm.misc.fn_cache import CacheMethod


K = TypeVar("K")
V = TypeVar("V")


class StructureFeature:
    """Extract structured features from the candidate graph

    Args:
        idmap: IDMap
        table: LinkedTable
        cg: Candidate graph
        dg: Data graph
        wdentities: Wikidata entities
        wdentity_labels: Wikidata entity labels
        wdclasses: Wikidata classes
        wdprops: Wikidata properties
        wd_num_prop_stats: Wikidata number property stats
        sim_fn: Similarity function
        candidate_types: mapping from a node id (in candidate graph) to its candidate types.
    """

    def __init__(
        self,
        idmap: IDMap,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        wdentities: Mapping[str, WDEntity],
        wdentity_labels: Mapping[str, WDEntityLabel],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
        wd_num_prop_stats: Mapping[str, WDQuantityPropertyStats],
        sim_fn: Optional[Callable[[str, str], float]],
        candidate_types: Dict[str, List[str]],
    ):
        self.idmap = idmap
        self.table = table
        self.cg = cg
        self.dg = dg
        self.wdentities = wdentities
        self.wdclasses = wdclasses
        self.wdprops = wdprops
        self.wd_num_prop_stats = wd_num_prop_stats

        self.cg_nodes = self.cg.nodes()
        self.cg_edges = self.cg.edges()
        self.candidate_types = candidate_types

    def extract_features(self, features: List[str]) -> Dict[str, list]:
        feat_data = {}
        for feat in features:
            fn = getattr(self, feat)
            feat_data[feat] = fn()
        return feat_data

    def REL(self) -> List[Tuple[str, str, str]]:
        """Extract relationships in the candidate graph"""
        idmap = self.idmap
        return [
            (idmap.m(e.source), idmap.m(e.target), idmap.m(e.predicate))
            for e in self.cg_edges
        ]

    def TYPE(self) -> List[Tuple[str, str]]:
        """Extract types of nodes in the candidate graph"""
        idmap = self.idmap
        return [
            (idmap.m(uid), idmap.m(type))
            for uid, types in self.candidate_types.items()
            for type in types
        ]

    def STATEMENT(self) -> List[Tuple[str]]:
        """Extract nodes that are statements in CG"""
        idmap = self.idmap
        return [
            (idmap.m(u.id),) for u in self.cg_nodes if isinstance(u, CGStatementNode)
        ]

    def NOT_STATEMENT(self) -> List[Tuple[str]]:
        """Extract nodes that are not statements in CG"""
        idmap = self.idmap
        return [
            (idmap.m(u.id),)
            for u in self.cg_nodes
            if not isinstance(u, CGStatementNode)
        ]

    def SUB_PROP(self) -> List[Tuple[str, str]]:
        """Extract subproperty of relationship for properties used in the candidate graph"""
        idmap = self.idmap
        props = self.get_props()
        return [
            (idmap.m(prop.id), idmap.m(sub.id))
            for prop in props.values()
            for sub in props.values()
            if sub.id in prop.ancestors
        ]

    def SUB_TYPE(self) -> List[Tuple[str, str]]:
        """Extract subclass of relationship of types used in the candidate graph"""
        idmap = self.idmap
        classes = self.get_classes()
        return [
            (idmap.m(cls.id), idmap.m(parent.id))
            for cls in classes.values()
            for parent in classes.values()
            if parent.id in cls.ancestors
        ]

    def OBJECT_PROPERTY(self) -> List[Tuple[str]]:
        "Extract object properties used in CG"
        props = self.get_props()
        idmap = self.idmap
        return [(idmap.m(p.id),) for p in props.values() if p.is_object_property()]

    def DATA_PROPERTY(self) -> List[Tuple[str]]:
        "Extract data properties used in CG"
        props = self.get_props()
        idmap = self.idmap
        return [(idmap.m(p.id),) for p in props.values() if p.is_data_property()]

    def DOMAIN_OF_PROPERTY(self) -> List[Tuple[str, str]]:
        assert False

    def RANGE_OF_PROPERTY(self) -> List[Tuple[str, str]]:
        assert False

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_props(self) -> Dict[str, WDProperty]:
        """Get properties used in the candidate graph"""
        prop_ids = {edge.predicate for edge in self.cg_edges}
        return {prop_id: self.wdprops[prop_id] for prop_id in prop_ids}

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_classes(self) -> Dict[str, WDClass]:
        """Get classes used in the candidate graph"""
        class_ids = {type for types in self.candidate_types.values() for type in types}
        return {class_id: self.wdclasses[class_id] for class_id in class_ids}

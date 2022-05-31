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
    Union,
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
from kgdata.wikidata.models.wdproperty import WDPropertyDomains, WDPropertyRanges
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

    # list of properties that all classes can be its domain.
    # these properties are typically not included in the databases to
    # keep the database size reasonable
    NO_DOMAIN_PROPS = {"P31", "P279", "P1647"}
    NO_RANGE_PROPS = {"P31", "P279", "P1647"}

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
        wdprop_domains: Optional[Mapping[str, WDPropertyDomains]],
        wdprop_ranges: Optional[Mapping[str, WDPropertyRanges]],
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
        self.wdprop_domains = wdprop_domains
        self.wdprop_ranges = wdprop_ranges
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

    def HAS_TYPE(self) -> List[Union[Tuple[str], Tuple[str, float]]]:
        idmap = self.idmap
        lst: List[Union[Tuple[str], Tuple[str, float]]] = [
            (idmap.m(uid),) for uid in self.candidate_types.keys()
        ]
        for node in self.cg_nodes:
            if isinstance(node, CGEntityValueNode):
                lst.append((idmap.m(node.id), 1.0))
            elif isinstance(node, CGLiteralValueNode):
                lst.append((idmap.m(node.id), 0.0))
            elif isinstance(node, CGColumnNode) and node.id not in self.candidate_types:
                lst.append((idmap.m(node.id), 0.0))
        return lst

    def COLUMN(self) -> List[Tuple[str]]:
        idmap = self.idmap
        return [(idmap.m(u.id),) for u in self.cg_nodes if isinstance(u, CGColumnNode)]

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

    def PROPERTY_DOMAIN(self) -> List[Tuple[str, str]]:
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
        return list(out)

    def PROPERTY_RANGE(self) -> List[Tuple[str, str]]:
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
        return list(out)

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

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_prop_domains(self) -> Dict[str, WDPropertyDomains]:
        prop_ids = {edge.predicate for edge in self.cg_edges}
        assert self.wdprop_domains is not None, "Property domains not provided"
        return {
            prop_id: self.wdprop_domains[prop_id]
            for prop_id in prop_ids
            if prop_id not in self.NO_DOMAIN_PROPS
        }

    @CacheMethod.cache(CacheMethod.as_is_posargs)
    def get_prop_ranges(self) -> Dict[str, WDPropertyRanges]:
        props = self.get_props()
        assert self.wdprop_ranges is not None, "Property ranges not provided"
        return {
            prop_id: self.wdprop_ranges[prop_id]
            for prop_id, prop in props.items()
            if prop_id not in self.NO_RANGE_PROPS and prop.is_object_property()
        }

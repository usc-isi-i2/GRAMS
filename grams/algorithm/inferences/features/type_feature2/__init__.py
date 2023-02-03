from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    EdgeFlowTarget,
    EntityValueNode,
    LinkGenMethod,
    LiteralValueNode,
)
from grams.algorithm.inferences.features.graph_helper import GraphHelper
from grams.algorithm.inferences.features.tree_utils import TreeStruct
from grams.algorithm.inferences.psl_lib import IDMap
from sm.misc.fn_cache import CacheMethod
import numpy as np
from grams.algorithm.data_graph import CellNode
from grams.algorithm.candidate_graph.cg_graph import CGColumnNode, CGGraph
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import (
    WDEntity,
    WDProperty,
    WDQuantityPropertyStats,
    WDClass,
)
from operator import itemgetter


class TypeFeaturesConfigs:
    EXTENDED_DISTANCE: int = 1


class TypeFeatures2:
    INSTANCE_OF = "P31"
    HIERARCHY_PROPS = {"P131", "P276"}

    def __init__(
        self,
        idmap: IDMap,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        context: AlgoContext,
        sim_fn: Optional[Callable[[str, str], float]] = None,
        graph_helper: Optional[GraphHelper] = None,
    ):
        self.idmap = idmap
        self.table = table
        self.cg = cg
        self.dg = dg
        self.wdentities = context.wdentities
        self.wdclasses = context.wdclasses
        self.wdprops = context.wdprops
        self.wd_num_prop_stats = context.wd_num_prop_stats
        self.sim_fn = sim_fn
        self.graph_helper = graph_helper or GraphHelper(table, cg, dg, context)

    def extract_features(self, features: List[str]) -> Dict[str, list]:
        # gather the list of entity columns (columns will be tagged with type)
        tagged_columns = []
        for u in self.cg.iter_nodes():
            if not isinstance(u, CGColumnNode):
                continue

            cells = self.get_cells(u)

            # using heuristic to determine if we should tag this column
            covered_fractions = [
                sum(
                    span.length
                    for spans in cell.entity_spans.values()
                    for span in spans
                )
                / max(len(cell.value), 1)
                for cell in cells
                if len(cell.entity_ids) > 0
            ]

            if len(covered_fractions) == 0:
                continue

            avg_covered_fractions = np.mean(covered_fractions)
            if avg_covered_fractions < 0.8:
                if avg_covered_fractions == 0:
                    avg_cell_len = np.mean(
                        [len(cell.value) for cell in cells if len(cell.entity_ids) > 0]
                    )
                    if avg_cell_len < 1:
                        # links are likely to be image such as national flag, so we still model them
                        tagged_columns.append(u)
                continue

            tagged_columns.append(u)

        # extract features
        feat_data = {}
        for feat in features:
            fn = getattr(self, feat)
            feat_data[feat] = [v for u in tagged_columns for v in fn(u)]
        return feat_data

    def TYPE_DISCOVERED_PROP_FREQ_OVER_ROW(
        self, u: CGColumnNode
    ) -> List[Tuple[str, float]]:
        """The frequency of whether an entity of a type in a row has been used to construct an edge in DGraph (connecting to other nodes)."""
        # gather all incoming and outgoing edges
        inedges = self.cg.in_edges(u.id)
        outedges = self.cg.out_edges(u.id)

        # list of types of the columns that we wish to find out
        can_types = list(self.get_type_freq(u).keys())
        # type2row[type][row] = 1 when an entity of type in a row is used, 0 otherwise
        type2row = {type: [0] * self.graph_helper.nrows for type in can_types}

        # for outgoing edges, gather the entities from the column that we discovered the relationship.
        for outedge in outedges:
            cg_stmt = self.cg.get_statement_node(outedge.target)
            for (source, target), dg_stmts in cg_stmt.flow.items():
                if (
                    source.sg_source_id != u.id
                    or source.sg_source_id == target.sg_target_id
                ):
                    continue
                dgu = self.dg.get_cell_node(source.dg_source_id)
                dgv = self.dg.get_node(target.dg_target_id)
                if not (
                    isinstance(dgv, CellNode)
                    or (
                        isinstance(dgv, (LiteralValueNode, EntityValueNode))
                        and dgv.is_context
                    )
                ):
                    continue

                row = dgu.row
                # for each type
                for dgsid in dg_stmts:
                    ent_types = self.graph_helper.get_dg_statement_source_entity_types(
                        dgsid
                    )
                    for type in can_types:
                        # check here
                        if type2row[type][row] == 1:
                            continue
                        if type in ent_types:
                            type2row[type][row] = 1

        for inedge in inedges:
            cg_stmt = self.cg.get_statement_node(inedge.source)
            for (source, target), dg_stmts in cg_stmt.flow.items():
                if (
                    target.sg_target_id != u.id
                    or source.sg_source_id == target.sg_target_id
                ):
                    continue

                dgu = self.dg.get_cell_node(source.dg_source_id)
                is_qualifier = source.edge_id != target.edge_id
                for dgsid, dgsprovs in dg_stmts.items():
                    matched_eids = self.graph_helper.get_dg_statement_target_kgentities(
                        dgsid, EdgeFlowTarget(target.dg_target_id, target.edge_id)
                    )
                    if len(matched_eids) == 0:
                        continue

                    matched_ent_types = {
                        etype
                        for eid in matched_eids
                        for etype in self.graph_helper.get_entity_types(eid)
                    }

                    for type in can_types:
                        if type2row[type][dgu.row] == 1:
                            continue
                        if type in matched_ent_types:
                            type2row[type][dgu.row] = 1

        output = []
        for type in can_types:
            freq = sum(type2row[type]) / self.graph_helper.nrows
            output.append((self.idmap.m(u.id), self.idmap.m(type), freq))
        return output

    def TYPE_FREQ_OVER_ROW(self, u: CGColumnNode) -> List[Tuple[str, str, float]]:
        type_freq = self.get_type_freq(u)
        output = []
        n_rows = self.table.size()
        for c, freq in type_freq.items():
            output.append((self.idmap.m(u.id), self.idmap.m(c), freq / n_rows))
        return output

    def EXTENDED_TYPE_FREQ_OVER_ROW(
        self, u: CGColumnNode
    ) -> List[Tuple[str, str, float]]:
        extended_type_freq = self.get_extended_type_freq(u)
        output = []
        n_rows = self.table.size()
        for c, freq in extended_type_freq.items():
            output.append((self.idmap.m(u.id), self.idmap.m(c), freq / n_rows))
        return output

    def TYPE_FREQ_OVER_ENT_ROW(self, u: CGColumnNode) -> List[Tuple[str, str, float]]:
        type_freq = self.get_type_freq(u)
        output = []
        n_ent_rows = self.get_num_ent_rows(u)
        for c, freq in type_freq.items():
            output.append((self.idmap.m(u.id), self.idmap.m(c), freq / n_ent_rows))
        return output

    def EXTENDED_TYPE_FREQ_OVER_ENT_ROW(
        self, u: CGColumnNode
    ) -> List[Tuple[str, str, float]]:
        extended_type_freq = self.get_extended_type_freq(u)
        output = []
        n_ent_rows = self.get_num_ent_rows(u)
        for c, freq in extended_type_freq.items():
            output.append((self.idmap.m(u.id), self.idmap.m(c), freq / n_ent_rows))
        return output

    def HAS_SUB_TYPE(self, u: CGColumnNode):
        output = set()
        classes = self.get_extended_type_freq(u)
        for c in classes:
            for pc in self.wdclasses[c].parents:
                if pc in classes:
                    output.add((self.idmap.m(u.id), self.idmap.m(pc)))
        return list(output)

    def ABS_TYPE_DISTANCE(self, u: CGColumnNode):
        """Get types and distance from the class to its most specific type."""
        output = set()
        classes = self.get_extended_type_freq(u)
        tree = self.get_candidate_types_as_maps(u)

        # start with leaf nodes
        start_nodes: List[Tuple[str, int]] = [
            (c, 0) for c in classes if len(tree.p2cs[c]) == 0
        ]
        best_distance = {}
        while len(start_nodes) > 0:
            node, distance = start_nodes.pop()
            best_distance[node] = min(distance, best_distance.get(node, float("inf")))
            for child in tree.c2ps[node]:
                start_nodes.append((child, distance + 1))

        # longest distance, can't use distance to root because root node
        # can even have shorter distance than one of its children due to varied tree length
        longest_distance = max(1, max(best_distance.values()))
        for c, distance in best_distance.items():
            output.add(
                (self.idmap.m(u.id), self.idmap.m(c), distance / longest_distance)
            )
        return list(output)

    def TYPE_DISTANCE(self, u: CGColumnNode):
        output = set()
        extended_classes = self.get_extended_type_freq(u)
        if len(extended_classes) == 0:
            return []

        classes = self.get_type_freq(u)
        tree = self.get_candidate_types_as_maps(u)

        top_classes = sorted(classes.items(), key=itemgetter(1), reverse=True)[:2]
        # now calculate the distance from the top classes into its parents
        base = set(c for c, f in top_classes)
        best_distance: Dict[str, float] = {c: 0 for c in base}
        for c in base:
            stack: List[Tuple[str, float]] = [
                (p, 1) for p in tree.c2ps[c] if p not in base
            ]
            while len(stack) > 0:
                node, distance = stack.pop()
                best_distance[node] = min(
                    distance, best_distance.get(node, float("inf"))
                )
                for p in tree.c2ps[node]:
                    if p not in base:
                        stack.append((p, distance + 1))

        longest_distance = max(1, max(best_distance.values()))

        for c in extended_classes:
            if c not in best_distance:
                # we can't reach these classes from the top classes
                # TODO: what should be the value? for now, we just set it to be half of the max value (50%)
                best_distance[c] = longest_distance / 2

        for c, distance in best_distance.items():
            output.add(
                (self.idmap.m(u.id), self.idmap.m(c), distance / longest_distance)
            )
        return list(output)

    def TYPE_HEADER_SIMILARITY(self, u: CGColumnNode):
        if self.sim_fn is None:
            return []

        extended_type_freq = self.get_extended_type_freq(u)
        output = []

        for c, freq in extended_type_freq.items():
            sim = self.sim_fn(
                u.label,
                self.wdclasses[c].label,
            )
            output.append((self.idmap.m(u.id), self.idmap.m(c), sim))
        return output

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_type_freq(self, u: CGColumnNode) -> Dict[str, float]:
        """Calculating frequency of types in a column.
        Each time a type appears in a cell, instead of counting 1, we count its probability
        """
        cells = self.get_cells(u)
        cell2qnodes = self.get_cell_to_qnodes(u)

        type2freq = {}

        for cell in cells:
            classes = {}
            for qnode, prob in cell2qnodes[cell.id]:
                for stmt in qnode.props.get(self.INSTANCE_OF, []):
                    assert stmt.value.is_entity_id(stmt.value), stmt.value
                    stmt_value_ent_id = stmt.value.as_entity_id()
                    if stmt_value_ent_id not in self.wdclasses:
                        # sometimes people just tag things incorrectly, e.g.,
                        # Q395 is not instanceof Q41511 (Q41511 is not a class)
                        continue
                    classes[stmt_value_ent_id] = max(
                        prob, classes.get(stmt_value_ent_id, 0)
                    )

            for c, prob in classes.items():
                if c not in type2freq:
                    type2freq[c] = 0.0
                type2freq[c] += prob

        return type2freq

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_extended_type_freq(self, u: CGColumnNode) -> Dict[str, float]:
        """Calculating frequency of types in a column.
        Each time a type appears in a cell, instead of counting 1, we count its probability
        """
        cells = self.get_cells(u)
        cell2qnodes = self.get_cell_to_qnodes(u)

        extended_type2freq = {}

        for cell in cells:
            classes = {}
            for qnode, prob in cell2qnodes[cell.id]:
                for stmt in qnode.props.get(self.INSTANCE_OF, []):
                    stmt_value_ent_id = stmt.value.as_entity_id_safe()
                    if stmt_value_ent_id not in self.wdclasses:
                        # sometimes people just tag things incorrectly, e.g.,
                        # Q395 is not instanceof Q41511 (Q41511 is not a class)
                        continue
                    classes[stmt_value_ent_id] = max(
                        prob, classes.get(stmt_value_ent_id, 0)
                    )
            classes = self._get_extended_type_freq_of_cell(
                classes, TypeFeaturesConfigs.EXTENDED_DISTANCE
            )
            for c, prob in classes.items():
                if c not in extended_type2freq:
                    extended_type2freq[c] = 0.0
                extended_type2freq[c] += prob

        return extended_type2freq

    def _get_extended_type_freq_of_cell(self, classes: Dict[str, float], distance: int):
        """Get parents of classes within distance from the node.
        Distance 1 is the parent, Distance 2 is the grandparent.
        Return the classes and their parents
        """
        if distance <= 0:
            return classes
        output = classes.copy()
        for klass, prob in classes.items():
            for parent_klass in self.wdclasses[klass].parents:
                output[parent_klass] = max(output.get(parent_klass, 0), prob)
        if distance > 1:
            return self._get_extended_type_freq_of_cell(output, distance - 1)
        return output

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_candidate_types_as_maps(self, u: CGColumnNode) -> TreeStruct:
        """Get candidate types of a column in a form of two maps: one from parent to children, and another one from child to parents
        If there are cycles, we break them
        """
        candidate_types = self.get_extended_type_freq(u)
        return TreeStruct.construct(
            candidate_types, lambda x: self.wdclasses[x].parents
        ).ensure_tree()

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_cells(self, u: CGColumnNode) -> List[CellNode]:
        """Get cells of a column node"""
        return [self.dg.get_cell_node(cid) for cid in u.nodes]

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_num_ent_rows(self, u: CGColumnNode) -> int:
        """Get number of rows of entities in a column"""
        return sum([1 for cell in self.get_cells(u) if len(cell.entity_ids) > 0])

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def get_cell_to_qnodes(
        self, u: CGColumnNode
    ) -> Dict[str, List[Tuple[WDEntity, float]]]:
        """Get a mapping to associated qnodes of a cell"""
        cells = self.get_cells(u)
        cell2qnodes = {}
        for cell in cells:
            self._add_merge_qnodes(cell, cell2qnodes)
        return cell2qnodes

    def _add_merge_qnodes(
        self, cell: CellNode, cell2qnodes: Dict[str, List[Tuple[WDEntity, float]]]
    ):
        """merge qnodes that are sub of each other
        attempt to merge qnodes (spatial) if they are contained in each other
        we should go even higher order
        """
        assert cell.id not in cell2qnodes
        if len(cell.entity_ids) > 1:
            # attempt to merge qnodes (spatial) if they are contained in each other
            # we should go even higher order
            ignore_qnodes = set()
            for q0_id in cell.entity_ids:
                q0 = self.wdentities[q0_id]
                vals = {
                    stmt.value.as_entity_id_safe()
                    for p in self.HIERARCHY_PROPS
                    for stmt in q0.props.get(p, [])
                }
                for q1_id in cell.entity_ids:
                    if q0_id == q1_id:
                        continue
                    if q1_id in vals:
                        # q0 is inside q1, ignore q1
                        ignore_qnodes.add(q1_id)
            qnode_lst = [
                self.wdentities[q_id]
                for q_id in cell.entity_ids
                if q_id not in ignore_qnodes
            ]
        elif len(cell.entity_ids) > 0:
            qnode_lst = [self.wdentities[cell.entity_ids[0]]]
        else:
            qnode_lst = []

        qnode2prob = {}
        for link in self.table.links[cell.row][cell.column]:
            for c in link.candidates:
                qnode2prob[c.entity_id] = max(
                    qnode2prob.get(c.entity_id, 0), c.probability
                )

        cell2qnodes[cell.id] = [(qnode, qnode2prob[qnode.id]) for qnode in qnode_lst]

    def _print_class_hierarchy(self, classes: List[str]):
        """Print a tree"""
        p2c = {c: [] for c in classes}
        c2p = {c: [] for c in classes}
        for c in classes:
            for p in self.wdclasses[c].parents:
                if p in classes:
                    p2c[p].append(c)
                    c2p[c].append(p)
        roots = [c for c, ps in c2p.items() if len(ps) == 0]
        label = lambda r: f"{self.wdclasses[r].label} ({r})"
        visited = {}
        counter = 0

        for root in roots:
            stack: List[Tuple[str, int]] = [(root, 0)]
            while len(stack) > 0:
                counter += 1
                node, depth = stack.pop()
                postfix = f" (visited at {visited[node]})" if node in visited else ""
                if depth == 0:
                    print(f"{counter}.\t{label(node)}{postfix}")
                else:
                    indent = "│   " * (depth - 1)
                    print(f"{counter}.\t{indent}├── {label(node)}{postfix}")

                if node in visited:
                    continue
                else:
                    visited[node] = counter

                for child in p2c[node]:
                    stack.append((child, depth + 1))

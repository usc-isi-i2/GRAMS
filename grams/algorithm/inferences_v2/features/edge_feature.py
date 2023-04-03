from __future__ import annotations

import numpy as np
from nptyping import Bool, Float64, Int32, NDArray, Shape
from ream.data_model_helper import NumpyDataModel
from timer import watch_and_report
from tqdm.auto import tqdm

from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import CellNode, DGGraph, EntityValueNode
from grams.algorithm.inferences_v2.features.detect_contradicted_info import (
    ContradictedInformationDetector,
)
from grams.algorithm.inferences_v2.features.func_dependency import (
    FunctionalDependencyDetector,
)
from grams.algorithm.inferences_v2.features.graph_traversal_helper import (
    GraphTraversalHelper,
)
from grams.algorithm.inferences_v2.features.helper import MISSING_VALUE, IDMap
from grams.inputs.linked_table import LinkedTable
from sm.misc.fn_cache import CacheMethod


class EdgeFeature(NumpyDataModel):
    # fmt: off
    __slots__ = [
        "source", "target", "statement", "inprop", "outprop", "is_qualifier",
        "freq_over_row", "freq_over_ent_row", "freq_over_pos_rel", "freq_unmatch_over_ent_row",
        "freq_unmatch_over_pos_rel", "not_func_dependency"
    ]
    # fmt: on

    source: NDArray[Shape["*"], Int32]
    target: NDArray[Shape["*"], Int32]
    statement: NDArray[Shape["*"], Int32]
    inprop: NDArray[Shape["*"], Int32]  # incoming property of the statement
    outprop: NDArray[Shape["*"], Int32]  # outgoing property of the statement
    is_qualifier: NDArray[Shape["*"], Bool]
    freq_over_row: NDArray[Shape["*"], Float64]
    freq_over_ent_row: NDArray[Shape["*"], Float64]
    freq_over_pos_rel: NDArray[Shape["*"], Float64]
    freq_unmatch_over_ent_row: NDArray[Shape["*"], Float64]
    freq_unmatch_over_pos_rel: NDArray[Shape["*"], Float64]
    not_func_dependency: NDArray[Shape["*"], Int32]

    def shift_id(self, offset: int):
        return EdgeFeature(
            source=self.source + offset,
            target=self.target + offset,
            statement=self.statement + offset,
            inprop=self.inprop + offset,
            outprop=self.outprop + offset,
            is_qualifier=self.is_qualifier,
            freq_over_row=self.freq_over_row,
            freq_over_ent_row=self.freq_over_ent_row,
            freq_over_pos_rel=self.freq_over_pos_rel,
            freq_unmatch_over_ent_row=self.freq_unmatch_over_ent_row,
            freq_unmatch_over_pos_rel=self.freq_unmatch_over_pos_rel,
            not_func_dependency=self.not_func_dependency,
        )


EdgeFeature.init()


class EdgeFeatureExtractor:
    """Extracting edge features in a candidate graph."""

    VERSION = 101

    def __init__(
        self,
        idmap: IDMap,
        table: LinkedTable,
        cg: CGGraph,
        dg: DGGraph,
        context: AlgoContext,
    ):
        self.idmap = idmap
        self.table = table
        self.cg = cg
        self.dg = dg
        self.context = context

        self.wdprops = context.wdprops
        self.wdentities = context.wdentities

        self.traversal = GraphTraversalHelper(table, cg, dg)
        self.contradicted_info_detector = ContradictedInformationDetector(
            correct_entity_threshold=0.8, traversal=self.traversal, context=context
        )
        self.functional_dependency_detector = FunctionalDependencyDetector(table)

    def extract(
        self,
    ) -> EdgeFeature:
        idmap = self.idmap
        rels: list[tuple[CGStatementNode, CGEdge, CGEdge]] = []

        for s in self.cg.iter_nodes():
            if not isinstance(s, CGStatementNode):
                continue
            (inedge,) = self.cg.in_edges(s.id)
            for outedge in self.cg.out_edges(s.id):
                rels.append((s, inedge, outedge))

        # extract edge features
        source = []
        target = []
        statement = []
        inprop = []
        outprop = []
        is_qualifier = []

        for s, inedge, outedge in rels:
            source.append(idmap.m(inedge.source))
            target.append(idmap.m(outedge.target))
            statement.append(idmap.m(s.id))
            inprop.append(idmap.m(inedge.predicate))
            outprop.append(idmap.m(outedge.predicate))
            # TODO: store is_qualifier in the CGEdge
            is_qualifier.append(outedge.predicate != inedge.predicate)

        with watch_and_report("freq over row", preprint=True):
            freq_over_row = self.FREQ_OVER_ROW(rels)
        with watch_and_report("freq over ent row", preprint=True):
            freq_over_ent_row = self.FREQ_OVER_ENT_ROW(rels)
        with watch_and_report("freq over pos rel", preprint=True):
            freq_over_pos_rel = self.FREQ_OVER_POS_REL(rels)
        with watch_and_report("freq unmatch over ent row", preprint=True):
            freq_unmatch_over_ent_row = self.FREQ_UNMATCH_OVER_ENT_ROW(rels)
        with watch_and_report("freq unmatch over pos rel", preprint=True):
            freq_unmatch_over_pos_rel = self.FREQ_UNMATCH_OVER_POS_REL(rels)
        with watch_and_report("not func dependency", preprint=True):
            not_func_dependency = self.NOT_FUNC_DEPENDENCY(rels)

        return EdgeFeature(
            source=np.array(source, dtype=np.int32),
            target=np.array(target, dtype=np.int32),
            statement=np.array(statement, dtype=np.int32),
            inprop=np.array(inprop, dtype=np.int32),
            outprop=np.array(outprop, dtype=np.int32),
            is_qualifier=np.array(is_qualifier, dtype=np.bool_),
            freq_over_row=np.array(freq_over_row, dtype=np.float64),
            freq_over_ent_row=np.array(freq_over_ent_row, dtype=np.float64),
            freq_over_pos_rel=np.array(freq_over_pos_rel, dtype=np.float64),
            freq_unmatch_over_ent_row=np.array(
                freq_unmatch_over_ent_row, dtype=np.float64
            ),
            freq_unmatch_over_pos_rel=np.array(
                freq_unmatch_over_pos_rel, dtype=np.float64
            ),
            not_func_dependency=np.array(not_func_dependency, dtype=np.int32),
        )

    def FREQ_OVER_ROW(self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]):
        """The frequency of the relation over the row."""
        n_rows = self.table.size()
        output = []
        for s, inedge, outedge in rels:
            ratio = self.get_rel_freq(s, outedge) / n_rows
            output.append(ratio)
        return output

    def FREQ_OVER_ENT_ROW(self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]):
        # what is the maximum possible links we can have? this ignore the the link so this is used to calculate FreqOverEntRow
        output = []
        for s, inedge, outedge in rels:
            max_pos_ent_rows = self.get_maximum_possible_ent_links_between_two_nodes(
                s, inedge, outedge
            )
            freq = self.get_rel_freq(s, outedge)
            ratio = 0 if max_pos_ent_rows == 0 else freq / max_pos_ent_rows
            output.append(ratio)
        return output

    def FREQ_OVER_POS_REL(self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]):
        """The frequency of the relation over all possible relations that the two nodes can have."""
        output = [0.0] * len(rels)
        tmp = {}
        max_possible_links = {}

        for reli, (s, inedge, outedge) in enumerate(tqdm(rels)):
            freq = self.get_rel_freq(s, outedge)
            n_possible_links = self.get_unmatch_discovered_links(
                s, inedge, outedge
            ) + len(self.traversal.get_rel_dg_pairs(s, outedge))

            max_possible_links[inedge.source, outedge.target] = max(
                n_possible_links,
                max_possible_links.get((inedge.source, outedge.target), 0),
            )

            tmp[
                (
                    self.idmap.m(inedge.source),
                    self.idmap.m(outedge.target),
                    self.idmap.m(s.id),
                    self.idmap.m(outedge.predicate),
                )
            ] = (reli, freq, (inedge.source, outedge.target))

        for key, (reli, freq, pair) in tmp.items():
            prob = (
                0.0
                if max_possible_links[pair] == 0
                else freq / max_possible_links[pair]
            )
            output[reli] = prob

        return output

    def FREQ_UNMATCH_OVER_ENT_ROW(
        self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]
    ):
        output = []
        for s, inedge, outedge in tqdm(rels):
            max_pos_ent_rows = self.get_maximum_possible_ent_links_between_two_nodes(
                s, inedge, outedge
            )
            n_unmatch_links = len(
                self.contradicted_info_detector.get_contradicted_information(
                    s, inedge, outedge
                )
            )
            ratio = 0 if max_pos_ent_rows == 0 else n_unmatch_links / max_pos_ent_rows
            output.append(ratio)
        return output

    def FREQ_UNMATCH_OVER_POS_REL(
        self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]
    ):
        output = [0.0] * len(rels)
        tmp = {}
        max_possible_links = {}

        for reli, (s, inedge, outedge) in enumerate(tqdm(rels)):
            n_unmatch_links = len(
                self.contradicted_info_detector.get_contradicted_information(
                    s, inedge, outedge
                )
            )
            n_possible_links = self.get_unmatch_discovered_links(
                s, inedge, outedge
            ) + len(self.traversal.get_rel_dg_pairs(s, outedge))

            max_possible_links[inedge.source, outedge.target] = max(
                n_possible_links,
                max_possible_links.get((inedge.source, outedge.target), 0),
            )

            tmp[
                (
                    self.idmap.m(inedge.source),
                    self.idmap.m(outedge.target),
                    self.idmap.m(s.id),
                    self.idmap.m(outedge.predicate),
                )
            ] = (reli, n_unmatch_links, (inedge.source, outedge.target))

        for key, (reli, freq, pair) in tmp.items():
            prob = (
                0.0
                if max_possible_links[pair] == 0
                else freq / max_possible_links[pair]
            )
            output[reli] = prob

        return output

    def NOT_FUNC_DEPENDENCY(self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]):
        output = []
        notfuncdep: dict[tuple[int, int], int] = {}

        for s, inedge, outedge in rels:
            u = self.cg.get_node(inedge.source)
            v = self.cg.get_node(outedge.target)
            if not isinstance(u, CGColumnNode) or not isinstance(v, CGColumnNode):
                output.append(MISSING_VALUE)
                continue

            if (u.column, v.column) not in notfuncdep:
                notfuncdep[u.column, v.column] = int(
                    not self.functional_dependency_detector.is_functional_dependency(
                        u.column, v.column
                    )
                )

            output.append(notfuncdep[u.column, v.column])
        return output

    def REL_HEADER_SIMILARITY(self, rels: list[tuple[CGStatementNode, CGEdge, CGEdge]]):
        return []

    @CacheMethod.cache(CacheMethod.two_object_args)
    def get_rel_freq(self, s: CGStatementNode, outedge: CGEdge):
        """Get frequency of the link, which is sum of links all rows between two nodes.
        This is weighted count so we should not use this for calculating the number of links"""
        # sum_prob = {}
        # for (source_flow, target_flow), dgstmt2provenances in s.flow.items():
        #     # the flow does not go through this outedge
        #     if target_flow.sg_target_id != outedge.target or target_flow.edge_id != outedge.predicate:
        #         continue

        #     # we now group them based on the entity type
        #     # there can be multiple stmts with the same entity type
        #     # so we need to group them.

        #     sum_prob[]

        sum_prob = 0.0
        for source_flow, target_flow in s.flow:
            if (
                target_flow.sg_target_id == outedge.target
                and target_flow.edge_id == outedge.predicate
            ):
                sum_prob += max(
                    p.prob
                    for provenances in s.flow[source_flow, target_flow].values()
                    for p in provenances
                )
        # dg_flows = s.get_edges_provenance([outedge])
        # for source_flow, target_flows in dg_flows.items():
        #     # regardless of entity or column node, we count all links between rows
        #     # for column node, we only have one for each row (one target_flows),
        #     # for entity node, we only have one source flow, but multiple target flows (max one per row)
        #     for target_flow, provenances in target_flows.items():
        #         sum_prob += max(provenances, key=attrgetter("prob")).prob
        return sum_prob

    @CacheMethod.cache(CacheMethod.three_object_args)
    def get_maximum_possible_ent_links_between_two_nodes(
        self, s: CGStatementNode, inedge: CGEdge, outedge: CGEdge
    ):
        """Find the maximum possible links between two nodes (ignore the possible predicates):

        Let M be the maximum possible links we want to find, N is the number of rows in the table.
        1. If two nodes are not columns, M is 1 because it's entity to entity link.
        2. If one node is a column, M = N - U, where U is the number of pairs that cannot have KG discovered links. A
        pair that cannot have KG discovered links is:
            a. If both nodes are columns, and the link is
                * data predicate: the source cell links to no entity.
                * object predicate: the source or target cell link to no entity
            b. If only one node is column, and the link is
                * data predicate:
                    - if the source node must be an entity, then the target must be a column. U is always 0
                    - else then the source node must be is a column and target is a literal value: a cell in the column links to no entity
                * object predicate: a cell in the column links to no entity.
        """
        u = self.cg.get_node(inedge.source)
        v = self.cg.get_node(outedge.target)

        if not isinstance(u, CGColumnNode) and not isinstance(v, CGColumnNode):
            return 1

        # instead of going through each node attach to the node in the semantic graph, we avoid by directly generating
        # the data node ID
        n_rows = self.table.size()
        n_null_entities = 0
        is_data_predicate = self.wdprops[outedge.predicate].is_data_property()

        if not (isinstance(u, CGColumnNode) and isinstance(v, CGColumnNode)):
            if is_data_predicate:
                if isinstance(u, (CGEntityValueNode, CGLiteralValueNode)):
                    assert isinstance(u, CGEntityValueNode) and isinstance(
                        v, CGColumnNode
                    )
                    return n_rows

                assert isinstance(u, CGColumnNode) and isinstance(
                    v, (CGEntityValueNode, CGLiteralValueNode)
                )

            if isinstance(u, CGColumnNode):
                ci = u.column
            else:
                assert isinstance(
                    v, CGColumnNode
                ), "Always true -- type system is not smart enough"
                ci = v.column

            for ri in range(n_rows):
                if len(self.dg.get_cell_node(f"{ri}-{ci}").entity_ids) == 0:
                    n_null_entities += 1
        else:
            uci = u.column
            vci = v.column

            for ri in range(n_rows):
                ucell_unk = len(self.dg.get_cell_node(f"{ri}-{uci}").entity_ids) == 0
                vcell_unk = len(self.dg.get_cell_node(f"{ri}-{vci}").entity_ids) == 0

                if is_data_predicate:
                    if ucell_unk:
                        n_null_entities += 1
                elif ucell_unk or vcell_unk:
                    n_null_entities += 1

        return n_rows - n_null_entities

    @CacheMethod.cache(CacheMethod.three_object_args)
    def get_unmatch_discovered_links(
        self, s: CGStatementNode, inedge: CGEdge, outedge: CGEdge
    ):
        """Get number of discovered links that don't match due to value differences. This function do not count if:
        * the link between two DG nodes is impossible
        * the property/qualifier do not exist in the WDEntity
        """
        u = self.cg.get_node(inedge.source)
        v = self.cg.get_node(outedge.target)
        uv_links = self.traversal.get_rel_dg_pairs(s, outedge)
        is_outpred_qualifier = inedge.predicate != outedge.predicate
        is_outpred_data_predicate = self.wdprops[outedge.predicate].is_data_property()

        n_unmatch_links = 0
        for dgu, dgv in self.traversal.iter_dg_pair(inedge.source, outedge.target):
            # if has link, then we don't have to count
            if (dgu.id, dgv.id) in uv_links:
                continue

            # ignore pairs that can't have any links
            if not self.traversal.dg_pair_has_possible_ent_links(
                dgu, dgv, is_outpred_data_predicate
            ):
                continue

            if isinstance(dgu, CellNode):
                # the source is cell node
                # property doesn't exist in any qnode
                dgu_qnodes = [self.wdentities[qnode_id] for qnode_id in dgu.entity_ids]
                if all(inedge.predicate not in qnode.props for qnode in dgu_qnodes):
                    continue

                if is_outpred_qualifier:
                    # qualifier doesn't exist in any qnode
                    has_qual = False
                    for qnode in dgu_qnodes:
                        for stmt in qnode.props.get(inedge.predicate, []):
                            if outedge.predicate in stmt.qualifiers:
                                has_qual = True
                    if not has_qual:
                        continue

                # # #########################
                # # TODO: remove me, modify on Apr 29, should have a better solution
                # # this is to say like if that properties have multiple values, then it's more likely to miss
                # if all(len(qnode.props.get(inpred, [])) > 2 for qnode in dgu_qnodes):
                #     continue
                # # #########################
            else:
                # the source is entity
                assert isinstance(dgu, EntityValueNode)
                dgu_qnode = self.wdentities[dgu.qnode_id]
                if inedge.predicate not in dgu_qnode.props:
                    continue
                if is_outpred_qualifier:
                    if all(
                        outedge.predicate not in stmt.qualifiers
                        for stmt in dgu_qnode.props[inedge.predicate]
                    ):
                        continue
                # # #########################
                # # TODO: remove me, modify on Apr 29, should have a better solution
                # # this is to say like if that properties have multiple values, then it's more likely to miss
                # if len(dgu_qnode.props.get(inpred, [])) > 2:
                #     continue
                # # #########################
            n_unmatch_links += 1
        return n_unmatch_links

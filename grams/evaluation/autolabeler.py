from __future__ import annotations
from collections import defaultdict

from collections.abc import Mapping
from typing import Literal, Sequence, overload
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.inputs.linked_table import LinkedTable
from sm.dataset import Example
from sm.misc.fn_cache import CacheMethod

from sm.outputs.semantic_model import (
    LiteralNodeDataType,
    Node,
    DataNode,
    LiteralNode,
)
from rdflib import RDFS
from grams.evaluation.evaluator import Evaluator, WrappedSemanticModel


class AutoLabeler:
    """Automatically label relationships and types in the candidate graph"""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.wdns = evaluator.wdns

    def label_example_types(
        self, pred_types: Sequence[tuple[int, str]], example: Example[LinkedTable]
    ):
        col2types = self._get_example_gold_types(example)
        return [ctype in col2types[ci] for ci, ctype in pred_types]

    def label_types(
        self,
        pred_types: Sequence[tuple[int, str]],
        gold_sms: list[WrappedSemanticModel],
    ) -> list[bool]:
        col2types: dict[int, set[str]] = {ci: set() for ci, _ in pred_types}
        for wsm in gold_sms:
            for ci, _ in pred_types:
                col2types[ci].update(
                    self.wdns.get_entity_id(stype.class_abs_uri)
                    for stype in wsm.sm.get_semantic_types_of_column(ci)
                    if stype.predicate_abs_uri == str(RDFS.label)
                )
        return [ctype in col2types[ci] for ci, ctype in pred_types]

    @overload
    def label_relationships(
        self,
        cg: CGGraph,
        gold_sms: list[WrappedSemanticModel],
        return_edge: Literal[True] = True,
    ) -> dict[int, bool]:
        ...

    @overload
    def label_relationships(
        self,
        cg: CGGraph,
        gold_sms: list[WrappedSemanticModel],
        return_edge: Literal[False] = False,
    ) -> dict[tuple[str, str, str, str], bool]:
        ...

    def label_relationships(
        self,
        cg: CGGraph,
        gold_sms: list[WrappedSemanticModel],
        return_edge: bool = True,
    ) -> dict[int, bool] | dict[tuple[str, str, str, str], bool]:
        """Return mapping from edge id to the label"""
        gold_cpa_sms = [self.evaluator.convert_sm_for_cpa(sm) for sm in gold_sms]

        edge_label: dict[int, bool] = {}
        rels: dict[tuple[str, str, str, str], bool] = {}

        for node in cg.iter_nodes():
            if not isinstance(node, CGStatementNode):
                continue

            # search for the relationship and label them
            (inedge,) = cg.in_edges(node.id)
            outedges = cg.out_edges(node.id)

            # we do not have more than one (inedge -> node -> outedge.target) where outedge.predicate = inedge.predicate
            # due to how cg is constructed, therefore, we do not need to group statements
            best_match_score = 0
            best_match = [False] * len(outedges)
            for sm in gold_cpa_sms:
                match = self._rel_label(cg, inedge, outedges, sm)
                if sum(match) > best_match_score:
                    best_match = match
                    best_match_score = sum(match)

            if return_edge:
                if best_match_score == 0:
                    edge_label[inedge.id] = False
                else:
                    edge_label[inedge.id] = True
                for i, outedge in enumerate(outedges):
                    edge_label[outedge.id] = best_match[i]
            else:
                for i, outedge in enumerate(outedges):
                    rels[
                        (inedge.source, outedge.target, node.id, outedge.predicate)
                    ] = best_match[i]

        if return_edge:
            return edge_label
        return rels

    def _rel_label(
        self,
        cg: CGGraph,
        inedge: CGEdge,
        outedges: list[CGEdge],
        wrapped_sm: WrappedSemanticModel,
    ):
        best_match_score = 0
        best_match = [False] * len(outedges)

        sm = wrapped_sm.sm
        assert wrapped_sm.is_normalized and wrapped_sm.is_cpa_transformed

        # get the corresponding statement
        cg_u = cg.get_node(inedge.source)
        if isinstance(cg_u, CGColumnNode):
            if not sm.has_data_node(cg_u.column):
                return best_match
            sm_u = sm.get_data_node(cg_u.column)
        else:
            assert isinstance(
                cg_u, CGEntityValueNode
            ), "Cannot be literal node because this contains outgoing edges."
            cg_u_enturi = self.wdns.get_entity_abs_uri(cg_u.qnode_id)
            if not sm.has_literal_node(cg_u_enturi):
                return best_match
            sm_u = sm.get_literal_node(cg_u_enturi)

        # select the maximum match
        inedge_abs_uri = self.wdns.get_prop_abs_uri(inedge.predicate)
        uri_to_outedge_index = {
            self.wdns.get_prop_abs_uri(outedge.predicate): i
            for i, outedge in enumerate(outedges)
        }

        for sm_us_edge in sm.out_edges(sm_u.id):
            if sm_us_edge.abs_uri != inedge_abs_uri:
                continue

            outedge_labels = [False] * len(outedges)
            for sm_sv_edge in sm.out_edges(sm_us_edge.target):
                if sm_sv_edge.abs_uri in uri_to_outedge_index:
                    # found an outedge that match
                    outedge_i = uri_to_outedge_index[sm_sv_edge.abs_uri]
                    cg_v = cg.get_node(outedges[outedge_i].target)
                    sm_v = sm.get_node(sm_sv_edge.target)
                    assert not isinstance(cg_v, CGStatementNode)
                    outedge_labels[outedge_i] = self._is_node_match(
                        cg, cg_v, wrapped_sm, sm_v
                    )

            if (
                outedge_labels[uri_to_outedge_index[inedge_abs_uri]]
                and sum(outedge_labels) > best_match_score
            ):
                # the main must match before we consider how many edges are matched
                best_match_score = sum(outedge_labels)
                best_match = outedge_labels

        return best_match

    def _is_node_match(
        self,
        cg: CGGraph,
        cg_node: CGColumnNode | CGEntityValueNode | CGLiteralValueNode,
        wrapped_sm: WrappedSemanticModel,
        sm_node: Node,
    ):
        if isinstance(cg_node, CGColumnNode):
            return isinstance(sm_node, DataNode) and sm_node.col_index == cg_node.column
        elif isinstance(cg_node, CGEntityValueNode):
            return (
                isinstance(sm_node, LiteralNode)
                and sm_node.datatype == LiteralNodeDataType.Entity
                and sm_node.value == self.wdns.get_entity_abs_uri(cg_node.qnode_id)
            )
        else:
            assert isinstance(cg_node, CGLiteralValueNode)
            return (
                isinstance(sm_node, LiteralNode)
                and sm_node.datatype == LiteralNodeDataType.String
                and sm_node.value == cg_node.value.to_string_repr()
            )

    @CacheMethod.cache(CacheMethod.single_object_arg)
    def _get_example_gold_types(self, example: Example[LinkedTable]):
        gold_sms = self.evaluator.get_example_gold_sms(example)
        cols = [col.index for col in example.table.table.columns]
        col2types: dict[int, set[str]] = {ci: set() for ci in cols}
        for wsm in gold_sms:
            for ci in cols:
                col2types[ci].update(
                    self.wdns.get_entity_id(stype.class_abs_uri)
                    for stype in wsm.sm.get_semantic_types_of_column(ci)
                    if stype.predicate_abs_uri == str(RDFS.label)
                )
        return col2types

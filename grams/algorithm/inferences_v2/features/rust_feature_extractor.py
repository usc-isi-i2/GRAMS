from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGGraph,
    CGEntityValueNode,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    CellNode,
    EntityValueNode,
    LiteralValueNode,
    StatementNode,
)
import grams.core as gcore
import grams.core.datagraph as gcore_dg
import grams.core.cangraph as gcore_cg
import grams.core.features as gcore_feat
from pathlib import Path
import grams.core.table as gcore_tbl

from sm.misc.fn_cache import CacheMethod


class RustFeatureExtractor:
    def __init__(
        self,
        table: gcore_tbl.LinkedTable,
        dg: DGGraph,
        cg: CGGraph,
        context: gcore.AlgoContext,
        db: gcore.GramsDB,
    ):
        # prepare data
        self.dgnodes = []
        self.dgidmap = {}
        for node in dg.iter_nodes():
            self.dgidmap[node.id] = len(self.dgnodes)
            if isinstance(node, CellNode):
                self.dgnodes.append(
                    gcore_dg.DGNode.cell(
                        gcore_dg.CellNode(
                            node.id,
                            node.value,
                            node.column,
                            node.row,
                            node.entity_ids,
                            [
                                (eid, [gcore_dg.Span(sp.start, sp.end) for sp in sps])
                                for eid, sps in node.entity_spans.items()
                            ],
                            list(node.entity_probs.items()),
                        )
                    )
                )
            elif isinstance(node, EntityValueNode):
                self.dgnodes.append(
                    gcore_dg.DGNode.entity(
                        gcore_dg.EntityValueNode(
                            node.id,
                            node.qnode_id,
                            node.qnode_prob,
                            gcore_dg.ContextSpan(
                                node.context_span.text,
                                gcore_dg.Span(
                                    node.context_span.span.start,
                                    node.context_span.span.end,
                                ),
                            )
                            if node.context_span is not None
                            else None,
                        )
                    )
                )
            elif isinstance(node, LiteralValueNode):
                self.dgnodes.append(
                    gcore_dg.DGNode.literal(
                        gcore_dg.LiteralValueNode(
                            node.id,
                            node.value.to_rust(gcore.Value),
                            gcore_dg.ContextSpan(
                                node.context_span.text,
                                gcore_dg.Span(
                                    node.context_span.span.start,
                                    node.context_span.span.end,
                                ),
                            )
                            if node.context_span is not None
                            else None,
                        )
                    )
                )
            else:
                assert isinstance(node, StatementNode)

        self.cgidmap = {}
        self.cgnodes = []
        self.cg2dg = []

        cg_nodes = cg.nodes()
        for node in cg_nodes:
            self.cgidmap[node.id] = len(self.cgidmap)
            if isinstance(node, (CGEntityValueNode, CGLiteralValueNode)):
                # those nodes have the same ids as in the dg
                self.cg2dg.append(self.dgidmap[node.id])
            else:
                # the nodes do not have the same ids as in the dg
                self.cg2dg.append(None)

        for node in cg_nodes:
            if isinstance(node, CGStatementNode):
                newnode = gcore_cg.CGNode.statement_node(
                    self.cgidmap[node.id],
                    [
                        gcore_cg.CGStatementFlow(
                            gcore_cg.CGEdgeFlowSource(
                                self.dgidmap[sourceflow.dg_source_id],
                                self.cgidmap[sourceflow.sg_source_id],
                                sourceflow.edge_id,
                            ),
                            gcore_cg.CGEdgeFlowTarget(
                                self.dgidmap[targetflow.dg_target_id],
                                self.cgidmap[targetflow.sg_target_id],
                                targetflow.edge_id,
                            ),
                            list(dgstmts.keys()),
                        )
                        for (sourceflow, targetflow), dgstmts in node.flow.items()
                    ],
                )
            elif isinstance(node, CGColumnNode):
                newnode = gcore_cg.CGNode.column_node(
                    self.cgidmap[node.id], node.label, node.column
                )
            elif isinstance(node, CGEntityValueNode):
                newnode = gcore_cg.CGNode.entity_node(
                    self.cgidmap[node.id],
                    node.qnode_id,
                    gcore_dg.ContextSpan(
                        node.context_span.text,
                        gcore_dg.Span(
                            node.context_span.span.start,
                            node.context_span.span.end,
                        ),
                    )
                    if node.context_span is not None
                    else None,
                )
            else:
                assert isinstance(node, CGLiteralValueNode)
                newnode = gcore_cg.CGNode.literal_node(
                    self.cgidmap[node.id],
                    node.value.to_rust(gcore.Value),
                    gcore_dg.ContextSpan(
                        node.context_span.text,
                        gcore_dg.Span(
                            node.context_span.span.start,
                            node.context_span.span.end,
                        ),
                    )
                    if node.context_span is not None
                    else None,
                )

            self.cgnodes.append(newnode)

        self.cgedges: dict[int, gcore_cg.CGEdge] = {}
        for edge in cg.iter_edges():
            self.cgedges[edge.id] = gcore_cg.CGEdge(
                edge.id,
                self.cgidmap[edge.source],
                self.cgidmap[edge.target],
                edge.predicate,
            )

        self.rustextractor = gcore_feat.FeatureExtractorContext(
            table, self.dgnodes, self.cg2dg, db, context
        )

    # @CacheMethod.cache(CacheMethod.three_object_args)
    def get_unmatch_discovered_links(
        self, s: CGStatementNode, inedge: CGEdge, outedge: CGEdge
    ):
        cgu = self.cgnodes[self.cgidmap[inedge.source]]
        news = self.cgnodes[self.cgidmap[s.id]]
        cgv = self.cgnodes[self.cgidmap[outedge.target]]
        ine = self.cgedges[inedge.id]
        oute = self.cgedges[outedge.id]

        return self.rustextractor.get_unmatch_discovered_links(
            cgu, news, cgv, ine, oute
        )

    def get_len_contradicted_information(
        self,
        s: CGStatementNode,
        inedge: CGEdge,
        outedge: CGEdge,
        correct_entity_threshold: float,
    ):
        cgu = self.cgnodes[self.cgidmap[inedge.source]]
        news = self.cgnodes[self.cgidmap[s.id]]
        cgv = self.cgnodes[self.cgidmap[outedge.target]]
        ine = self.cgedges[inedge.id]
        oute = self.cgedges[outedge.id]

        return self.rustextractor.get_len_contradicted_information(
            cgu, news, cgv, ine, oute, correct_entity_threshold
        )

    def get_contradicted_information(
        self,
        s: CGStatementNode,
        inedge: CGEdge,
        outedge: CGEdge,
        correct_entity_threshold: float,
    ):
        cgu = self.cgnodes[self.cgidmap[inedge.source]]
        news = self.cgnodes[self.cgidmap[s.id]]
        cgv = self.cgnodes[self.cgidmap[outedge.target]]
        ine = self.cgedges[inedge.id]
        oute = self.cgedges[outedge.id]

        return self.rustextractor.get_contradicted_information(
            cgu, news, cgv, ine, oute, correct_entity_threshold
        )


# def to_rust(self):
#     if self.is_string(self):
#         return gcore.Value.string(self.value)
#     if self.is_entity_id(self):
#         return gcore.Value.entity_id(
#             self.value["id"], self.value["entity-type"], self.value["numeric-id"]
#         )
#     if self.is_time(self):
#         return gcore.Value.time(
#             self.value["time"],
#             self.value["timezone"],
#             self.value["before"],
#             self.value["after"],
#             self.value["precision"],
#             self.value["calendarmodel"],
#         )
#     if self.is_quantity(self):
#         return gcore.Value.quantity(
#             self.value["amount"],
#             self.value.get("lowerBound", None),
#             self.value.get("upperBound", None),
#             self.value["unit"],
#         )
#     if self.is_globe_coordinate(self):
#         return gcore.Value.globe_coordinate(
#             self.value["latitude"],
#             self.value["longitude"],
#             self.value["precision"],
#             self.value["altitude"],
#             self.value["globe"],
#         )
#     if self.is_mono_lingual_text(self):
#         return gcore.Value.monolingual_text(self.value["text"], self.value["language"])
#     raise ValueError(f"Unknown type: {self.type}")

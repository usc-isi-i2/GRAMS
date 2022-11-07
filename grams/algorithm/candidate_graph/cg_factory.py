from typing import Dict, List, Mapping
from grams.algorithm.candidate_graph.cg_graph import (
    CGColumnNode,
    CGEdge,
    CGEdgeFlowSource,
    CGEdgeFlowTarget,
    CGEntityValueNode,
    CGGraph,
    CGLiteralValueNode,
    CGStatementNode,
)
from grams.algorithm.data_graph.dg_graph import (
    CellNode,
    DGGraph,
    DGNode,
    EdgeFlowSource,
    EdgeFlowTarget,
    EntityValueNode,
    LiteralValueNode,
    StatementNode,
)
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDEntity, WDClass, WDProperty
from sm.prelude import M


class CGFactory:
    def __init__(
        self,
        wdentities: Mapping[str, WDEntity],
        wdentity_labels: Mapping[str, str],
        wdclasses: Mapping[str, WDClass],
        wdprops: Mapping[str, WDProperty],
    ):
        self.wdentities = wdentities
        self.wdentity_labels = wdentity_labels
        self.wdclasses = wdclasses
        self.wdprops = wdprops

    def create_cg(self, table: LinkedTable, dg: DGGraph) -> CGGraph:
        """Create candidate graph from data graph"""
        cg = CGGraph()
        # first step: add node to the graph
        # for uid, udata in dg.iter_nodes():
        #     u: DGNode = udata["data"]
        for u in dg.iter_nodes():
            if isinstance(u, StatementNode):
                # we can have more than one predicates per entity column, these are represented in the statement
                continue

            sgi = self.get_cg_node_id(u)
            if not cg.has_node(sgi):
                label = self.get_sg_node_label(table, u)
                if isinstance(u, CellNode):
                    sgu = CGColumnNode(
                        id=sgi, label=label, column=u.column, nodes=set()
                    )
                elif isinstance(u, EntityValueNode):
                    sgu = CGEntityValueNode(
                        id=sgi,
                        label=label,
                        qnode_id=u.qnode_id,
                        context_span=u.context_span,
                    )
                else:
                    sgu = CGLiteralValueNode(
                        id=sgi, label=label, value=u.value, context_span=u.context_span
                    )
                cg.add_node(sgu)
            else:
                sgu = cg.get_node(sgi)

            if isinstance(sgu, CGColumnNode):
                sgu.nodes.add(u.id)

        # second step: add link
        # for uid, udata in dg.nodes(data=True):
        #     u: DGNode = udata["data"]
        for u in dg.iter_nodes():
            if isinstance(u, StatementNode):
                continue

            # add statement
            p2stmts: Dict[str, List[StatementNode]] = {}
            # for _, sid, us_eid, us_edata in dg.out_edges(uid, data=True, keys=True):
            for us_edge in dg.out_edges(u.id):
                if us_edge.predicate not in p2stmts:
                    p2stmts[us_edge.predicate] = []
                p2stmts[us_edge.predicate].append(dg.get_statement_node(us_edge.target))

            sgi = self.get_cg_node_id(u)
            for p, stmts in p2stmts.items():
                for stmt in stmts:
                    # we duplicate based on SG column node, not by DG node
                    children_values = {}
                    children_non_values = []

                    # for _, vid, sv_eid, sv_edata in dg.out_edges(
                    #     stmt.id, data=True, keys=True
                    # ):
                    #     v: DGNode = dg.nodes[vid]["data"]
                    for sv_edge in dg.out_edges(stmt.id):
                        v = dg.get_node(sv_edge.target)
                        if (v.id, sv_edge.predicate) not in stmt.forward_flow[u.id, p]:
                            # because of re-use statement, we need to only consider the children of this u node only
                            continue

                        if sv_edge.predicate == p:
                            tgi = self.get_cg_node_id(v)
                            if tgi not in children_values:
                                children_values[tgi] = []
                            children_values[tgi].append(v)
                        else:
                            children_non_values.append((sv_edge.predicate, v))

                    for tgi, vals in children_values.items():
                        stmt_id = CGStatementNode.get_id(sgi, p, tgi)
                        if not cg.has_node(stmt_id):
                            cg.add_node(CGStatementNode.new(stmt_id))

                        if not cg.has_edge_between_nodes(sgi, stmt_id, p):
                            cg.add_edge(
                                CGEdge(
                                    source=sgi,
                                    target=stmt_id,
                                    predicate=p,
                                    features={},
                                )
                            )
                        if not cg.has_edge_between_nodes(stmt_id, tgi, p):
                            cg.add_edge(
                                CGEdge(
                                    source=stmt_id,
                                    target=tgi,
                                    predicate=p,
                                    features={},
                                )
                            )

                        sg_stmt = cg.get_statement_node(stmt_id)
                        for v in vals:
                            sg_stmt.track_provenance(
                                CGEdgeFlowSource(u.id, sgi, p),
                                stmt.id,
                                CGEdgeFlowTarget(v.id, tgi, p),
                                stmt.get_provenance(
                                    EdgeFlowSource(u.id, p), EdgeFlowTarget(v.id, p)
                                ),
                            )

                        for qp, qv in children_non_values:
                            qtgi = self.get_cg_node_id(qv)
                            if not dg.has_edge_between_nodes(stmt_id, qtgi, qp):
                                cg.add_edge(
                                    CGEdge(
                                        source=stmt_id,
                                        target=qtgi,
                                        predicate=qp,
                                        features={},
                                    )
                                )

                            sg_stmt.track_provenance(
                                CGEdgeFlowSource(u.id, sgi, p),
                                stmt.id,
                                CGEdgeFlowTarget(qv.id, qtgi, qp),
                                stmt.get_provenance(
                                    EdgeFlowSource(u.id, p), EdgeFlowTarget(qv.id, qp)
                                ),
                            )
        return cg

    @staticmethod
    def get_cg_node_id(u: DGNode):
        if isinstance(u, CellNode):
            return f"column-{u.column}"
        return u.id

    def get_sg_node_label(self, table: LinkedTable, u: DGNode):
        if isinstance(u, CellNode):
            return table.table.columns[u.column].name or ""
        if isinstance(u, EntityValueNode):
            qnode_label = self.wdentity_labels.get(u.qnode_id, u.qnode_id)
            return f"{qnode_label} ({u.qnode_id})"
        if isinstance(u, LiteralValueNode):
            return u.value.to_string_repr()
        raise M.UnreachableError(
            "This function doesn't supposed to be called with statement node"
        )

    def get_label(
        self,
        porq: str,
    ):
        """Get label for a property or qualifier"""
        if porq.startswith("Q"):
            if porq in self.wdclasses:
                item = self.wdclasses[porq]
            elif porq in self.wdentities:
                item = self.wdentities[porq]
            else:
                assert False
        else:
            item = self.wdprops[porq]

        return f"{item.label} ({item.id})"

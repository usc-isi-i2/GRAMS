from grams.algorithm.literal_matchers import TextParser, LiteralMatch
from grams.inputs.linked_table import LinkedTable
import networkx as nx
from typing import Dict, Mapping, Optional, Set, Union, cast
from kgdata.wikidata.models import WDEntity, WDProperty, WDClass
from grams.algorithm.data_graph.dg_graph import (
    DGGraph,
    CellNode,
    ContextSpan,
    DGEdge,
    DGNode,
    DGPath,
    DGPathNode,
    DGPathEdge,
    EdgeFlowSource,
    EdgeFlowTarget,
    EntityValueNode,
    DGPathNodeStatement,
    DGPathNodeEntity,
    DGPathNodeLiteralValue,
    DGPathExistingNode,
    LiteralValueNode,
    Span,
    StatementNode,
)
from grams.algorithm.data_graph.dg_inference import KGInference
from grams.algorithm.data_graph.dg_pruning import DGPruning
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.kg_index import KGObjectIndex
from tqdm import tqdm
import sm.misc as M


class DGFactory:
    def __init__(
        self, wdentities: Mapping[str, WDEntity], wdprops: Mapping[str, WDProperty]
    ):
        self.wdentities = wdentities
        self.wdprops = wdprops
        self.textparser = TextParser()
        self.literal_match = LiteralMatch(wdentities)

        if DGConfigs.USE_KG_INDEX:
            self._path_object_search = self._path_object_search_v2
        else:
            self._path_object_search = self._path_object_search_v1

    def create_dg(
        self,
        table: LinkedTable,
        kg_object_index: KGObjectIndex,
        ignore_columns: Optional[Set[int]] = None,
        max_n_hop: int = 2,
        verbose: bool = False,
    ):
        """Build data graph from a linked table"""
        dg = DGGraph()
        context_node_id = None

        for ci, col in enumerate(table.table.columns):
            for ri, val in enumerate(col.values):
                # =====================================================
                # NOTE: old code
                # cell_qnodes = set()
                # cell_qnode_spans = {}
                # for link in table.links[ri][ci]:
                #     if link.entity_id is not None:
                #         cell_qnodes.add(link.entity_id)
                #         if link.entity_id not in cell_qnode_spans:
                #             cell_qnode_spans[link.entity_id] = []
                #         cell_qnode_spans[link.entity_id].append(Span(link.start, link.end))
                # assert all(len(spans) == len(set(spans)) for spans in cell_qnode_spans.values())
                # TODO: new code, doesn't handle the qnode_spans correctly
                cell_qnodes = {
                    candidate.entity_id
                    for link in table.links[ri][ci]
                    for candidate in link.candidates
                }
                cell_qnode_spans = {}
                for link in table.links[ri][ci]:
                    # TODO: old code
                    # if link.entity_id is not None or len(link.candidates) > 0:
                    #     if len(link.candidates) > 0:
                    #         tmpid = link.candidates[0].entity_id
                    #     else:
                    #         tmpid = link.entity_id
                    #     if tmpid not in cell_qnode_spans:
                    #         cell_qnode_spans[tmpid] = []
                    #     cell_qnode_spans[tmpid].append(Span(link.start, link.end))
                    if len(link.candidates) > 0:
                        tmpid = link.candidates[0].entity_id
                        if tmpid not in cell_qnode_spans:
                            cell_qnode_spans[tmpid] = []
                        cell_qnode_spans[tmpid].append(Span(link.start, link.end))
                assert all(
                    len(spans) == len(set(spans)) for spans in cell_qnode_spans.values()
                )
                # =====================================================

                node = CellNode(
                    id=f"{ri}-{ci}",
                    value=val,
                    column=ci,
                    row=ri,
                    entity_ids=list(cell_qnodes),
                    entity_spans=cell_qnode_spans,
                )
                dg.add_node(node)

        if DGConfigs.USE_CONTEXT and table.context.page_entity_id is not None:
            assert table.context.page_title is not None
            context_node_id = DGPathNodeEntity(table.context.page_entity_id).get_id()
            node = EntityValueNode(
                id=context_node_id,
                qnode_id=table.context.page_entity_id,
                context_span=ContextSpan(
                    text=table.context.page_title,
                    span=Span(0, len(table.context.page_title)),
                ),
            )
            dg.add_node(node)

        # find all paths
        n_rows = len(table.table.columns[0].values)
        kg_path_discovering_tasks = []
        new_paths = []
        for ri in range(n_rows):
            for ci, col in enumerate(table.table.columns):
                if ignore_columns is not None and ci in ignore_columns:
                    continue

                u = dg.get_node(f"{ri}-{ci}")
                for cj in range(ci + 1, len(table.table.columns)):
                    if ignore_columns is not None and cj in ignore_columns:
                        continue

                    v = dg.get_node(f"{ri}-{cj}")
                    kg_path_discovering_tasks.append((u, v))
                if context_node_id is not None and dg.has_node(context_node_id):
                    kg_path_discovering_tasks.append((u, dg.get_node(context_node_id)))

        for u, v in (
            tqdm(kg_path_discovering_tasks, desc="KG searching")
            if verbose
            else kg_path_discovering_tasks
        ):
            new_paths += self.kg_path_discovering(
                kg_object_index, dg, u, v, max_n_hop=max_n_hop
            )

        # add paths to the graph
        for path in new_paths:
            curr_nodeid = path.sequence[0].id
            tmp_path = [curr_nodeid]
            for i in range(1, len(path.sequence), 2):
                prop: DGPathEdge = path.sequence[i]
                value: DGPathNode = path.sequence[i + 1]

                if isinstance(value, DGPathExistingNode):
                    nodeid = value.id
                else:
                    nodeid = value.get_id()
                    if not dg.has_node(nodeid):
                        if isinstance(value, DGPathNodeStatement):
                            dg.add_node(
                                StatementNode(
                                    id=nodeid,
                                    qnode_id=value.qnode_id,
                                    predicate=value.predicate,
                                    is_in_kg=True,
                                )
                            )
                        elif isinstance(value, DGPathNodeEntity):
                            dg.add_node(
                                EntityValueNode(
                                    id=nodeid,
                                    qnode_id=value.qnode_id,
                                    context_span=None,
                                ),
                            )
                        else:
                            dg.add_node(
                                LiteralValueNode(
                                    id=nodeid, value=value.value, context_span=None
                                ),
                            )

                if dg.has_edge_between_nodes(curr_nodeid, nodeid, prop.value):
                    edge = dg.get_edge_between_nodes(curr_nodeid, nodeid, prop.value)
                else:
                    edge = DGEdge(
                        source=curr_nodeid,
                        target=nodeid,
                        predicate=prop.value,
                        is_qualifier=prop.is_qualifier,
                    )
                    dg.add_edge(edge)

                tmp_path.append(edge)
                tmp_path.append(nodeid)
                curr_nodeid = nodeid

            # path format: u - e - (s) - e - v - e - (s2) - e - v2
            for i in range(2, len(tmp_path), 4):
                u_node_id = tmp_path[i - 2]
                u_edge = tmp_path[i - 1]
                snode = dg.get_node(tmp_path[i])
                assert isinstance(snode, StatementNode)
                v_edge = tmp_path[i + 1]
                v_node_id = tmp_path[i + 2]

                edge_source = EdgeFlowSource(u_node_id, u_edge.predicate)
                edge_target = EdgeFlowTarget(v_node_id, v_edge.predicate)
                snode.track_provenance(
                    edge_source, edge_target, [path.sequence[i].provenance]
                )

        # DEBUG code
        # for uid, udata in dg.nodes(data=True):
        #     u = udata['data']
        #     if not u.is_cell:
        #         continue
        #     if u.column == 0:
        #         for _, sid, eid, edata in dg.out_edges(uid, data=True, keys=True):
        #             for _1, vid, e2, e2data in dg.out_edges(sid, data=True, keys=True):
        #                 v = dg.nodes[vid]['data']
        #                 if v.is_cell and v.column == 1:
        #                     print(uid, sid, vid, eid, e2)

        KGInference(
            dg, self.wdentities, self.wdprops
        ).infer_subproperty().kg_transitive_inference()

        # pruning unnecessary paths
        if DGConfigs.PRUNE_REDUNDANT_ENT:
            DGPruning(dg).prune_hidden_entities()

        M.log("grams", data_graph=dg)
        return dg

    def kg_path_discovering(
        self,
        kg_object_index: KGObjectIndex,
        dg: DGGraph,
        u: Union[EntityValueNode, CellNode],
        v: Union[EntityValueNode, CellNode],
        max_n_hop: int = 2,
        bidirection: bool = True,
    ):
        """Find all paths between two nodes in the graph.

        Parameters
        ----------
        qnodes: dictionary of qnodes
        dg: the input data graph
        u: node
        v: node
        max_n_hop: hop start at 1 (current node), n_hop = 2 mean the path can go through an hidden node
        bidirection: set to false telling if we want to just find paths going from u -> v.
            This parameter just for re-using code, as we will call the function changing the order of u & v

        Returns
        -------
        """

        paths = []

        # no qnode in the source
        u_qnodes = (
            [self.wdentities[u.qnode_id]]
            if isinstance(u, EntityValueNode)
            else [self.wdentities[qnode_id] for qnode_id in u.entity_ids]
        )
        if len(u_qnodes) == 0:
            if bidirection:
                return self.kg_path_discovering(
                    kg_object_index,
                    dg,
                    v,
                    u,
                    max_n_hop=max_n_hop,
                    bidirection=False,
                )
            return []

        v_qnodes = (
            [self.wdentities[v.qnode_id]]
            if isinstance(v, EntityValueNode)
            else [self.wdentities[qnode_id] for qnode_id in v.entity_ids]
        )
        if isinstance(v, EntityValueNode):
            # entity in the context
            assert v.context_span is not None
            v_value = v.context_span.get_text_span()
        else:
            v_value = v.value

        for n1 in u_qnodes:
            # count = timer.start("object discovery")
            for n2 in v_qnodes:
                for newpath in self._path_object_search(
                    kg_object_index, n1, n2, max_n_hop=max_n_hop
                ):
                    newpath.sequence.insert(0, DGPathExistingNode(u.id))
                    if isinstance(newpath.sequence[-1], DGPathEdge):
                        newpath.sequence.append(DGPathExistingNode(v.id))
                    paths.append(newpath)
            # count.stop()

            # count = timer.start("literal discovery")
            for newpath in self._path_value_search(n1, self.textparser.parse(v_value)):
                newpath.sequence.insert(0, DGPathExistingNode(u.id))
                if isinstance(newpath.sequence[-1], DGPathEdge):
                    newpath.sequence.append(DGPathExistingNode(v.id))
                paths.append(newpath)
            # count.stop()

        if bidirection:
            paths += self.kg_path_discovering(
                kg_object_index,
                dg,
                v,
                u,
                max_n_hop=max_n_hop,
                bidirection=False,
            )

        return paths

    def _path_value_search(self, source: WDEntity, value):
        matches = []
        for p, stmts in source.props.items():
            if p == "P31":
                # no need to search in the instanceOf property, as the ontology is removed from the databased as they are huge
                continue

            for stmt_i, stmt in enumerate(stmts):
                has_stmt_value = False
                for fn, (match, confidence) in self.literal_match.match(
                    stmt.value, value, skip_unmatch=True
                ):
                    matches.append(
                        DGPath(
                            sequence=[
                                DGPathEdge.p(p),
                                DGPathNodeStatement.from_FromLiteralMatchingFunc(
                                    source.id,
                                    p,
                                    stmt_i,
                                    {"func": fn.__name__, "value": stmt.value},
                                    confidence,
                                ),
                                DGPathEdge.p(p),
                            ]
                        )
                    )
                    has_stmt_value = True

                for q, qvals in stmt.qualifiers.items():
                    for qval in qvals:
                        for fn, (match, confidence) in self.literal_match.match(
                            qval, value, skip_unmatch=True
                        ):
                            if not has_stmt_value:
                                if stmt.value.is_qnode(stmt.value):
                                    pn_stmt_value = DGPathNodeEntity(
                                        stmt.value.as_entity_id()
                                    )
                                else:
                                    pn_stmt_value = DGPathNodeLiteralValue(stmt.value)
                                matches.append(
                                    DGPath(
                                        sequence=[
                                            DGPathEdge.p(p),
                                            DGPathNodeStatement.from_FromWikidataLink(
                                                source.id, p, stmt_i
                                            ),
                                            DGPathEdge.p(p),
                                            pn_stmt_value,
                                        ]
                                    )
                                )
                                has_stmt_value = True

                            matches.append(
                                DGPath(
                                    sequence=[
                                        DGPathEdge.p(p),
                                        DGPathNodeStatement.from_FromLiteralMatchingFunc(
                                            source.id,
                                            p,
                                            stmt_i,
                                            {
                                                "func": fn.__name__,
                                                "value": qval,
                                            },
                                            confidence,
                                        ),
                                        DGPathEdge.q(q),
                                    ]
                                )
                            )
        return matches

    def _path_object_search_v1(
        self,
        kg_object_index: KGObjectIndex,
        source: WDEntity,
        target: WDEntity,
        max_n_hop=2,
    ):
        # max_n_hop = 2 mean we will find a path that go from source to target through an hidden node
        # hop start at 1 (current node)

        if not DGConfigs.ALLOW_SAME_ENT_SEARCH and source.id == target.id:
            return []

        matches = []
        iter = source.props.items()

        for p, stmts in iter:
            if p == "P31":
                # no need to search in the instanceOf property, as the ontology is removed from the databased as they are huge
                continue

            for stmt_i, stmt in enumerate(stmts):
                # add the cell id so that we have different statement for different cell.
                has_stmt_value = False

                # to simplify the design, we do not consider a statement that its value do not exist in KQ
                # due to an error on KG
                if stmt.value.is_qnode(stmt.value):
                    if stmt.value.as_entity_id() not in self.wdentities:
                        # this can happen due to some of the qnodes is in the link, but is missing in the KG
                        # this is very rare so we can employ some check to make sure this is not due to
                        # our wikidata subset
                        is_error_in_kg = any(
                            any(
                                _s.value.is_qnode(_s.value)
                                and _s.value.as_entity_id() in self.wdentities
                                for _s in _stmts
                            )
                            for _p, _stmts in source.props.items()
                        ) or stmt.value.as_entity_id().startswith("L")
                        if not is_error_in_kg:
                            raise Exception(
                                f"Missing qnodes in your KG subset: {stmt.value.as_entity_id()}"
                            )
                        continue

                if stmt.value.is_qnode(stmt.value):
                    # found by match entity
                    if stmt.value.as_entity_id() == target.id:
                        matches.append(
                            DGPath(
                                sequence=[
                                    DGPathEdge.p(p),
                                    DGPathNodeStatement.from_FromWikidataLink(
                                        source.id, p, stmt_i
                                    ),
                                    DGPathEdge.p(p),
                                ]
                            )
                        )
                        has_stmt_value = True
                    elif max_n_hop > 1:
                        stmt_value_qnode_id = stmt.value.as_entity_id()
                        if stmt_value_qnode_id.startswith("L"):
                            assert (
                                stmt_value_qnode_id not in self.wdentities
                            ), "The L nodes (lexical) is not in the Wikidata dump"
                            continue

                        for nextpath in self._path_object_search(
                            kg_object_index,
                            self.wdentities[stmt_value_qnode_id],
                            target,
                            max_n_hop - 1,
                        ):
                            matches.append(
                                DGPath(
                                    sequence=[
                                        DGPathEdge.p(p),
                                        DGPathNodeStatement.from_FromWikidataLink(
                                            source.id, p, stmt_i
                                        ),
                                        DGPathEdge.p(p),
                                        DGPathNodeEntity(stmt_value_qnode_id),
                                    ]
                                    + nextpath.sequence
                                )
                            )
                            has_stmt_value = True

                for q, qvals in stmt.qualifiers.items():
                    for qval in qvals:
                        if qval.is_qnode(qval):
                            if qval.as_entity_id() == target.id:
                                if not has_stmt_value:
                                    if stmt.value.is_qnode(stmt.value):
                                        pn_stmt_value = DGPathNodeEntity(
                                            stmt.value.as_entity_id()
                                        )
                                    else:
                                        pn_stmt_value = DGPathNodeLiteralValue(
                                            stmt.value
                                        )
                                    matches.append(
                                        DGPath(
                                            sequence=[
                                                DGPathEdge.p(p),
                                                DGPathNodeStatement.from_FromWikidataLink(
                                                    source.id, p, stmt_i
                                                ),
                                                DGPathEdge.p(p),
                                                pn_stmt_value,
                                            ]
                                        )
                                    )
                                    has_stmt_value = True

                                matches.append(
                                    DGPath(
                                        sequence=[
                                            DGPathEdge.p(p),
                                            DGPathNodeStatement.from_FromWikidataLink(
                                                source.id, p, stmt_i
                                            ),
                                            DGPathEdge.q(q),
                                        ]
                                    )
                                )
                            elif max_n_hop > 1:
                                qval_qnode_id = qval.as_entity_id()
                                if qval_qnode_id.startswith("L"):
                                    assert (
                                        qval_qnode_id not in self.wdentities
                                    ), "The L nodes (lexical) is not in the Wikidata dump"
                                    continue

                                if qval_qnode_id not in self.wdentities:
                                    # this can happen due to some of the qnodes is in the link, but is missing in the KG
                                    # this is very rare so we can employ some check to make sure this is not due to
                                    # our wikidata subset
                                    is_error_in_kg = any(
                                        any(
                                            _s.value.is_qnode(_s.value)
                                            and _s.value.as_entity_id()
                                            in self.wdentities
                                            for _s in _stmts
                                        )
                                        for _p, _stmts in source.props.items()
                                    ) or qval_qnode_id.startswith("L")
                                    if not is_error_in_kg:
                                        raise Exception(
                                            f"Missing qnodes in your KG subset: {qval_qnode_id}"
                                        )
                                    continue

                                _n_matches = len(matches)
                                for nextpath in self._path_object_search(
                                    kg_object_index,
                                    self.wdentities[qval_qnode_id],
                                    target,
                                    max_n_hop - 1,
                                ):
                                    matches.append(
                                        DGPath(
                                            sequence=[
                                                DGPathEdge.p(p),
                                                DGPathNodeStatement.from_FromWikidataLink(
                                                    source.id, p, stmt_i
                                                ),
                                                DGPathEdge.q(q),
                                                DGPathNodeEntity(qval.as_entity_id()),
                                            ]
                                            + nextpath.sequence
                                        )
                                    )

                                if len(matches) > _n_matches and not has_stmt_value:
                                    if stmt.value.is_qnode(stmt.value):
                                        pn_stmt_value = DGPathNodeEntity(
                                            stmt.value.as_entity_id()
                                        )
                                    else:
                                        pn_stmt_value = DGPathNodeLiteralValue(
                                            stmt.value
                                        )
                                    matches.append(
                                        DGPath(
                                            sequence=[
                                                DGPathEdge.p(p),
                                                DGPathNodeStatement.from_FromWikidataLink(
                                                    source.id, p, stmt_i
                                                ),
                                                DGPathEdge.p(p),
                                                pn_stmt_value,
                                            ]
                                        )
                                    )
                                    has_stmt_value = True

        return matches

    def _path_object_search_v2(
        self,
        kg_object_index: KGObjectIndex,
        source: WDEntity,
        target: WDEntity,
        max_n_hop=2,
    ):
        # max_n_hop = 2 mean we will find a path that go from source to target through an hidden node
        # hop start at 1 (current node)

        if not DGConfigs.ALLOW_SAME_ENT_SEARCH and source.id == target.id:
            return []

        matches = []
        for hop1_path in kg_object_index.iter_hop1_props(source.id, target.id):
            stmt_i = hop1_path.statement_index
            rel = hop1_path.relationship
            stmt = source.props[rel.prop][stmt_i]

            if len(rel.quals) > 0 and not rel.both:
                # the prop doesn't match, have to add it
                if stmt.value.is_qnode(stmt.value):
                    pn_stmt_value = DGPathNodeEntity(stmt.value.as_entity_id())
                else:
                    pn_stmt_value = DGPathNodeLiteralValue(stmt.value)
                matches.append(
                    DGPath(
                        sequence=[
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                source.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.p(rel.prop),
                            pn_stmt_value,
                        ]
                    )
                )
            else:
                # the prop match
                matches.append(
                    DGPath(
                        sequence=[
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                source.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.p(rel.prop),
                        ]
                    )
                )
            for qual in rel.quals:
                matches.append(
                    DGPath(
                        sequence=[
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                source.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.q(qual),
                        ]
                    )
                )

        if max_n_hop > 1:
            # TODO: need to refactor the current method of searching for paths
            # as we may generate many duplicated segments
            for match_item in kg_object_index.iter_hop2_props(source.id, target.id):
                if match_item.qnode == target.id:
                    continue
                hop1_paths = match_item.hop1
                hop2_paths = match_item.hop2
                middle_qnode = self.wdentities[match_item.qnode]

                # we don't care about transitive here, so we do a cross product
                hop1_seqs = []
                for hop1_path in hop1_paths:
                    rel = hop1_path.relationship
                    stmt_i = hop1_path.statement_index
                    stmt = source.props[rel.prop][stmt_i]

                    if len(rel.quals) > 0 and not rel.both:
                        # we don't have the statement value yet, add it to the matches
                        # the prop doesn't match, have to add it, we don't worry about duplication
                        # as it is resolve during the merge provenance phase
                        if stmt.value.is_qnode(stmt.value):
                            pn_stmt_value = DGPathNodeEntity(stmt.value.as_entity_id())
                        else:
                            pn_stmt_value = DGPathNodeLiteralValue(stmt.value)
                        matches.append(
                            DGPath(
                                sequence=[
                                    DGPathEdge.p(rel.prop),
                                    DGPathNodeStatement.from_FromWikidataLink(
                                        source.id, rel.prop, stmt_i
                                    ),
                                    DGPathEdge.p(rel.prop),
                                    pn_stmt_value,
                                ]
                            )
                        )
                    else:
                        # add prop to the seqs that we need to expand next, and so stmt.value must be a qnode
                        # as it is the middle qnode
                        assert stmt.value.as_entity_id_safe() == middle_qnode.id
                        hop1_seqs.append(
                            [
                                DGPathEdge.p(rel.prop),
                                DGPathNodeStatement.from_FromWikidataLink(
                                    source.id, rel.prop, stmt_i
                                ),
                                DGPathEdge.p(rel.prop),
                                DGPathNodeEntity(middle_qnode.id),
                            ]
                        )

                    for qual in rel.quals:
                        hop1_seqs.append(
                            [
                                DGPathEdge.p(rel.prop),
                                DGPathNodeStatement.from_FromWikidataLink(
                                    source.id, rel.prop, stmt_i
                                ),
                                DGPathEdge.p(qual),
                                DGPathNodeEntity(middle_qnode.id),
                            ]
                        )

                for hop2_path in hop2_paths:
                    rel = hop2_path.relationship
                    stmt_i = hop2_path.statement_index
                    stmt = middle_qnode.props[rel.prop][stmt_i]

                    if len(rel.quals) > 0 and not rel.both:
                        if stmt.value.is_qnode(stmt.value):
                            pn_stmt_value = DGPathNodeEntity(stmt.value.as_entity_id())
                        else:
                            pn_stmt_value = DGPathNodeLiteralValue(stmt.value)

                        hop2_seq = [
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                middle_qnode.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.p(rel.prop),
                            pn_stmt_value,
                        ]
                        for hop1_seq in hop1_seqs:
                            matches.append(DGPath(sequence=hop1_seq + hop2_seq))
                    else:
                        hop2_seq = [
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                middle_qnode.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.p(rel.prop),
                        ]
                        for hop1_seq in hop1_seqs:
                            matches.append(DGPath(sequence=hop1_seq + hop2_seq))

                    for qual in rel.quals:
                        hop2_seq = [
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                middle_qnode.id, rel.prop, stmt_i
                            ),
                            DGPathEdge.p(qual),
                        ]
                        for hop1_seq in hop1_seqs:
                            matches.append(DGPath(sequence=hop1_seq + hop2_seq))

        return matches

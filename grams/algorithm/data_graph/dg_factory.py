from typing import Mapping, Optional, Set, Union
from grams.algorithm.literal_matchers.literal_match import LiteralMatchConfigs
from kgdata.wikidata.models.wdvalue import WDValueKind

from tqdm import tqdm
import grams.core as gcore
from grams.core.table import LinkedTable as gcore_LinkedTable
from grams.algorithm.data_graph.dg_graph import (
    CellNode,
    ContextSpan,
    DGEdge,
    DGGraph,
    DGPath,
    DGPathEdge,
    DGPathExistingNode,
    DGPathNode,
    DGPathNodeEntity,
    DGPathNodeLiteralValue,
    DGPathNodeStatement,
    DGStatementID,
    EdgeFlowSource,
    EdgeFlowTarget,
    EntityValueNode,
    FlowProvenance,
    LinkGenMethod,
    LiteralValueNode,
    Span,
    StatementNode,
)
from grams.algorithm.data_graph.dg_config import DGConfigs
from grams.algorithm.data_graph.dg_inference import KGInference
from grams.algorithm.data_graph.dg_pruning import DGPruning
from grams.algorithm.kg_index import KGObjectIndex
from grams.algorithm.literal_matchers import LiteralMatch, TextParser
from grams.inputs.linked_table import LinkedTable
from kgdata.wikidata.models import WDEntity, WDProperty


class DGFactory:
    def __init__(
        self,
        wdentities: Mapping[str, WDEntity],
        wdprops: Mapping[str, WDProperty],
        text_parser: TextParser,
        literal_match: LiteralMatch,
        cfg: DGConfigs,
    ):
        self.cfg: DGConfigs = cfg
        self.wdentities = wdentities
        self.wdprops = wdprops
        self.textparser = text_parser
        self.literal_match = literal_match

        if self.cfg.USE_KG_INDEX:
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
                cell_qnodes = {
                    candidate.entity_id
                    for link in table.links[ri][ci]
                    for candidate in link.candidates
                }
                cell_qnode_spans = {}
                cell_qnode_scores = {}
                for link in table.links[ri][ci]:
                    if len(link.candidates) > 0:
                        for can in link.candidates:
                            if can.entity_id not in cell_qnode_spans:
                                cell_qnode_spans[can.entity_id] = []
                            cell_qnode_spans[can.entity_id].append(
                                Span(link.start, link.end)
                            )
                            cell_qnode_scores[can.entity_id] = can.probability

                assert all(
                    len(spans) == len({s.to_tuple() for s in spans})
                    for spans in cell_qnode_spans.values()
                )
                # =====================================================

                node = CellNode(
                    id=CellNode.get_id(ri, ci),
                    value=val,
                    column=ci,
                    row=ri,
                    entity_ids=list(cell_qnodes),
                    entity_spans=cell_qnode_spans,
                    entity_probs=cell_qnode_scores,
                )
                dg.add_node(node)

        if self.cfg.USE_CONTEXT and len(table.context.page_entities) > 0:
            assert table.context.page_title is not None
            for page_entity in table.context.page_entities:
                context_node_id = DGPathNodeEntity(page_entity).get_id()
                node = EntityValueNode(
                    id=context_node_id,
                    qnode_id=page_entity,
                    qnode_prob=1.0,
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
                                    forward_flow={},
                                    reversed_flow={},
                                    flow={},
                                )
                            )
                        elif isinstance(value, DGPathNodeEntity):
                            dg.add_node(
                                EntityValueNode(
                                    id=nodeid,
                                    qnode_id=value.qnode_id,
                                    # the prob of this node should depend on the how it is generated, but since we won't discover
                                    # the link between this node and other node, having the prob 1.0 is fine (we should only this prob. later
                                    # in discovering unmatched links)
                                    qnode_prob=1.0,
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
                        id=-1,  # will be assigned later
                        source=curr_nodeid,
                        target=nodeid,
                        predicate=prop.value,
                        is_qualifier=prop.is_qualifier,
                        is_inferred=False,
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

        if self.cfg.RUN_INFERENCE:
            KGInference(
                dg, self.wdentities, self.wdprops
            ).infer_subproperty().kg_transitive_inference()

        # pruning unnecessary paths
        if self.cfg.PRUNE_REDUNDANT_ENT:
            DGPruning(dg, self.cfg).prune_hidden_entities()

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
            for newpath in self._path_value_search(
                n1, self.textparser.parse(v_value), v_qnodes
            ):
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

    def _path_value_search(self, source: WDEntity, value, target_ents: list[WDEntity]):
        target_ent_ids = None
        matches = []
        for p, stmts in source.props.items():
            # if p == "P31":
            #     # no need to search in the instanceOf property, as the ontology is removed from the databased as they are huge
            #     continue

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
                                    None,
                                    fn.__name__,
                                    confidence,
                                ),
                                DGPathEdge.p(p),
                            ]
                        )
                    )
                    has_stmt_value = True

                for q, qvals in stmt.qualifiers.items():
                    for qual_i, qval in enumerate(qvals):
                        for fn, (match, confidence) in self.literal_match.match(
                            qval, value, skip_unmatch=True
                        ):
                            if not has_stmt_value:
                                if stmt.value.is_qnode(stmt.value):
                                    if target_ent_ids is None:
                                        target_ent_ids = {e.id for e in target_ents}

                                    target_ent_id = stmt.value.as_entity_id()
                                    has_stmt_value = target_ent_id in target_ent_ids

                                    pn_stmt_value = DGPathNodeEntity(target_ent_id)
                                else:
                                    pn_stmt_value = DGPathNodeLiteralValue(stmt.value)

                                if not has_stmt_value:
                                    matches.append(
                                        DGPath(
                                            sequence=[
                                                DGPathEdge.p(p),
                                                DGPathNodeStatement.from_FromWikidataLink(
                                                    source.id, p, stmt_i, None
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
                                            qual_i,
                                            fn.__name__,
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

        if not self.cfg.ALLOW_SAME_ENT_SEARCH and source.id == target.id:
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
                                        source.id, p, stmt_i, None
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
                                            source.id, p, stmt_i, None
                                        ),
                                        DGPathEdge.p(p),
                                        DGPathNodeEntity(stmt_value_qnode_id),
                                    ]
                                    + nextpath.sequence
                                )
                            )
                            has_stmt_value = True

                for q, qvals in stmt.qualifiers.items():
                    for qual_i, qval in enumerate(qvals):
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
                                                    source.id, p, stmt_i, None
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
                                                source.id, p, stmt_i, qual_i
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
                                                    source.id, p, stmt_i, qual_i
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
                                                    source.id, p, stmt_i, None
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

        if not self.cfg.ALLOW_SAME_ENT_SEARCH and source.id == target.id:
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
                                source.id, rel.prop, stmt_i, None
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
                                source.id, rel.prop, stmt_i, None
                            ),
                            DGPathEdge.p(rel.prop),
                        ]
                    )
                )
            for qual, qual_i in rel.quals:
                matches.append(
                    DGPath(
                        sequence=[
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                source.id, rel.prop, stmt_i, qual_i
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
                                        source.id, rel.prop, stmt_i, None
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
                                    source.id, rel.prop, stmt_i, None
                                ),
                                DGPathEdge.p(rel.prop),
                                DGPathNodeEntity(middle_qnode.id),
                            ]
                        )

                    for qual, qual_i in rel.quals:
                        hop1_seqs.append(
                            [
                                DGPathEdge.p(rel.prop),
                                DGPathNodeStatement.from_FromWikidataLink(
                                    source.id, rel.prop, stmt_i, qual_i
                                ),
                                DGPathEdge.q(qual),
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
                                middle_qnode.id, rel.prop, stmt_i, None
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
                                middle_qnode.id, rel.prop, stmt_i, None
                            ),
                            DGPathEdge.p(rel.prop),
                        ]
                        for hop1_seq in hop1_seqs:
                            matches.append(DGPath(sequence=hop1_seq + hop2_seq))

                    for qual, qual_i in rel.quals:
                        hop2_seq = [
                            DGPathEdge.p(rel.prop),
                            DGPathNodeStatement.from_FromWikidataLink(
                                middle_qnode.id, rel.prop, stmt_i, qual_i
                            ),
                            DGPathEdge.q(qual),
                        ]
                        for hop1_seq in hop1_seqs:
                            matches.append(DGPath(sequence=hop1_seq + hop2_seq))

        return matches


def create_dg(
    table: LinkedTable,
    rust_table: gcore_LinkedTable,
    rust_context: gcore.AlgoContext,
    wdentities: Mapping[str, WDEntity],
    wdprops: Mapping[str, WDProperty],
    text_parser: TextParser,
    literal_matcher_cfg: LiteralMatchConfigs,
    cfg: DGConfigs,
):
    nrows, ncols = table.shape()
    cells = [
        [text_parser.parse(table.table[ri, ci]).to_rust() for ci in range(ncols)]
        for ri in range(nrows)
    ]
    literal_matcher = gcore.literal_matchers.LiteralMatcher(
        literal_matcher_cfg.to_rust()
    )
    matches = gcore.steps.data_matching.matching(
        rust_table,
        cells,
        rust_context,
        literal_matcher,
        [],
        # ["P31"],
        [],
        cfg.ALLOW_SAME_ENT_SEARCH,
        cfg.USE_CONTEXT,
    )

    dg = DGGraph()
    dg_nodeids = []
    for i in range(matches.get_n_nodes()):
        if matches.is_cell_node(i):
            rustcellnode = matches.get_cell_node(i)
            # =====================================================
            cell_qnodes = {
                candidate.entity_id
                for link in table.links[rustcellnode.row][rustcellnode.col]
                for candidate in link.candidates
            }
            cell_qnode_spans = {}
            cell_qnode_scores = {}
            for link in table.links[rustcellnode.row][rustcellnode.col]:
                if len(link.candidates) > 0:
                    for can in link.candidates:
                        if can.entity_id not in cell_qnode_spans:
                            cell_qnode_spans[can.entity_id] = []
                        cell_qnode_spans[can.entity_id].append(
                            Span(link.start, link.end)
                        )
                        cell_qnode_scores[can.entity_id] = can.probability

            assert all(
                len(spans) == len({s.to_tuple() for s in spans})
                for spans in cell_qnode_spans.values()
            )
            # =====================================================

            node = CellNode(
                id=CellNode.get_id(rustcellnode.row, rustcellnode.col),
                value=table.table[rustcellnode.row, rustcellnode.col],
                column=rustcellnode.col,
                row=rustcellnode.row,
                entity_ids=list(cell_qnodes),
                entity_spans=cell_qnode_spans,
                entity_probs=cell_qnode_scores,
            )
            dg.add_node(node)
        else:
            rustentnode = matches.get_entity_node(i)
            node = EntityValueNode(
                id=DGPathNodeEntity(rustentnode.entity_id).get_id(),
                qnode_id=rustentnode.entity_id,
                qnode_prob=1.0,
                context_span=None,
            )
            dg.add_node(node)
        dg_nodeids.append(node.id)

    # add paths to the graph
    for rels in matches.iter_rels():
        uid = dg_nodeids[rels.source_id]
        vid = dg_nodeids[rels.target_id]

        for rel in rels.iter_rels():
            for stmt in rel.iter_statements():
                stmt_property = stmt.property
                sid = DGStatementID(
                    rel.source_entity_id, stmt_property, stmt.statement_index
                ).get_id()
                if not dg.has_node(sid):
                    dg.add_node(
                        StatementNode(
                            id=sid,
                            qnode_id=rel.source_entity_id,
                            predicate=stmt_property,
                            is_in_kg=True,
                            forward_flow={},
                            reversed_flow={},
                            flow={},
                        )
                    )

                if not dg.has_edge_between_nodes(uid, sid, stmt_property):
                    dg.add_edge(
                        DGEdge(
                            id=-1,  # will be assigned later
                            source=uid,
                            target=sid,
                            predicate=stmt_property,
                            is_qualifier=False,
                            is_inferred=False,
                        )
                    )

                snode = dg.get_statement_node(sid)
                edge_source = EdgeFlowSource(uid, stmt_property)

                if stmt.property_matched_score is not None:
                    # we have a matched property
                    if not dg.has_edge_between_nodes(sid, vid, stmt_property):
                        dg.add_edge(
                            DGEdge(
                                id=-1,  # will be assigned later
                                source=sid,
                                target=vid,
                                predicate=stmt_property,
                                is_qualifier=False,
                                is_inferred=False,
                            )
                        )

                    edge_target = EdgeFlowTarget(vid, stmt_property)
                    # TODO: determine the provenance
                    snode.track_provenance(
                        edge_source,
                        edge_target,
                        [
                            FlowProvenance(
                                gen_method=LinkGenMethod.FromWikidataLink,
                                gen_method_arg={"value_index": stmt.statement_index},
                                prob=stmt.property_matched_score,
                            )
                        ],
                    )
                else:
                    # we do not have a matched property, so we need to add a new node
                    # to represent the property's value
                    stmt_value = (
                        wdentities[rel.source_entity_id]
                        .props[stmt_property][stmt.statement_index]
                        .value
                    )
                    target_id = add_stmt_value_node(dg, stmt_value)

                    if not dg.has_edge_between_nodes(sid, target_id, stmt_property):
                        dg.add_edge(
                            DGEdge(
                                id=-1,  # will be assigned later
                                source=sid,
                                target=target_id,
                                predicate=stmt_property,
                                is_qualifier=False,
                                is_inferred=False,
                            )
                        )

                    edge_target = EdgeFlowTarget(target_id, stmt_property)
                    # TODO: determine the provenance
                    snode.track_provenance(
                        edge_source,
                        edge_target,
                        [
                            FlowProvenance(
                                gen_method=LinkGenMethod.FromWikidataLink,
                                gen_method_arg={"value_index": stmt.statement_index},
                                prob=1.0,
                            )
                        ],
                    )

                for qual in stmt.iter_qualifier_matched_scores():
                    qual_qualifier = qual.qualifier
                    if not dg.has_edge_between_nodes(sid, vid, qual_qualifier):
                        dg.add_edge(
                            DGEdge(
                                id=-1,  # will be assigned later
                                source=sid,
                                target=vid,
                                predicate=qual_qualifier,
                                is_qualifier=True,
                                is_inferred=False,
                            )
                        )
                    edge_target = EdgeFlowTarget(vid, qual_qualifier)
                    # TODO: determine the provenance
                    snode.track_provenance(
                        edge_source,
                        edge_target,
                        [
                            FlowProvenance(
                                gen_method=LinkGenMethod.FromWikidataLink,
                                gen_method_arg={"value_index": qual.qualifier_index},
                                prob=qual.matched_score,
                            )
                        ],
                    )

    if cfg.ADD_MISSING_PROPERTY:
        add_missing_property(dg, wdentities)

    if cfg.RUN_INFERENCE:
        KGInference(
            dg, wdentities, wdprops
        ).infer_subproperty().kg_transitive_inference()

    # pruning unnecessary paths
    if cfg.PRUNE_REDUNDANT_ENT:
        DGPruning(dg, cfg).prune_hidden_entities()

    return dg


def add_missing_property(dg: DGGraph, wdentities: Mapping[str, WDEntity]):
    """There is a case where a statement have both qualifier and property pointing the same node. Unless we have another property, the qualifier is unlikely to be chosen. Therefore, we add
    a new property node to give the qualifier the chance to be chosen."""
    for snode in dg.iter_nodes():
        if isinstance(snode, StatementNode):
            outedges = dg.out_edges(snode.id)
            # get the property, and
            prop_edges: list[DGEdge] = []
            qual_edges: list[DGEdge] = []
            for outedge in outedges:
                if snode.predicate == outedge.predicate:
                    prop_edges.append(outedge)
                else:
                    qual_edges.append(outedge)
            if len(prop_edges) == 1:
                prop_target = prop_edges[0].target
                if any(e.target == prop_target for e in qual_edges):
                    qual_edge = next(
                        qual_edge
                        for qual_edge in qual_edges
                        if qual_edge.target == prop_target
                    )

                    # we need a new property node
                    stmt_value = (
                        wdentities[snode.qnode_id]
                        .props[snode.predicate][snode.statement_index]
                        .value
                    )
                    target_id = add_stmt_value_node(dg, stmt_value)

                    if not dg.has_edge_between_nodes(
                        snode.id, target_id, snode.predicate
                    ):
                        dg.add_edge(
                            DGEdge(
                                id=-1,  # will be assigned later
                                source=snode.id,
                                target=target_id,
                                predicate=snode.predicate,
                                is_qualifier=False,
                                is_inferred=False,
                            )
                        )

                    # the new node will have the same flow as one of the qualifier
                    for edge_source in snode.reversed_flow[
                        EdgeFlowTarget(qual_edge.target, qual_edge.predicate)
                    ]:
                        edge_target = EdgeFlowTarget(target_id, snode.predicate)
                        # TODO: determine the provenance
                        snode.track_provenance(
                            edge_source,
                            edge_target,
                            [
                                FlowProvenance(
                                    gen_method=LinkGenMethod.FromWikidataLink,
                                    gen_method_arg={
                                        "value_index": snode.statement_index
                                    },
                                    prob=1.0,
                                )
                            ],
                        )


def add_stmt_value_node(dg: DGGraph, stmt_value: WDValueKind):
    if stmt_value.is_entity_id(stmt_value):
        target_entid = stmt_value.as_entity_id()
        target_id = DGPathNodeEntity(target_entid).get_id()
        if not dg.has_node(target_id):
            dg.add_node(
                EntityValueNode(
                    id=target_id,
                    qnode_id=target_entid,
                    # the prob of this node should depend on the how it is generated, but since we won't discover
                    # the link between this node and other node, having the prob 1.0 is fine (we should only this prob. later
                    # in discovering unmatched links)
                    qnode_prob=1.0,
                    context_span=None,
                )
            )
    else:
        target_id = DGPathNodeLiteralValue(stmt_value).get_id()
        if not dg.has_node(target_id):
            dg.add_node(
                LiteralValueNode(
                    id=target_id,
                    value=stmt_value,
                    context_span=None,
                )
            )
    return target_id


def assert_dg_equal(table_id: str, pydg: DGGraph, rudg: DGGraph):
    # handle special cases where we know its true

    def remove_extra_nodes(pydg2: DGGraph, rudg2: DGGraph):
        # rudg will have extra value nodes when it tries to fix qualifiers and property pointing to the same node
        pynodes1 = {u.id for u in pydg2.iter_nodes()}
        runodes2 = {u.id for u in rudg2.iter_nodes()}

        assert pynodes1.issubset(runodes2)
        for node in runodes2.difference(pynodes1):
            assert isinstance(
                node, (LiteralValueNode)
            )  # it's rare to see an entity in both property and qualifiers of a statement
            (inedge,) = rudg2.in_edges(node.id)
            # make sure that the source node
            outedges = pydg2.out_edges(inedge.source)
            (prop_edge,) = [e for e in outedges if e.predicate == inedge.predicate]
            assert any(
                e.target == prop_edge.target
                for e in outedges
                if e.predicate != inedge.predicate
            )
            rudg2.remove_node(node.id)

    def remove_node(dg: DGGraph, uid: str):
        (inedge,) = dg.in_edges(uid)
        stmt = dg.get_statement_node(inedge.source)
        stmt.untrack_target_flow(EdgeFlowTarget(uid, inedge.predicate))
        dg.remove_node(uid)

    if table_id == "Miss_Heritage_2014":
        remove_node(pydg, 'val:{"type":"string","value":"Argentina"}')
    elif table_id == "List_of_women's_Twenty20_International_cricket_grounds":
        remove_node(
            pydg,
            'val:{"type":"monolingualtext","value":{"text":"Bristol","language":"en"}}',
        )
    elif table_id == "Major_professional_sports_teams_of_the_United_States_and_Canada":
        remove_node(
            pydg,
            'val:{"type":"monolingualtext","value":{"text":"Dallas","language":"cs"}}',
        )

    nodes1 = {u.id for u in pydg.iter_nodes()}
    nodes2 = {u.id for u in rudg.iter_nodes()}

    # check if the nodes are the same
    diff_nodes = list(nodes1.symmetric_difference(nodes2))
    if len(diff_nodes) > 0:
        print("-----\nDG1 contains the following nodes that are not in DG2:")
        print(nodes1.difference(nodes2))
        print("-----\nDG2 contains the following nodes that are not in DG1:")
        print(nodes2.difference(nodes1))
        assert False

    # check if the edges are the same
    edges1 = {(u.source, u.target, u.predicate) for u in pydg.iter_edges()}
    edges2 = {(u.source, u.target, u.predicate) for u in rudg.iter_edges()}

    diff_edges = edges1.symmetric_difference(edges2)
    if len(diff_edges) > 0:
        print("-----\nDG1 contains the following edges that are not in DG2:")
        print(edges1.difference(edges2))
        print("-----\nDG2 contains the following edges that are not in DG1:")
        print(edges2.difference(edges1))
        assert False

    # check the flow
    for snode in pydg.iter_nodes():
        if isinstance(snode, StatementNode):
            flows1 = set(snode.flow.keys())
            flows2 = set(rudg.get_statement_node(snode.id).flow.keys())
            diff_flows = flows1.symmetric_difference(flows2)
            if len(diff_flows) > 0:
                print("-----\nDG1 contains the following flows that are not in DG2:")
                print(flows1.difference(flows2))
                print("-----\nDG2 contains the following flows that are not in DG1:")
                print(flows2.difference(flows1))
                assert False

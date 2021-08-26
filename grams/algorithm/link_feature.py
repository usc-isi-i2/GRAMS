import functools
from collections import defaultdict
from enum import Enum
from operator import xor, attrgetter

import networkx as nx
from typing import Dict, Iterable, Tuple, Set, List, Optional, Callable

from kgdata.wikidata.models import QNode, WDProperty, WDQuantityPropertyStats
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.data_graph import DGNode
from grams.algorithm.semantic_graph import SGNode, SGStatementNode, SGEdge
from grams.algorithm.literal_match import TextParser


class QuantityType(Enum):
    Integer = "int"
    Float = "float"


class LinkFeatureExtraction:
    FreqOverRow = "FrequencyOfLinkOverRow"
    FreqOverEntRow = "FrequencyOfLinkOverEntRow"
    FreqOverPossibleLink = "FrequencyOfLinkOverPossibleLink"
    FreqUnmatchOverEntRow = "FrequencyOfUnmatchLinkOverEntRow"
    FreqUnmatchOverPossibleLink = "FrequencyOfUnmatchLinkOverPossibleLink"
    GTE5Link = "GTE5Link"
    GTE10Link = "GTE10Link"
    GTE15Link = "GTE15Link"
    GTE20Link = "GTE20Link"
    GTE30Link = "GTE30Link"
    GTE40Link = "GTE40Link"
    GTE50Link = "GTE50Link"
    NotFuncDep = "NotFuncDep"
    DataTypeMismatch = "DataTypeMismatch"
    HeaderSimilarity = "HeaderSimimilarity"
    
    def __init__(self,
                 table: LinkedTable, sg: nx.MultiDiGraph, dg: nx.MultiDiGraph,
                 qnodes: Dict[str, QNode], wdprops: Dict[str, WDProperty],
                 wd_num_prop_stats: Dict[str, WDQuantityPropertyStats],
                 sim_fn: Optional[Callable[[str, str], float]] = None):
        self.table = table
        self.sg = sg
        self.dg = dg
        self.qnodes = qnodes
        self.wdprops = wdprops
        self.wd_num_prop_stats = wd_num_prop_stats
        self.text_parser = TextParser()
        self.sim_fn = sim_fn

        self.cache_get_value_map = {}

    def extract_features(self):
        n_rows = self.table.size()
        freq_over_row = {}
        freq_over_ent_row = {}
        freq_over_pos_link = {}
        freq_unmatch_over_ent_row = {}
        freq_unmatch_over_pos_link = {}
        not_fd_links = {}
        dtype_mismatch_links = {}
        # make none to force setting sim_fn to not None when we want to use the features
        header_sim_links = None if self.sim_fn is None else {}

        gte_link_thresholds = [5, 10, 15, 20, 30, 40, 50]
        gte_link_bins = {k: {} for k in gte_link_thresholds}
        pair2max_possible_link = defaultdict(int)

        for sid, sdata in self.sg.nodes(data=True):
            stmt: SGStatementNode = self.sg.nodes[sid]['data']
            if not stmt.is_statement:
                continue

            in_edge: SGEdge
            out_edge: SGEdge

            (uid, _, in_edge), = list(self.sg.in_edges(stmt.id, data='data'))
            unode: SGNode = self.sg.nodes[uid]['data']
            for _, vid, out_edge in self.sg.out_edges(stmt.id, data='data'):
                vnode: SGNode = self.sg.nodes[vid]['data']
                dg_flows = stmt.get_edges_provenance([out_edge])
                dg_links: Set[Tuple[str, str]] = set()

                sum_prob = 0.0
                for source_flow, target_flows in dg_flows.items():
                    assert source_flow.sg_source_id == uid
                    if unode.is_column:
                        # we should only have one target flow as we consider row by row only
                        target_flow, provenances = self._unpack_size1_dict(target_flows)
                        dg_links.add((source_flow.dg_source_id, target_flow.dg_target_id))
                        sum_prob += max(provenances, key=attrgetter('prob')).prob
                    else:
                        # this is entity and we may have more than one target flow (to different row)
                        assert unode.is_entity_value
                        for target_flow, provenances in target_flows.items():
                            dg_links.add((source_flow.dg_source_id, target_flow.dg_target_id))
                            sum_prob += max(provenances, key=attrgetter('prob')).prob

                # what is the maximum possible links we can have? this ignore the the link so this is used to calculate FreqOverEntRow
                max_possible_ent_rows = self._get_maximum_possible_ent_links_between_two_nodes(
                    uid, vid, self.wdprops[out_edge.predicate].is_data_property())

                # calculate the number of links that the source node doesn't have the values as in the target nodes
                # we should consider on what kind of missing. e.g., object missing is more obvious than value mis-match
                # in value mis-match: datetime mis-match is more profound than area/population mis-match
                n_unmatch_links = self._get_n_unmatch_discovered_links(
                    uid, vid, in_edge.predicate, out_edge.predicate,
                    dg_links, self.wdprops[out_edge.predicate].is_data_property())
                n_possible_links = n_unmatch_links + len(dg_links)
                pair2max_possible_link[uid, vid] = max(n_possible_links, pair2max_possible_link[uid, vid])

                # compute the features
                pair_freq_over_row = sum_prob / n_rows
                pair_freq_over_ent_row = (sum_prob / max_possible_ent_rows) if max_possible_ent_rows > 0 else 0
                # re-calculate it later
                pair_freq_over_pos_link = (sum_prob, (uid, vid))
                pair_freq_unmatch_over_ent_row = (n_unmatch_links / max_possible_ent_rows) if max_possible_ent_rows > 0 else 0
                # re-calculate it later
                pair_freq_unmatch_over_pos_link = (n_unmatch_links, (uid, vid))
                dtype_mismatch = None
                header_sim = None

                if vnode.is_column:
                    assert out_edge.predicate[0] == 'P', 'Sanity Check'
                    if in_edge.predicate in self.wd_num_prop_stats and (in_edge.predicate == out_edge.predicate or out_edge.predicate in self.wd_num_prop_stats[in_edge.predicate].qualifiers):
                        if in_edge.predicate == out_edge.predicate:
                            stat = self.wd_num_prop_stats[in_edge.predicate].value
                        else:
                            stat = self.wd_num_prop_stats[in_edge.predicate].qualifiers[out_edge.predicate]

                        dtype = self._estimate_col_quantity_type(vnode.column)
                        if dtype is not None and stat.size > 0:
                            # if the property is not quantity, their size is 0
                            is_prop_int = (stat.int_size / stat.size) > 0.99
                            if dtype == QuantityType.Integer:
                                dtype_mismatch = int(not is_prop_int)
                            else:
                                dtype_mismatch = int(is_prop_int)
                    if self.sim_fn is not None:
                        header_sim = self.sim_fn(self.wdprops[out_edge.predicate].label, vnode.label)

                if in_edge.predicate == out_edge.predicate:
                    # copy to the parent property as well
                    key = (uid, sid, in_edge.predicate)
                    assert key not in freq_over_row
                    freq_over_row[key] = pair_freq_over_row
                    freq_over_ent_row[key] = pair_freq_over_ent_row
                    freq_over_pos_link[key] = pair_freq_over_pos_link
                    freq_unmatch_over_ent_row[key] = pair_freq_unmatch_over_ent_row
                    freq_unmatch_over_pos_link[key] = pair_freq_unmatch_over_pos_link

                    if unode.is_column and vnode.is_column:
                        not_fd_links[key] = int(not self._functional_dependency_test(unode.column, vnode.column))

                    for threshold, indicator in zip(gte_link_thresholds, self._categorize_num_gte(len(dg_links), gte_link_thresholds)):
                        gte_link_bins[threshold][key] = indicator

                    if dtype_mismatch is not None:
                        dtype_mismatch_links[key] = dtype_mismatch

                    if header_sim is not None:
                        header_sim_links[key] = header_sim

                key = (sid, vid, out_edge.predicate)
                assert key not in freq_over_row
                freq_over_row[key] = pair_freq_over_row
                freq_over_ent_row[key] = pair_freq_over_ent_row
                freq_over_pos_link[key] = pair_freq_over_pos_link
                freq_unmatch_over_ent_row[key] = pair_freq_unmatch_over_ent_row
                freq_unmatch_over_pos_link[key] = pair_freq_unmatch_over_pos_link

                for threshold, indicator in zip(gte_link_thresholds,
                                                self._categorize_num_gte(len(dg_links), gte_link_thresholds)):
                    gte_link_bins[threshold][key] = indicator

                if unode.is_column and vnode.is_column:
                    not_fd_links[key] = int(not self._functional_dependency_test(unode.column, vnode.column))

                if dtype_mismatch is not None:
                    dtype_mismatch_links[key] = dtype_mismatch

                if header_sim is not None:
                    header_sim_links[key] = header_sim

        # re-adjust the feature over possible links
        for key, (freq, pair) in freq_over_pos_link.items():
            if pair2max_possible_link[pair] == 0:
                freq_over_pos_link[key] = 0
            else:    
                freq_over_pos_link[key] = freq / pair2max_possible_link[pair]
        for key, (freq, pair) in freq_unmatch_over_pos_link.items():
            if pair2max_possible_link[pair] == 0:
                freq_unmatch_over_pos_link[key] = 0
            else:
                freq_unmatch_over_pos_link[key] = freq / pair2max_possible_link[pair]

        return {
            self.FreqOverRow: freq_over_row,
            self.FreqOverEntRow: freq_over_ent_row,
            self.FreqOverPossibleLink: freq_over_pos_link,
            self.FreqUnmatchOverEntRow: freq_unmatch_over_ent_row,
            self.FreqUnmatchOverPossibleLink: freq_unmatch_over_pos_link,
            self.GTE5Link: gte_link_bins[5],
            self.GTE10Link: gte_link_bins[10],
            self.GTE15Link: gte_link_bins[15],
            self.GTE20Link: gte_link_bins[20],
            self.GTE30Link: gte_link_bins[30],
            self.GTE40Link: gte_link_bins[40],
            self.GTE50Link: gte_link_bins[50],
            self.NotFuncDep: not_fd_links,
            self.DataTypeMismatch: dtype_mismatch_links,
            self.HeaderSimilarity: header_sim_links
        }

    def add_debug_info(self, features: dict, sg=None):
        if sg is None:
            sg = self.sg
        """Add features to edges in the graph for debugging. Features are from the extract_features function"""
        # add the features to edges for debugging purpose
        for feat, feat_data in features.items():
            if feat_data is None:
                continue
            for (uid, vid, eid), val in feat_data.items():
                if sg.has_edge(uid, vid, eid):
                    edge: SGEdge = sg.edges[uid, vid, eid]['data']
                    assert feat not in edge.features
                    edge.features[feat] = val

    def _unpack_size1_dict(self, odict: dict):
        """Unpack a dictionary of size 1"""
        lst = list(odict.items())
        assert len(lst) == 1, lst
        return lst[0]

    def _categorize_num_gte(self, num: int, thresholds: List[int]):
        feats = []
        for i in range(len(thresholds)):
            if num >= thresholds[i]:
                feats.append(1)
            else:
                feats.append(0)
        return feats

    def _iter_dg_pair(self, uid: str, vid: str) -> Iterable[Tuple[DGNode, DGNode]]:
        """This function iterate through each pair of data graph nodes between two semantic graph nodes.

                    If both sg nodes are entities, we only have one pair.
                    If one or all of them are columns, the number of pairs will be the size of the table.
                    Otherwise, not support iterating between nodes & statements
                    """
        u: SGNode = self.sg.nodes[uid]['data']
        v: SGNode = self.sg.nodes[vid]['data']

        if u.is_column and v.is_column:
            uci = u.column
            vci = v.column
            for ri in range(self.table.size()):
                ucell = self.dg.nodes[f"{ri}-{uci}"]['data']
                vcell = self.dg.nodes[f"{ri}-{vci}"]['data']
                yield ucell, vcell
        elif u.is_column:
            assert v.is_value
            uci = u.column
            vcell = self.dg.nodes[v.id]['data']
            for ri in range(self.table.size()):
                ucell = self.dg.nodes[f"{ri}-{uci}"]['data']
                yield ucell, vcell
        elif v.is_column:
            assert u.is_value
            vci = v.column
            ucell = self.dg.nodes[u.id]['data']
            for ri in range(self.table.size()):
                vcell = self.dg.nodes[f"{ri}-{vci}"]['data']
                yield ucell, vcell
        else:
            assert not u.is_column and not v.is_column
            yield self.dg.nodes[u.id]['data'], self.dg.nodes[v.id]['data']

    def _get_maximum_possible_ent_links_between_two_nodes(self, uid: str, vid: str, is_data_predicate: bool):
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
        u: SGNode = self.sg.nodes[uid]['data']
        v: SGNode = self.sg.nodes[vid]['data']

        if not u.is_column and not v.is_column:
            return 1

        # instead of going through each node attach to the node in the semantic graph, we cheat by directly generating
        # the data node ID
        n_rows = self.table.size()
        n_null_entities = 0
        if xor(u.is_column, v.is_column):
            if is_data_predicate:
                if u.is_value:
                    assert u.is_entity_value and v.is_column
                    return n_rows

                assert u.is_column and v.is_value

            ci = u.column if u.is_column else v.column
            for ri in range(n_rows):
                if len(self.dg.nodes[f"{ri}-{ci}"]['data'].qnode_ids) == 0:
                    n_null_entities += 1
        else:
            uci = u.column
            vci = v.column

            for ri in range(n_rows):
                ucell_unk = len(self.dg.nodes[f"{ri}-{uci}"]['data'].qnode_ids) == 0
                vcell_unk = len(self.dg.nodes[f"{ri}-{vci}"]['data'].qnode_ids) == 0
                
                if is_data_predicate:
                    if ucell_unk:
                        n_null_entities += 1
                elif ucell_unk or vcell_unk:
                    n_null_entities += 1

        return n_rows - n_null_entities

    def _dg_pair_has_possible_ent_links(self, dgu: DGNode, dgv: DGNode, is_data_predicate: bool):
        if dgu.is_cell and dgv.is_cell:
            # both are cells
            if is_data_predicate:
                # data predicate: source cell must link to some entities to have possible links
                return len(dgu.qnode_ids) > 0
            else:
                # object predicate: source cell and target cell must link to some entities to have possible links
                return len(dgu.qnode_ids) > 0 and len(dgv.qnode_ids) > 0
        elif dgu.is_cell:
            # the source is cell, the target will be literal/entity value
            # we have link when source cell link to some entities, doesn't depend on type of predicate
            return len(dgu.qnode_ids) > 0
        elif dgv.is_cell:
            # the target is cell, the source will be literal/entity value
            if is_data_predicate:
                # data predicate: always has possibe links
                return True
            else:
                # object predicate: have link when the target cell link to some entities
                return len(dgv.qnode_ids) > 0
        else:
            # all cells are values, always have link due to how the link is generated in the first place
            return True

    def _get_n_unmatch_discovered_links(self, uid: str, vid: str, inpred: str, outpred: str, uv_links: Set[Tuple[str, str]],
                                       is_outpred_data_predicate: bool):
        """Get number of discovered links that don't match due to value differences. This function do not count if:
        * the link between two DG nodes is impossible
        * the property/qualifier do not exist in the QNode
        """
        u: SGNode = self.sg.nodes[uid]['data']
        v: SGNode = self.sg.nodes[vid]['data']
        is_outpred_qualifier = inpred != outpred

        n_unmatch_links = 0
        for dgu, dgv in self._iter_dg_pair(uid, vid):
            # if has link, then we don't have to count
            if (dgu.id, dgv.id) in uv_links:
                continue

            # ignore pairs that can't have any links
            if not self._dg_pair_has_possible_ent_links(dgu, dgv, is_outpred_data_predicate):
                continue

            if dgu.is_cell:
                # the source is cell node
                # property doesn't exist in any qnode
                dgu_qnodes = [self.qnodes[qnode_id] for qnode_id in dgu.qnode_ids]
                if all(inpred not in qnode.props for qnode in dgu_qnodes):
                    continue

                if is_outpred_qualifier:
                    # qualifier doesn't exist in any qnode
                    has_qual = False
                    for qnode in dgu_qnodes:
                        for stmt in qnode.props.get(inpred, []):
                            if outpred in stmt.qualifiers:
                                has_qual = True
                    if not has_qual:
                        continue

                # #########################
                # TODO: remove me, modify on Apr 29, should have a better solution
                # this is to say like if that properties have multiple values, then it's more likely to miss
                if all(len(qnode.props.get(inpred, [])) > 2 for qnode in dgu_qnodes):
                    continue
                # #########################
            else:
                # the source is entity
                assert dgu.is_entity_value
                dgu_qnode = self.qnodes[dgu.qnode_id]
                if inpred not in dgu_qnode.props:
                    continue
                if is_outpred_qualifier:
                    if all(outpred not in stmt.qualifiers for stmt in dgu_qnode.props[inpred]):
                        continue
                # #########################
                # TODO: remove me, modify on Apr 29, should have a better solution
                # this is to say like if that properties have multiple values, then it's more likely to miss
                if len(dgu_qnode.props.get(inpred, [])) > 2:
                    continue
                # #########################
            n_unmatch_links += 1
        return n_unmatch_links

    def _functional_dependency_test(self, source_column_index: int, target_column_index: int):
        """Test whether values in the target column is uniquely determiend by the values in the source column. True if
        it's FD.

        Parameters
        ----------
        source_column_index
        target_column_index

        Returns
        -------
        """
        sci = source_column_index
        tci = target_column_index

        # find a mapping from
        source_map = self._get_value_map(source_column_index)
        target_map = {
            ri: key
            for key, rows in self._get_value_map(target_column_index).items()
            for ri in rows
        }

        n_violate_fd = 0
        for key, rows in source_map.items():
            target_keys = {target_map[ri] for ri in rows}
            if len(target_keys) > 1:
                n_violate_fd += 1

        if len(source_map) == 0:
            return True

        if n_violate_fd / len(source_map) > 0.01:
            return False
        return True

    def _estimate_col_quantity_type(self, column_index: int) -> Optional[QuantityType]:
        """These function should be replaced by a ML method"""
        col = self.table.table.get_column_by_index(column_index)
        n_int, n_float = 0, 0
        n_values = 0
        for val in col.values:
            nval = self.text_parser.parse(val)
            if nval.number is not None:
                if isinstance(nval.number, int):
                    n_int += 1
                else:
                    n_float += 1
            if nval.normed_string.lower() not in {"", "na", "n/a"}:
                n_values += 1
        # more than 80% is considered to be the numeric column
        if n_values > 0 and (n_int + n_float) / n_values > 0.95:
            if n_int / (n_int + n_float) > 0.95:
                return QuantityType.Integer
            return QuantityType.Float
        return None

    def _get_value_map(self, column_index: int):
        """Get a map of values in a column to its row numbers (possible duplication).
        This function is not perfect now. The value of the column is considered to be list of entities (if exist) or
        just the value of the cell
        """
        if column_index not in self.cache_get_value_map:
            map = defaultdict(list)
            col = self.table.table.get_column_by_index(column_index)
            for ri in range(self.table.size()):
                links = self.table.links[ri][column_index]
                ents = [link.qnode_id for link in links if link.qnode_id is not None]
                if len(ents) > 0:
                    key = tuple(ents)
                else:
                    key = col.values[ri].strip()
                map[key].append(ri)
            self.cache_get_value_map[column_index] = dict(map)
        return self.cache_get_value_map[column_index]

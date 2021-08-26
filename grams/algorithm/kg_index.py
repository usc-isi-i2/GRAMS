import enum
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set, Union, List, NamedTuple, Optional

import sm.misc as M
from loguru import logger
from tqdm.auto import tqdm
from kgdata.wikidata.models import QNode, WDProperty

Relationship = NamedTuple('Relationship', [('prop', str), ('quals', List[str]), ('both', bool)])
OneHopIndexPath = NamedTuple('OneHopIndexPath', [('relationship', Relationship), ('statement_index', int)])
OneHopIndex = Dict[str, Dict[str, List[OneHopIndexPath]]]

# to find all paths between hop1 & hop2, you need to do cross product, but if you consider only transitive props, then
# you need to filter the second hop to be only equal
TwoHopIndexPath = NamedTuple('TwoHopIndexPath', [('qnode', str), ('hop1', List[OneHopIndexPath]), ('hop2', List[OneHopIndexPath])])
TwoHopIndex = Dict[str, Dict[str, List[TwoHopIndexPath]]]


class TraversalOption(enum.Enum):
    NoConstraint = "no_constraint"
    TransitiveOnly = "transitive_only"


class KGObjectIndex:
    """An index in KG that help speed up searching for entities
    """
    def __init__(self, one_hop_index: OneHopIndex, two_hop_index: TwoHopIndex, n_hop: int, traversal_option: TraversalOption):
        self.one_hop_index = one_hop_index
        self.two_hop_index = two_hop_index
        self.n_hop = n_hop
        self.traversal_option = traversal_option

    @staticmethod
    def from_qnodes(index_qnode_ids: List[str], qnodes: Dict[str, QNode], wdprops: Dict[str, WDProperty],
                    n_hop: int = 1, traversal_option: TraversalOption = TraversalOption.TransitiveOnly, verbose: bool = False):
        # one hop index
        hop1_index: OneHopIndex = {}
        hop2_index: TwoHopIndex = {}

        assert 1 <= n_hop <= 2
        if n_hop > 1 and traversal_option == TraversalOption.TransitiveOnly:
            transitive_props = {pid for pid, p in wdprops.items() if p.is_transitive()}
        else:
            transitive_props = None

        if verbose:
            pbar = tqdm(total=len(index_qnode_ids) * (1 + int(n_hop > 1)), desc=f'build kg object index for {len(index_qnode_ids)} qnodes')
        else:
            pbar = M.FakeTQDM()

        for index_qnode_id in index_qnode_ids:
            qnode = qnodes[index_qnode_id]
            hop1_index[qnode.id] = dict(build_one_hop_index(qnode))
            pbar.update(1)

        if n_hop > 1:
            for index_qnode_id in index_qnode_ids:
                qnode_hop2_index = defaultdict(list)
                for uid, hop1_paths in hop1_index[index_qnode_id].items():
                    # TODO: ignore P31 and not discovering outside of it until to fix the method that discovers incoming&outgoing edges to include classes
                    # which currently has scalability issue since they have too many edges.
                    hop1_paths = [
                        hop1_path for hop1_path in hop1_paths
                        if hop1_path.relationship.prop != 'P31' and (transitive_props is None or hop1_path.relationship.prop in transitive_props)
                    ]
                    if len(hop1_paths) == 0:
                        continue
                    if transitive_props is not None:
                        hop1_props = {path.relationship.prop for path in hop1_paths}
                    
                    if uid not in qnodes:
                        # this can happen due to some of the qnodes is in the link, but is missing in the KG
                        # this is very rare so we can employ some check to make sure this is not due to
                        # our wikidata subset
                        is_error_in_kg = any(
                            any(_s.value.is_qnode() and _s.value.as_qnode_id() in qnodes for _s in _stmts)
                            for _p, _stmts in qnodes[index_qnode_id].props.items()
                        )
                        if not is_error_in_kg:
                            raise Exception(f"Missing qnodes in your KG subset: {uid}")
                        continue
                    
                    for vid, hop2_paths in build_one_hop_index(qnodes[uid], transitive_props).items():
                        hop2_paths = [
                            hop2_path for hop2_path in hop2_paths
                            if transitive_props is None or hop2_path.relationship.prop in hop1_props
                        ]
                        if len(hop2_paths) > 0:
                            qnode_hop2_index[vid].append(TwoHopIndexPath(uid, hop1_paths, hop2_paths))
                pbar.update(1)
                hop2_index[index_qnode_id] = dict(qnode_hop2_index)
        pbar.close()
        return KGObjectIndex(hop1_index, hop2_index, n_hop, traversal_option)

    @staticmethod
    def deserialize(infile: Union[Path, str], verbose: bool = False):
        start = time.time()
        index = M.deserialize_pkl(infile)
        if verbose:
            logger.info("Deserialize KG object index takes {} seconds", f"{time.time() - start:.3f}")
        return index

    def serialize(self, outfile: Union[Path, str]):
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
        M.serialize_pkl(self, outfile)

    def iter_hop1_props(self, source_qnode_id: str, target_qnode_id: str) -> Iterable[OneHopIndexPath]:
        return self.one_hop_index[source_qnode_id].get(target_qnode_id, [])

    def iter_hop2_props(self, source_qnode_id: str, target_qnode_id: str) -> Iterable[TwoHopIndexPath]:
        return self.two_hop_index[source_qnode_id].get(target_qnode_id, [])


def build_one_hop_index(qnode: QNode, filter_props: Set[str]=None):
    hop1_index = defaultdict(list)
    if filter_props is None:
        iter = qnode.props.items()
    else:
        iter = [(p, qnode.props.get(p, [])) for p in filter_props]

    for p, stmts in iter:
        for stmt_i, stmt in enumerate(stmts):
            lst = defaultdict(list)
            if stmt.value.is_qnode():
                target_qnode_id = stmt.value.as_qnode_id()
                lst[target_qnode_id].append(None)

            for q, qvals in stmt.qualifiers.items():
                for target_qnode_id in {qval.as_qnode_id() for qval in qvals if qval.is_qnode()}:
                    lst[target_qnode_id].append(q)

            for target_qnode_id, rels in lst.items():
                assert all(x is not None for x in rels[1:])
                # (p & q) is guarantee to be unique
                if rels[0] is not None or len(rels) > 1:
                    # we have qualifiers
                    qs = rels if rels[0] is not None else rels[1:]
                else:
                    qs = []
                rel = Relationship(p, qs, both=rels[0] is None and len(rels) > 1)
                hop1_index[target_qnode_id].append(OneHopIndexPath(rel, stmt_i))
    return hop1_index

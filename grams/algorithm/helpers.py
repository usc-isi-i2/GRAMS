from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, Set, List, Optional, Callable


@dataclass
class Tree:
    @dataclass
    class HierarchyRecord:
        id: str
        duplicated: bool
        depth: int

    id: str
    depth: int
    children: List['Tree']
    score: Optional[float] = None

    def get_flatten_hierarchy(self, dedup: bool = False) -> List[HierarchyRecord]:
        """Flatten the tree into a flatten hierarchy"""
        stack = [self]
        output = []
        seen_ids = set()
        while len(stack) > 0:
            node = stack.pop()
            dup = node.id in seen_ids
            output.append(Tree.HierarchyRecord(node.id, dup, node.depth))
            seen_ids.add(node.id)

            if not dedup or not dup:
                # do not travel the children
                for c in reversed(node.children):
                    stack.append(c)
        return output

    def sort(self, key=None, reverse: bool = False):
        self.children.sort(key=key or attrgetter('score'), reverse=reverse)
        for c in self.children:
            c.sort(key, reverse)

    def adjust_depth(self, depth: int):
        self.depth = depth
        for c in self.children:
            c.adjust_depth(depth + 1)
        return self

    def update_score(self, score_fn: Callable[['Tree'], float]):
        """Use to adjust score of a node in the tree based on its children. The score is going to be used for sorting"""
        for c in self.children:
            c.update_score(score_fn)
        self.score = score_fn(self)
        return self

    def preorder(self, fn: Callable[['Tree', List['Tree']], None], path: List['Tree'] = None):
        if path is None:
            path = []
        fn(self, path)
        path.append(self)
        for child in self.children:
            child.preorder(fn, path)
        path.pop()

    def clone(self):
        return Tree(self.id, self.depth, [c.clone() for c in self.children], self.score)


@dataclass
class Forest:
    trees: List[Tree]

    def get_flatten_hierarchy(self, dedup: bool = False) -> List[Tree.HierarchyRecord]:
        return Tree(None, -1, self.trees).get_flatten_hierarchy(dedup)[1:]

    def sort(self, key=None, reverse: bool = False):
        for tree in self.trees:
            tree.sort(key, reverse)
        self.trees.sort(key=key or attrgetter('score'), reverse=reverse)
        return self

    def update_score(self, score_fn: Callable[['Tree'], float]):
        for tree in self.trees:
            tree.update_score(score_fn)
        return self

    def preorder(self, fn: Callable[['Tree'], None]):
        for tree in self.trees:
            tree.preorder(fn)


class IndirectDictAccess:

    def __init__(self, odict, access):
        self.odict = odict
        self.access = access

    def __getitem__(self, item):
        return self.access(self.odict[item])

    def __contains__(self, item):
        return item in self.odict


def reorder2tree(lst: List[str], super_relationships: Dict[str, Set[str]]) -> Forest:
    """This function is very forgiving as it doesn't throw exception
    when the qnode is not in the super_relationship dictionary
    """
    if len(lst) == 0:
        return Forest([])

    roots = []
    graph: Dict[str, Tree] = {}

    # unknown class; we do have this case in some dumps (tag to wrong qnodes)
    lst = [u for u in lst if u in super_relationships]

    for i, u in enumerate(lst):
        parents = set()
        for j, v in enumerate(lst):
            if i == j:
                continue
            if v in super_relationships[u]:
                parents.add(v)

        if len(parents) > 1:
            # remove grand parents
            dparents = set()
            for v1 in parents:
                if all(v1 == v2 or v1 not in super_relationships[v2] for v2 in parents):
                    dparents.add(v1)
            parents = dparents

        if len(parents) == 0:
            roots.append(u)

        if u not in graph:
            graph[u] = Tree(u, -1, [])
        for v in parents:
            if v not in graph:
                graph[v] = Tree(v, -1, [])
            graph[v].children.append(graph[u])

    if len(roots) == 0:
        # we have cycle.. try to break it, or we can just throw exception
        # this is going to be very rare based on previous analysis (6 cycles found)
        raise Exception("Found cycles")

    # want to make sure that we have is not a graph
    visited = set()

    def is_acyclic(uid):
        if uid in visited:
            return False
        visited.add(uid)
        for c in graph[uid].children:
            if not is_acyclic(c.id):
                return False
        visited.remove(uid)
        return True

    for root in roots:
        if not is_acyclic(root):
            raise Exception("Found cycles")

    # clone so that we can get correct depth
    return Forest([graph[root].clone().adjust_depth(0) for root in roots])

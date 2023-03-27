from __future__ import annotations
from enum import Enum
from hugedict.misc import identity
import orjson
from pathlib import Path
from typing import Callable, Mapping, Union
from hugedict.sqlitedict import SqliteDict, SqliteDictFieldType
from kgdata.wikidata.models.wdclass import WDClass
from kgdata.wikidata.models.wdproperty import WDProperty
from sm.evaluation.hierarchy_scoring_fn import (
    INF_DISTANCE,
    HierarchyScoringFn,
    MAX_ANCESTOR_DISTANCE,
    MAX_DESCENDANT_DISTANCE,
)
from sm.namespaces.namespace import OutOfNamespace
from sm.namespaces.wikidata import WikidataNamespace


MAX_DISTANCE = max(MAX_ANCESTOR_DISTANCE, MAX_DESCENDANT_DISTANCE)


class ItemType(str, Enum):
    CLASS = "class"
    PROPERTY = "property"


class SqliteItemDistance(SqliteDict[str, int]):
    def __init__(
        self,
        path: Union[str, Path],
        items: Mapping[str, WDClass] | Mapping[str, WDProperty],
        item_type: ItemType,
    ):
        super().__init__(
            path,
            keytype=SqliteDictFieldType.str,
            ser_value=identity,
            deser_value=identity,
            valuetype=SqliteDictFieldType.int,
        )
        self._items = items
        self._cache_distance = {}
        if item_type == ItemType.CLASS:
            self.uri2id = WikidataNamespace.get_entity_id
        elif item_type == ItemType.PROPERTY:
            self.uri2id = WikidataNamespace.get_prop_id
        else:
            raise ValueError(f"Unknown item type: {item_type}")

        self.is_valid_id = WikidataNamespace.is_valid_id
        self.is_valid_uri = WikidataNamespace.create().is_uri_in_ns

    def get_distance(self, pred_item: str, target_item: str) -> int:
        if pred_item == target_item:
            return 0

        if (pred_item, target_item) not in self._cache_distance:
            if self.is_valid_uri(target_item):
                assert self.is_valid_uri(pred_item)
                pred_id = self.uri2id(pred_item)
                target_id = self.uri2id(target_item)
            else:
                assert self.is_valid_id(pred_item)
                assert self.is_valid_id(target_item)

                pred_id = pred_item
                target_id = target_item

            key = orjson.dumps([pred_id, target_id]).decode()
            if key not in self:
                distance = self._calculate_distance(pred_id, target_id)
                self[key] = distance
            self._cache_distance[pred_item, target_item] = self[key]
        return self._cache_distance[pred_item, target_item]

    def _calculate_distance(self, pred_id: str, target_id: str) -> int:
        targ_obj = self._items[target_id]

        if pred_id in targ_obj.ancestors:
            # predicted item is the ancestor of the target
            return self._calculate_distance_to_ancestors(targ_obj, pred_id)

        pred_obj = self._items[pred_id]
        if target_id in pred_obj.ancestors:
            # predicted item is the descendant of the target
            return -self._calculate_distance_to_ancestors(pred_obj, target_id)

        return INF_DISTANCE

    def _calculate_distance_to_ancestors(
        self, source: WDClass | WDProperty, target_id: str
    ) -> int:
        visited = {}
        stack: list[tuple[str, int]] = [(uid, 1) for uid in source.parents]
        while len(stack) > 0:
            uid, distance = stack.pop()
            if uid in visited and distance >= visited[uid]:
                # we have visited this node before and since last time we visit
                # the previous route is shorter, so we don't need to visit it again
                continue

            visited[uid] = distance
            u = self._items[uid]
            if target_id == uid or target_id not in u.ancestors:
                # this is a dead-end path, don't need to explore
                continue
            for parent_id in u.parents:
                if distance < MAX_DISTANCE:
                    # still within the distance limit, we can explore further
                    stack.append((parent_id, distance + 1))
        # we may have not visited the target node because its too far away (> MAX_DISTANCE).
        return visited.get(target_id, INF_DISTANCE)


def get_hierarchy_scoring_fn(
    path: str | Path,
    items: Mapping[str, WDClass] | Mapping[str, WDProperty],
    item_type: ItemType,
) -> HierarchyScoringFn:
    return HierarchyScoringFn(SqliteItemDistance(path, items, item_type))

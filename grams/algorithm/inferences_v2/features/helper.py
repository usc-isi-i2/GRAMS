from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeVar


MISSING_VALUE = -1
K = TypeVar("K")


@dataclass
class IDMap(Generic[K]):
    map: dict[K, int] = field(default_factory=dict)
    invert_map: list[K] = field(default_factory=list)

    def add(self, key: K) -> int:
        assert key not in self.map
        newkey = len(self.invert_map)
        self.map[key] = newkey
        self.invert_map.append(key)
        return newkey

    def m(self, key: K) -> int:
        """Get a new key from old key"""
        if key not in self.map:
            self.map[key] = len(self.invert_map)
            self.invert_map.append(key)
        return self.map[key]

    def im(self, new_key: int) -> K:
        """Get the old key from the new key"""
        return self.invert_map[new_key]

    def __contains__(self, key: K) -> bool:
        return key in self.map

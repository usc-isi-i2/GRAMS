import pickle
from typing import TypeVar, Dict, Union, List, Callable, Any, Optional

import redis, orjson
import rocksdb

from grams.misc.misc import identity_func
from grams.misc.fn_cache import EagerSingleFileCache

"""
Provide a in-memory database for storing huge dictionary. Mainly used for development.
"""

K = TypeVar('K')
V = TypeVar('V')


class RedisStore(Dict[K, V]):
    instance = None

    def __init__(self, url):
        self.url = url
        self.redis = redis.Redis.from_url(url)

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __contains__(self, item: str):
        return self.redis.exists(item) == 1

    def __getitem__(self, item: str):
        assert item is not None, item
        resp = self.redis.get(item)
        if resp is None:
            raise KeyError(item)
        return self.deserialize(resp)

    def __setitem__(self, key: str, value: Union[str, bytes]):
        self.redis.set(key, value)

    def __len__(self):
        return self.redis.dbsize()

    def cache_dict(self) -> 'CacheDictStore[K, V]':
        """Return a version of the store that will cache all of the query for faster processing"""
        return CacheDictStore(self)

    def cache_file(self, cache_id) -> 'CacheFileStore[K, V]':
        """Return a version of the store that will cache all of the query for faster processing"""
        return CacheFileStore(self, cache_id)

    def cache_redis(self, url: str, ns: Optional[str]) -> 'CacheRedisStore[K, V]':
        return CacheRedisStore(self, ns, url)

    def deserialize(self, value: str):
        return value


class PickleRedisStore(RedisStore[K, V]):

    def __setitem__(self, key: str, value: Any):
        value = pickle.dumps(value)
        self.redis.set(key, value)

    def deserialize(self, value: bytes):
        return pickle.loads(value)


class InMemStore(Dict[K, V]):
    def __init__(self, rdict: Dict[K, V], deserialize: Callable[[str], V] = None):
        self.rdict = rdict
        self.deserialize = deserialize or identity_func

    def __contains__(self, item):
        return item in self.rdict

    def __getitem__(self, item):
        return self.deserialize(self.rdict[item])

    def __setitem__(self, key: str, value: Union[str, bytes]):
        self.rdict[key] = value

    def values(self):
        return (self.deserialize(x) for x in self.rdict.values())

    def items(self):
        return ((k, self.deserialize(v)) for k, v in self.rdict.items())

    def keys(self):
        return self.rdict.keys()

    def __len__(self):
        return len(self.rdict)

    def cache_dict(self) -> 'CacheDictStore[K, V]':
        return CacheDictStore(self)

    def cache_redis(self, url: str, ns: Optional[str]) -> 'CacheRedisStore[K, V]':
        return CacheRedisStore(self, ns, url)

    def as_dict(self):
        return {k: self.deserialize(v) for k, v in self.rdict.items()}


class CacheDictStore(Dict[K, V]):
    def __init__(self, store: Dict[K, V]):
        self.store = store
        self.cache = {}

    def __contains__(self, item: str):
        return item in self.cache or item in self.store

    def __getitem__(self, item: str):
        if item not in self.cache:
            self.cache[item] = self.store[item]
        return self.cache[item]

    def __setitem__(self, key: str, value: Union[str, bytes]):
        raise Exception("NotSupportedFunction")

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def keys(self):
        return self.store.keys()

    def __len__(self):
        return len(self.store)


class CacheFileStore(Dict[K, V]):
    def __init__(self, store: Dict[K, V], cache_id: str):
        self.store = store
        self.cache = EagerSingleFileCache.get_instance(cache_id)

    def __contains__(self, item: str):
        return item in self.cache or item in self.store

    def __getitem__(self, item: str):
        if item not in self.cache:
            self.cache[item] = self.store[item]
        return self.cache[item]

    def __setitem__(self, key: str, value: Union[str, bytes]):
        raise Exception("NotSupportedFunction")

    def as_dict(self):
        """Turn this cache store into a dictionary. Use this function when all of the data has been cached,
        and you want to run the code in parallel
        """
        return self.cache.as_dict()


class CacheRedisStore(Dict[K, V]):
    def __init__(self, store: Dict[K, V], ns: Optional[str], url: str):
        self.store = store
        self.cache = PickleRedisStore(url)
        self.ns = ns
        if self.ns is not None:
            self._key = self._get_key
        else:
            self._key = identity_func

    def __contains__(self, item: str):
        return self._key(item) in self.cache or item in self.store

    def __getitem__(self, item: str):
        key2 = self._key(item)
        if key2 not in self.cache:
            self.cache[key2] = self.store[item]
        return self.cache[key2]

    def __setitem__(self, key: str, value: Union[str, bytes]):
        raise Exception("NotSupportedFunction")

    def _get_key(self, item):
        return self.ns + ":" + item

    def clear_cache(self):
        self.cache.redis.flushdb()


class RocksDBStore(Dict[K, V]):
    def __init__(self, inpath: str):
        self.db = rocksdb.DB(str(inpath), rocksdb.Options(create_if_missing=True))

    def __contains__(self, key):
        return self.db.get(key.encode()) is not None

    def __getitem__(self, key):
        item = self.db.get(key.encode())
        if item is None:
            raise KeyError(key)
        return self.deserialize(item)

    def __setitem__(self, key, value):
        self.db.put(key.encode(), value.encode())

    def __len__(self):
        assert False, "Does not support this operator"

    def get(self, key: str, default=None):
        item = self.db.get(key.encode())
        if item is None:
            return None
        return self.deserialize(item)

    def cache_dict(self) -> 'CacheDictStore[K, V]':
        return CacheDictStore(self)

    def deserialize(self, value):
        return value


class JSONRocksDBStore(RocksDBStore[str, dict]):
    def __setitem__(self, key, value):
        self.db.put(key.encode(), orjson.dumps(value))

    def deserialize(self, value):
        return orjson.loads(value)
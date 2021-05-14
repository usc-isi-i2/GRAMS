import functools
import logging
import os, glob, orjson, pickle
import atexit
from hashlib import md5
from pathlib import Path

from typing import List, Dict, Tuple, Callable, Any, Optional

from slugify import slugify

from grams.misc.read_file import get_open_fn
from grams.misc.deser import deserialize_pkl, serialize_pkl


class BasicFileCache:
    logger = logging.getLogger("file_cache.basic")
    instances = {}

    def __init__(self, outfile_prefix: str):
        self.outfile_prefix = outfile_prefix
        # store max `self.max_chunk_size` calls per file
        self.max_chunk_size = 100000
        self.cache_chunks = []
        self.has_init = False

    def lazy_init(self):
        assert not self.has_init
        self.logger.debug("Initialize cache...")
        # load old cache
        for infile in sorted(glob.glob(self.outfile_prefix + ".*")):
            # TODO: handle corrupted cache
            self.cache_chunks.append(deserialize_pkl(infile))

        # if the last one is full, we need to create new chunk
        if len(self.cache_chunks) == 0 or len(self.cache_chunks[-1]) >= self.max_chunk_size:
            self.cache_chunks.append({})
        self.last_save = len(self.cache_chunks) - 1
        self.has_init = True
        self.logger.debug("Initialize cache... Finish!")

    @staticmethod
    def get_instance(id: str):
        if id not in BasicFileCache.instances:
            # force users to set up the CACHE_DIR if they want to use it
            # and only if the id is not starts with /
            BasicFileCache.instances[id] = BasicFileCache(id)

        return BasicFileCache.instances[id]

    def flush(self):
        if not self.has_init:
            # only flush when the cache has been inited
            return
        # flush all the changes to disk when we terminate
        for i in range(self.last_save, len(self.cache_chunks)):
            serialize_pkl(self.cache_chunks[i], self.outfile_prefix + f".{i:03}")
        self.last_save = len(self.cache_chunks) - 1

    def get(self, args: Tuple[Any, ...]):
        if not self.has_init:
            self.lazy_init()
        return next(chunk[args] for chunk in self.cache_chunks if args in chunk)

    def set(self, args: Tuple[Any, ...], val: Any):
        if not self.has_init:
            self.lazy_init()
        self.cache_chunks[-1][args] = val
        if len(self.cache_chunks[-1]) >= self.max_chunk_size:
            self.cache_chunks.append({})

    def has(self, args: Tuple[Any, ...]):
        if not self.has_init:
            self.lazy_init()
        return any(args in chunk for chunk in self.cache_chunks)

    @staticmethod
    def cache_func(id: str):
        """
        Cache a function, however, require that the arguments need hashable
        """
        cache = BasicFileCache.get_instance(id)
        def wrapper_fn(func):
            @functools.wraps(func)
            def fn(*args, **kwargs):
                if len(kwargs) == 0:
                    key_args = args
                else:
                    key_args = tuple(list(args) + sorted(kwargs.items(), key=lambda x: x[0]))

                if not cache.has(key_args):
                    cache.set(key_args, func(*args, **kwargs))
                return cache.get(key_args)

            return fn

        return wrapper_fn


class EagerSingleFileCache:
    logger = logging.getLogger("file_cache.eager")
    instances = {}

    def __init__(self, outfile: str):
        self.outfile = outfile
        self.cache = {}
        self.open_fn = get_open_fn(self.outfile)
        self.has_init = False

    def lazy_init(self):
        assert not self.has_init
        # load old cache
        if os.path.exists(self.outfile):
            self.logger.debug("Initialize cache...")
            with get_open_fn(self.outfile)(self.outfile, "rb") as f:
                reader = pickle.Unpickler(f)
                while True:
                    try:
                        (key, value) = reader.load()
                        self.cache[key] = value
                    except EOFError:
                        break
            self.logger.debug("Initialize cache... Finish!")
        self.has_init = True

    @staticmethod
    def get_instance(id: str):
        if id not in EagerSingleFileCache.instances:
            # force users to set up the CACHE_DIR if they want to use it
            # and only if the id is not starts with /
            if id[0] != "/":
                if not os.path.exists(CACHE_DIR):
                    os.mkdir(CACHE_DIR)
                EagerSingleFileCache.instances[id] = EagerSingleFileCache(os.path.join(CACHE_DIR, id))
            else:
                EagerSingleFileCache.instances[id] = EagerSingleFileCache(id)

        return EagerSingleFileCache.instances[id]

    def get(self, key):
        if not self.has_init:
            self.lazy_init()
        return self.cache[key]
    
    def has(self, key):
        if not self.has_init:
            self.lazy_init()
        return key in self.cache

    def set(self, key, value):
        if not self.has_init:
            self.lazy_init()
        self.cache[key] = value
        with self.open_fn(self.outfile, "ab") as f:
            writer = pickle.Pickler(f)
            writer.dump((key, value))

    def __getitem__(self, item):
        if not self.has_init:
            self.lazy_init()
        return self.cache[item]
    
    def __setitem__(self, item, value):
        if not self.has_init:
            self.lazy_init()
        self.set(item, value)
    
    def __contains__(self, item):
        if not self.has_init:
            self.lazy_init()
        return item in self.cache

    def as_dict(self):
        """Turn this cache into a dictionary

        Returns
        -------
        """
        if not self.has_init:
            self.lazy_init()
        return self.cache

    @staticmethod
    def cache_func(id: str):
        """
        Cache a function, however, require that the arguments need hashable and pass by positions only (i.e., only *args, no **kwargs)
        """
        cache = EagerSingleFileCache.get_instance(id)
        def wrapper_fn(func):
            @functools.wraps(func)
            def fn(*args):
                if not cache.has(args):
                    cache.set(args, func(*args))
                return cache.get(args)

            return fn

        return wrapper_fn


def redis_cache_func(url: str = "redis://localhost:6379", namespace: str = "", instance_method: bool = False):
    """
    - Cache a function, however, require that the arguments need hashable and pass by positions only (i.e., only *args, no **kwargs)
    - Allow an option to specify the namespace in case we have many function to cache.
    - Support caching instance method. However, it cannot observe all parameters that hidden in an instance
        so use it with caution
    """
    from grams.misc.map import PickleRedisStore
    store = PickleRedisStore(url)

    if instance_method:
        def wrapper_instance_fn(func):
            @functools.wraps(func)
            def fn(*args):
                key = namespace + ":" + orjson.dumps(args[1:]).decode()
                if key not in store:
                    store[key] = func(*args)
                return store[key]
            return fn

        return wrapper_instance_fn

    def wrapper_fn(func):
        @functools.wraps(func)
        def fn(*args):
            key = namespace + ":" + orjson.dumps(args).decode()
            if key not in store:
                store[key] = func(*args)
            return store[key]
        return fn

    return wrapper_fn


def individual_file_cache(cache_dir: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    def wrapper_fn(func):
        @functools.wraps(func)
        def fn(*args):
            key = orjson.dumps(args)
            key = slugify(key) + "_" + md5(key).hexdigest()
            outfile = cache_dir / f"{key}.pkl"
            if not outfile.exists():
                resp = func(*args)
                serialize_pkl(resp, outfile)
                return resp
            return deserialize_pkl(outfile)
        return fn

    return wrapper_fn


@atexit.register
def basic_file_cache_deconstruct():
    for cache in BasicFileCache.instances.values():
        cache.flush()

from pathlib import Path

from sm.misc import RocksDBStore
from typing import Optional, Union


class Wikipedia2WikidataDB(RocksDBStore[str, str]):
    instance = None

    @staticmethod
    def get_instance(dbfile: Optional[Union[Path, str]] = None, read_only: bool = False):
        if Wikipedia2WikidataDB.instance is None:
            Wikipedia2WikidataDB.instance = Wikipedia2WikidataDB(dbfile, read_only=read_only)
        return Wikipedia2WikidataDB.instance

    def deserialize(self, value):
        return value.decode()



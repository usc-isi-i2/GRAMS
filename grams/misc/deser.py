import csv
import gzip
import pickle
import orjson
import json
from pathlib import Path
from typing import Union, List, Sequence, Optional

import ujson
from ruamel.yaml import YAML

from grams.misc.read_file import get_open_fn


def serialize_pkl(object, fpath: Union[str, Path]):
    """Serialize an object to a file"""
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        pickle.dump(object, f)


def deserialize_pkl(fpath: Union[str, Path]):
    """Deserialize an object from a file"""
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        return pickle.load(f)


def serialize_jl(objects, fpath: Union[str, Path]):
    """Serialize a list of objects to a file in JSON line format"""
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        for obj in objects:
            f.write(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_DATACLASS))
            f.write(b"\n")


def deserialize_jl(fpath: Union[str, Path], n_lines: Optional[int]=None):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        if n_lines is None:
            return [orjson.loads(line) for line in f]
        lst = []
        for line in f:
            lst.append(orjson.loads(line))
            if len(lst) >= n_lines:
                break
        return lst


def serialize_json(
    obj, fpath: Union[str, Path], use_pyjson: bool = False, indent: Optional[int] = None
):
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        if use_pyjson:
            f.write(json.dumps(obj, indent=indent).encode())
        else:
            if indent is not None:
                f.write(
                    ujson.dumps(
                        obj, indent=indent, escape_forward_slashes=False
                    ).encode()
                )
            else:
                f.write(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_DATACLASS))


def deserialize_json(fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        return orjson.loads(f.read())


def serialize_byte_lines(objects: Sequence[bytes], fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        for obj in objects:
            f.write(obj)
            f.write(b"\n")


def serialize_lines(objects: Sequence[str], fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        for obj in objects:
            f.write(obj.encode())
            f.write(b"\n")


def deserialize_lines(fpath: Union[str, Path], n_lines: Optional[int]=None, trim: bool=False):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        if n_lines is None:
            if trim:
                return [line.decode().strip() for line in f]    
            return [line.decode() for line in f]

        lst = []
        if trim:
            for line in f:
                lst.append(line.decode().strip())
                if len(lst) >= n_lines:
                    break
        else:
            for line in f:
                lst.append(line.decode())
                if len(lst) >= n_lines:
                    break
        return lst


def deserialize_key_val_lines(fpath: Union[str, Path], delimiter="\t"):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        return [line.decode().split(delimiter, 1) for line in f]


def deserialize_key_json_val_lines(fpath: Union[str, Path], delimiter="\t"):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        results = []
        for line in f:
            k, v = line.decode().split(delimiter, 1)
            results.append((k, orjson.loads(v)))
        return results


def serialize_yml(objects, fpath: Union[str, Path]):
    with get_open_fn(fpath)(fpath, "wb") as f:
        yaml = YAML()
        yaml.dump(objects, f)


def deserialize_yml(fpath: Union[str, Path]):
    with get_open_fn(fpath)(fpath, "rb") as f:
        yaml = YAML()
        return yaml.load(f)


def serialize_csv(rows, fpath: Union[str, Path], mode="w", delimiter=","):
    with open(fpath, mode, newline="") as f:
        writer = csv.writer(
            f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        for row in rows:
            writer.writerow(row)


def deserialize_csv(fpath: Union[str, Path], delimiter=","):
    with get_open_fn(fpath)(fpath, "rt") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return [row for row in reader]


def serialize_text(text: str, fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        f.write(text.encode())


def serialize_bytes(text: bytes, fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "wb") as f:
        f.write(text)


def deserialize_text(fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        return f.read().decode()


def deserialize_bytes(fpath: Union[str, Path]):
    with get_open_fn(str(fpath))(str(fpath), "rb") as f:
        return f.read()
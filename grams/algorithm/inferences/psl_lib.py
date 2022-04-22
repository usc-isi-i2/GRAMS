from __future__ import annotations
import re, os, uuid, logging
from typing import (
    Dict,
    Generic,
    List,
    MutableMapping,
    Optional,
    TypeVar,
)
from datetime import datetime
from numpy import ndarray
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pslpython.model import Model, ModelError
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from copy import copy
from functools import partial


class PSLModel:
    def __init__(
        self,
        predicates: List[Predicate],
        rules: RuleContainer,
        temp_dir: Optional[str] = None,
        ignore_predicates_not_in_rules: bool = False,
    ):
        self.model = Model("psl-model")
        self.rules = rules
        self.name2predicate: Dict[str, Predicate] = {}
        self.temp_dir = (
            temp_dir
            or f"/tmp/psl-python-{datetime.now().strftime('%y%m%d-%H%M%S')}-{str(uuid.uuid4()).replace('-', '')}"
        )

        if ignore_predicates_not_in_rules:
            using_predicates = []
            for p in predicates:
                pname = p.name() + "("
                if any(
                    r._rule_body.find(pname) != -1
                    for r in rules.values()
                    if r._rule_body is not None
                ):
                    using_predicates.append(p)
                else:
                    logger.debug(
                        "Ignoring predicate {} as it can't be found in any rule",
                        p.name(),
                    )
            predicates = using_predicates

        for p in predicates:
            self.model.add_predicate(p)
            self.name2predicate[p.name()] = p

        for r in rules.values():
            self.model.add_rule(r)

        # setup custom logger
        self.logs = []
        self.logger = logging.getLogger(name="psl-model")
        self.logger.addHandler(InMemHandler(self.logs))

    def parameters(self) -> Dict[str, float]:
        """Return parameters of the model"""
        params = {}
        for name, rule in self.rules.items():
            params[name] = rule.weight()
        return params

    def set_parameters(self, params: Dict[str, float]):
        """Set parameters of the model"""
        for name, rule in self.rules.items():
            rule.set_weight(params[name])

    def set_data(
        self,
        observations: Dict[str, list],
        targets: Dict[str, list],
        truth: Dict[str, list],
        force_setall: bool = True,
    ):
        """Set data of the current model"""
        pnames = set(observations.keys()).union(targets, truth)

        for pname in pnames:
            p = self.name2predicate[pname]
            p.clear_data()

            for partition, partition_data in [
                (Partition.OBSERVATIONS, observations),
                (Partition.TARGETS, targets),
                (Partition.TRUTH, truth),
            ]:
                if pname in partition_data and len(partition_data[pname]) > 0:
                    p.add_data(partition, partition_data[pname])

        if force_setall and len(self.name2predicate) != len(pnames):
            raise Exception(
                "Missing predicates: "
                + ", ".join(list(set(self.name2predicate.keys()).difference(pnames)))
            )

    def predict(
        self,
        observations: Dict[str, list],
        targets: Dict[str, list],
        truth: Optional[Dict[str, list]] = None,
        force_setall: bool = True,
    ):
        try:
            self.set_data(observations, targets, truth or {}, force_setall)
            resp = self.model.infer(
                logger=self.logger,
                additional_cli_optons=["--h2path", os.path.join(self.temp_dir, "h2")],
                temp_dir=self.temp_dir,
                # cleanup_temp=False,
            )
        except ModelError as e:
            self.log_errors(e)
            raise

        output = {}
        for p, df in resp.items():
            psize = len(p._types)
            output[p.name()] = {
                tuple((r[i] for i in range(psize))): r["truth"]
                for _, r in df.iterrows()
            }
        return output

    def debug(self, idmap: Optional[IDMap] = None):
        """Write data of predicates into a file for debugging"""
        outdir = Path(self.temp_dir) / "debug"
        outdir.mkdir(exist_ok=True, parents=True)

        def remap(row: ndarray, size: int, idmap: IDMap):
            row = copy(row)
            for i in range(size):
                row[i] = idmap.im(row[i])
            return row

        for pname, p in self.name2predicate.items():
            psize = len(p._types)
            for partition, df in p.data().items():
                if df.size == 0:
                    continue

                if idmap is not None:
                    df = df.apply(
                        partial(remap, size=psize, idmap=idmap), axis=1, raw=True
                    )
                df.to_csv(outdir / f"{pname}.{partition}.csv", index=False)

    def log_errors(self, error: Exception):
        logger.error(str(error))
        print("\n".join(self.logs))


class InMemHandler(logging.Handler):
    def __init__(self, storage: list):
        logging.Handler.__init__(self)
        self.storage = storage

    def emit(self, record):
        self.storage.append(self.format(record))


K = TypeVar("K")


class IDMap(Generic[K]):
    def __init__(self, counter: int = 0):
        self.counter = counter
        self.map: Dict[K, str] = {}
        self.invert_map: Dict[str, K] = {}

    def m(self, key: K) -> str:
        """Get a new key from old key"""
        if key not in self.map:
            new_key = f"i-{self.counter}"
            self.map[key] = new_key
            self.invert_map[new_key] = key
            self.counter += 1
        return self.map[key]

    def im(self, new_key: str) -> K:
        """Get the old key from the new key"""
        return self.invert_map[new_key]

    def to_readable(self) -> dict[str, str]:
        """Transform the IDMap so that the keys reserve the original values as much as possible

        Returns:
            dict: A dictionary mapping the original keys to the new keys so that you can
                  update objects that has been mapped before
        """
        map, invert_map = {}, {}
        trans = {}
        for v, k in self.map.items():
            newk = "n-" + re.sub(r"""[:"'{}+=]""", "-", str(v))
            assert newk not in map
            map[v] = newk
            invert_map[newk] = v
            trans[k] = newk

        self.map = map
        self.invert_map = invert_map

        return trans


class RuleContainer(MutableMapping[str, Rule]):
    def __init__(self, rules: Optional[Dict[str, Rule]] = None):
        self.rules: Dict[str, Rule] = rules if rules is not None else {}

    def __len__(self):
        return len(self.rules)

    def __iter__(self):
        return self.rules.__iter__()

    def items(self):
        return self.rules.items()

    def keys(self):
        return self.rules.keys()

    def values(self):
        return self.rules.values()

    def __contains__(self, key: str) -> bool:
        return key in self.rules

    def __setitem__(self, key: str, value: Rule):
        assert key not in self.rules
        self.rules[key] = value

    def __getitem__(self, key: str) -> Rule:
        return self.rules[key]

    def __delitem__(self, __v: str) -> None:
        raise Exception("Not support operation")

    def print(self):
        """Print list of rules"""
        print("=" * 10, "Rules")
        for name, rule in self.rules.items():
            print(f"{name}: {rule._rule_body}")
        print("=" * 10)

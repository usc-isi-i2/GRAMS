from __future__ import annotations
import re, os, uuid, logging
import shutil
from typing import (
    Any,
    Dict,
    Generic,
    List,
    MutableMapping,
    Optional,
    Set,
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
from operator import itemgetter


K = TypeVar("K")


class OverrideModel(Model):
    def _write_data(self, temp_dir):
        super()._write_data(temp_dir)


class PSLModel:
    def __init__(
        self,
        predicates: List[Predicate],
        rules: RuleContainer,
        temp_dir: Optional[str] = None,
        ignore_predicates_not_in_rules: bool = False,
        required_predicates: Optional[Set[str]] = None,
    ):
        self.model = Model("psl-model")
        self.rules = rules
        self.name2predicate: Dict[str, Predicate] = {}

        if temp_dir is not None:
            self.temp_dir = temp_dir
        else:
            Path("/tmp/pslpython").mkdir(exist_ok=True, parents=True)
            self.temp_dir = f"/tmp/pslpython/r{datetime.now().strftime('%y%m%d-%H%M%S')}-{str(uuid.uuid4()).replace('-', '')}"
        assert not Path(self.temp_dir).exists(), f"{self.temp_dir} already exists"

        if ignore_predicates_not_in_rules:
            if required_predicates is None:
                required_predicates = set()
            using_predicates = []
            for p in predicates:
                if p.name() in required_predicates:
                    using_predicates.append(p)
                    continue

                pname = p.name() + "("
                subrules = [
                    r
                    for r in rules.values()
                    if r._rule_body is not None and r._rule_body.find(pname) != -1
                ]
                ignore = True

                if len(subrules) > 0:
                    for r in subrules:
                        body: str = r._rule_body  # type: ignore
                        start = 0
                        while (idx := body.find(pname, start)) != -1:
                            if idx == 0:
                                ignore = False
                                break
                            if re.match("[a-zA-Z0-9_]", body[idx - 1]) is None:
                                ignore = False
                                break
                            start = idx + 1
                        if not ignore:
                            break

                if not ignore:
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
            if rule.weighted():
                rule.set_weight(params[name])

    def extend_rules(self, rules: RuleContainer, predicates: List[Predicate]):
        """Extend the model with new rules and predicates. This is usually used
        to add rules that were pre-grounded.

        Note: this function must be called before `set_data` or `predict`.
        """
        self.rules.update(rules)
        for r in rules.values():
            self.model.add_rule(r)

        for p in predicates:
            if p.name() not in self.name2predicate:
                self.model.add_predicate(p)
                self.name2predicate[p.name()] = p

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
        cleanup_tempdir: bool = True,
        retry: bool = True,
        provenance: str = "",
    ):
        self.set_data(observations, targets, truth or {}, force_setall)

        for _ in range(3):
            try:
                resp = self.model.infer(
                    logger=self.logger,
                    additional_cli_optons=[
                        "--h2path",
                        os.path.join(self.temp_dir, "h2"),
                    ],
                    temp_dir=self.temp_dir,
                    cleanup_temp=False,
                )
                break
            except ModelError as e:
                # error related to parser (not jvm), no retry
                # non_recoverable_error = any(
                #     line.find(k) != -1
                #     for line in self.logs
                #     for k in [
                #         "org.linqs.psl.parser.antlr.PSLParser",
                #         "Unique index or primary key violation",
                #     ]
                # )
                non_recoverable_error = False
                self.log_errors(e)
                if not retry or non_recoverable_error:
                    raise

            logger.info("Retry PSL inference...")
        else:
            raise ModelError("PSL inference failed")

        if cleanup_tempdir:
            shutil.rmtree(self.temp_dir)

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

    def log_errors(self, error: Exception, clear: bool = True):
        logger.error(str(error))
        print("\n".join(self.logs))
        if clear:
            self.logs = []

    @staticmethod
    def normalize_probs(
        target_values: Dict[K, float], eps: float = 0.001, threshold: float = 0.0
    ) -> Dict[K, float]:
        """The predicted probabilities can be noisy due to numerical optimizations, i.e., equal variables
        can have slightly different scores. This function groups values that are close
        within the range [-eps, +eps] together, and replace them with the average value
        to reduce the noises.

        Args:
            target_values: probability of grounded values of a predicate
            eps: group values within [-eps, +eps] together
            threshold: remove values with probability less than threshold

        Return:
            A dictionary of normalized probabilities >= threshold
        """
        if eps == 0.0 and threshold == 0.0:
            return target_values

        norm_probs = {}
        lst = sorted(
            [x for x in target_values.items() if x[1] >= threshold], key=itemgetter(1)
        )

        if len(lst) == 0:
            return {}

        clusters = []
        pivot = 1
        clusters = [[lst[0]]]
        while pivot < len(lst):
            x = lst[pivot - 1][1]
            y = lst[pivot][1]
            if (y - x) <= eps:
                # same clusters
                clusters[-1].append(lst[pivot])
            else:
                # different clusters
                clusters.append([lst[pivot]])
            pivot += 1
        for cluster in clusters:
            avg_prob = sum([x[1] for x in cluster]) / len(cluster)
            for k, _prob in cluster:
                norm_probs[k] = avg_prob
        return norm_probs


class InMemHandler(logging.Handler):
    def __init__(self, storage: list):
        logging.Handler.__init__(self)
        self.storage = storage

    def emit(self, record):
        self.storage.append(self.format(record))


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


class ReadableIDMap(IDMap[K]):
    def __init__(self):
        self.map: Dict[K, str] = {}
        self.invert_map: Dict[str, K] = {}

    def m(self, key: K) -> str:
        """Get a new key from old key"""
        if key not in self.map:
            new_key = "n-" + re.sub(r"""[:"'{}+=]""", "-", str(key))
            self.map[key] = new_key
            self.invert_map[new_key] = key
        return self.map[key]

    def im(self, new_key: str) -> K:
        """Get the old key from the new key"""
        return self.invert_map[new_key]


class IdentityIDMap(IDMap[str]):
    def m(self, key: str) -> str:
        return key

    def im(self, new_key: str) -> str:
        return new_key


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
        assert key not in self.rules, key
        self.rules[key] = value

    def __getitem__(self, key: str) -> Rule:
        return self.rules[key]

    def __delitem__(self, key: str) -> None:
        self.rules.pop(key)

    def print(self):
        """Print list of rules"""
        print("=" * 10, "Rules")
        for name, rule in self.rules.items():
            print(f"{name}: {rule._rule_body}")
        print("=" * 10)

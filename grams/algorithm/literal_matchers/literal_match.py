from typing import List, Mapping, Tuple, Callable, Dict, Union
from grams.algorithm.literal_matchers.text_parser import (
    ParsedTextRepr,
)
import grams.core.literal_matchers as gcore_matcher
from kgdata.wikidata.models import WDValue, WDValueType, WDEntity
from grams.algorithm.literal_matchers.string_match import (
    string_exact_test,
    monolingual_exact_test,
)
from grams.algorithm.literal_matchers.number_match import quantity_test
from grams.algorithm.literal_matchers.coordinate_match import globecoordinate_test
from grams.algorithm.literal_matchers.datetime_match import time_test
from grams.algorithm.literal_matchers.entity_match import entity_similarity_test
from grams.algorithm.literal_matchers.types import LiteralMatchKit
from sm.misc.funcs import import_func
from dataclasses import dataclass, field

MatchFunc = Callable[[WDValue, ParsedTextRepr, LiteralMatchKit], Tuple[bool, float]]


@dataclass
class LiteralMatchConfigs:
    """Configuration for literal matcher functions"""

    STRING: str = field(
        default=".string_exact_test", metadata={"help": "list of functions "}
    )
    QUANTITY: str = field(default=".quantity_test", metadata={"help": ""})
    GLOBECOORDINATE: str = field(default=".globecoordinate_test", metadata={"help": ""})
    TIME: str = field(default=".time_test", metadata={"help": ""})
    MONOLINGUAL_TEXT: str = field(
        default=".monolingual_exact_test", metadata={"help": ""}
    )
    ENTITY: str = field(default="", metadata={"help": ""})

    def to_rust(self) -> gcore_matcher.LiteralMatcherConfig:
        def py2ru(ident):
            if ident == "":
                return "always_fail_test"
            assert ident.startswith("."), ident
            return ident[1:]

        return gcore_matcher.LiteralMatcherConfig(
            py2ru(self.STRING),
            py2ru(self.QUANTITY),
            py2ru(self.GLOBECOORDINATE),
            py2ru(self.TIME),
            py2ru(self.MONOLINGUAL_TEXT),
            py2ru(self.ENTITY),
        )


class LiteralMatch:
    def __init__(self, wdentities: Mapping[str, WDEntity], cfg: LiteralMatchConfigs):
        self.match_kit = LiteralMatchKit(wdentities)
        self.cfg = cfg

        self.type2func: Dict[
            WDValueType,
            List[MatchFunc],
        ] = {}

        self.type2func["string"] = self.cfg2funcs(cfg.STRING)
        self.type2func["quantity"] = self.cfg2funcs(cfg.QUANTITY)
        self.type2func["globecoordinate"] = self.cfg2funcs(cfg.GLOBECOORDINATE)
        self.type2func["time"] = self.cfg2funcs(cfg.TIME)
        self.type2func["monolingualtext"] = self.cfg2funcs(cfg.MONOLINGUAL_TEXT)
        self.type2func["wikibase-entityid"] = self.cfg2funcs(cfg.ENTITY)

    def match(
        self, pval: WDValue, val: ParsedTextRepr, skip_unmatch: bool = True
    ) -> List[Tuple[MatchFunc, Tuple[bool, float]]]:
        if skip_unmatch:
            return [
                (func, m)
                for func in self.type2func[pval.type]
                if (m := func(pval, val, self.match_kit))[0]
            ]
        return [
            (func, func(pval, val, self.match_kit))
            for func in self.type2func[pval.type]
        ]

    @staticmethod
    def cfg2funcs(cfg: str) -> List[MatchFunc]:
        funcs = []
        if len(cfg) == 0:
            return funcs
        for path in cfg.split(","):
            if path[0] == ".":
                # relative import
                funcs.append(
                    {
                        "string_exact_test": string_exact_test,
                        "monolingual_exact_test": monolingual_exact_test,
                        "quantity_test": quantity_test,
                        "globecoordinate_test": globecoordinate_test,
                        "time_test": time_test,
                        "entity_similarity_test": entity_similarity_test,
                    }[path[1:]]
                )
            else:
                funcs.append(import_func(path))
        return funcs

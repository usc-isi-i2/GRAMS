from typing import List, Mapping, Tuple, Callable, Dict, Union
from grams.algorithm.literal_matchers.text_parser import (
    ParsedTextRepr,
)
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


MatchFunc = Callable[[WDValue, ParsedTextRepr, LiteralMatchKit], Tuple[bool, float]]


class LiteralMatch:
    STRING = ".string_exact_test"
    QUANTITY = ".quantity_test"
    GLOBECOORDINATE = ".globecoordinate_test"
    TIME = ".time_test"
    MONOLINGUAL_TEXT = ".monolingual_exact_test"
    ENTITY = ""

    def __init__(self, wdentities: Mapping[str, WDEntity]):
        self.match_kit = LiteralMatchKit(wdentities)

        self.type2func: Dict[
            WDValueType,
            List[MatchFunc],
        ] = {}

        self.type2func["string"] = self.cfg2funcs(LiteralMatch.STRING)
        self.type2func["quantity"] = self.cfg2funcs(LiteralMatch.QUANTITY)
        self.type2func["globecoordinate"] = self.cfg2funcs(LiteralMatch.GLOBECOORDINATE)
        self.type2func["time"] = self.cfg2funcs(LiteralMatch.TIME)
        self.type2func["monolingualtext"] = self.cfg2funcs(
            LiteralMatch.MONOLINGUAL_TEXT
        )
        self.type2func["wikibase-entityid"] = self.cfg2funcs(LiteralMatch.ENTITY)

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

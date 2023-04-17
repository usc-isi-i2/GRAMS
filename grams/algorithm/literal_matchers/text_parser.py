from __future__ import annotations
import re
from numparser.fsm.parser import DEFAULT_NOT_UNITS
from dataclasses import dataclass, field
from datetime import datetime, MINYEAR
from typing import Optional
import grams.core.literal_matchers as gcore_literal_matchers
import ftfy
from dateutil.parser import parse as dt_parse, ParserError
from numparser import ParsedNumber, FSMParser


@dataclass
class TextParserConfigs:
    """Configurations for the parser"""

    NUM_PARSER: str = field(
        default="grams.algorithm.literal_matchers.text_parser.BasicNumberParser",
        metadata={"help": "number parser"},
    )
    DATETIME_PARSER: str = field(
        default="grams.algorithm.literal_matchers.text_parser.BasicDatetimeParser",
        metadata={"help": "datetime parser"},
    )


@dataclass
class ParsedTextRepr:
    origin: str
    normed_string: str
    number: Optional[ParsedNumber]
    datetime: Optional[ParsedDatetimeRepr]

    def to_rust(self) -> gcore_literal_matchers.ParsedTextRepr:
        return gcore_literal_matchers.ParsedTextRepr(
            self.origin,
            self.normed_string,
            None
            if self.number is None
            else gcore_literal_matchers.ParsedNumberRepr(
                self.number.number,
                self.number.number_string,
                isinstance(self.number.number, int),
                self.number.unit,
                self.number.prob,
            ),
            None if self.datetime is None else self.datetime.to_rust(),
        )


@dataclass
class ParsedDatetimeRepr:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def has_only_year(self):
        return (
            self.year is not None
            and self.month is None
            and self.day is None
            and self.hour is None
            and self.minute is None
            and self.second is None
        )

    def first_day_of_year(self):
        return (
            self.year is not None
            and self.month == 1
            and self.day == 1
            and self.hour is None
            and self.minute is None
            and self.second is None
        )

    def to_rust(self) -> gcore_literal_matchers.ParsedDatetimeRepr:
        return gcore_literal_matchers.ParsedDatetimeRepr(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )


class TextParser:
    def __init__(self, cfg: TextParserConfigs):
        self.cache = {}
        self.cfg = cfg

        if (
            cfg.DATETIME_PARSER
            == "grams.algorithm.literal_matchers.text_parser.BasicDatetimeParser"
        ):
            self.dt_parser = BasicDatetimeParser()
        else:
            raise NotImplementedError()

        if (
            cfg.NUM_PARSER
            == "grams.algorithm.literal_matchers.text_parser.BasicNumberParser"
        ):
            self.number_parser = BasicNumberParser()
        else:
            raise NotImplementedError()

        self.number_chars = re.compile(r"[^0-9\.+-]")

    @staticmethod
    def default():
        return TextParser(TextParserConfigs())

    def parse(self, text: str) -> ParsedTextRepr:
        if text not in self.cache:
            normed_text = self._norm_string(text)
            dt = self.dt_parser.parse(text, normed_text)
            number = self.number_parser.parse(text, normed_text)

            self.cache[text] = ParsedTextRepr(
                origin=text,
                normed_string=normed_text,
                number=number,
                datetime=dt,
            )
        return self.cache[text]

    def _norm_string(self, text: str):
        return ftfy.fix_text(text).replace("\xa0", " ").strip()


class BasicNumberParser:
    parser = FSMParser(not_units=DEFAULT_NOT_UNITS.union(["st", "rd", "nd", "th"]))

    def parse(self, text: str, normed_text: str) -> Optional[ParsedNumber]:
        lst = self.parser.parse_value(normed_text)
        if len(lst) == 0:
            return None
        else:
            return lst[0]


class BasicDatetimeParser:
    def __init__(self) -> None:
        self.default_dt = datetime(MINYEAR, 1, 1)
        self.default_dt2 = datetime(MINYEAR + 3, 2, 28)

    def parse(self, text: str, normed_text: str) -> Optional[ParsedDatetimeRepr]:
        try:
            dt = dt_parse(normed_text, default=self.default_dt)
            year = dt.year
            month = dt.month
            day = dt.day

            if (
                dt.year == self.default_dt.year
                or dt.month == self.default_dt.month
                or dt.day == self.default_dt.day
            ):
                dt2 = dt_parse(text, default=self.default_dt2)

                if (
                    dt.year == self.default_dt.year
                    and dt2.year == self.default_dt2.year
                ):
                    year = None
                if (
                    dt.month == self.default_dt.month
                    and dt2.month == self.default_dt2.month
                ):
                    month = None
                if dt.day == self.default_dt.day and dt2.day == self.default_dt2.day:
                    day = None

            dt = ParsedDatetimeRepr(year=year, month=month, day=day)
        except (ParserError, TypeError, OverflowError):
            dt = None

        return dt

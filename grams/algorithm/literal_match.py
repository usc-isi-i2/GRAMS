import re
import rltk
from dataclasses import dataclass
from datetime import datetime, MINYEAR
from enum import Enum
from typing import Tuple, Optional

import fastnumbers
import ftfy
from dateutil.parser import parse as dt_parse, ParserError

from kgdata.wikidata.models import DataValue, QNode


class WikidataValueType(Enum):
    string = "string"
    time = "time"
    quantity = "quantity"
    mono_lingual_text = "monolingualtext"
    globe_coordinate = "globecoordinate"
    entity_id = "wikibase-entityid"


@dataclass
class DatetimeParsedRepr:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None

    def has_only_year(self):
        return self.year is not None and \
               self.month is None and \
               self.day is None and \
               self.hour is None and \
               self.minute is None and \
               self.second is None

    def first_day_of_year(self):
        return self.year is not None and \
               self.month == 1 and \
               self.day == 1 and \
               self.hour is None and \
               self.minute is None and \
               self.second is None


@dataclass
class ParsedTextRepr:
    origin: str
    normed_string: str
    number_string: str
    number: Optional[float]
    datetime: Optional[DatetimeParsedRepr]


class TextParser:
    def __init__(self):
        self.cache = {}
        self.default_dt = datetime(MINYEAR, 1, 1)
        self.default_dt2 = datetime(MINYEAR+3, 2, 28)
        self.number_chars = re.compile("[^0-9\.+-]")

    def parse(self, text: str) -> ParsedTextRepr:
        if text not in self.cache:
            self.cache[text] = self._parse(text)
        return self.cache[text]

    def _parse(self, text: str):
        try:
            dt = dt_parse(text, default=self.default_dt)
            year = dt.year
            month = dt.month
            day = dt.day

            if dt.year == self.default_dt.year or dt.month == self.default_dt.month or dt.day == self.default_dt.day:
                dt2 = dt_parse(text, default=self.default_dt2)

                if dt.year == self.default_dt.year and dt2.year == self.default_dt2.year:
                    year = None
                if dt.month == self.default_dt.month and dt2.month == self.default_dt2.month:
                    month = None
                if dt.day == self.default_dt.day and dt2.day == self.default_dt2.day:
                    day = None

            dt = DatetimeParsedRepr(
                year=year,
                month=month,
                day=day
            )
        except (ParserError, TypeError, OverflowError):
            dt = None

        number_string = self._parse_number_string(text)
        normed_string = self._norm_string(text)
        if fastnumbers.isfloat(number_string) and (len(number_string) / min(1, len(normed_string))) > 0.95:
            number = fastnumbers.fast_real(number_string, coerce=False)
        else:
            number = None
        return ParsedTextRepr(
            origin=text,
            normed_string=normed_string,
            number_string=number_string,
            number=number,
            datetime=dt
        )

    def _parse_number_string(self, text: str):
        num_string = self.number_chars.sub("", text)
        return num_string

    def _norm_string(self, text: str):
        return ftfy.fix_text(text).replace("\xa0", " ").strip()


class LiteralMatcher:
    literal_types = {
        WikidataValueType.string.value, WikidataValueType.time.value,
        WikidataValueType.quantity.value, WikidataValueType.mono_lingual_text.value,
        WikidataValueType.globe_coordinate.value}
    non_literal_types = {WikidataValueType.entity_id.value}

    @classmethod
    def string_test_exact(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        if val.normed_string == p_val.value:
            return True, 1.0
        return False, 0.0

    @classmethod
    def string_test_fuzzy(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        return cls.match_string(val.normed_string, p_val.value)

    @classmethod
    def globe_coordinate_test(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        return False, 0.0

    @classmethod
    def time_test(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        if val.datetime is None:
            return False, 0.0
        timestr = p_val.value['time']
        # there are two calendar:
        # gregorian calendar, and julian calendar (just 13 days behind gregorian)
        # https://www.timeanddate.com/calendar/julian-gregorian-switch.html#:~:text=13%20Days%20Behind%20Today,days%20behind%20the%20Gregorian%20calendar.
        # TODO: we need to consider a range for julian calendar
        assert p_val.value['calendarmodel'] in {'http://www.wikidata.org/entity/Q1985786',
                                                'http://www.wikidata.org/entity/Q1985727'}, p_val.value['calendarmodel']
        # TODO: handle timezone, before/after and precision
        # pass

        # parse timestring
        match = re.match(r"([-+]\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z", timestr)

        target_val = DatetimeParsedRepr(
            year=int(match.group(1)),
            month=int(match.group(2)),
            day=int(match.group(3)),
            hour=int(match.group(4)),
            minute=int(match.group(5)),
            second=int(match.group(6)),
        )
        if timestr[0] == '-':
            target_val.year = -target_val.year
        else:
            assert timestr[0] == '+'
        for p in ['year', 'month', 'day', 'hour', 'minute', 'second']:
            if getattr(target_val, p) == 0:
                setattr(target_val, p, None)

        if target_val == val.datetime:
            return True, 1.0
        if val.datetime.has_only_year() and target_val.year == val.datetime.year:
            return True, 0.8
        if val.datetime.first_day_of_year() and target_val.year == val.datetime.year:
            return True, 0.75
        return False, 0.0

    @classmethod
    def quantity_test(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        if val.number is None:
            return False, 0.0

        target_val = float(p_val.value["amount"])
        if abs(target_val - val.number) < 1e-5:
            return True, 1.0
        if target_val == 0:
            diff_percentage = abs(target_val - val.number) / 1e-7
        else:
            diff_percentage = abs((target_val - val.number) / target_val)

        if diff_percentage < 0.05:
            # within 5%
            return True, 0.95 - diff_percentage
        # if diff_percentage < 0.1:
        #     return True, 0.9 - diff_percentage
        return False, 0.0

    @classmethod
    def mono_lingual_text_test_exact(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        target_val = p_val.value['text']
        if val.normed_string == target_val:
            return True, 1.0
        return False, 0.0

    @classmethod
    def mono_lingual_text_test_fuzzy(cls, p_val: DataValue, val: ParsedTextRepr) -> Tuple[bool, float]:
        target_val = p_val.value['text']
        return cls.match_string(val.normed_string, target_val)

    @classmethod
    def entity_id_test_fuzzy(cls, qnode: QNode, val: ParsedTextRepr) -> Tuple[bool, float]:
        string = val.normed_string
        lst = [x.strip() for x in string.split(",")]
        for label in [qnode.label] + qnode.aliases:
            for item in lst:
                result = cls.match_string(label, item)
                if result[0] is True:
                    return result
        return False, 0.0

    @classmethod
    def match_string(cls, s1: str, s2: str):
        if s1 == s2:
            return True, 1.0
        # calculate edit distance
        distance = rltk.levenshtein_distance(s1, s2)
        if distance <= 1:
            return True, 0.95
        elif distance > 1 and distance / max(1, min(len(s1), len(s2))) < 0.03:
            return True, 0.85
        return False, 0.0


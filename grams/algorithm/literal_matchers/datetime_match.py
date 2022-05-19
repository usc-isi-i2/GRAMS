import re
from typing import Tuple, cast
from grams.algorithm.literal_matchers.text_parser import (
    ParsedTextRepr,
    ParsedDatetimeRepr,
)
from grams.algorithm.literal_matchers.types import LiteralMatchKit
from kgdata.wikidata.models import WDValueTime


def time_test(
    kgvalue: WDValueTime, value: ParsedTextRepr, kit: LiteralMatchKit
) -> Tuple[bool, float]:
    """Compare if the value in KG matches with value in the cell

    Args:
        kgvalue: the value in KG
        value: the value in the cell
        kit: objects that may be needed for matching
    Returns:
        a tuple of (whether it's matched, confidence)
    """
    celldt = value.datetime
    if celldt is None:
        return False, 0.0

    kgtime = kgvalue.value
    timestr = kgtime["time"]
    # there are two calendar:
    # gregorian calendar, and julian calendar (just 13 days behind gregorian)
    # https://www.timeanddate.com/calendar/julian-gregorian-switch.html#:~:text=13%20Days%20Behind%20Today,days%20behind%20the%20Gregorian%20calendar.
    # TODO: we need to consider a range for julian calendar
    assert kgtime["calendarmodel"] in {
        "http://www.wikidata.org/entity/Q1985786",
        "http://www.wikidata.org/entity/Q1985727",
    }, kgtime["calendarmodel"]
    # TODO: handle timezone, before/after and precision
    # pass

    # parse timestring
    match = re.match(r"([-+]\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z", timestr)
    assert match is not None
    kgdt = ParsedDatetimeRepr(
        year=int(match.group(1)),
        month=int(match.group(2)),
        day=int(match.group(3)),
        hour=int(match.group(4)),
        minute=int(match.group(5)),
        second=int(match.group(6)),
    )
    if timestr[0] == "-":
        kgdt.year = -kgdt.year  # type: ignore
    else:
        assert timestr[0] == "+"
    for p in ["year", "month", "day", "hour", "minute", "second"]:
        if getattr(kgdt, p) == 0:
            setattr(kgdt, p, None)

    if kgdt == celldt:
        return True, 1.0
    if celldt.has_only_year() and kgdt.year == celldt.year:
        return True, 0.8
    if celldt.first_day_of_year() and kgdt.year == celldt.year:
        return True, 0.75
    return False, 0.0

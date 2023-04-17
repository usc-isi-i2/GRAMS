from typing import Optional

class LiteralMatcherConfig:
    string: str
    quantity: str
    globecoordinate: str
    time: str
    monolingual_text: str
    entity: str

    def __init__(
        self,
        string: str,
        quantity: str,
        globecoordinate: str,
        time: str,
        monolingual_text: str,
        entity: str,
    ): ...

class LiteralMatcher:
    def __init__(self, cfg: LiteralMatcherConfig): ...

class ParsedTextRepr:
    origin: str
    normed_string: str
    number: Optional[ParsedNumberRepr]
    datetime: Optional[ParsedDatetimeRepr]

    def __init__(
        self,
        origin: str,
        normed_string: str,
        number: Optional[ParsedNumberRepr],
        datetime: Optional[ParsedDatetimeRepr],
    ): ...

class ParsedNumberRepr:
    number: float
    number_string: str
    is_integer: bool
    unit: Optional[str]
    prob: Optional[float]

    def __init__(
        self,
        number: float,
        number_string: str,
        is_integer: bool,
        unit: Optional[str],
        prob: Optional[float],
    ): ...

class ParsedDatetimeRepr:
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]
    hour: Optional[int]
    minute: Optional[int]
    second: Optional[int]

    def __init__(
        self,
        year: Optional[int],
        month: Optional[int],
        day: Optional[int],
        hour: Optional[int],
        minute: Optional[int],
        second: Optional[int],
    ): ...

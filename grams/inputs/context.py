from typing import (
    Dict,
    List,
    Set,
    TypedDict,
    Union,
)
from dataclasses import asdict, dataclass, field


class Attribute(TypedDict, total=False):
    href: str


@dataclass
class Text:
    id: str
    value: str = ""
    tags: List[str] = field(default_factory=list)
    id2attrs: Dict[str, Attribute] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "value": self.value,
            "tags": self.tags,
            "id2attrs": {k: v for k, v in self.id2attrs.items()},
        }

    @staticmethod
    def from_dict(obj: dict):
        return Text(
            id=obj["id"],
            value=obj["value"],
            tags=obj["tags"],
            id2attrs=obj["id2attrs"],
        )

    @staticmethod
    def from_tuple(obj: tuple):
        return Text(obj[0], obj[1], obj[2], obj[3])

    def __str__(self):
        return self.value


@dataclass
class Linebreak:
    n_lines: int = 1

    def to_dict(self):
        return {"n_lines": self.n_lines}

    @staticmethod
    def from_dict(obj: dict):
        return Linebreak(
            n_lines=obj["n_lines"],
        )

    def __str__(self):
        return f"\n{{{self.n_lines}}}"


@dataclass
class ContentHierarchy:
    """Content at each level that leads to the table"""

    level: int  # level of the heading, level 0 indicate the beginning of the document but should not be used
    heading: str  # title of the level (header)
    # partially HTML content, normalized <a>, <b>, <i> tags (breaklines or block text such as div, p are converted to line breaks)
    # other HTML containing content such as <table>, <img>, <video>, <audio> is kept as empty tag.
    content_before: List[Union[Text, Linebreak]] = field(default_factory=list)
    # only not empty if this is not the same level as the table.
    content_after: List[Union[Text, Linebreak]] = field(default_factory=list)

    def to_dict(self):
        return {
            "level": self.level,
            "heading": self.heading,
            "content_before": [c.to_dict() for c in self.content_before],
            "content_after": [c.to_dict() for c in self.content_after],
        }

    @staticmethod
    def from_dict(obj: dict):
        return ContentHierarchy(
            level=obj["level"],
            heading=obj["heading"],
            content_before=[
                Text.from_dict(c) if "id" in c else Linebreak.from_dict(c)
                for c in obj["content_before"]
            ],
            content_after=[
                Text.from_dict(c) if "id" in c else Linebreak.from_dict(c)
                for c in obj["content_after"]
            ],
        )

    @staticmethod
    def from_tuple(obj: tuple):
        return ContentHierarchy(
            level=obj[0],
            heading=obj[1],
            content_before=[
                Text.from_tuple(c) if len(c) > 1 else Linebreak(c[0]) for c in obj[2]
            ],
            content_after=[
                Text.from_tuple(c) if len(c) > 1 else Linebreak(c[0]) for c in obj[3]
            ],
        )

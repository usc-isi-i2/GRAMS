from __future__ import annotations
from collections import defaultdict
import re
from dataclasses import dataclass, asdict, field
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse

import fastnumbers
import orjson
from slugify import slugify

import sm.misc as M
from sm.inputs.table import ColumnBasedTable, Column
from grams.inputs.context import ContentHierarchy


@dataclass
class LinkedTable:
    # the table that we are working on
    table: ColumnBasedTable

    # context of the table
    context: Context

    # a mapping from (row id, column id) to the list of links attach to that cell (not include header)
    links: List[List[List[Link]]]

    @property
    def id(self):
        return self.table.table_id

    def shape(self):
        return self.table.shape()

    def remove_nonexistent_entities(self, nonexisting_entities: Set[str]):
        nrows, ncols = self.table.shape()
        for ri in range(nrows):
            for ci in range(ncols):
                for link in self.links[ri][ci]:
                    link.remove_nonexisting_entities(nonexisting_entities)

    def size(self):
        """Number of rows of the table"""
        if len(self.table.columns) == 0:
            return 0
        return len(self.table.columns[0].values)

    def iter_cell_index(self):
        nrows, ncols = self.table.shape()
        for ri in range(nrows):
            for ci in range(ncols):
                yield ri, ci

    def to_dict(self):
        return {
            "version": "1.2",
            "table": self.table.to_dict(),
            "context": self.context.to_dict(),
            "links": [
                [[link.to_dict() for link in links] for links in rlinks]
                for rlinks in self.links
            ],
        }

    def get_friendly_fs_id(self):
        return LinkedTable._get_friendly_fs_id(self.id)

    @staticmethod
    def _get_friendly_fs_id(id: str):
        if id.startswith("http://") or id.startswith("https://"):
            if id.find("dbpedia.org") != -1:
                return (
                    slugify(
                        urlparse(id).path.replace("/resource/", "").replace("/", "_")
                    ).replace("-", "_")
                    + "_"
                    + md5(id.encode()).hexdigest()
                )

            if id.find("wikipedia.org") != -1:
                return (
                    slugify(
                        urlparse(id).path.replace("/wiki/", "").replace("/", "_")
                    ).replace("-", "_")
                    + "_"
                    + md5(id.encode()).hexdigest()
                )

            raise NotImplementedError()
        return slugify(id.replace("/", "_")).replace("-", "_")

    @staticmethod
    def from_dict(odict: dict):
        assert float(odict.get("version", "0.0")) >= 1.1
        tbl = ColumnBasedTable.from_dict(odict["table"])
        context = Context.from_dict(odict["context"])
        links = [
            [[Link.from_dict(link) for link in links] for links in rlinks]
            for rlinks in odict["links"]
        ]
        return LinkedTable(tbl, context, links)

    @staticmethod
    def from_column_based_table(tbl: ColumnBasedTable):
        links = []
        if len(tbl.columns) > 0:
            links = [
                [[] for ci in range(len(tbl.columns))]
                for ri in range(len(tbl.columns[0].values))
            ]
        return LinkedTable(tbl, Context(None, None, None), links)

    @staticmethod
    def from_csv_file(
        infile: Union[Path, str],
        link_file: Optional[Union[Path, str]] = None,
        first_row_header: bool = True,
        table_id: Optional[str] = None,
        top_k: int = 100,
    ) -> LinkedTable:
        """Load table from a csv file, and its links from a tsv file (if exist).

        For format of the link file, see `LinkedTable.parse_link_file`

        Args:
            infile: csv file
            link_file: if not provided, this function will look for a file of same name of  `infile` in the same folder but ends with `.links.tsv`.
            first_row_header: whether the first row of the `infile` is header of the table
            table_id: if None, the table id will be file name of `infile`
            top_k: top k candidate entity to keep.
        """
        infile = Path(infile)
        if link_file is None:
            link_file = infile.parent / f"{infile.stem}.links.tsv"
        else:
            link_file = Path(link_file)

        if table_id is None:
            table_id = infile.stem
        rows = M.deserialize_csv(infile)

        assert len(rows) > 0, "Empty table"
        columns = []
        if first_row_header:
            headers = rows[0]
            rows = rows[1:]
        else:
            headers = [f"column-{i:03d}" for i in range(len(rows[0]))]

        for ci, cname in enumerate(headers):
            columns.append(Column(ci, cname, [r[ci] for r in rows]))
        table = ColumnBasedTable(table_id, columns)

        if link_file.exists():
            links = LinkedTable.parse_link_file(table, link_file, top_k)
        else:
            links = []
            for ri in range(len(rows)):
                links.append([[] for _ci in range(len(headers))])

        return LinkedTable(table, Context(), links)

    @staticmethod
    def parse_link_file(
        table: ColumnBasedTable, infile: Union[Path, str], top_k: int = 100
    ) -> List[List[List[Link]]]:
        """
        Each row of a link file has the following format: `<row_index>\t<col_index>\t(<link>|(<entity_id>(\t<entity_id>)*))`, where:
            * `row_index` and `col_index` start from 0
            * `row_index` does not count the header of the table (i.e., skip the first row of `infile` if it's the header)
            * `(<link>|(<entity_id>(\t<entity_id>)*))` is either:
                - `<link>` a json string encoding Link object and can be deserialized using `Link.from_dict` function
                - or (<entity_id>(\t<entity_id>)*) a list of entity ids joined by `\t` tab character, each entity id can be a wikidata qnode id (e.g., Q414) or a full qnode uri (e.g., "http://www.wikidata.org/entity/Q414"). The first entity is considered as the correct entity of the cell, and the rest are considered as the candidate entities of the cell
        Note that a pair `<row_index>`, `<col_index>` don't have to be unique.
        """
        links = []
        nrows, ncols = table.shape()

        for ri in range(nrows):
            links.append([[] for _ci in range(ncols)])

        rows = []
        for row in M.deserialize_csv(infile, delimiter="\t"):
            ri, ci, ents = int(row[0]), int(row[1]), row[2:]
            rows.append((ri, ci, ents))

        has_only_correct_entity = all(len(ents) == 1 for _, _, ents in rows)

        for ri, ci, ents in rows:
            if len(ents) == 1 and ents[0].startswith("{"):
                link = Link.from_dict(orjson.loads(ents[0]))
            else:
                pents = []
                for ent in ents[: top_k + 1]:
                    if ent.startswith("http"):
                        assert ent.startswith("http://www.wikidata.org/entity/")
                        qnode_id = ent.replace("http://www.wikidata.org/entity/", "")
                    else:
                        qnode_id = ent

                    if ":" in qnode_id:
                        qnode_id, qnode_prob = qnode_id.split(":", 1)
                        assert fastnumbers.isfloat(qnode_prob)
                        qnode_prob = fastnumbers.float(qnode_prob)
                    else:
                        qnode_prob = 1.0
                    pents.append((qnode_id, qnode_prob))

                if has_only_correct_entity:
                    candidates = [CandidateEntity(pents[0][0], pents[0][1])]
                else:
                    candidates = [CandidateEntity(e[0], e[1]) for e in pents[1:]]

                link = Link(
                    start=0,
                    end=len(table.columns[ci][ri]),
                    url=f"http://www.wikidata.org/entity/{pents[0][0]}",
                    entity_id=pents[0][0] if pents[0][0].strip() != "" else None,
                    candidates=candidates,
                )
            links[ri][ci].append(link)

        return links


@dataclass
class Context:
    """Table's context"""

    page_title: Optional[str] = None
    page_url: Optional[str] = None
    page_entity_id: Optional[str] = None

    content_hierarchy: List[ContentHierarchy] = field(default_factory=list)

    def to_dict(self):
        return {
            "page_title": self.page_title,
            "page_url": self.page_url,
            "page_entity_id": self.page_entity_id,
            "content_hierarchy": [c.to_dict() for c in self.content_hierarchy],
        }

    @staticmethod
    def from_dict(odict: dict):
        return Context(
            page_title=odict.get("page_title"),
            page_url=odict.get("page_url"),
            page_entity_id=odict.get("page_entity_id"),
            content_hierarchy=[
                ContentHierarchy.from_dict(c)
                for c in odict.get("content_hierarchy", [])
            ],
        )


@dataclass
class CandidateEntity:
    entity_id: str
    probability: float


@dataclass
class Link:
    start: int
    end: int  # exclusive
    url: Optional[str]  # none when there is no link, and the entity is not mapped yet
    entity_id: Optional[str]
    candidates: List[CandidateEntity]

    @staticmethod
    def from_dict(obj: dict) -> Link:
        return Link(
            start=obj["start"],
            end=obj["end"],
            url=obj["url"],
            entity_id=obj["entity_id"],
            candidates=[CandidateEntity(**c) for c in obj.get("candidates", [])],
        )

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "url": self.url,
            "entity_id": self.entity_id,
            "candidates": [asdict(c) for c in self.candidates],
        }

    def remove_nonexisting_entities(self, nonexisting_entities: Set[str]):
        if self.entity_id in nonexisting_entities:
            self.entity_id = None
        self.candidates = [
            c for c in self.candidates if c.entity_id not in nonexisting_entities
        ]

    def length(self) -> int:
        return self.end - self.start

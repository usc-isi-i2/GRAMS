from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Union

import fastnumbers
import orjson
import serde.csv

from sm.dataset import FullTable, get_friendly_fs_id
from sm.inputs.prelude import (
    WIKIDATA,
    Column,
    ColumnBasedTable,
    Context,
    EntityId,
    Link,
)
from sm.misc.matrix import Matrix
import grams.core.table as gcore


@dataclass
class LinkedTable(FullTable):
    links: Matrix[List[ExtendedLink]]

    @property
    def id(self):
        return self.table.table_id

    def shape(self):
        return self.table.shape()

    def size(self):
        """Number of rows of the table"""
        if len(self.table.columns) == 0:
            return 0
        return len(self.table.columns[0].values)

    def remove_nonexistent_entities(self, nonexisting_entities: Set[str]):
        nrows, ncols = self.table.shape()
        for links in self.links.flat_iter():
            for link in links:
                link.remove_nonexisting_entities(nonexisting_entities)

    def iter_cell_index(self):
        nrows, ncols = self.table.shape()
        for ri in range(nrows):
            for ci in range(ncols):
                yield ri, ci

    @classmethod
    def from_dict(cls, obj: dict):
        version = obj["version"]
        if not version == 2:
            raise ValueError(f"Unknown version: {version}")

        return cls(
            table=ColumnBasedTable.from_dict(obj["table"]),
            context=Context.from_dict(obj["context"]),
            links=Matrix(
                [
                    [[ExtendedLink.from_dict(link) for link in cell] for cell in row]
                    for row in obj["links"]
                ]
            ),
        )

    def get_friendly_fs_id(self):
        return get_friendly_fs_id(self.id)

    @staticmethod
    def from_column_based_table(tbl: ColumnBasedTable):
        links = []
        if len(tbl.columns) > 0:
            links = [
                [[] for ci in range(len(tbl.columns))]
                for ri in range(len(tbl.columns[0].values))
            ]
        return LinkedTable(tbl, Context(), Matrix(links))

    @staticmethod
    def from_full_table(tbl: FullTable):
        return LinkedTable(
            tbl.table,
            tbl.context,
            Matrix(
                [
                    [
                        [
                            ExtendedLink(
                                link.start, link.end, link.url, link.entities, []
                            )
                            for link in cell
                        ]
                        for cell in row
                    ]
                    for row in tbl.links.data
                ]
            ),
        )

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
        rows = serde.csv.deser(infile)

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
            links = Matrix(links)
        return LinkedTable(table, Context(), links)

    @staticmethod
    def parse_link_file(
        table: ColumnBasedTable, infile: Union[Path, str], top_k: int = 100
    ) -> Matrix[List[ExtendedLink]]:
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
        for row in serde.csv.deser(infile, delimiter="\t"):
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
                    candidates = [
                        CandidateEntityId(EntityId(pents[0][0], WIKIDATA), pents[0][1])
                    ]
                else:
                    candidates = [
                        CandidateEntityId(EntityId(e[0], WIKIDATA), e[1])
                        for e in pents[1:]
                    ]

                link = ExtendedLink(
                    start=0,
                    end=len(table.columns[ci][ri]),
                    url=f"http://www.wikidata.org/entity/{pents[0][0]}",
                    entities=[EntityId(pents[0][0], WIKIDATA)]
                    if pents[0][0].strip() != ""
                    else [],
                    candidates=candidates,
                )
            links[ri][ci].append(link)

        return Matrix(links)

    def to_rust(self) -> gcore.LinkedTable:
        def to_col(col: Column) -> gcore.Column:
            values = []
            for v in col.values:
                if isinstance(v, str):
                    values.append(v)
                elif v is None:
                    values.append("")
                else:
                    raise ValueError(f"Unsupported value type: {type(v)}")
            return gcore.Column(col.index, col.clean_multiline_name, values)

        def to_link(link: ExtendedLink) -> gcore.Link:
            return gcore.Link(
                link.start,
                link.end,
                link.url,
                [gcore.EntityId(str(eid)) for eid in link.entities],
                [
                    gcore.CandidateEntityId(
                        gcore.EntityId(str(c.entity_id)), c.probability
                    )
                    for c in link.candidates
                ],
            )

        return gcore.LinkedTable(
            self.table.table_id,
            [
                [[to_link(link) for link in cell] for cell in row]
                for row in self.links.data
            ],
            [to_col(col) for col in self.table.columns],
            gcore.Context(
                self.context.page_title,
                self.context.page_url,
                [gcore.EntityId(str(eid)) for eid in self.context.page_entities],
            ),
        )


class CandidateEntityId:
    __slots__ = ("entity_id", "probability")

    def __init__(self, entity_id: EntityId, probability: float):
        self.entity_id = entity_id
        self.probability = probability

    def to_dict(self):
        return {
            "id": self.entity_id.to_dict(),
            "prob": self.probability,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> CandidateEntityId:
        return CandidateEntityId(
            entity_id=EntityId.from_dict(obj["id"]),
            probability=obj["prob"],
        )


class ExtendedLink(Link):
    __slots__ = ("candidates",)

    def __init__(
        self,
        start: int,
        end: int,
        url: Optional[str],
        entities: List[EntityId],
        candidates: List[CandidateEntityId],
    ):
        super().__init__(start, end, url, entities)
        self.candidates = candidates

    def clone(self):
        return ExtendedLink(
            self.start,
            self.end,
            self.url,
            self.entities.copy(),
            self.candidates.copy(),
        )

    def to_dict(self):
        obj = super().to_dict()
        obj["candidates"] = [c.to_dict() for c in self.candidates]
        return obj

    @classmethod
    def from_dict(cls, obj: dict) -> ExtendedLink:
        version = obj.get("version")
        if version == 2:
            return ExtendedLink(
                start=obj["start"],
                end=obj["end"],
                url=obj["url"],
                entities=[EntityId.from_dict(e) for e in obj["entities"]],
                candidates=[CandidateEntityId.from_dict(c) for c in obj["candidates"]],
            )
        if version is None:
            return ExtendedLink(
                start=obj["start"],
                end=obj["end"],
                url=obj["url"],
                entities=[EntityId(id=eid, type="wikidata")]
                if (eid := obj["entity_id"]) is not None
                else [],
                candidates=[
                    CandidateEntityId(
                        EntityId(id=c["entity_id"], type="wikidata"),
                        probability=c["probability"],
                    )
                    for c in obj["candidates"]
                ]
                if "candidates" in obj
                else [],
            )
        raise ValueError(f"Unknown version: {version}")

    def remove_nonexisting_entities(self, nonexisting_entities: Set[str]):
        self.entities = [
            entid for entid in self.entities if entid not in nonexisting_entities
        ]
        self.candidates = [
            c for c in self.candidates if c.entity_id not in nonexisting_entities
        ]

    def length(self) -> int:
        return self.end - self.start

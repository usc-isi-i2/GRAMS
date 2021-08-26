from dataclasses import dataclass, asdict
from hashlib import md5
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import orjson
from slugify import slugify

import sm.misc as M
from sm.inputs.table import ColumnBasedTable, Column


@dataclass
class LinkedTable:
    # the table that we are working on
    table: ColumnBasedTable

    context: "Context"

    # a mapping from (row id, column id) to the list of links attach to that cell
    links: List[List[List["Link"]]]

    @property
    def id(self):
        return self.table.table_id

    def size(self):
        if len(self.table.columns) == 0:
            return 0
        return len(self.table.columns[0].values)

    def to_dict(self):
        return {
            "version": 1,
            "table": self.table.to_dict(),
            "context": asdict(self.context),
            "links": [
                [[asdict(link) for link in links] for links in rlinks]
                for rlinks in self.links
            ],
        }

    def get_friendly_fs_id(self):
        id = self.id
        if id.startswith("http://") or id.startswith("https://"):
            if id.find("dbpedia.org") != -1:
                id = slugify(
                    urlparse(id).path.replace("/resource/", "").replace("/", "_")
                ).replace("-", "_")
                id += "_" + md5(self.id.encode()).hexdigest()
            elif id.find("wikipedia.org") != -1:
                id = slugify(
                    urlparse(id).path.replace("/wiki/", "").replace("/", "_")
                ).replace("-", "_")
                id += "_" + md5(self.id.encode()).hexdigest()
            else:
                raise NotImplementedError()
        return id

    @staticmethod
    def from_dict(odict: dict):
        tbl = ColumnBasedTable.from_dict(odict["table"])
        context = Context(**odict["context"])
        links = [
            [[Link(**link) for link in links] for links in rlinks]
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
        link_file: Optional[str] = None,
        first_row_header: bool = True,
        table_id: Optional[str] = None,
    ):
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
        links = []
        for ri in range(len(rows)):
            links.append([[] for _ci in range(len(headers))])

        if link_file.exists():
            for row in M.deserialize_csv(link_file, delimiter="\t"):
                ri, ci, ents = int(row[0]), int(row[1]), row[2:]
                for ent in ents:
                    if ent.startswith("{"):
                        # it's json, encoding the hyperlinks
                        link = Link(**orjson.loads(ent))
                    elif ent.startswith("http"):
                        assert ent.startswith("http://www.wikidata.org/entity/")
                        link = Link(
                            0,
                            len(table.columns[ci][ri]),
                            ent,
                            ent.replace("http://www.wikidata.org/entity/", ""),
                        )
                    else:
                        link = Link(
                            0,
                            len(table.columns[ci][ri]),
                            f"http://www.wikidata.org/entity/{ent}",
                            ent,
                        )
                    links[ri][ci].append(link)
        return LinkedTable(table, Context(), links)


@dataclass
class Context:
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    page_qnode: Optional[str] = None


@dataclass
class Link:
    start: int
    end: int
    url: str
    qnode_id: Optional[str]

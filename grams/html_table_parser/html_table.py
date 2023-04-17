from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from operator import itemgetter
from typing import Callable, Dict, List, Mapping, Optional
from urllib.parse import unquote_plus, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from rsoup.python.models.context import ContentHierarchy

import grams.inputs as GI
from grams.config import DATA_DIR
from grams.html_table_parser.parsing_exception import (
    InvalidCellSpanException,
    InvalidColumnSpanException,
    OverlapSpanException,
)
from kgdata.wikipedia.misc import get_title_from_url
from sm.prelude import I, M


@dataclass
class HTMLTableCellHTMLElement:
    # html tag (lower case)
    tag: str
    start: int
    # end (exclusive)
    end: int
    # html attributes
    attrs: Dict[str, str]
    children: List["HTMLTableCellHTMLElement"]

    def post_order(self):
        for c in self.children:
            for el in c.post_order():
                yield el
        yield self

    def clone(self):
        return HTMLTableCellHTMLElement(
            self.tag,
            self.start,
            self.end,
            copy.copy(self.attrs),
            [c.clone() for c in self.children],
        )


@dataclass
class HTMLTableCell:
    # text value of the cell
    value: str
    rowspan: int
    colspan: int
    # html of the cell
    html: str
    # list of html elements that created this cell
    # except that:
    # - BR and HR are replaced by `\n` character and not in this list
    # - div, span are container and won't in the list
    # for more details, look at the HTMLTableParser._extract_cell_recur function
    elements: List[HTMLTableCellHTMLElement]

    # original row & col span
    original_rowspan: Optional[int] = None
    original_colspan: Optional[int] = None

    def travel_elements_post_order(self):
        for el in self.elements:
            for pointer in el.post_order():
                yield pointer

    def clone(self):
        return HTMLTableCell(
            value=self.value,
            rowspan=self.rowspan,
            colspan=self.colspan,
            html=self.html,
            elements=[el.clone() for el in self.elements],
            original_rowspan=self.original_rowspan,
            original_colspan=self.original_colspan,
        )


@dataclass
class HTMLTableRow:
    cells: List[HTMLTableCell]
    # html attributes of the tr elements
    attrs: Dict[str, str]


@dataclass
class HTMLTable:
    page_url: str
    # value of html caption
    caption: str
    # html attributes of the table html element
    attrs: Dict[str, str]
    # context
    context: List[ContentHierarchy]
    # list of rows in the table
    rows: List[HTMLTableRow]

    def span(self) -> HTMLTable:
        """Span the table by copying values to merged field"""
        if len(self.rows) == 0:
            return self

        pi = 0
        data = []
        pending_ops = {}

        # >>> begin find the max #cols
        # calculate the number of columns as some people may actually set unrealistic colspan as they are lazy..
        # I try to make its behaviour as much closer to the browser as possible.
        # one thing I notice that to find the correct value of colspan, they takes into account the #cells of rows below the current row
        # so we may have to iterate several times
        cols = [0 for _ in range(len(self.rows))]
        for i, row in enumerate(self.rows):
            cols[i] += len(row.cells)
            for cell in row.cells:
                if cell.rowspan > 1:
                    for j in range(1, cell.rowspan):
                        if i + j < len(cols):
                            cols[i + j] += 1

        _row_index, max_ncols = max(enumerate(cols), key=itemgetter(1))
        # sometimes they do show an extra cell for over-colspan row, but it's not consistent or at least not easy for me to find the rule
        # so I decide to not handle that. Hope that we don't have many tables like that.
        # >>> finish find the max #cols

        for row in self.rows:
            new_row = []
            pj = 0
            for cell_index, cell in enumerate(row.cells):
                cell = cell.clone()
                cell.original_colspan = cell.colspan
                cell.original_rowspan = cell.rowspan
                cell.colspan = 1
                cell.rowspan = 1

                # adding cell from the top
                while (pi, pj) in pending_ops:
                    new_row.append(pending_ops[pi, pj].clone())
                    pending_ops.pop((pi, pj))
                    pj += 1

                # now add cell and expand the column
                for _ in range(cell.original_colspan):
                    if (pi, pj) in pending_ops:
                        # exception, overlapping between colspan and rowspan
                        raise OverlapSpanException()
                    new_row.append(cell.clone())
                    for ioffset in range(1, cell.original_rowspan):
                        # no need for this if below
                        # if (pi+ioffset, pj) in pending_ops:
                        #     raise OverlapSpanException()
                        pending_ops[pi + ioffset, pj] = cell
                    pj += 1

                    if pj >= max_ncols:
                        # our algorithm cannot handle the case where people are bullying the colspan system, and only can handle the case
                        # where the span that goes beyond the maximum number of columns is in the last column.
                        if cell_index != len(row.cells) - 1:
                            raise InvalidColumnSpanException()
                        else:
                            break

            # add more cells from the top since we reach the end
            while (pi, pj) in pending_ops and pj < max_ncols:
                new_row.append(pending_ops[pi, pj].clone())
                pending_ops.pop((pi, pj))
                pj += 1

            data.append(HTMLTableRow(cells=new_row, attrs=copy.copy(row.attrs)))
            pi += 1

        # len(pending_ops) may > 0, but fortunately, it doesn't matter as the browser also does not render that extra empty lines
        return HTMLTable(
            page_url=self.page_url,
            caption=self.caption,
            attrs=copy.copy(self.attrs),
            context=[copy.deepcopy(c) for c in self.context],
            rows=data,
        )

    def pad(self) -> HTMLTable:
        """Pad the irregular table (missing cells) to make it become regular table.

        This function only return new table when it's padded
        """
        if len(self.rows) == 0:
            return self

        ncols = len(self.rows[0].cells)
        is_regular_table = all(len(r.cells) == ncols for r in self.rows)
        if is_regular_table:
            return self

        max_ncols = max(len(r.cells) for r in self.rows)
        default_cell = HTMLTableCell(
            value="",
            rowspan=1,
            colspan=1,
            html="",
            elements=[],
            original_rowspan=1,
            original_colspan=1,
        )

        rows = []
        for r in self.rows:
            row = HTMLTableRow(
                cells=[c.clone() for c in r.cells], attrs=copy.copy(r.attrs)
            )
            while len(row.cells) < max_ncols:
                row.cells.append(default_cell.clone())
            rows.append(row)

        return HTMLTable(
            page_url=self.page_url,
            caption=self.caption,
            attrs=copy.copy(self.attrs),
            context=[copy.deepcopy(c) for c in self.context],
            rows=rows,
        )

    def as_df(self):
        return pd.DataFrame([[c.value for c in r.cells] for r in self.rows])

    def as_relational_linked_table(
        self,
        table_id: Optional[str] = None,
        get_entity_from_url: Optional[Callable[[str], Optional[str]]] = None,
        get_entity_from_title: Optional[Callable[[str], Optional[str]]] = None,
    ):
        assert len(self.rows) > 0
        header = [c.value for c in self.rows[0].cells]
        table = I.ColumnBasedTable(
            table_id or self.page_url,
            [
                I.Column(
                    ci,
                    cname,
                    [self.rows[ri].cells[ci].value for ri in range(1, len(self.rows))],
                )
                for ci, cname in enumerate(header)
            ],
        )

        if get_entity_from_url is None:
            if get_entity_from_title is not None:
                get_entity_from_url = lambda url: get_entity_from_title(
                    get_title_from_url(url)
                )
            else:
                get_entity_from_url = lambda _: None

        links = []
        for ri, row in enumerate(self.rows[1:]):
            rlinks = []
            for ci in range(len(header)):
                clinks = []
                for el in row.cells[ci].travel_elements_post_order():
                    if el.tag == "a":
                        entity_id = get_entity_from_url(el.attrs["href"])
                        if entity_id is not None:
                            candidates = [GI.CandidateEntity(entity_id, 1.0)]
                        else:
                            candidates = []
                        clinks.append(
                            GI.Link(
                                el.start,
                                el.end,
                                el.attrs["href"],
                                entity_id,
                                candidates,
                            )
                        )
                rlinks.append(clinks)
            links.append(rlinks)
        return GI.LinkedTable(
            table,
            GI.Context(
                page_url=self.page_url,
                page_title=get_title_from_url(self.page_url),
                page_entity_id=get_entity_from_url(self.page_url),
                content_hierarchy=self.context,
            ),
            links,
        )

import copy
from dataclasses import dataclass
from operator import itemgetter
from urllib.parse import urlparse, unquote_plus
from functools import partial
import requests, re, pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from typing import Iterable, List, Dict, Optional, Union, cast

from sm.prelude import I, M
import grams.inputs as GI
from grams.config import DATA_DIR
from grams.html_table_parser.html_table import (
    HTMLTable,
    HTMLTableCell,
    HTMLTableRow,
    HTMLTableCellHTMLElement,
)
from grams.html_table_parser.parsing_exception import (
    InvalidCellSpanException,
    InvalidColumnSpanException,
    OverlapSpanException,
)
from grams.html_table_parser.context_extractor import ContextExtractor


# bs4 PageElement is a subclass and only has two children: Tag and NavigableString
# re-define the type here as pylance can't infer it correctly (maybe because PageElement isn't abstract class)
PageElement = Union[Tag, NavigableString]


class HTMLTableParser:
    NUM_REGEX = re.compile(r"(\d+)")
    IGNORE_TAGS = set(["div", "span"])

    @dataclass
    class Header:
        level: int
        value: str

    @dataclass
    class Text:
        value: str

    """Extract tables from html, which may contains multiple nested tables. I assume that the outer tables are used for
    formatting and the inner tables are the interested ones. So this extractor will discard the outer tables and only keep
    the most inner tables.
    """

    def __init__(self, page_url: str, html: str) -> None:
        self.page_url = page_url
        self.doc = BeautifulSoup(html, "html5lib")
        self.context_extractor = ContextExtractor(self.doc)

    def extract_tables(
        self, auto_span: bool = True, auto_pad: bool = True
    ) -> List[HTMLTable]:
        tables = []
        table_els = self.doc.find_all("table")
        for table_el in table_els:
            tables += self.extract(table_el)

        if auto_span:
            new_tables = []
            for tbl in tables:
                try:
                    tbl = tbl.span()
                    new_tables.append(tbl)
                except OverlapSpanException:
                    pass
                except InvalidColumnSpanException:
                    pass
            tables = new_tables
        if auto_pad:
            tables = [tbl.pad() for tbl in tables]
        return tables

    def extract(self, table_el: Tag) -> List[HTMLTable]:
        """Extract tables from html, which may contains multiple nested tables. I assume that the outer tables are used for
        formatting and the inner tables are the interested ones. So this function will discard the outer tables and only keep
        the most inner tables.

        This function right now ignore tables with invalid rowspan or cellspan
        """
        results = []
        self._extract_table(table_el, results)
        if len(results) > 0:
            context = self.context_extractor.extract(table_el)
            for r in results:
                r["context"] = context

        tables = [HTMLTable(self.page_url, **r) for r in results]

        # convert relative links to absolute links
        parsed_resp = urlparse(self.page_url)
        domain = f"{parsed_resp.scheme}://{parsed_resp.netloc}"
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    for el in cell.travel_elements_post_order():
                        if el.tag == "a":
                            if "href" in el.attrs:
                                href = el.attrs["href"]
                                if href[0] == "/":
                                    el.attrs["href"] = domain + href

        if parsed_resp.netloc.endswith("wikipedia.org"):
            for table in tables:
                self._postprocess_wikipedia(table)

        return tables

    def _extract_table(self, el: Tag, results: List[dict]):
        """Extract tables from the table tag. Ignore tables that contain other tables

        Parameters
        ----------
        el : Tag
            html table tag
        results : List[dict]
            list of results
        """
        assert el.name == "table"
        caption = None
        rows = []
        contain_nested_table = any(
            c.find("table") is not None is not None
            for c in cast(List[PageElement], el.contents)
        )

        for c in cast(List[PageElement], el.contents):
            if isinstance(c, NavigableString):
                continue
            if c.name == "caption":
                caption = c.get_text().strip()
                continue
            if c.name == "style":
                continue
            assert (
                c.name == "thead" or c.name == "tbody"
            ), f"not implemented {c.name} tag"
            for row_el in cast(List[PageElement], c.contents):
                if isinstance(row_el, NavigableString):
                    continue
                if row_el.name == "style":
                    continue

                assert row_el.name == "tr", f"Invalid tag: {row_el.name}"
                cells = []
                for cell_el in cast(List[PageElement], row_el.contents):
                    if isinstance(cell_el, NavigableString):
                        continue
                    if cell_el.name == "style":
                        continue

                    assert (
                        cell_el.name == "th" or cell_el.name == "td"
                    ), f"Invalid tag: {row_el.name}"

                    if contain_nested_table:
                        nested_tables = []
                        self._extract_tbl_tags(cell_el, nested_tables)
                        for tag in nested_tables:
                            self._extract_table(tag, results)
                    else:
                        try:
                            cell = self._extract_cell(cell_el)
                            cells.append(cell)
                        except InvalidCellSpanException:
                            # not extracting this table, this table is not table at the intermediate level
                            return

                if not contain_nested_table:
                    rows.append(HTMLTableRow(cells=cells, attrs=dict(row_el.attrs)))

        if not contain_nested_table:
            results.append(dict(rows=rows, caption=caption, attrs=el.attrs))

    def _extract_tbl_tags(self, el: Tag, tags: List[Tag]):
        """Recursive find the first table node

        Parameters
        ----------
        el : Tag
            root node that we start finding
        tags : List[Tag]
            list of results
        """
        for c in cast(Iterable[PageElement], el.children):
            if isinstance(c, NavigableString):
                continue
            elif c.name == "table":
                tags.append(c)
            elif len(c.contents) > 0:
                self._extract_tbl_tags(c, tags)

    def _extract_cell(self, el: Tag) -> HTMLTableCell:
        """Extract cell from td/th tag. This function does not expect a nested table in the cell

        Parameters
        ----------
        tbl : RawHTMLTable
            need the table to get the page URI in case we have cell reference
        el : Tag
            cell tag (th or td)
        """
        assert el.name == "th" or el.name == "td"

        # extract other attributes
        is_header = el.name == "th"
        attrs = el.attrs

        colspan = attrs.get("colspan", "1").strip()
        rowspan = attrs.get("rowspan", "1").strip()

        if colspan == "":
            colspan = 1
        else:
            m = self.NUM_REGEX.search(colspan)
            if m is None:
                raise InvalidCellSpanException()
            colspan = int(m.group(0))
        if rowspan == "":
            rowspan = 1
        else:
            m = self.NUM_REGEX.search(rowspan)
            if m is None:
                # this is not correct, but
                raise InvalidCellSpanException()
            rowspan = int(m.group(0))

        # extract value
        result = {"text": "", "elements": []}
        self._extract_cell_recur(el, result)
        value, elements = result["text"], result["elements"]
        if len(value) > 0 and value[-1] == " ":
            value = value[:-1]

        return HTMLTableCell(
            value=value,
            html=str(el),
            elements=elements,
            colspan=colspan,
            rowspan=rowspan,
        )

    def _extract_cell_recur(self, el: Tag, result: dict):
        """Real implementation of the _extract_cell function"""
        for c in cast(Iterable[PageElement], el.children):
            if isinstance(c, NavigableString):
                result["text"] += c.strip()
                result["text"] += " "
            elif c.name == "br" or c.name == "hr":
                result["text"] += "\n"
            elif c.name in self.IGNORE_TAGS:
                self._extract_cell_recur(c, result)
            else:
                start = len(result["text"])
                children_result = {"text": result["text"], "elements": []}
                self._extract_cell_recur(c, children_result)
                # -1 for the trailing space
                if (
                    len(children_result["text"]) > 0
                    and children_result["text"][-1] == " "
                ):
                    if start == len(children_result["text"]):
                        # no new text
                        start -= 1
                    end = len(children_result["text"]) - 1
                else:
                    end = len(children_result["text"])
                assert end >= start
                result["text"] = children_result["text"]
                result["elements"].append(
                    HTMLTableCellHTMLElement(
                        tag=c.name,
                        start=start,
                        end=end,
                        attrs=dict(c.attrs),
                        children=children_result["elements"],
                    )
                )

    def _postprocess_wikipedia(self, table: HTMLTable):
        for row in table.rows:
            for cell in row.cells:
                for el in cell.travel_elements_post_order():
                    if el.tag == "a":
                        if "href" not in el.attrs:
                            if "selflink" in el.attrs["class"]:
                                el.attrs["href"] = table.page_url
                            else:
                                raise Exception(
                                    f"An anchor in a Wikipedia table does not have href: {el}"
                                )

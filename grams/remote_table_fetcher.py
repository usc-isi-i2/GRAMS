import copy
from dataclasses import dataclass
from operator import itemgetter
from urllib.parse import urlparse, unquote_plus

import requests, re, pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Dict, Optional

from sm.prelude import I, M
import grams.inputs as GI
from grams.config import DATA_DIR
from grams.db import Wikipedia2WikidataDB


@dataclass
class RemoteExtractedTableCellHTMLElement:
    # html tag (lower case)
    tag: str
    start: int
    # end (exclusive)
    end: int
    # html attributes
    attrs: Dict[str, str]
    children: List['RemoteExtractedTableCellHTMLElement']

    def post_order(self):
        for c in self.children:
            for el in c.post_order():
                yield el
        yield self

    def clone(self):
        return RemoteExtractedTableCellHTMLElement(self.tag, self.start, self.end,
                                                   copy.copy(self.attrs), [c.clone() for c in self.children])


@dataclass
class RemoteExtractedTableCell:
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
    # for more details, look at the _extract_cell_recur function
    elements: List[RemoteExtractedTableCellHTMLElement]

    # original row & col span
    original_rowspan: Optional[int] = None
    original_colspan: Optional[int] = None

    def travel_elements_post_order(self):
        for el in self.elements:
            for pointer in el.post_order():
                yield pointer

    def clone(self):
        return RemoteExtractedTableCell(
            value=self.value,
            rowspan=self.rowspan,
            colspan=self.colspan,
            html=self.html,
            elements=[el.clone() for el in self.elements],
            original_rowspan=self.original_rowspan,
            original_colspan=self.original_colspan
        )


@dataclass
class RemoteExtractedTableRow:
    cells: List[RemoteExtractedTableCell]
    # html attributes of the tr elements
    attrs: Dict[str, str]


@dataclass
class ContextLevel:
    level: int
    header: str
    content: str

    def clone(self):
        return ContextLevel(self.level, self.header, self.content)


class OverlapSpanException(Exception):
    """Indicating the table has cell rowspan and cell colspan overlaps"""
    pass


class InvalidColumnSpanException(Exception):
    """Indicating that the column span is not used in a standard way. In particular, the total of columns' span is beyond the maximum number of columns is considered
    to be non standard with one exception that only the last column spans more than the maximum number of columns
    """
    pass


@dataclass
class RemoteExtractedTable:
    page_url: str
    # value of html caption
    caption: str
    # html attributes of the table html element
    attrs: Dict[str, str]
    # context
    context: List[ContextLevel]
    # list of rows in the table
    rows: List[RemoteExtractedTableRow]

    def span(self) -> "RemoteExtractedTable":
        """Span the table by copying values to merged field
        """
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

            data.append(RemoteExtractedTableRow(cells=new_row, attrs=copy.copy(row.attrs)))
            pi += 1

        # len(pending_ops) may > 0, but fortunately, it doesn't matter as the browser also does not render that extra empty lines
        return RemoteExtractedTable(
            page_url=self.page_url,
            caption=self.caption,
            attrs=copy.copy(self.attrs),
            context=[c.clone() for c in self.context],
            rows=data)

    def pad(self) -> "RemoteExtractedTable":
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
        default_cell = RemoteExtractedTableCell(
            value="", rowspan=1, colspan=1, html="", elements=[], original_rowspan=1, original_colspan=1
        )

        rows = []
        for r in self.rows:
            row = RemoteExtractedTableRow(cells=[c.clone() for c in r.cells], attrs=copy.copy(r.attrs))
            while len(row.cells) < max_ncols:
                row.cells.append(default_cell.clone())
            rows.append(row)

        return RemoteExtractedTable(
            page_url=self.page_url,
            caption=self.caption,
            attrs=copy.copy(self.attrs),
            context=[c.clone() for c in self.context],
            rows=rows)

    def as_df(self):
        return pd.DataFrame([
            [c.value for c in r.cells]
            for r in self.rows
        ])

    def as_relational_linked_table(self, table_id=None):
        assert len(self.rows) > 0
        header = [c.value for c in self.rows[0].cells]
        table = I.ColumnBasedTable(table_id or self.page_url, [
            I.Column(ci, cname, [self.rows[ri].cells[ci].value for ri in range(1, len(self.rows))])
            for ci, cname in enumerate(header)
        ])

        wikidb = Wikipedia2WikidataDB.get_instance(DATA_DIR / "enwiki_links.db", read_only=True)
        links = [
            [
                [
                    GI.Link(el.start, el.end, el.attrs['href'],
                            wikidb.get(self.get_title_from_url(el.attrs['href']), None))
                    for el in row.cells[ci].travel_elements_post_order()
                    if el.tag == 'a'
                ]
                for ci in range(len(header))
            ]
            for ri, row in enumerate(self.rows[1:])
        ]
        return GI.LinkedTable(table, GI.Context(), links)

    def get_title_from_url(self, url: str) -> str:
        """This function converts a wikipedia page/article's URL to its title. The function is tested manually in `20200425-wikipedia-links` notebook in section 2.2.

        Parameters
        ----------
        url : str
            a wikipedia page/article's URL

        Returns
        -------
        str
            a wikipedia page/article's title
        """
        path = urlparse(url).path
        if not path.startswith("/wiki/"):
            return ""

        assert path.startswith("/wiki/"), path
        path = path[6:]
        title = unquote_plus(path).replace("_", " ")
        return title.strip()


class InvalidCellSpanException(Exception):
    """Indicating that the html colspan or rowspan is wrong
    """
    pass


class HTMLTableExtractor:
    NUM_REGEX = re.compile("(\d+)")
    IGNORE_TAGS = set(["div", "span"])

    @dataclass
    class Header:
        level: int
        value: str

    @dataclass
    class Text:
        value: str

    """Extract tables from an html table, which may contains multiple nested tables. I assume that the outer tables are used for
    formatting and the inner tables are the interested ones. So this extractor will discard the outer tables and only keep
    tables at the bottom level.

    The main function of this extractor are extract
    """

    def extract(self, page_url: str, html: str) -> List[RemoteExtractedTable]:
        """Extract just one html table, which may contains multiple nested tables. I assume that the outer tables are used for
        formatting and the inner tables are the interested ones. So this function will discard the outer tables and only keep
        tables at the bottom level.

        This function right now ignore tables with invalid rowspan or cellspan
        """
        el = BeautifulSoup(html, "html5lib")
        table_els = el.find_all("table")
        results = []
        for table_el in table_els:
            temp_results = []
            self._extract_table(table_el, temp_results)
            if len(temp_results) > 0:
                context = self._locate_context(table_el)
                for r in temp_results:
                    r['context'] = context
                results += temp_results

        tables = [
            RemoteExtractedTable(page_url, **r)
            for r in results
        ]

        # convert relative links to absolute links
        parsed_resp = urlparse(page_url)
        domain = f"{parsed_resp.scheme}://{parsed_resp.netloc}"
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    for el in cell.travel_elements_post_order():
                        if el.tag == "a":
                            if 'href' in el.attrs:
                                href = el.attrs['href']
                                if href[0] == '/':
                                    el.attrs['href'] = domain + href

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
            c.find("table") is not None is not None for c in el.contents
        )

        for c in el.contents:
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
            for row_el in c.contents:
                if isinstance(row_el, NavigableString):
                    continue
                if row_el.name == "style":
                    continue

                assert row_el.name == "tr", f"Invalid tag: {row_el.name}"
                cells = []
                for cell_el in row_el.contents:
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
                    rows.append(RemoteExtractedTableRow(cells=cells, attrs=row_el.attrs))

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
        for c in el.children:
            if c.name == "table":
                tags.append(c)
            elif isinstance(c, NavigableString):
                continue
            elif len(c.contents) > 0:
                self._extract_tbl_tags(c, tags)

    def _extract_cell(self, el: Tag) -> RemoteExtractedTableCell:
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

        return RemoteExtractedTableCell(
            value=value,
            html=str(el),
            elements=elements,
            colspan=colspan,
            rowspan=rowspan,
        )

    def _extract_cell_recur(self, el: Tag, result: dict):
        """Real implementation of the _extract_cell function"""
        for c in el.children:
            c_name = c.name
            if c_name is None:
                result["text"] += c.strip()
                result["text"] += " "
            elif c_name == "br" or c_name == "hr":
                result["text"] += "\n"
            elif c_name in self.IGNORE_TAGS:
                self._extract_cell_recur(c, result)
            else:
                start = len(result["text"])
                children_result = {"text": result['text'], "elements": []}
                self._extract_cell_recur(c, children_result)
                # -1 for the trailing space
                if len(children_result['text']) > 0 and children_result['text'][-1] == " ":
                    if start == len(children_result['text']):
                        # no new text
                        start -= 1
                    end = len(children_result['text']) - 1
                else:
                    end = len(children_result['text'])
                assert end >= start
                result['text'] = children_result['text']
                result["elements"].append(RemoteExtractedTableCellHTMLElement(
                    tag=c_name, start=start, end=end, attrs=c.attrs, children=children_result['elements']
                ))

    def _locate_context_flatten_hierarchy(self, el):
        if el.name is None:
            # navigable string
            return [self.Text(str(el))]

        # flatten the tree to list of text? -> (header -> text)*
        if el.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            return [self.Header(level=int(el.name[1:]), value=el.get_text())]
        else:
            assert el.name == "hr" or not el.name.startswith("h")
            lst = []
            for c in el.contents:
                lst += self._locate_context_flatten_hierarchy(c)
            return lst

    def _locate_context_optimize_flatten_hierarchy(self, lst):
        if len(lst) == 0:
            return lst
        new_lst = []
        for item in lst:
            if isinstance(item, self.Header):
                new_lst.append(item)
            else:
                if len(new_lst) > 0 and isinstance(new_lst[-1], self.Text):
                    new_lst[-1].value += "\n" + item.value
                else:
                    new_lst.append(item)
        return new_lst

    def _locate_context_recur(self, el):
        if el.name in {'body', 'html'}:
            # hit the top
            return []

        parent = el.parent
        prev_sibs = []
        for i, e in enumerate(parent.contents):
            if e == el:
                # this is the index
                break
            prev_sibs.append(e)

        lst = []
        for sib in prev_sibs:
            lst += self._locate_context_flatten_hierarchy(sib)
        lst = self._locate_context_optimize_flatten_hierarchy(lst)
        return self._locate_context_optimize_flatten_hierarchy(self._locate_context_recur(el.parent) + lst)

    def _locate_context(self, el):
        """Locate the context of the elements"""
        context = self._locate_context_recur(el)
        # optimize the context
        optimized_context = []
        last_header = None
        if isinstance(context[-1], self.Header):
            last_header = ContextLevel(context[-1].level, header=context[-1].value, content="")
            context.pop()
        if isinstance(context[0], self.Text):
            assert isinstance(context[1], self.Header) and context[1].level == 1
            context.pop(0)
        else:
            assert isinstance(context[0], self.Header) and context[0].level == 1

        for i in range(0, len(context), 2):
            header, text = context[i], context[i + 1]
            optimized_context.append(ContextLevel(level=header.level, header=header.value, content=text.value))
        if last_header is not None:
            optimized_context.append(last_header)

        lst = [optimized_context[-1]]
        for i in range(len(optimized_context) - 2, -1, -1):
            if optimized_context[i].level < lst[-1].level:
                lst.append(optimized_context[i])
        return list(reversed(lst))

    def _postprocess_wikipedia(self, table: RemoteExtractedTable):
        for row in table.rows:
            for cell in row.cells:
                for el in cell.travel_elements_post_order():
                    if el.tag == "a":
                        if "href" not in el.attrs:
                            if 'selflink' in el.attrs['class']:
                                el.attrs['href'] = table.page_url
                            else:
                                raise Exception(f"An anchor in a Wikipedia table does not have href: {el}")


@M.cache_func()
def requests_get(url):
    return requests.get(url)


def fetch_tables(url: str, auto_span: bool = True, auto_pad: bool = True, cache: bool = False):
    if cache:
        resp = requests_get(url)
    else:
        resp = requests.get(url)

    assert resp.status_code == 200
    html = resp.text
    tables = HTMLTableExtractor().extract(url, html)
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


if __name__ == '__main__':
    # tables = fetch_tables("https://en.wikipedia.org/wiki/President_of_the_National_Council_(Austria)")
    # tables = fetch_tables("https://en.wikipedia.org/wiki/List_of_largest_selling_pharmaceutical_products")
    tables = fetch_tables("https://en.wikipedia.org/wiki/President_of_the_National_Council_(Austria)")
    tables[10].as_relational_linked_table()
    # print(tables[1].as_df())
    # M.serialize_json(tables[6].as_relational_linked_table().to_dict(),
    #                  "/workspace/sm-dev/grams/examples/misc/table_01.json")

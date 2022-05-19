from typing import Callable, Optional
import requests
from grams.html_table_parser.table_parser import HTMLTableParser


def fetch_tables(
    url: str,
    auto_span: bool = True,
    auto_pad: bool = True,
    fetch: Optional[Callable[[str], str]] = None,
):
    """Fetch tables from a webpage"""
    if fetch is None:
        fetch = default_fetch
    html = fetch(url)
    return HTMLTableParser(url, html).extract_tables(auto_span, auto_pad)


def default_fetch(url: str):
    resp = requests.get(url)
    assert resp.status_code == 200
    return resp.text

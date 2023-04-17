from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from grams.html_table_parser.context_extractor import ContextExtractor, Tree
from rsoup.python.models.context import Linebreak, Text


class E:
    @staticmethod
    def string(id, value):
        return Tree(id=id, value=value)

    @staticmethod
    def span(id, value, children=None):
        return Tree(id=id, value=value, tag="span", children=children or [])

    @staticmethod
    def h3(id, value, children=None):
        return Tree(id=id, value=value, block=True, tag="h3", children=children or [])

    @staticmethod
    def p(id, value, children=None):
        return Tree(id=id, value=value, block=True, tag="p", children=children or [])

    @staticmethod
    def b(id, value, children=None):
        return Tree(id=id, value=value, tag="b", children=children or [])

    @staticmethod
    def div(id, value, children=None):
        return Tree(id=id, value=value, block=True, tag="div", children=children or [])


def test_create_tree(html_files: dict):
    html = html_files["ContextExtractor_GetTree.html"]
    doc = BeautifulSoup(html, "html5lib")
    context_extractor = ContextExtractor(doc)

    el = doc.select_one("div#app")
    assert el is not None
    tree = context_extractor.get_tree(el)

    assert tree == E.div(
        0,
        "",
        [
            E.string(1, "\n      "),
            E.h3(2, "Hello world"),
            E.string(3, "\n      "),
            E.p(
                4,
                "",
                [
                    E.string(5, "This is a paragraph. "),
                    E.b(6, "This is bold text."),
                ],
            ),
            E.string(7, "\n    "),
        ],
    )


def test_optimize_flatten_tree(html_files: dict):
    html = html_files["ContextExtractor_GetTree.html"]
    doc = BeautifulSoup(html, "html5lib")
    context_extractor = ContextExtractor(doc)

    el = doc.select_one("div#app")
    assert el is not None
    tree = context_extractor.get_tree(el)
    trees = tree.fix_tree()
    context = [item for tree in trees for item in context_extractor.flatten_tree(tree)]
    assert [str(s) for s in context] == [
        "\n      ",
        "Hello world",
        "\n{1}",
        "\n      ",
        "This is a paragraph. ",
        "This is bold text.",
        "\n{1}",
        "\n    ",
    ]

    context1 = context_extractor.optimize_flatten_tree(context, merge_empty_lines=False)
    assert context1 == [
        Text(id="1", value="\n      ", tags=["div"]),
        Text(id="2", value="Hello world", tags=["h3", "div"]),
        Linebreak(n_lines=1),
        Text(id="3", value="\n      ", tags=["div"]),
        Text(id="5", value="This is a paragraph. ", tags=["p", "div"]),
        Text(id="6", value="This is bold text.", tags=["b", "p", "div"]),
        Linebreak(n_lines=1),
        Text(id="7", value="\n    ", tags=["div"]),
    ]

    context2 = context_extractor.optimize_flatten_tree(context)
    assert context2 == [
        Linebreak(n_lines=1),
        Text(id="2", value="Hello world", tags=["h3", "div"]),
        Linebreak(n_lines=2),
        Text(id="5", value="This is a paragraph. ", tags=["p", "div"]),
        Text(id="6", value="This is bold text.", tags=["b", "p", "div"]),
        Linebreak(n_lines=2),
    ]

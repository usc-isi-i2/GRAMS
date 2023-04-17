from __future__ import annotations
import copy
from operator import attrgetter
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from dataclasses import dataclass, field
from rsoup.python.models.context import Attribute, Text, Linebreak, ContentHierarchy


@dataclass
class Tree:
    id: int
    value: str = ""
    block: bool = False
    tag: Optional[str] = None
    # attrs is empty when it's a block
    attrs: Attribute = field(default_factory=dict)  # type: ignore
    children: List[Tree] = field(default_factory=list)
    # telling if the tree is verified that the structure is correct (inline only contains inline)
    is_verified: bool = False

    def descendants(self) -> Iterable[Tree]:
        yield self
        for c in self.children:
            yield from c.descendants()

    def clone_without_children(self) -> Tree:
        return Tree(
            id=self.id,
            value=self.value,
            block=self.block,
            tag=self.tag,
            attrs=copy.copy(self.attrs),
            children=[],
            is_verified=self.is_verified,
        )

    def fix_tree(self) -> List[Tree]:
        """An inline element can't contain any block element. if it happens,
        we need to split it
        """
        # no children, no need to fix
        if len(self.children) == 0:
            newself = self.clone_without_children()
            newself.is_verified = True
            return [newself]

        if self.block:
            newself = self.clone_without_children()
            children = []
            for c in self.children:
                children += c.fix_tree()
            newself.children = children
            newself.is_verified = True
            return [newself]

        # this tree is an inline element,
        trees: List[Tree] = [self.clone_without_children()]
        trees[-1].is_verified = True

        for c in self.children:
            subtrees = c.fix_tree()
            if c.block:
                trees += subtrees
            else:
                for subtree in subtrees:
                    if subtree.block:
                        trees += subtree.fix_tree()
                    else:
                        if trees[-1].block:
                            # parent of this subtree should still be the current node
                            trees.append(
                                Tree(
                                    id=self.id,
                                    value="",
                                    block=False,
                                    tag=self.tag,
                                    attrs=self.attrs,
                                    children=[],
                                    is_verified=True,
                                )
                            )
                        trees[-1].children.append(subtree)

        return trees


PageElement = Union[Tag, NavigableString]


class ContextExtractor:
    """Extracting context that leads to an element in an HTML page

    Assuming that the page follows tree structure. Each header element
    represents a level (section) in the tree.

    This extractor tries to does it best to detect which text should be kept in the same line
    and which one is not. However, it does not take into account the style of element (display: block)
    and hence has to rely on some heuristics. For example, <canvas> is an inline element, however, it
    is often used as block element so this extractor put it in another line.
    """

    # list of inline elements that will be rendered in same line except <br> tags
    # https://developer.mozilla.org/en-US/docs/Web/HTML/Inline_elements
    # fmt: off
    INLINE_ELEMENTS = {
        "a", "abbr", "acronym", "audio", "b",
        "bdi", "bdo", "big", "button", "cite", "canvas",
        "code", "data", "datalist", "del", "dfn", "em",
        "embed", "i", "iframe", "img", "input", "ins",
        "kbd", "label", "map", "mark", "meter",
        "object", "output", "picture", "progress", "q",
        "ruby", "s", "samp", "select", "slot",
        "small", "span", "strong", "sub", "sup", "svg", "template", 
        "textarea", "time", "u", "tt", "var", "video", "wbr"
    }
    BLOCK_ELEMENTS = {
        "body", "br", "address", "article", "aside", 
        "blockquote", "details", "dialog", "dd", "div", 
        "dl", "dt", "fieldset", "figcaption", "figure", 
        "footer", "form", "h1", "h2", "h3", "h4", "h5", 
        "h6", "header", "hgroup", "hr", "li", "main", 
        "nav", "ol", "p", "pre", "section", "table", "ul"
    }
    IGNORE_ELEMENTS = {"script", "style", "noscript"}
    SKIP_CONTENT_BLOCK_ELEMENTS = {"table"}
    SAME_CONTENT_LEVEL_ELEMENTS = {"table", "h1", "h2", "h3", "h4", "h5", "h6"}
    HEADER_ELEMENTS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    RICH_ELEMENTS = {"a", "b", "i", "h1", "h2", "h3", "h4", "h5", "h6"}
    # fmt: on

    def __init__(self, doc: BeautifulSoup):
        self.doc = doc

    def extract(self, el: PageElement) -> List[ContentHierarchy]:
        """Extract context tree that leads to the given element"""
        # travel up the tree to find the the parents
        tree_before, tree_after = self.locate_content_before_and_after(el)
        if tree_before is None:
            context_before = []
        else:
            tree = self.get_tree(tree_before)
            # fix errors in the tree, so that inline element only contains inline elements
            trees = tree.fix_tree()
            context_before = [
                item for tree in trees for item in self.flatten_tree(tree)
            ]
            context_before = self.optimize_flatten_tree(context_before)

        if tree_after is None:
            context_after = []
        else:
            tree = self.get_tree(tree_after)
            trees = tree.fix_tree()
            context_after = [item for tree in trees for item in self.flatten_tree(tree)]
            context_after = self.optimize_flatten_tree(context_after)

        context = [ContentHierarchy(level=0, heading="")]
        i = 0
        while i < len(context_before):
            c = context_before[i]
            if isinstance(c, Linebreak):
                context[-1].content_before.append(c)
                i += 1
                continue
            headers = self.HEADER_ELEMENTS.intersection(c.tags)
            if len(headers) == 0:
                context[-1].content_before.append(c)
                i += 1
                continue

            assert len(headers) == 1
            header = list(headers)[0]
            context.append(ContentHierarchy(level=int(header[1:]), heading=c.value))
            i += 1
            while i < len(context_before):
                # now if the next one is not line break and is the same header, we can merge them
                nc = context_before[i]
                if isinstance(nc, Linebreak):
                    break
                assert self.HEADER_ELEMENTS.intersection(nc.tags) == headers
                context[-1].heading += nc.value
                i += 1

        # we do another filter to make sure the content is related to the element
        # that the header leading to this element must be increasing
        rev_context = []
        header = 10
        for i in range(len(context) - 1, -1, -1):
            if context[i].level < header:
                rev_context.append(context[i])
                header = context[i].level
        context = list(reversed(rev_context))
        context[-1].content_after = context_after

        return context

    def locate_content_before_and_after(
        self, element: PageElement
    ) -> Tuple[Optional[Tag], Optional[Tag]]:
        """Finding surrounding content of the element.

        Assuming elements in the document is rendered from top to bottom and
        left to right. In other words, there is no CSS that do float right/left
        to make pre/after elements to be appeared out of order.

        Currently, (the logic is not good)
            * to determine the content before the element, we just keep all elements rendered
        before this element (we are doing another filter outside of this function in `self.extract`).
            * to determine the content after the element, we consider only the siblings
        and stop before they hit a block element (not all block elements) that may be in the same level such as table, etc.
        """
        tree_before = None
        tree_after = None

        el = element

        while el.parent is not None and el.parent.name != "html":
            parent = self.copy_node_without_children(el.parent)
            for e in el.parent.contents:
                if e is el:
                    # this is the index
                    break
                parent.append(copy.copy(e))
            if tree_before is not None:
                parent.append(tree_before)
            tree_before = parent
            el = el.parent

        el = element
        if el.parent is not None:
            for i, e in enumerate(el.parent.contents):
                if e is el:
                    if i < len(el.parent.contents) - 1:
                        tree_after = self.copy_node_without_children(el.parent)
                        for e2 in el.parent.contents[i + 1 :]:
                            if (
                                isinstance(e2, Tag)
                                and e2.name in self.SAME_CONTENT_LEVEL_ELEMENTS
                            ):
                                break
                            tree_after.append(copy.copy(e2))
                    break

        assert tree_before is not None and isinstance(tree_before, Tag)
        return tree_before, tree_after

    def get_tree(
        self,
        el: PageElement,
        id_container: Optional[dict] = None,
    ) -> Tree:
        """Convert element to tree"""
        if id_container is None:
            id_container = {"id": -1}

        if isinstance(el, NavigableString):
            id_container["id"] += 1
            return Tree(id=id_container["id"], value=el.get_text())

        id_container["id"] += 1
        tree_id = id_container["id"]
        value = ""

        if el.name in self.SKIP_CONTENT_BLOCK_ELEMENTS:
            children = []
        else:
            children = [
                self.get_tree(c, id_container)
                for c in cast(List[PageElement], el.contents)
                if isinstance(c, NavigableString) or c.name not in self.IGNORE_ELEMENTS
            ]
            if len(children) == 1 and children[0].tag is None:
                # one child and it's a string, so we undo it to reduce the tree depth
                value = children[0].value
                children = []
                id_container["id"] -= 1

        if el.name in self.RICH_ELEMENTS:
            attrs = self.extract_rich_text_attrs(el)
        else:
            attrs: Attribute = {}

        return Tree(
            id=tree_id,
            value=value,
            block=el.name in self.BLOCK_ELEMENTS,
            tag=el.name,
            attrs=attrs,
            children=children,
        )

    def flatten_tree(self, tree: Tree) -> List[Union[Text, Linebreak]]:
        """Assuming that the tree is already fixed"""
        assert tree.is_verified
        output = []
        if tree.block:
            assert tree.tag is not None
            if tree.value != "":
                assert len(tree.attrs) == 0
                output.append(
                    Text(
                        id=str(tree.id),
                        value=tree.value,
                        tags=[tree.tag],
                        id2attrs={},
                    )
                )

            for c in tree.children:
                for sc in self.flatten_tree(c):
                    if isinstance(sc, Text):
                        sc.tags.append(tree.tag)
                    output.append(sc)
                if c.block:
                    output.append(Linebreak())

            # remove the last line break
            if len(output) > 0 and isinstance(output[-1], Linebreak):
                output.pop()
            return output

        # inline element
        if len(tree.children) == 0:
            tags = []
            id2attrs = {}

            if tree.tag is not None:
                tags.append(tree.tag)
                if len(tree.attrs) > 0:
                    id2attrs[str(tree.id)] = tree.attrs
            return [
                Text(id=str(tree.id), value=tree.value, tags=tags, id2attrs=id2attrs)
            ]

        assert tree.tag is not None
        for c in tree.children:
            for subc in self.flatten_tree(c):
                assert isinstance(subc, Text)
                subc = copy.deepcopy(subc)
                subc.tags.append(tree.tag)
                if len(tree.attrs) > 0:
                    subc.id2attrs[str(tree.id)] = tree.attrs
                output.append(subc)
        return output

    def optimize_flatten_tree(
        self, lst: List[Union[Text, Linebreak]], merge_empty_lines: bool = True
    ) -> List[Union[Text, Linebreak]]:
        """Merge consecutive elements to make the list more compact"""
        if len(lst) == 0:
            return lst

        if (
            merge_empty_lines
            and isinstance(lst[0], Text)
            and lst[0].value.strip() == ""
        ):
            new_lst: List[Union[Text, Linebreak]] = [
                Linebreak(lst[0].value.count("\n"))
            ]
        else:
            new_lst: List[Union[Text, Linebreak]] = [lst[0]]

        for i in range(1, len(lst)):
            last_item = new_lst[-1]
            item = lst[i]
            if (
                merge_empty_lines
                and isinstance(item, Text)
                and item.value.strip() == ""
            ):
                item = Linebreak(item.value.count("\n"))

            if isinstance(item, Linebreak):
                if isinstance(last_item, Linebreak):
                    last_item.n_lines += item.n_lines
                else:
                    new_lst.append(item)
            else:
                if isinstance(last_item, Linebreak):
                    new_lst.append(item)
                else:
                    # same text, merge them if they share the same tags and non empty id2attrs
                    mergable = (
                        item.tags == last_item.tags and item.id2attrs == last_item.id
                    )
                    if mergable:
                        last_item.value += item.value
                    else:
                        new_lst.append(item)
        return new_lst

    def extract_rich_text_attrs(self, e: Tag) -> Attribute:
        if e.name == "a":
            return {"href": e.attrs.get("href", "")}
        return {}

    def copy_node_without_children(self, node: Tag) -> Tag:
        contents = node.contents
        node.contents = []
        newnode = copy.copy(node)
        node.contents = contents
        return newnode

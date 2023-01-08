from __future__ import annotations
from typing import Mapping
from grams.algorithm.candidate_graph.cg_graph import CGGraph
from kgdata.wikidata.models import WDEntity


from osin.types.pyobject import OHTML, OTable
from osin.types.pyobject.html import OListHTML
from sm.namespaces.namespace import OutOfNamespace
from sm.namespaces.wikidata import WikidataNamespace
from sm.outputs.semantic_model import (
    SemanticModel,
    ClassNode,
    LiteralNode,
    Edge,
    LiteralNodeDataType,
)

from grams.inputs.linked_table import LinkedTable
from sm.dataset import Example


class AuxComplexObjectBase:
    def __init__(self, wdentities: Mapping[str, WDEntity]) -> None:
        self.wdns = WikidataNamespace.create()
        self.wdentities = wdentities

    def get_ent_label(self, entid: str):
        if entid not in self.wdentities:
            return entid
        return f"{self.wdentities[entid].label} ({entid})"


class AuxComplexTableObject(AuxComplexObjectBase):
    def get_table(self, example: Example[LinkedTable]) -> OTable:
        nrows, ncols = example.table.table.shape()
        columns = [
            f"{ci}. " + (c.clean_name or "")
            for ci, c in enumerate(example.table.table.columns)
        ]
        return OTable(
            [
                {columns[ci]: self._get_cell(example, ri, ci) for ci in range(ncols)}
                for ri in range(nrows)
            ]
        )

    def _get_cell(self, example: Example[LinkedTable], row: int, col: int):
        from htbuilder import div, ul, li, a, b, span, pre, code

        links = example.table.links[row, col]
        if len(links) == 0:
            return str(example.table.table[row, col])

        cell = str(example.table.table[row, col])
        ent_el = []
        can_el = []
        for link in links:
            x = div(style="margin-left: 8px")(
                "-&nbsp;",
                a(href=link.url)(f"{cell[link.start:link.end]}: "),
                *[
                    span(
                        ", " if i > 0 else "",
                        a(href=self.wdns.get_entity_abs_uri(entid))(entid),
                    )
                    for i, entid in enumerate(link.entities)
                ],
            )
            popx = div(self.get_ent_label(entid) for entid in link.entities)
            ent_el.append(OHTML(str(x), str(popx)))

            gold_ents = set(link.entities)

            # showing the top 5 candidates, as longer won't give more info.
            discans = [(i, can) for i, can in enumerate(link.candidates)][:5]
            if all(can.entity_id not in gold_ents for _, can in discans):
                # add missing pred gold
                discans.extend(
                    [
                        (i, can)
                        for i, can in enumerate(link.candidates)
                        if can.entity_id in gold_ents
                    ]
                )

            for i, canid in discans:
                x = div(style="margin-left: 8px")(
                    f"- {i}.&nbsp;",
                    span(
                        "(",
                        code(style="color: green; font-weight: bold")("C"),
                        ")&nbsp;",
                    )
                    if canid.entity_id in gold_ents
                    else "",
                    a(href=self.wdns.get_entity_abs_uri(canid.entity_id))(
                        canid.entity_id
                    ),
                    ": ",
                    span(title=canid.probability)(round(canid.probability, 4)),
                )
                popx = self.get_ent_label(canid.entity_id)
                can_el.append(OHTML(str(x), str(popx)))

        items = [
            OHTML(
                str(
                    span(
                        b("Cell: "),
                        cell,
                    )
                ),
            ),
            OHTML(str(b("Ground-truth: "))),
            *ent_el,
            OHTML(str(b("Candidates: "))),
            *can_el,
            # OHTML(
            #     str(
            #         div(
            #             *[b("Candidates: "), ul(can_el)] if len(can_el) > 0 else [],
            #         )
            #     )
            # ),
        ]
        return OListHTML(items)


class AuxComplexSmObject(AuxComplexObjectBase):
    def get_sm(self, example: Example[LinkedTable], sm: SemanticModel) -> OHTML:
        html = (
            sm.deep_copy()
            .add_readable_label(self.update_readable_label)
            .print(env="browser")
        )
        assert html is not None
        return OHTML(html)

    def get_sms(self, example: Example[LinkedTable], sms: list[SemanticModel]) -> OHTML:
        from htbuilder import div, b

        htmls = []
        for i, sm in enumerate(sms):
            html = (
                sm.deep_copy()
                .add_readable_label(self.update_readable_label)
                .print(env="browser")
            )
            assert html is not None
            htmls.append(b(f"Semantic Model {i}:"))
            htmls.append(html)

        return OHTML(str(div(*htmls)))

    def update_readable_label(self, item: ClassNode | LiteralNode | Edge):
        if isinstance(item, (ClassNode, Edge)):
            abs_uri = item.abs_uri
        elif (
            isinstance(item, LiteralNode)
            and item.datatype == LiteralNodeDataType.Entity
        ):
            abs_uri = item.value
        else:
            return

        try:
            entid = self.wdns.get_entity_id(abs_uri)
        except OutOfNamespace:
            try:
                entid = self.wdns.get_prop_id(abs_uri)
            except OutOfNamespace:
                return

        if entid in self.wdentities:
            item.readable_label = self.get_ent_label(entid)


# class AuxComplexCandidateGraph(AuxComplexObjectBase):
#     def get_graph(self, example: Example[LinkedTable], cg: CGGraph) -> OHTML:
#         rows = []
#         for edge in cg.iter_edges():
#             rows.append({"source": edge.source, "target": edge.target})
#         return OHTML(rows)

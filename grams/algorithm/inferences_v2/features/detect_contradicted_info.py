from __future__ import annotations

from grams.algorithm.context import AlgoContext
from grams.algorithm.data_graph.dg_graph import (
    EntityValueNode,
    LiteralValueNode,
)
from grams.algorithm.inferences_v2.features.detect_missing_info import (
    MissingInformationDetector,
)
from grams.algorithm.inferences_v2.features.graph_traversal_helper import (
    GraphTraversalHelper,
)

from grams.algorithm.data_graph import CellNode
from grams.algorithm.candidate_graph.cg_graph import (
    CGEdge,
    CGStatementNode,
)
from kgdata.wikidata.models import (
    WDEntity,
)
from kgdata.wikidata.models.wdvalue import WDValue, WDValueEntityId
from sm.misc.fn_cache import CacheMethod
from dataclasses import dataclass
from loguru import logger


@dataclass
class ContradictedInformation:
    __slots__ = (
        "row",
        "source_ents",
        "inedge_predicate",
        "outedge_predicate",
        "target_values",
    )
    row: int
    source_ents: list[WDEntity]
    inedge_predicate: str
    outedge_predicate: str
    target_values: str | list[str] | WDValue


class ContradictedInformationDetector:
    def __init__(
        self,
        correct_entity_threshold: float,
        traversal: GraphTraversalHelper,
        context: AlgoContext,
    ):
        self.correct_entity_threshold = correct_entity_threshold
        self.wdentities = context.wdentities
        self.wdprops = context.wdprops
        self.cg = traversal.cg
        self.dg = traversal.dg
        self.traversal = traversal
        self.missing_info_detector = MissingInformationDetector(context)

    @CacheMethod.cache(CacheMethod.three_object_args)
    def get_contradicted_information(
        self, s: CGStatementNode, inedge: CGEdge, outedge: CGEdge
    ) -> list[ContradictedInformation]:
        """Get the DG pairs that may contain contradicted information with the relationship inedge -> s -> outedge

        A pair that contain contradicted information when both:
        (1) It does not found in data graph
        (2) The n-ary relationships (property and (optionally) qualifier) exist in the entity.

        Because of (1), the (2) says the value in KG is different from the value in the table. However, we need to be
        careful because of missing values. To combat this, we need to distinguish when we actually have missing values.

        Also, we need to be careful to not use entities that the threshold is way too small (e.g., the cell is Bob but the candidate
        is John).

        For detecting missing values, we can use some information such as if the relationship has exactly one value, or
        we can try to detect if we can find the information in some pages.
        """
        u = self.cg.get_node(inedge.source)
        uv_links = self.traversal.get_rel_dg_pairs(s, outedge)

        inedge_predicate = inedge.predicate
        outedge_predicate = outedge.predicate

        is_outpred_qualifier = inedge.predicate != outedge.predicate
        is_outpred_data_predicate = self.wdprops[outedge.predicate].is_data_property()

        contradicted_info = []

        for dgu, dgv in self.traversal.iter_dg_pair(inedge.source, outedge.target):
            if (dgu.id, dgv.id) in uv_links:
                continue

            row = -1
            if isinstance(dgu, CellNode):
                row = dgu.row
                dgu_ents = [
                    self.wdentities[eid]
                    for eid in dgu.entity_ids
                    if dgu.entity_probs[eid] >= self.correct_entity_threshold
                ]
            else:
                assert isinstance(dgu, EntityValueNode)
                if dgu.qnode_prob >= self.correct_entity_threshold:
                    dgu_ents = [self.wdentities[dgu.qnode_id]]
                else:
                    dgu_ents = []

            do_not_have_info = (
                (len(dgu_ents) == 0)
                or all(inedge_predicate not in ent.props for ent in dgu_ents)
                or (
                    is_outpred_qualifier
                    and all(
                        outedge_predicate not in stmt.qualifiers
                        for ent in dgu_ents
                        for stmt in ent.props.get(inedge_predicate, [])
                    )
                )
            )
            if do_not_have_info:
                continue

            if is_outpred_qualifier:
                dgu_ents = [
                    ent
                    for ent in dgu_ents
                    if inedge_predicate in ent.props
                    and any(
                        outedge_predicate in stmt.qualifiers
                        for stmt in ent.props[inedge_predicate]
                    )
                ]
            else:
                dgu_ents = [ent for ent in dgu_ents if inedge_predicate in ent.props]

            # at this moment, it should have some info. different than the one in the table
            # but it can be due to missing values, so we check it here.
            if is_outpred_data_predicate:
                has_contradicted_info = True
                if isinstance(dgv, CellNode):
                    dgv_value = dgv.value
                    row = dgv.row
                elif isinstance(dgv, LiteralValueNode):
                    if WDValue.is_string(dgv.value):
                        dgv_value = dgv.value.value
                    elif WDValue.is_quantity(dgv.value):
                        dgv_value = dgv.value.value["amount"]
                    elif WDValue.is_mono_lingual_text(dgv.value):
                        dgv_value = dgv.value.value["text"]
                    else:
                        dgv_value = dgv.value
                else:
                    # we do have this case that data predicate such as P2561
                    # that values link to entity value, we do not handle it for now so
                    # we set the value to WDValue so it is skipped
                    assert isinstance(dgv, EntityValueNode)
                    dgv_value = WDValueEntityId(
                        "wikibase-entityid",
                        {
                            "entity-type": "item",
                            "id": dgv.qnode_id,
                            "numeric-id": int(dgv.qnode_id[1:]),
                        },
                    )

                if not isinstance(dgv_value, WDValue):
                    for dgu_ent in dgu_ents:
                        if self.missing_info_detector.is_missing_data_info(
                            dgu_ent, inedge_predicate, outedge_predicate, dgv_value
                        ):
                            has_contradicted_info = False
                            break

                if has_contradicted_info:
                    contradicted_info.append(
                        ContradictedInformation(
                            row=row,
                            source_ents=dgu_ents,
                            inedge_predicate=inedge_predicate,
                            outedge_predicate=outedge_predicate,
                            target_values=dgv_value,
                        )
                    )
            else:
                # object property, check an external db
                if isinstance(dgv, CellNode):
                    row = dgv.row
                    dgv_ent_ids = [
                        eid
                        for eid in dgv.entity_ids
                        if dgv.entity_probs[eid] >= self.correct_entity_threshold
                    ]
                elif isinstance(dgv, LiteralValueNode):
                    # this can happens due to inconsistency in the KG.
                    logger.warning(
                        "Found a literal value {} for an object property {} -> {} in one of the entities: {}",
                        str(dgv),
                        inedge_predicate,
                        outedge_predicate,
                        ",".join([f"{e.label} ({e.id})" for e in dgu_ents]),
                    )
                    continue
                else:
                    # it throws error in a table -- check it out.
                    assert isinstance(dgv, EntityValueNode)
                    if dgv.qnode_prob >= self.correct_entity_threshold:
                        dgv_ent_ids = [dgv.qnode_id]
                    else:
                        dgv_ent_ids = []

                # this is similar to do_not_have_info condition above, but we can only check this if
                # is_outpred_qualifier is False
                if len(dgv_ent_ids) == 0:
                    continue

                has_contradicted_info = True
                for dgu_ent in dgu_ents:
                    # any here because it's like discover one potential link between any candidates
                    if any(
                        self.missing_info_detector.is_missing_object_info(
                            dgu_ent, inedge_predicate, outedge_predicate, dgv_ent_id
                        )
                        for dgv_ent_id in dgv_ent_ids
                    ):
                        has_contradicted_info = False
                if has_contradicted_info:
                    contradicted_info.append(
                        ContradictedInformation(
                            row=row,
                            source_ents=dgu_ents,
                            inedge_predicate=inedge_predicate,
                            outedge_predicate=outedge_predicate,
                            target_values=dgv_ent_ids,
                        )
                    )

        return contradicted_info

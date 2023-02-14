from __future__ import annotations
from grams.algorithm.context import AlgoContext
from kgdata.wikidata.models.wdentity import WDEntity
from kgdata.wikidata.models.wdvalue import WDValueKind


class MissingInformationDetector:
    def __init__(self, context: AlgoContext):
        self.context = context
        self.wdentity_wikilinks = context.wdentity_wikilinks

    def is_missing_object_info(
        self, ent: WDEntity, property: str, qualifier: str, value: str
    ) -> bool:
        # found two entities some where on the internet
        # so it may be a missing information
        if ent.id in self.wdentity_wikilinks:
            return value in self.wdentity_wikilinks[ent.id].targets
        return False

    def is_missing_data_info(
        self, ent: WDEntity, property: str, qualifier: str, value: str
    ) -> bool:
        # TODO: implement this via infobox search
        return False

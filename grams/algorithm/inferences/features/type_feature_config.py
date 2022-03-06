from typing import Literal


class TypeFeatureConfigs:
    """Holding configuration for type feature extraction"""

    # methods for fixing cycles in KG's ontology classes
    # cycles rarely happen so the recommended method is to fix them manually
    # for example, in Wikidata 2020, we found only 9 cycles out of millions of classes
    # set it to "auto" if you want to automatically fix cycles
    FIX_TYPE_CYCLE: Literal["auto", "manually"] = "manually"

from __future__ import annotations

from typing import Optional

from grams.core import AlgoContext
from grams.core.table import LinkedTable
import grams.core.steps.data_matching as data_matching

class CandidateLocalSearchConfig:
    strsim: str
    threshold: float
    use_column_name: bool
    use_language: Optional[str]
    search_all_columns: bool

    def __init__(
        self,
        strsim: str,
        threshold: float,
        use_column_name: bool,
        use_language: Optional[str],
        search_all_columns: bool,
    ) -> None: ...

def candidate_local_search(
    table: LinkedTable, context: AlgoContext, cfg: CandidateLocalSearchConfig
) -> LinkedTable: ...

__all__ = ["CandidateLocalSearchConfig", "candidate_local_search", "data_matching"]

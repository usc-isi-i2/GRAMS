from typing import Optional
from grams.core import AlgoContext
from grams.core.table import LinkedTable

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

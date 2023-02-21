from grams.core import AlgoContext
from grams.core.table import LinkedTable

class AugCanConfig:
    strsim: str
    threshold: float
    use_column_name: bool

    def __init__(
        self, strsim: str, threshold: float, use_column_name: bool
    ) -> None: ...

def augment_candidates(
    table: LinkedTable, context: AlgoContext, cfg: AugCanConfig
) -> LinkedTable: ...

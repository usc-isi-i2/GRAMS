from grams.core.table import LinkedTable
import grams.core.literal_matchers as literal_matchers
import grams.core.steps as steps
from kgdata.core.models import StatementView, Value as KGValue

class GramsDB:
    @staticmethod
    def init(datadir: str) -> None: ...
    @staticmethod
    def get_instance() -> GramsDB: ...
    def get_algo_context(self, table: LinkedTable, n_hop: int) -> AlgoContext: ...

class AlgoContext:
    def get_entity_statement(
        self, entity_id: str, prop: str, stmt_index: int
    ) -> StatementView: ...

Value = KGValue

__all__ = ["GramsDB", "AlgoContext", "literal_matchers", "steps", "Value"]

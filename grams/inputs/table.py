from collections import OrderedDict, defaultdict
from typing import List, Optional

import pandas as pd

from grams.inputs.column import Column


class TableMetadata:
    def __init__(self, table_id: str, page_title: str, table_name: str, text_before: str, text_after: str):
        self.table_id = table_id
        self.page_title = page_title
        self.table_name = table_name
        self.text_before = text_before
        self.text_after = text_after


class ColumnBasedTable:
    def __init__(self, columns: List[Column], metadata: TableMetadata):
        self.columns = columns
        self.metadata = metadata
        self.df = self.as_dataframe()

    def get_column_by_index(self, col_idx: int) -> Column:
        (col,) = [col for col in self.columns if col.index == col_idx]
        return col

    def as_dataframe(self) -> pd.DataFrame:
        d = OrderedDict()
        dup = defaultdict(int)
        for i, col in enumerate(self.columns):
            if col.name is None:
                cname = f"unk_{i:02}"
            else:
                cname = col.name

            if cname in d:
                dup[cname] += 1
                cname += f" ({dup[cname]})"
            d[cname] = col.values
        return pd.DataFrame(d)

    def subset(self, start_row: int, end_row: int):
        return ColumnBasedTable(
            [Column(c.index, c.name, c.values[start_row:end_row]) for c in self.columns],
            self.metadata
        )

    def to_json(self):
        return {
            "columns": [col.__dict__ for col in self.columns],
            "metadata": self.metadata.__dict__
        }

    @staticmethod
    def from_json(record: dict):
        return ColumnBasedTable([Column(**col) for col in record['columns']], TableMetadata(**record['metadata']))

    @staticmethod
    def from_dataframe(df: pd.DataFrame, table_metadata: Optional[TableMetadata] = None):
        if table_metadata is None:
            table_metadata = TableMetadata("", "", "", "", "")
        columns = []
        for ci, c in enumerate(df.columns):
            values = [r[ci] for ri, r in df.iterrows()]
            column = Column(ci, c, values)
            columns.append(column)
        return ColumnBasedTable(columns, table_metadata)


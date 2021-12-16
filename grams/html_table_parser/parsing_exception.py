class InvalidCellSpanException(Exception):
    """Indicating that the html colspan or rowspan is wrong"""

    pass


class OverlapSpanException(Exception):
    """Indicating the table has cell rowspan and cell colspan overlaps"""

    pass


class InvalidColumnSpanException(Exception):
    """Indicating that the column span is not used in a standard way. In particular, the total of columns' span is beyond the maximum number of columns is considered
    to be non standard with one exception that only the last column spans more than the maximum number of columns
    """

    pass

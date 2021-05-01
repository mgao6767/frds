# from .data_manager import DataManager

# from .dataset import Dataset
# from .datascope.trth import Connection as TRTHConnection
"""Define the base structure of a dataset"""
from dataclasses import dataclass
from typing import List


@dataclass
class Dataset:
    """The base class of a dataset"""

    source: str
    library: str
    table: str
    vars: List[str]
    date_vars: List[str]

    def __hash__(self):
        return id(self)

    @property
    def table_id(self):
        """Return the tuple of (source, library, table)

        Returns
        -------
        TableID
            (source, library, table)
        """
        return (self.source, self.library, self.table)

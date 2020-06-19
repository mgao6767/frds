"""Define the base structure of a dataset"""
from dataclasses import dataclass
from typing import List
from frds.typing import TableID


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
    def table_id(self) -> TableID:
        """Return the tuple of (source, library, table)

        Returns
        -------
        TableID
            (source, library, table)
        """
        return TableID((self.source, self.library, self.table))

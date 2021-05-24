from abc import ABC
import os
from typing import Optional, Union, List

from frds.data import DATA_DIRECTORY


class WRDSDataset(ABC):
    source = "wrds"
    library: str
    table: str
    index_col: Union[str, List[str]]
    date_cols: Optional[Union[str, List[str]]]

    @classmethod
    @property
    def local_path(cls) -> str:
        """Full path to the local Sqlite dataset"""
        stata_file_name = cls.table + ".db"
        lib_dir = os.path.join(DATA_DIRECTORY, cls.source, cls.library)
        os.makedirs(lib_dir, exist_ok=True)
        return os.path.join(lib_dir, stata_file_name)

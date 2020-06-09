from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass
class Dataset:
    source: str
    library: str
    table: str
    vars: List[str]
    date_vars: List[str]
    start_date: datetime = datetime.today() - timedelta(days=365)
    end_date: datetime = datetime.today()

    def __hash__(self):
        return id(self)

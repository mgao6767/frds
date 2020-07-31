from typing import List
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure

NAME = "StockDelisting"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="crsp",
        table="dse",
        vars=["date", "permno", "permco", "dlstcd", "event"],
        date_vars=["date"],
    )
]
VARIABLE_LABELS = {}


class StockDelisting(Measure):
    def __init__(self):
        super().__init__(NAME, DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):

        dse = pd.DataFrame.from_records(nparrays[0])

        cond = np.in1d(dse.event, ["DELIST"]) & (
            ((500 <= dse.dlstcd) & (dse.dlstcd <= 599))
            | ((200 <= dse.dlstcd) & (dse.dlstcd <= 299))
        )

        return dse[cond], VARIABLE_LABELS

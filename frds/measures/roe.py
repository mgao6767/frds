from typing import List, Tuple, Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda


NAME = "ROE"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "ceq",
            "ib",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABELS = {
    NAME: "Income Before Extraordinary Items scaled by Common Equity (Total)"
}


class ROE(Measure):

    def __init__(self):
        super().__init__("ROE", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        nparray = filter_funda(nparrays[0])
        roa = np.true_divide(nparray.ib, nparray.ceq, where=(nparray.ceq != 0))
        roa[np.isnan(nparray.ceq)] = np.nan
        nparray = rfn.rec_append_fields(nparray, NAME, roa)
        # keep only useful columns
        cols = set(rfn.get_names_flat(nparray.dtype))
        nparray.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "ib", "ceq", NAME])
        return (
            pd.DataFrame.from_records(nparray, exclude=exclude_cols),
            VARIABLE_LABELS,
        )

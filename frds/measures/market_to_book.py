from typing import List, Tuple, Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

NAME = "MarketToBookRatio"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "csho",
            "prcc_f",
            "ceq",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABELS = {NAME: "(PRCC_F*CSHO)/CEQ"}


class MarketToBookRatio(Measure):
    """Market to book ratio
        common shares outstanding * share price at fiscal year end
     =  ----------------------------------------------------------
        book value common equity
    """

    def __init__(self):
        super().__init__("Market to Book Ratio", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        nparray = filter_funda(nparrays[0])
        # market value at fiscal year
        mv = nparray.prcc_f * nparray.csho
        # market-to-book = market value of equity / common equity
        mtb = np.true_divide(mv, nparray.ceq, where=(nparray.ceq != 0))
        # set mtb to missing if common equity is somehow missing
        mtb[np.isnan(nparray.ceq)] = np.nan
        # add book leverage to the result
        nparray = rfn.rec_append_fields(nparray, NAME, mtb)
        # keep only useful columns
        cols = set(rfn.get_names_flat(nparray.dtype))
        nparray.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "prcc_f", "csho", "ceq", NAME])
        return (
            pd.DataFrame.from_records(nparray, exclude=exclude_cols),
            VARIABLE_LABELS,
        )

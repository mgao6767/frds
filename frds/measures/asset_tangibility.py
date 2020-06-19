from typing import List, Tuple, Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda


NAME = "AssetTangibility"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "at",
            "ppent",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABELS = {
    NAME: "Property, Plant and Equipment (Net) scaled by Assets (Total)"
}


class AssetTangibility(Measure):
    """Asset tangibility:
        Plant, Property and Equipment (Net)
     =  -----------------------------------
        Assets (Total)
    """

    def __init__(self):
        super().__init__("Asset Tangibility", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        nparray = filter_funda(nparrays[0])
        tangibility = np.true_divide(
            nparray.ppent, nparray.at, where=(nparray.at != 0)
        )
        tangibility[np.isnan(nparray.at)] = np.nan
        nparray = rfn.rec_append_fields(nparray, NAME, tangibility)
        # keep only useful columns
        cols = set(rfn.get_names_flat(nparray.dtype))
        nparray.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "ppent", "at", NAME])
        return (
            pd.DataFrame.from_records(nparray, exclude=exclude_cols),
            VARIABLE_LABELS,
        )

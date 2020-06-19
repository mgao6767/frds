from typing import List, Tuple, Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

NAME = "CapitalExpenditure"
DATASETS_REQUIRED = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "at",
            "capx",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABELS: Dict[str, str] = {
    NAME: "Capital Expenditures scaled by Assets (Total)"
}


class CapitalExpenditure(Measure):
    """Capial expenditure scaled by total assets

        capital expenditures
     =  --------------------
        total assets
    """

    def __init__(self):
        super().__init__("Capital Expendture", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        """Compute the capital expenditures scaled by total assets

        Parameters
        ----------
        nparrays : List[np.recarray]
            Input datasets, same order as specified in `DATASETS_REQUIRED`

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, str]]
            Output dataset and the `VARIABLE_LABELS`
        """
        nparray = filter_funda(nparrays[0])
        capx = np.true_divide(nparray.capx, nparray.at, where=(nparray.at != 0))
        capx[np.isnan(nparray.at)] = np.nan
        nparray = rfn.rec_append_fields(nparray, NAME, capx)
        # keep only useful columns
        cols = set(rfn.get_names_flat(nparray.dtype))
        nparray.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "capx", "at", NAME])
        return (
            pd.DataFrame.from_records(nparray, exclude=exclude_cols),
            VARIABLE_LABELS,
        )

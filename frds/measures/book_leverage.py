from typing import List, Tuple, Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

NAME = "BookLeverage"
DATASETS_REQUIRED = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "dltt",
            "dlc",
            "ceq",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABEL: Dict[str, str] = {NAME: "(DLTT+DLC)/(DLTT+DLC+CEQ)"}


class BookLeverage(Measure):
    """Book leverage:
        (long-term debt + debt in current liabilities) 
     =  --------------------------------------------------------------
        (long-term debt + debt in current liabilities + common equity)

    If CEQ is missing, the book leverage is treated as missing. 
    This avoids a book leverage of 100%."
    """

    def __init__(self):
        super().__init__("Book Leverage", datasets_required=DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        """Compute the book leverage

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
        # debts = long-term debt + debt in current liabilities
        debts = np.nansum([nparray.dltt, nparray.dlc], axis=0)
        # assets = debts + common equity
        assets = np.nansum([debts, nparray.ceq], axis=0)
        # book leverage = debts / assets
        bleverage = np.true_divide(debts, assets, where=(assets != 0))
        # set book leverage to missing if common equity is somehow missing
        bleverage[np.isnan(nparray.ceq)] = np.nan
        # add book leverage to the result
        nparray = rfn.rec_append_fields(nparray, NAME, bleverage)
        # keep only useful columns
        cols = set(rfn.get_names_flat(nparray.dtype))
        nparray.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "dltt", "dlc", "ceq", NAME])
        return (
            pd.DataFrame.from_records(nparray, exclude=exclude_cols),
            VARIABLE_LABEL,
        )

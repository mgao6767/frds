import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

NAME = "TobinQ"
DATASETS_REQUIRED = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "fyear",
            "gvkey",
            "csho",
            "prcc_c",
            "seq",
            "txdb",
            "itcb",
            "pstkrv",
            "pstkl",
            "pstk",
            "at",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
VARIABLE_LABELS = {NAME: "Tobin's Q"}


class TobinQ(Measure):
    def __init__(self):
        super().__init__(NAME, DATASETS_REQUIRED)

    def estimate(self, nparrays):
        funda = filter_funda(nparrays[0])
        # Use Daniel and Titman (JF 1997) Book of Equity Calculation
        # PSTKRV: Preferred stock Redemption Value. If missing, use PSTKL: Liquidating Value
        # If still missing, then use PSTK: Preferred stock - Carrying Value, Stock (Capital)
        pref = np.where(np.isnan(funda.pstkrv), funda.pstkl, funda.pstkrv)
        pref = np.where(np.isnan(pref), funda.pstk, pref)
        # BE = Stockholders Equity + Deferred Taxes + investment Tax Credit - Preferred Stock
        be = np.nansum([funda.seq, funda.txdb, funda.itcb, -pref], axis=0)
        # Calculate Market Value of Equity at Year End
        # use prrc_c at the calendar year end for a fair cross sectional comparison
        me = funda.prcc_c * funda.csho
        # Calculate Tobin's Q
        tobin_q = np.true_divide(
            np.nansum([funda.at, me, -be], axis=0),
            funda.at,
            where=(funda.at != 0),
        )
        # Keep Companies with Existing Shareholders' Equity
        tobin_q[funda.seq < 0] = np.nan
        funda = rfn.rec_append_fields(funda, NAME, tobin_q)
        # keep only useful columns
        cols = set(rfn.get_names_flat(funda.dtype))
        funda.sort(order=(keys := ["gvkey", "datadate"]))
        exclude_cols = cols - set([*keys, "fyear", NAME])
        result = pd.DataFrame.from_records(funda, exclude=exclude_cols)
        result[NAME].replace([np.inf, -np.inf], np.nan, inplace=True)
        return result, VARIABLE_LABELS

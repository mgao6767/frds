import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'BookLeverage'
description = 'Book leverage: (long-term debt + debt in current liabilities) / (long-term debt + debt in current liabilities + common equity)'
note = 'If CEQ is missing, the book leverage is treated as missing. This avoids a book leverage of 100%.'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'dltt', 'dlc', 'ceq', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: '(DLTT+DLC)/(DLTT+DLC+CEQ)'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

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
    nparray = rfn.rec_append_fields(nparray, name, bleverage)
    nparray.sort(order=['gvkey', 'datadate'])
    # keep all variables in case users want to verify
    return pd.DataFrame.from_records(nparray), variable_labels

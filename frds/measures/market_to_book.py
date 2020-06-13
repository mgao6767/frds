import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'MarketToBookRatio'
description = 'Market-to-book ratio'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'csho', 'prcc_f', 'ceq', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: '(PRCC_F*CSHO)/CEQ'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])
    # market value at fiscal year
    mv = nparray.prcc_f * nparray.csho
    # market-to-book = market value of equity / common equity
    mtb = np.true_divide(mv, nparray.ceq, where=(nparray.ceq != 0))
    # set mtb to missing if common equity is somehow missing
    mtb[np.isnan(nparray.ceq)] = np.nan
    # add book leverage to the result
    nparray = rfn.rec_append_fields(nparray, name, mtb)
    # keep only useful columns
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    exclude_cols = cols - set([*keys, 'prcc_f', 'csho', 'ceq', name])
    return pd.DataFrame.from_records(nparray, exclude=exclude_cols), variable_labels

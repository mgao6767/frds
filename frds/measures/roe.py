import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'ROE'
description = 'Return on Equity'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'ceq', 'ib', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: 'Income Before Extraordinary Items scaled by Common Equity (Total)'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])
    roa = np.true_divide(nparray.ib, nparray.ceq, where=(nparray.ceq != 0))
    roa[np.isnan(nparray.ceq)] = np.nan
    nparray = rfn.rec_append_fields(nparray, name, roa)
    # keep only useful columns
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    exclude_cols = cols - set([*keys, 'ib', 'ceq', name])
    return pd.DataFrame.from_records(nparray, exclude=exclude_cols), variable_labels

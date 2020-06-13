import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'CapitalExpenditure'
description = 'Capital expenditures scaled by total assets.'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'at', 'capx', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: 'Capital Expenditures scaled by Assets (Total)'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])
    capx = np.true_divide(nparray.capx, nparray.at, where=(nparray.at != 0))
    capx[np.isnan(nparray.at)] = np.nan
    nparray = rfn.rec_append_fields(nparray, name, capx)
    # keep only useful columns
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    exclude_cols = cols - set([*keys, 'capx', 'at', name])
    return pd.DataFrame.from_records(nparray, exclude=exclude_cols), variable_labels

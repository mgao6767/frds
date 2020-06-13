import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'Size'
description = 'Firm size defined as the natural logarithm of total assets.'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'at', 'indfmt', 'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: 'Natural logarithm of total assets'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])
    size = np.log(nparray.at, where=(nparrat.at>0))
    size[np.isnan(nparray.at)] = np.nan
    nparray = rfn.rec_append_fields(nparray, name, size)
    # keep only useful columns
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    exclude_cols = cols - set([*keys, 'at', name])
    return pd.DataFrame.from_records(nparray, exclude=exclude_cols), variable_labels

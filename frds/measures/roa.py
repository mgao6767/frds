import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from typing import List
from ..data import Dataset

name = 'ROA'
description = 'Return on Assets'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'at', 'ib', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]
variable_labels = {
    name: 'Income Before Extraordinary Items scaled by Assets (Total)'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])

    nparray = rfn.rec_append_fields(nparray, name, nparray.ib/nparray.at)
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    exclude_cols = cols - set([*keys, name])
    return pd.DataFrame.from_records(nparray, exclude=exclude_cols), variable_labels

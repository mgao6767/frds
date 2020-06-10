import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn
from datetime import datetime
from typing import List
from ..data import Dataset

name = 'Tangibility'
description = 'The asset tangibility for firms in the comp.funda sample.'
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'at', 'ppent', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate'])
]


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray = filter_funda(nparrays[0])

    nparray = rfn.rec_append_fields(nparray, name, nparray.ppent/nparray.at)
    cols = set(rfn.get_names_flat(nparray.dtype))
    nparray.sort(order=(keys := ['gvkey', 'datadate']))
    (cols_to_keep := set(keys)).add(name)
    return pd.DataFrame.from_records(nparray, exclude=list(cols-cols_to_keep))

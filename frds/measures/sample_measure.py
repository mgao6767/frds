import numpy as np
from numpy.lib import recfunctions as rfn
from datetime import datetime
from typing import List
from ..data import Dataset

name = 'Tangibility'
description = 'The asset tangibility for firms in the comp.funda sample.'
# TODO: `start_date` and `end_date` are not used right now.
# For demonstration, I deliberately separate it into two datasets.
datasets = [
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'at', 'indfmt',
                  'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate']),
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=['datadate', 'gvkey', 'ppent',
                  'indfmt', 'datafmt', 'popsrc', 'consol'],
            date_vars=['datadate']
            )
]


def estimate(nparrays: List[np.recarray]):
    nparray1 = nparrays[0]
    nparray2 = nparrays[1]

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    nparray1 = filter_funda(nparray1)
    nparray2 = filter_funda(nparray2)

    # The output of `rfn.rec_join()` is sorted along the keys.
    nparray = rfn.rec_join(keys := ['gvkey', 'datadate'], nparray1, nparray2)
    nparray = rfn.rec_append_fields(nparray, name, nparray.ppent/nparray.at)
    cols = set(rfn.get_names_flat(nparray.dtype))
    (cols_to_keep := set(keys)).add(name)
    return rfn.rec_drop_fields(nparray, cols-cols_to_keep)

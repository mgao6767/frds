import numpy as np
from numpy.lib import recfunctions as rfn
from datetime import datetime
from typing import List
from ..data import Dataset

name = 'Sample Measure'
description = 'Calculate the average firm size based on comp.funda sample.'
# TODO: `start_date` and `end_date` are not used right now.
datasets = [Dataset(source='wrds', library='comp',
                    table='funda',
                    vars=['datadate', 'gvkey', 'at', 'indfmt',
                          'datafmt', 'popsrc', 'consol'],
                    date_vars=['datadate'],
                    start_date=datetime(2019, 1, 1),
                    end_date=datetime(2019, 1, 5)),
            Dataset(source='wrds', library='comp',
                    table='funda',
                    vars=['datadate', 'gvkey', 'ppent',
                          'indfmt', 'datafmt', 'popsrc', 'consol'],
                    date_vars=['datadate'],
                    start_date=datetime(2019, 1, 1),
                    end_date=datetime(2019, 1, 5),
                    )]


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

    nparray = rfn.rec_join(['gvkey', 'datadate'], nparray1, nparray2)
    nparray2.sort(order=['gvkey', 'datadate'])
    print(f'{nparray=}')
    return np.nanmean(nparray.at)

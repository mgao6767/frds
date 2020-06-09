import numpy as np
from datetime import datetime
from ..data import Dataset

name = 'Sample Measure'
description = 'Calculate the average firm size based on comp.funda sample.'
# TODO: Complicated measures may use more than one dataset!
dataset = Dataset(source='wrds', library='comp',
                  table='funda', vars=['datadate', 'gvkey', 'at'],
                  date_vars=['datadate'],
                  start_date=datetime(2019, 1, 1),
                  end_date=datetime(2019, 1, 5))


def estimate(nparray: np.recarray):
    return np.nanmean(nparray.at)

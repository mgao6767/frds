from datetime import datetime
from ..data import Dataset

name = 'Sample Measure'
dataset = Dataset(source='wrds', library='comp',
                  table='funda', vars=['datadate', 'gvkey', 'at'],
                  date_vars=['datadate'],
                  start_date=datetime(2019, 1, 1),
                  end_date=datetime(2019, 1, 5))


def estimate():
    pass

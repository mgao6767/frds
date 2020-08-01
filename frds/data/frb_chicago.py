from datetime import datetime
import zipfile
import tempfile
import pandas as pd
import requests
from frds.data import Dataset


class Connection:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def get_table(self, library, table, columns, date_cols, obs):
        if library.lower() == "bhc" and table.lower() == "bhcf":
            return pd.concat(self.bhcf(columns, date_cols))

    @staticmethod
    def bhcf(columns, date_cols):
        # https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
        now = datetime.now()
        tempfiles, dfs = [], []
        for year in range(1986, now.year):
            for qtr in [1, 2, 3, 4]:
                url = f"https://www.chicagofed.org/api/sitecore/BHCHome/GetFile?SelectedQuarter={qtr}&SelectedYear={year}"
                resp = requests.get(url)
                tempfiles.append(f := tempfile.NamedTemporaryFile())
                f.write(resp.content)
                # print(year, qtr, url)
                if not zipfile.is_zipfile(f):
                    # starts from 1986Q3 so first two aren't zipfiles
                    continue
                df = pd.read_csv(
                    f,
                    sep="^",
                    skiprows=[1],
                    compression="zip",
                    low_memory=False,
                    usecols=lambda col: col.upper() in columns,
                    # parse_dates=date_cols,
                )
                df.columns = map(str.upper, df.columns)
                for date in date_cols:
                    df[date] = pd.to_datetime(df[date], format="%Y%m%d")
                dfs.append(df)
        for f in tempfiles:
            f.close()
        return dfs

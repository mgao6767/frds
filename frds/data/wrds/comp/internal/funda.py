from dataclasses import dataclass
import pandas as pd
from frds.data.wrds import WRDSDataset


@dataclass
class Funda(WRDSDataset):
    """Fundamentals Annual"""

    data: pd.DataFrame
    library = "comp"
    table = "funda"
    index_col = ["gvkey", "datadate"]
    date_cols = ["datadate"]

    def __post_init__(self):
        idx = [c.upper() for c in self.index_col]
        if set(self.data.index.names) != set(idx):
            self.data.reset_index(inplace=True, drop=True)
        self.data.rename(columns=str.upper, inplace=True)
        self.data.set_index(idx, inplace=True)

    @staticmethod
    def lag(series: pd.Series, lags: int = 1, *args, **kwargs):
        return series.shift(lags, *args, **kwargs)

    @staticmethod
    def lead(series: pd.Series, leads: int = 1, *args, **kwargs):
        return series.shift(-leads, *args, **kwargs)

    @property
    def FYEAR(self) -> pd.Series:
        """Fiscal year"""
        return self.data["FYEAR"].astype(int)

    @property
    def PPENT(self) -> pd.Series:
        """Plant, Property and Equipment (Net)"""
        return self.data["PPENT"]

    @property
    def AT(self) -> pd.Series:
        """Total Assets"""
        return self.data["AT"]

    @property
    def PRCC_F(self) -> pd.Series:
        """Share price at year end"""
        return self.data["PRCC_F"]

    @property
    def CSHO(self) -> pd.Series:
        """Common shares outstanding"""
        return self.data["CSHO"]

    @property
    def CEQ(self) -> pd.Series:
        """Common equity"""
        return self.data["CEQ"]

    @property
    def IB(self) -> pd.Series:
        """Income Before Extraordinary Items"""
        return self.data["IB"]

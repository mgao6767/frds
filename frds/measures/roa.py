from typing import Union
import pandas as pd
from frds.data.wrds.comp import funda, fundq


class ROA_variants:
    """Return on assets"""

    def __call__(self, data: Union[funda, fundq]) -> pd.Series:
        """Income before extraordinary items scaled by contemporaneous total assets."""
        if isinstance(data, funda):
            return data.IB / data.AT
        if isinstance(data, fundq):
            return data.IBQ / data.ATQ

    @classmethod
    def v2(cls, data: Union[funda, fundq]) -> pd.Series:
        """Income before extraordinary items scaled by lagged total assets."""
        if isinstance(data, funda):
            return data.IB / funda.lag(data.AT, lags=1)
        if isinstance(data, fundq):
            return data.IBQ / fundq.lag(data.ATQ, lags=1)


ROA = ROA_variants()

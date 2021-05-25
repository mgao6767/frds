from typing import Union
import pandas as pd
from frds.data.wrds.comp import funda, fundq


class ROE_variants:
    """Return on equity"""

    def __call__(self, data: Union[funda, fundq]) -> pd.Series:
        """Income before extraordinary items scaled by contemporaneous common equity."""
        if isinstance(data, funda):
            return data.IB / data.CEQ
        if isinstance(data, fundq):
            return data.IBQ / data.CEQQ

    @classmethod
    def v2(cls, data: Union[funda, fundq]) -> pd.Series:
        """Income before extraordinary items scaled by lagged common equity."""
        if isinstance(data, funda):
            return data.IB / funda.lag(data.CEQ, lags=1)
        if isinstance(data, fundq):
            return data.IBQ / fundq.lag(data.CEQQ, lags=1)


ROE = ROE_variants()

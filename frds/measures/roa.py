import pandas as pd
from frds.data.wrds.comp import funda


class ROA_variants:
    """Return on assets"""

    def __call__(self, data: funda) -> pd.Series:
        """Default method:
        Income before extraordinary items scaled by contemporaneous total assets.

        Args:
            data (funda): Fundamentals Annual

        Returns:
            pd.Series: ROA
        """
        return data.IB / data.AT

    @classmethod
    def v2(cls, data: funda) -> pd.Series:
        """Income before extraordinary items scaled by lagged total assets.

        Args:
            data (funda): Fundamentals Annual

        Returns:
            pd.Series: ROA
        """
        return data.IB / funda.lag(data.AT, lags=1)


ROA = ROA_variants()

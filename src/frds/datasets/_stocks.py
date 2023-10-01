import os
import pandas as pd

_current_file_path = os.path.abspath(__file__)
_current_directory = os.path.dirname(_current_file_path)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class _StockReturns(metaclass=SingletonMeta):
    """Some datasets about stock returns"""

    _stocks_us: pd.DataFrame = None

    @property
    def stocks_us(self) -> pd.DataFrame:
        """US stock returns 2010-2022, including Google, GS, JPM, and S&P500.

        Returns:
            pd.DataFrame: date-index DataFrame

        Construction:
            >>> import yfinance as yf
            >>> tickers = ['^GSPC', 'GOOGL', 'GS', 'JPM']
            >>> data = yf.download(tickers, start='2010-01-01', end='2022-12-31')['Adj Close']
            >>> daily_returns = data.pct_change().dropna()
            >>> daily_returns.to_csv('stocks_us.csv.zip', index=True, compression='zip')
        """
        # fmt: off
        path = f"{os.path.join(_current_directory, 'stocks_us.csv.zip')}"
        if self._stocks_us is None:
            self._stocks_us = pd.read_csv(path, index_col=['Date'], parse_dates=['Date'])
        return self._stocks_us


StockReturns = _StockReturns()

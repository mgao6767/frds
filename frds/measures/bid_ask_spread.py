"""
# Bid-Ask Spread
"""
import typing
from datetime import datetime
import numpy as np
import pandas as pd
from frds.measures import MeasureCategory, setup, update_progress
from frds.utils.data import get_market_microstructure_data

setup(
    measure_name="Bid-Ask Spread",
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/bid_ask_spread",
    author="Mingze Gao",
    author_email="mingze.gao@sydney.edu.au",
)


@update_progress()  # voluntarily update progress
def estimation(
    securities: typing.List[str],
    start_date: datetime,
    end_date: datetime,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Given a list of securities and start/end dates, return a dataframe of daily average
    bid-ask spreads:
        Date    Security    Bid-Ask Spread
        ...     ...         ...
    """
    data_period = pd.date_range(start_date, end_date)
    results, total_jobs, completed_jobs = [], len(data_period) * len(securities), 0
    for date, security, data in get_market_microstructure_data(
        securities, start_date, end_date
    ):
        result = _est(data)
        results.append({"Date": date, "Security": security, "Bid-Ask Spread": result})
        completed_jobs += 1
        progress(completed_jobs // total_jobs)  # noqa: F821
    return pd.DataFrame(results)


def _est(data: pd.DataFrame) -> float:
    """
    Compute the average bid-ask spread for the given intraday data of a security
    Required fields are:
        - Ask Price
        - Bid Price
    """
    ask = data["Ask Price"].to_numpy()
    bid = data["Bid Price"].to_numpy()
    midpt = (ask + bid) / 2
    spread = np.divide(ask - bid, midpt)
    del ask, bid, midpt
    return np.mean(spread)

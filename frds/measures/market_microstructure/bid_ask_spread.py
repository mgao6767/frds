"""
# Bid-Ask Spread
"""
import typing
from datetime import datetime
import numpy as np
import pandas as pd
from frds.measures import MeasureCategory, setup, update_progress
from frds.utils.data import (
    get_market_microstructure_data_from_disk,
    get_total_files_of_market_microstructure_data_on_disk,
)
from frds.utils.measures import save_results

name = "Bid-Ask Spread"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/bid_ask_spread",
    author="Mingze Gao",
    author_email="mingze.gao@sydney.edu.au",
)


@update_progress()  # voluntarily update progress
def estimation(
    securities: typing.List[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Given a list of securities and start/end dates, return a dataframe of daily average
    bid-ask spreads:
        Date    Security    Bid-Ask Spread
        ...     ...         ...
    """
    total_jobs = get_total_files_of_market_microstructure_data_on_disk()
    results, completed_jobs = [], 0
    for date, security, data in get_market_microstructure_data_from_disk():
        result = _est(data)
        results.append({"Date": date, "Security": security, "Bid-Ask Spread": result})
        completed_jobs += 1
        progress(int(completed_jobs / total_jobs * 100))  # noqa: F821
    df = pd.DataFrame(results)
    save_results(df, f"{name}.csv")


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
    spread = np.divide(ask - bid, midpt, where=(midpt != 0))
    del ask, bid, midpt
    return np.nanmean(spread)


"""
# Realized Spread
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

name = "Realized Spread"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/realized_spread",
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
        Date    Security    Realized Spread
        ...     ...         ...
    """
    total_jobs = get_total_files_of_market_microstructure_data_on_disk()
    results, completed_jobs = [], 0
    for date, security, data in get_market_microstructure_data_from_disk():
        result = _est(data)
        results.append({"Date": date, "Security": security, name: result})
        completed_jobs += 1
        progress(int(completed_jobs / total_jobs * 100))  # noqa: F821
    df = pd.DataFrame(results)
    save_results(df, f"{name}.csv")


def _est(data: pd.DataFrame) -> float:
    """
    Compute the dollar-volume-weighted average realized spread
    Required fields are:
        - Mid Point
        - Volume
        - Price
        - Direction
    """
    log_midpt = np.log(data["Mid Point"].to_numpy())
    price = data["Price"].to_numpy()
    log_price = np.log(price)
    timestamps = np.array(data.index, dtype="datetime64")
    # Find the Quote Mid Point 5 min later than each trade.
    matched_log_midpt = []
    last_ts = timestamps[0]
    for idx, ts1 in enumerate(timestamps):
        for i, ts2 in enumerate(timestamps[idx:]):
            if ts2 - ts1 >= np.timedelta64(5, "m"):
                matched_log_midpt.append(log_midpt[idx + i])
                break
    matched = len(matched_log_midpt)
    rspread = (
        2
        * data["Direction"].to_numpy()[:matched]
        * (log_price[:matched] - matched_log_midpt)
    )
    # Daily realized spread is the dollar-volume-weighted average
    # of the realized spread computed over all trades in the day.
    volume = data["Volume"].to_numpy()
    dolloar_volume = np.multiply(volume, price)[:matched]
    rsprd = np.sum(np.multiply(rspread, dolloar_volume) / np.sum(dolloar_volume))
    return None if np.isnan(rsprd) else rsprd

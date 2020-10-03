"""
# Effective Spread
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

name = "Effective Spread"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/effective_spread",
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
        Date    Security    Effective Spread
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
    Compute the dollar-volume-weighted average effective spread
    Required fields are:
        - Mid Point
        - Volume
        - Price
    """
    log_midpt = np.log(data["Mid Point"].to_numpy())
    price = data["Price"].to_numpy()
    log_price = np.log(price)
    espread = 2 * np.abs(log_price - log_midpt)
    # Daily effective spread is the dollar-volume-weighted average
    # of the effective spread computed over all trades in the day.
    volume = data["Volume"].to_numpy()
    dolloar_volume = np.multiply(volume, price)
    esprd = np.nansum(np.multiply(espread, dolloar_volume) / np.sum(dolloar_volume))
    del log_midpt, price, log_price, espread, volume, dolloar_volume
    return esprd

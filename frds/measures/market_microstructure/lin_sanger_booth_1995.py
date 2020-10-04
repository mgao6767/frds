"""
# Adverse selection measure as in Lin, Sanger and Booth (1995)
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

name = "LinSangerBooth1995"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/linsangerbooth1995",
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
        Date    Security    LinSangerBooth1995
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
    log_midpt = np.log(data["Mid Point"])
    diff_log_midpt = np.diff(log_midpt)
    signed_effective_sprd = data["Direction"] * np.abs(
        np.log(data["Price"]) - log_midpt
    )

    # lag signed effective spread
    x = np.vstack([np.ones(len(diff_log_midpt)), signed_effective_sprd[:-1]]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(x, diff_log_midpt, rcond=None)
    except np.linalg.LinAlgError:
        return None
    else:
        return coef[1]

"""
# Kyle's Lambda
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

name = "Kyle's Lambda"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/kyleslambda",
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
        Date    Security    Kyle's Lambda
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
    price = data["Price"].to_numpy()
    volume = data["Volume"].to_numpy()
    direction = data["Direction"].to_numpy()
    sqrt_dollar_volume = np.sqrt(np.multiply(price, volume))
    signed_sqrt_dollar_volume = np.abs(np.multiply(direction, sqrt_dollar_volume))
    # Find the total signed sqrt dollar volume and return per 5 min.
    timestamps = np.array(data.index, dtype="datetime64")
    last_ts, last_price = timestamps[0], price[0]
    bracket_ssdv = 0
    bracket = last_ts + np.timedelta64(5, "m")
    rets, ssdvs, = [], []
    for idx, ts in enumerate(timestamps):
        if ts <= bracket:
            bracket_ssdv += signed_sqrt_dollar_volume[idx]
        else:
            ret = np.log(price[idx - 1] / last_price)
            if not np.isnan(ret) and not np.isnan(bracket_ssdv):
                rets.append(ret)
                ssdvs.append(bracket_ssdv)
            # Reset bracket
            bracket = ts + np.timedelta64(5, "m")
            last_price = price[idx]
            bracket_ssdv = signed_sqrt_dollar_volume[idx]
    # Perform regression.
    x = np.vstack([np.ones(len(ssdvs)), np.array(ssdvs)]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(x, np.array(rets), rcond=None)
    except np.linalg.LinAlgError:
        return None
    else:
        return None if np.isnan(coef[1]) else coef[1] * 1e6


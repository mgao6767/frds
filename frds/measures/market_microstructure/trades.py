"""
# Buys, Sells and Total Trades
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

name = "Buys, Sells and Trades"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/trades",
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
        Date    Security    Number of Sells     Number of Buys      Total Trades
        ...     ...         ...                 ...                 ...
    """
    total_jobs = get_total_files_of_market_microstructure_data_on_disk()
    results, completed_jobs = [], 0
    for date, security, data in get_market_microstructure_data_from_disk():
        buys, sells, trades = _est(data)
        results.append(
            {
                "Date": date,
                "Security": security,
                "Number of Buys": buys,
                "Number of Selss": sells,
                "Total Trades": trades,
            }
        )
        completed_jobs += 1
        progress(int(completed_jobs / total_jobs * 100))  # noqa: F821
    df = pd.DataFrame(results)
    save_results(df, f"{name}.csv")


def _est(data: pd.DataFrame) -> float:
    """
    Required fields are:
        - Direction
    """
    directions = data["Direction"].to_numpy()
    directions[np.where(directions == -1)] = 0
    trades = len(directions)
    buys = np.sum(directions)
    sells = trades - buys
    return buys, sells, trades

"""
# Variance ratio test as in Lo and MacKinlay (1988)
"""
import typing
from datetime import datetime
import numpy as np
import pandas as pd
from numba import jit
from frds.measures import MeasureCategory, setup, update_progress
from frds.utils.data import (
    get_market_microstructure_data_from_disk,
    get_total_files_of_market_microstructure_data_on_disk,
)
from frds.utils.measures import save_results

name = "LoMackinlay1988"
setup(
    measure_name=name,
    measure_type=MeasureCategory.MARKET_MICROSTRUCTURE,
    doc_url="https://frds.io/measures/lomackinlay1988",
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
        Date    Security    LoMackinlay1988
        ...     ...         ...
    """
    total_jobs = get_total_files_of_market_microstructure_data_on_disk()
    results, completed_jobs = [], 0
    for date, security, data in get_market_microstructure_data_from_disk():
        result = _est(data)
        results.append({"Date": date, "Security": security, **result})
        completed_jobs += 1
        progress(int(completed_jobs / total_jobs * 100))  # noqa: F821
    df = pd.DataFrame(results)
    save_results(df, f"{name}.csv")


def _est(data: pd.DataFrame) -> float:
    "A fast estimation of Variance Ratio test statistics as in Lo and MacKinlay (1988)"
    # Prices array = [p1, p2, p3, p4, ..., pT]
    prices = data["Price"].to_numpy(dtype=np.float64)
    result = {}
    # Estimate many lags.
    for k in [5, 10, 20, 50, 100]:
        # Compute a constant array as np.array creation is not allowed in nopython mode.
        const_arr = np.arange(k - 1, 0, step=-1, dtype=np.int)
        vr, stat1, stat2 = _estimate(np.log(prices), k, const_arr)
        result.update(
            {
                f"Variance Ratio (k={k})": vr,
                f"Test Statistic (k={k}) Homoscedasticity Assumption": stat1,
                f"Test Statistic (k={k}) Heteroscedasticity Assumption": stat2,
            }
        )
    return result


@jit(nopython=True, nogil=True, cache=True)
def _estimate(log_prices, k, const_arr):
    # Log returns = [x2, x3, x4, ..., xT], where x(i)=ln[p(i)/p(i-1)]
    rets = np.diff(log_prices)
    # T is the length of return series
    T = len(rets)
    # mu is the mean log return
    mu = np.mean(rets)
    # sqr_demeaned_x is the array of squared demeaned log returns
    sqr_demeaned_x = np.square(rets - mu)
    # Var(1)
    # Didn't use np.var(rets, ddof=1) because
    # sqr_demeaned_x is calculated already and will be used many times.
    var_1 = np.sum(sqr_demeaned_x) / (T - 1)
    # Var(k)
    # Variance of log returns where x(i) = ln[p(i)/p(i-k)]
    # Before np.roll() - array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # After np.roll(,shift=2) - array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    # Discard the first k elements.
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1 / m * np.sum(np.square(rets_k - k * mu))

    # Variance Ratio
    vr = var_k / var_1

    # a_arr is an array of { (2*(k-j)/k)^2 } for j=1,2,...,k-1, fixed for a given k:
    #   When k=5, a_arr = array([2.56, 1.44, 0.64, 0.16]).
    #   When k=8, a_arr = array([3.0625, 2.25, 1.5625, 1., 0.5625, 0.25, 0.0625])
    # Without JIT it's defined as:
    #   a_arr = np.square(np.arange(k-1, 0, step=-1, dtype=np.int) * 2 / k)
    # But np.array creation is not allowed in nopython mode.
    # So const_arr=np.arange(k-1, 0, step=-1, dtype=np.int) is created outside.
    a_arr = np.square(const_arr * 2 / k)

    # b_arr is part of the delta_arr.
    b_arr = np.empty(k - 1, dtype=np.float64)
    for j in range(1, k):
        b_arr[j - 1] = np.sum((sqr_demeaned_x * np.roll(sqr_demeaned_x, j))[j + 1 :])

    delta_arr = b_arr / np.square(np.sum(sqr_demeaned_x))

    # Both arrarys are of length (k-1)
    assert len(delta_arr) == len(a_arr) == k - 1

    phi1 = 2 * (2 * k - 1) * (k - 1) / (3 * k * T)
    phi2 = np.sum(a_arr * delta_arr)

    # VR test statistics under two assumptions
    vr_stat_homoscedasticity = (vr - 1) / np.sqrt(phi1)
    vr_stat_heteroscedasticity = (vr - 1) / np.sqrt(phi2)

    return vr, vr_stat_homoscedasticity, vr_stat_heteroscedasticity

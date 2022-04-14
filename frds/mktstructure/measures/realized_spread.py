import numpy as np
import pandas as pd

from .exceptions import *

name = "RealizedSpread"
description = """
The realized spread is the temporary component of the effective spread. 
It measures the revenue to liquidity providers assuming that the liquidity provider is able to close
her position at the midpoint prevailing five minutes after the trade:
2 * q * (P - midpoint_5min)
where q is the trade direction (1 for buys and -1 for sells),
P is the transaction price, and midpoint_5min is the bid-ask midpoint 5mins later. 
"""
vars_needed = {"Price", "Volume", "Mid Point", "Direction"}


def estimate(data: pd.DataFrame) -> np.ndarray:
    log_midpt = np.log(data["Mid Point"].to_numpy())
    price = data["Price"].to_numpy()
    timestamps = np.array(data.index, dtype="datetime64")
    # Find the Quote Mid Point 5 min later than each trade.
    matched_midpt = []
    for idx, ts1 in enumerate(timestamps):
        for i, ts2 in enumerate(timestamps[idx:]):
            if ts2 - ts1 >= np.timedelta64(5, "m"):
                matched_midpt.append(log_midpt[idx + i])
                break
    matched = len(matched_midpt)
    rspread = (
        2 * data["Direction"].to_numpy()[:matched] * (price[:matched] - matched_midpt)
    )
    # Daily realized spread is the dollar-volume-weighted average
    # of the realized spread computed over all trades in the day.
    volume = data["Volume"].to_numpy()
    dolloar_volume = np.multiply(volume, price)[:matched]
    rsprd = np.sum(np.multiply(rspread, dolloar_volume) / np.sum(dolloar_volume))
    return np.nan if np.isnan(rsprd) else rsprd

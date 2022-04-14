import numpy as np
import pandas as pd

from .exceptions import *

name = "PriceImpact"
description = """
The price impact is 
2 * q * (midpoint_5min - midpoint) / midpoint
where q is the trade direction (1 for buys and -1 for sells),
midpoint is bid-ask midpoint, and midpoint_5min is the bid-ask midpoint 5min later. 
"""
vars_needed = {"Price", "Volume", "Mid Point", "Direction"}


def estimate(data: pd.DataFrame) -> np.ndarray:
    if not vars_needed.issubset(data.columns):
        raise MissingVariableError(name, vars_needed.difference(data.columns))

    midpt = data["Mid Point"].to_numpy()

    timestamps = np.array(data.index, dtype="datetime64")
    # Find the Quote Mid Point 5 min later than each trade.
    matched_midpt = []
    for idx, ts1 in enumerate(timestamps):
        for i, ts2 in enumerate(timestamps[idx:]):
            if ts2 - ts1 >= np.timedelta64(5, "m"):
                matched_midpt.append(midpt[idx + i])
                break
    matched = len(matched_midpt)
    directions = data["Direction"].to_numpy()[:matched]
    pimpact = 2 * directions * (matched_midpt - midpt[:matched]) / midpt[:matched]
    # Daily price impact is the dollar-volume-weighted average
    # of the price impact computed over all trades in the day.
    price = data["Price"].to_numpy()
    volume = data["Volume"].to_numpy()
    dolloar_volume = np.multiply(volume, price)[:matched]
    pimpact = np.sum(np.multiply(pimpact, dolloar_volume) / np.sum(dolloar_volume))
    return np.nan if np.isnan(pimpact) else pimpact

import numpy as np
import pandas as pd

from .exceptions import *

name = "EffectiveSpread"
description = """
The effective spread is 
2 * q * (P - midpoint) / midpoint
where q is the trade direction (1 for buys and -1 for sells),
P is the transaction price, and midpoint is the bid-ask midpoint. 
"""
vars_needed = {"Price", "Volume", "Mid Point", "Direction"}

# TODO: JFE 2021: Bias in the effective bid-ask spread https://doi.org/10.1016/j.jfineco.2021.04.018


def estimate(data: pd.DataFrame) -> np.ndarray:
    if not vars_needed.issubset(data.columns):
        raise MissingVariableError(name, vars_needed.difference(data.columns))

    midpt = data["Mid Point"].to_numpy()
    price = data["Price"].to_numpy()
    direction = data["Direction"].to_numpy()
    espread = 2 * direction * (price - midpt) / midpt

    # Daily effective spread is the dollar-volume-weighted average
    # of the effective spread computed over all trades in the day.
    volume = data["Volume"].to_numpy()
    dolloar_volume = np.multiply(volume, price)
    esprd = np.sum(np.multiply(espread, dolloar_volume) / np.sum(dolloar_volume))
    return np.nan if np.isnan(esprd) else esprd

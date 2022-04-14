import numpy as np
import pandas as pd

from .exceptions import *

name = "BidAskSpread"
description = "Simple average bid-ask spread"
vars_needed = {"Bid Price", "Ask Price", "Mid Point"}


def estimate(data: pd.DataFrame) -> np.ndarray:
    if not vars_needed.issubset(data.columns):
        raise MissingVariableError(name, vars_needed.difference(data.columns))

    spread = np.divide(
        data["Ask Price"].to_numpy() - data["Bid Price"].to_numpy(),
        data["Mid Point"].to_numpy(),
    )
    return np.mean(spread) if len(spread) else np.nan

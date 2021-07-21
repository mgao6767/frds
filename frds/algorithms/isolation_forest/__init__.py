"""Isolation Forest"""

import numpy as np
import pandas as pd

from .iforest_ext import iforest  # C++ extension


def anomaly_scores(
    data: pd.DataFrame,
    forest_size: int = 1000,
    tree_size: int = 256,
    exclude_cols: list = None,
    random_seed: int = 1,
    name: str = "AnomalyScore",
) -> pd.DataFrame:
    """Calculate the anomaly socres using Isolation Forest

    Args:
        data (pd.DataFrame): indexed DataFrame
        tree_size (int, optional): number of observations per Isolation Tree. Defaults to 256.
        forest_size (int, optional): number of trees. Defaults to 1_000.
        exclude_cols (list, optional): columns in the input `data` to ignore. Defaults to None.
        random_seed (int, optional): random seed for reproducibility. Defaults to 1.
        name (str, optional): column name of the return DataFrame. Defaults to "AnomalyScore".

    Returns:
        pd.DataFrame: [description]
    """
    exclude_cols = [] if exclude_cols is None else exclude_cols
    data = data.drop(columns=exclude_cols)
    obs = data.index
    # convert input dataset from an indexed `pd.DataFrame` to a `np.ndarray`
    data = data.to_numpy(dtype=np.double).T
    _, n_obs = data.shape
    # make an empty array to store calculated anomaly scores
    a_scores = np.empty(n_obs, dtype=np.double)
    # magic at work
    iforest(data, a_scores, forest_size, tree_size, random_seed)
    # the order is preserved
    return pd.DataFrame(a_scores, index=obs, columns=[name])

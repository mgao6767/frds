"""Some utility functions"""

import numpy as np


def filter_funda(funda: np.recarray) -> np.recarray:
    """Standard filter on the `wrds.comp.funda` dataset

    Parameters
    ----------
    funda : np.recarray
        `wrds.comp.funda` dataset

    Returns
    -------
    np.recarray
        Filtered dataset
    """
    return funda[
        np.in1d(funda.datafmt, ("STD"))
        & np.in1d(funda.indfmt, ("INDL"))
        & np.in1d(funda.popsrc, ("D"))
        & np.in1d(funda.consol, ("C"))
    ]

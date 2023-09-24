import numpy as np


def z_score(roa: float, capital_ratio: float, past_roas: np.ndarray) -> float:
    r""":doc:`/measures/z_score`

    Args:
        roa (float): the current bank ROA.
        capital_ratio (float): the current bank equity to asset ratio.
        past_roas (np.ndarray): ``(n_periods,)`` array of past bank ROAs used to calculate the standard deviation.

    Returns:
        float: The bank's Z-score
    """
    return (roa + capital_ratio) / np.std(past_roas)

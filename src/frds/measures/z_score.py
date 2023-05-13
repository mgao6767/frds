import numpy as np


def estimate(roa: float, capital_ratio: float, past_roas: np.ndarray) -> float:
    r"""Z-score

    Args:
        roa (float): the current bank ROA.
        capital_ratio (float): the current bank equity to asset ratio.
        past_roas (np.ndarray): (n_periods,) array of past bank ROAs used to calculate the standard deviation.

    Returns:
        float: The bank's Z-score

    Examples:
        >>> from frds.measures import z_score
        >>> import numpy as np
        >>> roas = np.array([0.1,0.2,0.15,0.18,0.2])
        >>> z_score.estimate(roa=0.2, capital_ratio=0.5, past_roas=roas)
        18.549962900111296

    """
    return (roa + capital_ratio) / np.std(past_roas)

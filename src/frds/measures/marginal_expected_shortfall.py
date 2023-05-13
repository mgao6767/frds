import numpy as np


def estimate(
    firm_returns: np.ndarray, market_returns: np.ndarray, q: float = 0.05
) -> float:
    """Marginal Expected Shortfall (MES).

    Args:
        firm_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the firm.
        market_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the market as a whole.
        q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.

    Returns:
        float: The marginal expected shortfall of firm $i$ at time $t$.

    Examples:
        >>> from numpy.random import RandomState
        >>> from frds.measures import marginal_expected_shortfall as mes

        Let's simulate some returns for the firm and the market
        >>> rng = RandomState(0)
        >>> firm_returns = rng.normal(0,1,100)
        >>> mkt_returns = rng.normal(0,1,100)

        Compute the MES.
        >>> mes.estimate(firm_returns, mkt_returns)
        0.13494025343324562

    """
    assert 0 <= q <= 1
    assert firm_returns.shape == market_returns.shape
    low_threshold = np.percentile(market_returns, q * 100)
    worst_days = np.argwhere(market_returns < low_threshold)
    return np.mean(firm_returns[worst_days])

import numpy as np


def marginal_expected_shortfall(
    firm_returns: np.ndarray, market_returns: np.ndarray, q: float = 0.05
) -> float:
    """Marginal Expected Shortfall (MES).

    The firm's average return during the 5% worst days for the market.

    MES measures how exposed a firm is to aggregate tail shocks and, interestingly, together with leverage, \
    it has a significant explanatory power for which firms contribute to a potential crisis as noted by \
    [Acharya, Pedersen, Philippon, and Richardson (2010)](https://doi.org/10.1093/rfs/hhw088).
    
    It is used to construct the [Systemic Expected Shortfall](/measures/systemic_expected_shortfall/).

    Args:
        firm_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the firm.
        market_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the market as a whole.
        q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.

    Returns:
        float: The marginal expected shortfall of firm $i$ at time $t$.

    Examples:
        >>> from numpy.random import RandomState
        >>> from frds.measures import marginal_expected_shortfall

        Let's simulate some returns for the firm and the market
        >>> rng = RandomState(0)
        >>> firm_returns = rng.normal(0,1,100)
        >>> mkt_returns = rng.normal(0,1,100)

        Compute the MES.
        >>> marginal_expected_shortfall(firm_returns, mkt_returns)
        0.13494025343324562

    References:
        * [Acharya, Pedersen, Philippon, and Richardson (2017)](https://doi.org/10.1093/rfs/hhw088),
            Measuring systemic risk, *The Review of Financial Studies*, 30, (1), 2-47.
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Absorption Ratio](/measures/absorption_ratio/)
        * [Contingent Claim Analysis](/measures/cca/)
        * [Distress Insurance Premium](/measures/distress_insurance_premium/)
        * [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)
    """
    assert 0 <= q <= 1
    assert firm_returns.shape == market_returns.shape
    low_threshold = np.percentile(market_returns, q * 100)
    worst_days = np.argwhere(market_returns < low_threshold)
    return np.mean(firm_returns[worst_days])

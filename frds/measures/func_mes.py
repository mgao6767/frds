import numpy as np


def marginal_expected_shortfall(
    firm_returns: np.ndarray, market_returns: np.ndarray, q: float = 0.05
) -> float:
    """Marginal Expected Shortfall (MES).

    Acharya, Pedersen, Philippon, and Richardson (2010) argue that each financial institutions contribution to \
    systemic risk can be measured as its systemic expected shortfall (SES), i.e., its propensity to be undercapitalized \
    when the system as a whole is undercapitalized. SES is a theoretical construct and the authors use the following 3 measures to proxy it:

    1. The outcome of stress tests performed by regulators. The SES metric of a firm here is defined as the recommended capital that \
        it was required to raise as a result of the stress test in February 2009.
    2. The decline in equity valuations of large financial firms during the crisis, as measured by their cumulative equity return \
        from July 2007 to December 2008.
    3. The widening of the credit default swap spreads of large financial firms as measured by their cumulative CDS spread increases \
        from July 2007 to December 2008.

    Given these proxies, the authors seek to develop leading indicators which "predict" an institutions SES; \
    these leading indicators are marginal expected shortfall (MES) and leverage (LVG).

    The description above is from Bisias, Lo, and Valavanis.

    Args:
        firm_returns (np.ndarray): The time series of returns (equity or CDS) for the firm.
        market_returns (np.ndarray): The time series of returns (equity or CDS) for the market as a whole. \
            It is necessary to have enough data points so that there exists a value.
        q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.

    Returns:
        float: The marginal expected shortfall of firm $i$ at time $t$.

    References:
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Absorption Ratio](/measures/absorption_ratio/)
        * [Distress Insurance Premium](/measures/distress_insurance_premium/)
        * [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)
    """
    assert firm_returns.shape == market_returns.shape
    low_threshold = np.percentile(market_returns, q * 100)
    worst_days = np.argwhere(market_returns < low_threshold)
    return np.mean(firm_returns[worst_days])

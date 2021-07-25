import numpy as np


def kyle_lambda(
    returns: np.ndarray, prices: np.ndarray, volumes: np.ndarray
) -> np.ndarray:
    r"""Kyle's Lambda

    This approach is motivated by Kyles (1985) model in which liquidity is measured by
    a linear-regression estimate of the volume required to move the price of a security
    by one dollar. Sometimes referred to as Kyles lambda, this measure is an inverse proxy
    of liquidity, with higher values of lambda implying lower liquidity and market depth.
    The authors estimate this measure on a daily basis by using all transactions during normal
    trading hours on each day.

    Given the sequence of intraday returns $[R_{i,1}, R_{i,2}, ..., R_{i,T}]$,
    prices $[p_{i,1}, p_{i,2}, ..., p_{i,T}]$, and volumes $[v_{i,1}, v_{i,2}, ..., v_{i,T}]$ for security $i$
    during a specific day, the following regression is estimated:

    $$
    R_{i,t} = \alpha_i + \lambda_i \text{Sgn}(t) \ln(v_{i,t}\times p_{i,t}) + \varepsilon_{i,t}
    $$

    where $\text{Sgn}(t)$ is -1 or + 1 depending on the direction of the trade, i.e., “buy” or “sell”,
    as determined according to the following rule: if $R_{i,t}$ is positive, the value +1 is assigned
    to that transaction (to indicate net buying), and if $R_{i,t}$ is negative, the value −1 is
    assigned to that transaction (to indicate net selling). Any interval with zero return
    receives the same sign as that of the most recent transaction with a non-zero return
    (using returns from the prior day, if necessary).

    The description above is from Bisias, Lo, and Valavanis.

    Args:
        returns (np.ndarray): (n_securities, n_periods) array of security returns
        prices (np.ndarray): (n_securities, n_periods) array of security prices
        volumes (np.ndarray): (n_securities, n_periods) array of security volumes

    Returns:
        np.ndarray: array of Kyle's Lambda estimates for the securities

    Examples:
        >>> import numpy as np
        >>> from frds.measures import kyle_lambda
        >>> volumes = np.array(
        ...     [[180, 900, 970, 430, 110], [250, 400, 590, 260, 600], [700, 220, 110, 290, 310]]
        ... )
        >>> price_raw = np.array(
        ...     [[44, 39, 36, 28, 23, 18], [82, 81, 79, 40, 26, 13], [55, 67, 13, 72, 10, 65]]
        ... )
        >>> returns = np.diff(price_raw, axis=1) / np.roll(price_raw, 1)[:, 1:]
        >>> prices = price_raw[:, 1:]
        >>> kyle_lambda(returns, prices, volumes)
        array([-0.02198189, -0.1951004 ,  0.22752204])
    """

    assert returns.shape == prices.shape
    assert returns.shape == volumes.shape
    n_securities, n_periods = returns.shape

    lambdas = np.zeros(n_securities)
    for i in range(n_securities):
        mod_signs = np.zeros(n_periods)
        np.sign(returns[i], out=mod_signs)
        for t in range(1, len(mod_signs)):
            if mod_signs[t] == 0:
                mod_signs[t] = mod_signs[t - 1]

        X = np.vstack([np.ones(n_periods), mod_signs * np.log(prices[i] * volumes[i])])
        betas = np.linalg.lstsq(X.T, returns[i], rcond=None)[0]
        lambdas[i] = betas[1]

    return lambdas

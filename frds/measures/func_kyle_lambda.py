import numpy as np


def kyle_lambda(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    r"""Kyle's Lambda

    A measure of market impact cost from [Kyle (1985)](https://doi.org/10.2307/1913210),
    which can be interpreted as the cost of demanding a certain amount of liquidity over a given time period.

    It is also used as a measure of market liquidity and can be estimated by the
    volume required to move the price of a security by one dollar. Sometimes referred
    to as Kyles lambda, this measure is an inverse proxy of liquidity, with higher
    values of lambda implying lower liquidity and market depth. The authors estimate
    this measure on a daily basis by using all transactions during normal trading hours
    on each day.

    Given the sequence of

    * intraday returns $[R_{i,1}, R_{i,2}, ..., R_{i,T}]$,
    * prices $[p_{i,1}, p_{i,2}, ..., p_{i,T}]$, and
    * volumes $[v_{i,1}, v_{i,2}, ..., v_{i,T}]$

    for security $i$ during a specific day, the following regression is estimated:

    $$
    R_{i,t} = \alpha_i + \lambda_i \text{Sgn}(t) \ln(v_{i,t}\times p_{i,t}) + \varepsilon_{i,t}
    $$

    where $\text{Sgn}(t)$ is -1 or + 1 depending on the direction of the trade, i.e., “buy” or “sell”,
    as determined according to the following rule:

    * if $R_{i,t}$ is positive, the value +1 is assigned to that transaction (to indicate net buying),
    * if $R_{i,t}$ is negative, the value −1 is assigned to that transaction (to indicate net selling).

    Any interval with zero return receives the same sign as that of the most recent transaction with a non-zero return
    (using returns from the prior day, if necessary).

    Args:
        prices (np.ndarray): (n_securities, n_periods) array of security prices
        volumes (np.ndarray): (n_securities, n_periods) array of security volumes

    Returns:
        np.ndarray: Array of Kyle's Lambda estimates for the securities

        !!! note
            The `prices` data is used to calculate the asset returns, which is 1 less than the number of
            prices. As such, the `volumes` data of the first period is not used in the calculation.

    Examples:
        >>> import numpy as np
        >>> from frds.measures import kyle_lambda

        3 assets daily volume for 6 days
        >>> volumes = np.array(
        ...     [[100, 180, 900, 970, 430, 110], [200, 250, 400, 590, 260, 600], [300, 700, 220, 110, 290, 310]]
        ... )

        Their daily prices
        >>> prices = np.array(
        ...     [[44, 39, 36, 28, 23, 18], [82, 81, 79, 40, 26, 13], [55, 67, 13, 72, 10, 65]]
        ... )

        Calculate their respective Kyle's Lambda estimates.
        >>> kyle_lambda(prices, volumes)
        array([-0.02198189, -0.1951004 ,  0.22752204])

    References:
        * [Kyle (1985)](https://doi.org/10.2307/1913210),
            Continuous auctions and insider trading, *Econometrica*, 53, 1315-1355.
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.
    """

    assert volumes.shape == prices.shape

    n_securities, n_periods = prices.shape
    # number of returns is 1 less than prices
    n_periods -= 1

    # calculate the asset returns
    returns = np.diff(prices, axis=1) / np.roll(prices, 1)[:, 1:]

    # volumes and prices of the first period are not used
    volumes = np.delete(volumes, (0), axis=1)  # volumes = volumes[:,1:]
    prices = np.delete(prices, (0), axis=1)  # prices = prices[:, 1:]

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

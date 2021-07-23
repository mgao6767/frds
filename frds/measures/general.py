import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd

from ..algorithms.isolation_forest import anomaly_scores


def anomaly_score(
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
        pd.DataFrame: single-column indexed DataFrame

    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from frds.measures.general import anomaly_score
    >>> n_obs, n_attrs = 1_000, 100
    >>> np.random.seed(0)
    >>> data = pd.DataFrame(
    ...     {f"Attr{i}": np.random.normal(0, 1, n_obs) for i in range(n_attrs)},
    ...     index=[f"obs.{i}" for i in range(n_obs)],
    ... )
    >>> data.loc["obs.o"] = 10  # create an outlier
    >>> data.head()
               Attr0      Attr1      Attr2      Attr3      Attr4      Attr5  ...     Attr94     Attr95     Attr96     Attr97     Attr98     Attr99
    obs.0  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000  ...  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000
    obs.1   0.400157   0.892474  -1.711970   0.568722   1.843700  -0.737456  ...  -0.293003   1.439150  -1.912125  -0.041582  -0.359808  -0.910595
    obs.2   0.978738  -0.422315   0.046135  -0.114487   0.271091  -1.536920  ...   1.873511  -1.500278  -1.073391   0.462102   1.425187   0.187550
    obs.3   2.240893   0.104714  -0.958374   0.251630   1.136448  -0.562255  ...   2.274601   0.851165   0.774576   0.087572  -0.013497  -0.514234
    obs.4   1.867558   0.228053  -0.080812  -1.210856  -1.738332  -1.599511  ...   0.821944   0.512349  -0.263450  -0.583916  -0.972011  -0.527498
    [5 rows x 100 columns]
    >>> anomaly_score(data)     # check anomaly scores
            AnomalyScore
    obs.0        0.681283
    obs.1        0.511212
    obs.2        0.500675
    obs.3        0.496884
    obs.4        0.505786
    ...               ...
    obs.995      0.486229
    obs.996      0.488048
    obs.997      0.479535
    obs.998      0.493547
    obs.999      0.498489
    >>> # verify that obs.0 has the highest anomaly score
    >>> anomaly_score(data).sort_values(by="AnomalyScore", ascending=False).head()
            AnomalyScore
    obs.0        0.681283
    obs.766      0.527717
    obs.911      0.525643
    obs.676      0.525509
    obs.71       0.525022
    """
    return anomaly_scores(data, forest_size, tree_size, exclude_cols, random_seed, name)


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

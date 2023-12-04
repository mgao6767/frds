import numpy as np


def kyle_lambda(returns: np.ndarray, signed_dollar_volume: np.ndarray) -> float:
    r""":doc:`/measures/kyle_lambda`

    Args:
        returns (np.ndarray): ``(n,)`` array of stock returns
        signed_dollar_volume (np.ndarray): ``(n,)`` array of signed dollar volume

    Returns:
        float: Kyle's lambda (*100000)

        .. note::
           The return value is the estimated coefficient of :math:`\lambda` in equation
           :math:numref:`kylelambda_regression` multiplied by 100,000. Otherwise, the
           value would be too small and may cause issues.
    """
    returns = np.asarray(returns)
    signed_dollar_volume = np.asarray(signed_dollar_volume)
    x = np.vstack(
        [np.ones(len(signed_dollar_volume)), np.array(signed_dollar_volume)]
    ).T
    try:
        coef, _, _, _ = np.linalg.lstsq(x, np.array(signed_dollar_volume), rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return np.nan if np.isnan(coef[1]) else coef[1] * 1e6

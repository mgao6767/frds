import numpy as np


def kyle_lambda(returns: np.ndarray, signed_dollar_volume: np.ndarray) -> float:
    r""":doc:`/measures/kyle_lambda`

    Args:
        returns (np.ndarray): ``(n,)`` array of stock returns
        signed_dollar_volume (np.ndarray): ``(n,)`` array of signed dollar volume

    Returns:
        float: Kyle's lambda (*1000000)

        .. note::
           The return value is the estimated coefficient of :math:`\lambda` in equation
           :math:numref:`kylelambda_regression` multiplied by 1,000,000.
    """
    y = np.asarray(returns)
    x = np.asarray(signed_dollar_volume)
    x = x[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return a[0] * 1_000_000

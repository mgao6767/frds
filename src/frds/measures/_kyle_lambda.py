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
           :math:numref:`kylelambda_regression` multiplied by 1,000,000. Otherwise, the
           value would be too small and may cause issues.
    """
    returns = np.asarray(returns)
    signed_dollar_volume = np.asarray(signed_dollar_volume)
    coeff, _ = np.polyfit(signed_dollar_volume, returns, 1)
    return coeff * 1_000_000

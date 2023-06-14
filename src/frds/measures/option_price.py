from typing import Tuple
import numpy as np

from frds.utils import normal_cdf


def blsprice(
    S: float, K: float, r: float, T: float, sigma: float, q=0
) -> Tuple[float, float]:
    """European option values from Black-Scholes model

    Args:
        S (float): Current price of the underlying asset.
        K (float): Strike (exercise) price of the option.
        r (float): Annualized continuously compounded risk-free rate of return over the life of the option, expressed as a positive decimal number
        T (float): Time to expiration of the option, expressed in years.
        sigma (float): Annualized asset price volatility (i.e., annualized standard deviation of the continuously compounded asset return), expressed as a positive decimal number.
        q (int, optional): Annualized continuously compounded yield of the underlying asset over the life of the option, expressed as a decimal number. Defaults to 0.

    Returns:
        Tuple[float, float]: Prices of European call and put options

    Examples:
        >>> import numpy as np
        >>> from frds.measures.option_price import blsprice
        >>> blsprice(0.67, 0.7, 0.01, 5.0, 0.33, 0.002)
        (0.19003370474049647, 0.1925609132790535)
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * normal_cdf(d1) - K * np.exp(-r * T) * normal_cdf(d2)
    put = K * np.exp(-r * T) * normal_cdf(-d2) - S * np.exp(-q * T) * normal_cdf(-d1)
    return call, put

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from typing import Tuple

from frds.measures.option_price import blsprice


def merton_solution(
    x: Tuple[float, float],
    E: float,
    D: float,
    r: float,
    d: float,
    T: float,
    sigE: float,
) -> Tuple[float, float]:
    """Objective function for Merton solution, solving x

    This function is translated from Nagel's Matlab code `MertonSolution.m`

    Args:
        x (Tuple[float, float]): (A,sig) for asset value and asset volatility
        E (float): equity value
        D (float): debt value
        r (float): risk-free rate
        d (float): dividend yield
        T (float): time to maturity
        sigE (float): equity volatility

    Returns:
        Tuple[float, float]: tuple of estimation errors for asset value and volatility, to be used by `fsolve`
    """
    A, sig = x

    # esig and eA are used to penalize negative estimates
    esig, eA = 0, 0
    if sig < 0:
        sig, esig = 0, 9999999
    if A < 0:
        A, eA = 0, 9999999

    C, P = blsprice(A, D, r, T, sig, d)

    # add PV(dividends) to get cum dividend value
    PVd = A * (1 - np.exp(-d * T))
    C = C + PVd

    d1 = (np.log(A) - np.log(D) + (r - d + sig**2 / 2) * T) / (sig * np.sqrt(T))
    v = (np.exp(-d * T) * norm.cdf(d1) + (1 - np.exp(-d * T))) * (A / E) * sig

    # v = ((A)/(C))*sig;
    # A includes PV(div), but B-S derivative prices don't
    # Merton asset volatility too high

    penalty = esig * sig**2 + eA * A**2
    err = (E - C + penalty, sigE - v + penalty)

    return err


if __name__ == "__main__":
    A, sig = (0.677834482362600, 0.334594844841385)
    x0 = (A, sig)
    E = 0.011939293102641
    D = 0.7
    r = 0.01
    d = 0.002
    T = 5.0
    sigE = 0.669189689682770
    merton = lambda x0: merton_solution(x0, E, D, r, d, T, sigE)

    a = fsolve(merton, x0, maxfev=100_000)

    print(a)

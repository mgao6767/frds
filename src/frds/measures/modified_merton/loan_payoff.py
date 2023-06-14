import numpy as np
from scipy.stats import norm


def loan_payoff(
    F: float, f: float, ival: float, rho: float, sig: float, T: float
) -> float:
    """Loan payoff

    This function is translated from Nagel's Matlab code `LoanPayoff.m`

    Args:
        F (float): loan face value (either a scalar or a matrix the same size as f)
        f (float): log asset value factor realizations
        ival (float): initial log asset value
        rho (float): correlation of asset values
        sig (float): volatility of log asset values
        T (float): loan maturity

    Returns:
        float: loan payoff
    """
    # expected asset value conditional on common factor
    EA = np.exp(f + ival + 0.5 * (1 - rho) * T * sig**2)
    s = sig * np.sqrt(T) * np.sqrt(1 - rho)
    a = (np.log(F) - f - ival) / s
    # loan portfolio payoff at maturity
    L = EA * (1 - norm.cdf(s - a)) + norm.cdf(-a) * F

    return L

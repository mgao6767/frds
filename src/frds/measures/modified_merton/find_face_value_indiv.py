import numpy as np
from frds.measures.option_price import blsprice


def find_face_value_indiv(
    mu: float, F: float, ival: float, sig: float, T: float, r: float, d: float
) -> float:
    """Objective function to solve for promised loan yield

    This function is translated from Nagel's Matlab code `FindFaceValueIndiv.m`

    Args:
        mu (float): yield
        F (float): loan face value
        ival (float): initial log asset value
        sig (float): volatility of log asset values
        T (float): loan maturity
        r (float): risk-free rate
        d (float): dividend yield

    Returns:
        # Tuple[float, float, float]: error, new yield, loan face value
        float: error to be used by `fsolve`
    """

    C, P = blsprice(np.exp(ival), F, r, T, sig, d)

    L = F * np.exp(-r * T) - P
    # to do with call we would also need to subtract PV(depreciation)
    newmu = (1 / T) * np.log(F / L)

    err = mu - newmu

    # return err, newmu, L
    return err

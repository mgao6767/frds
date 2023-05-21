from typing import Tuple
import numpy as np
from scipy.optimize import minimize


def calc_Q_avg(data: np.ndarray) -> np.ndarray:
    """Compute the unconditional correlation matrix

    Args:
        data (np.ndarray): (2,T) array of firm and market volatility-adjusted returns

    Returns:
        np.ndarray: (2,2) array of unconditional correlation
    """
    _, T = data.shape
    Q_avg = np.zeros([2, 2])
    for t in range(T):
        e = data[:, t]
        Q_avg += np.outer(e, e)
    Q_avg /= T
    return Q_avg


def calc_Q(data: np.ndarray, a: float, b: float) -> list[np.ndarray]:
    """Calculate list of Q, the quasi correlation matrix

    Args:
        data (np.ndarray): (2,T) array of firm and market volatility-adjusted returns
        a (float): parameter `a` in DCC model
        b (float): parameter `b` in DCC model

    Returns:
        list[np.ndarray]: list of the quasi correlation matrices
    """
    _, T = data.shape
    Q_avg = calc_Q_avg(data)
    Qs = [Q_avg]
    omega = (1.0 - a - b) * Q_avg
    for t in range(T):
        et_1 = data[:, t]
        Qt_1 = Qs[-1]  # Q(t-1)
        # overflow error may occur
        Qt = omega + a * np.outer(et_1, et_1) + b * Qt_1
        Qs.append(Qt)
    return Qs


def calc_R(data: np.ndarray, a: float, b: float) -> list[np.ndarray]:
    """Calculate list of R, the conditional correlation matrix

    Args:
        data (np.ndarray): (2,T) array of firm and market volatility-adjusted returns
        a (float): parameter `a` in DCC model
        b (float): parameter `b` in DCC model

    Returns:
        list[np.ndarray]: list of the conditional correlation matrices
    """
    Qs = calc_Q(data, a, b)
    Rs = []
    for q in Qs:
        tmp = 1.0 / np.sqrt(np.abs(q))
        tmp = tmp * np.eye(2)
        R = np.dot(np.dot(tmp, q), tmp)
        if abs(R[0, 1]) >= 1:
            R[0, 1] = 0.9999 * (1 if R[0, 1] >= 0 else -1)
            R[1, 0] = R[0, 1]
        Rs.append(R)
    return Rs


def dcc(data: np.ndarray) -> Tuple[float, float]:
    """Estimate DCC

    Args:
        data (np.ndarray): (2,T) array of firm and market volatility-adjusted returns

    Returns:
        Tuple[float, float]: (a, b) for the DCC model
    """
    _, T = data.shape

    _calc_R = lambda a, b: calc_R(data, a, b)

    def loss_func(ab: np.ndarray) -> float:
        """Negative loglikelihood as a function of (a,b)

        Args:
            ab (np.ndarray): (2,) array of (a, b)

        Returns:
            float: negative log likelihood given (a,b)
        """
        a, b = ab[0], ab[1]
        if a < 0 or b < 0 or a > 1 or b > 1:
            return np.inf

        R = _calc_R(a, b)

        loss = 0.0
        for t in range(T):
            Rt = R[t]
            Rt_ = np.linalg.inv(Rt)
            et = data[:, t]
            det = np.linalg.det(Rt)
            # certain combination of (a,b) may lead to incorrect Rt
            if det <= 0:
                return np.inf
            loss += np.log(det) + np.dot(np.dot(et, Rt_), et)
        return loss

    # Solve for a, b
    res = minimize(
        loss_func,  # -ve log likelihood as a function of (a,b)
        [0.5, 0.5],  # initial values for a, b
        method="SLSQP",  # allow for constraints below
        constraints=[
            {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]},  # a+b<1
            {"type": "ineq", "fun": lambda x: x[0]},  # a>0
            {"type": "ineq", "fun": lambda x: x[1]},  # b>0
        ],
        options={"disp": False},
    )

    a, b = res.x[:2]  # a, b

    return a, b

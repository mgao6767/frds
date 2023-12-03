import warnings
from typing import Generator
from itertools import product
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, OptimizeResult


class PIN:
    """:doc:`/measures/probability_of_informed_trading`
    based on the canonical Easley et al (2002).
    """

    @dataclass
    class Parameters:
        alpha: float = np.nan
        delta: float = np.nan
        epsilon_b: float = np.nan
        epsilon_s: float = np.nan
        mu: float = np.nan
        loglikelihood: float = np.nan
        pin: float = np.nan
        method: str = ""

        def calculate_pin(self) -> float:
            self.pin = (
                self.alpha
                * self.mu
                / (self.alpha * self.mu + self.epsilon_b + self.epsilon_s)
            )
            return self.pin

    def __init__(self, B: np.ndarray, S: np.ndarray) -> None:
        """__init__

        Args:
            B (np.ndarray): ``(T,)`` array of buys
            S (np.ndarray): ``(T,)`` array of sells
        """
        assert isinstance(B, np.ndarray)
        assert isinstance(S, np.ndarray)
        assert B.shape == S.shape
        self.B = B
        self.S = S
        self.T = len(B)
        self.estimation_success = False
        self.parameters = type(self).Parameters()

    def estimate(self, method="EHO2010") -> Parameters:
        """Estimate PIN

        As in Yan and Zhang (2012), a grid search is performed to select the best
        initial parameters giving the highest log likelihood.

        Args:
            method (str, optional): estimation method ("EHO2010", "LK2011"). Defaults to "EHO2010".

        Returns:
            params: :class:`frds.measures.PIN.Parameters`
        """
        if method not in ("EHO2010", "LK2011"):
            raise ValueError("Unsupported PIN estimation method")

        # MLE via minimizing the negative log-likelihood
        # Set bounds for parameters
        bounds = [
            (0.0, 1.0),  # Bounds for alpha
            (0.0, 1.0),  # Boudns for delta
            (0.0, np.inf),  # Lower bound for eB
            (0.0, np.inf),  # Lower bound for eS
            (0.0, np.inf),  # Lower bound for mu
        ]

        # Reset params
        self.parameters = type(self).Parameters()
        if method == "EHO2010":
            func = self.loglikelihood_EHO2010
        elif method == "LK2011":
            func = self.loglikelihood_LK2011

        # fmt: off
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "",
                RuntimeWarning,
            )
            for starting_vals in self.init_params_grid_search():
                # alpha, delta, eB, eS, mu
                opt: OptimizeResult = minimize(
                    func,
                    starting_vals,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if opt.success:
                    loglikelihood = self.parameters.loglikelihood
                    if np.isnan(loglikelihood) or (-opt.fun<np.inf and -opt.fun > loglikelihood):
                        self.parameters = type(self).Parameters(*list(opt.x), loglikelihood=-opt.fun)
                        self.parameters.calculate_pin()
                        self.parameters.method = method
                        self.estimation_success = True
        return self.parameters

    def init_params_grid_search(self) -> Generator[tuple, None, None]:
        """Yields initial parameters for MLE

        The initial values are selected as in Yan and Zhang (2012).

        Yields:
            Generator[tuple, None, None]: (alpha, delta, eB, eS, mu)
        """
        grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha, delta, gamma in product(*[grid, grid, grid]):
            B_bar = np.average(self.B)
            eB = gamma * B_bar
            mu = (B_bar - eB) / (alpha * (1 - delta))
            eS = np.average(self.S) - alpha * delta * mu
            if eS < 0:
                continue
            yield (alpha, delta, eB, eS, mu)

    def loglikelihood_EHO2010(self, params: np.ndarray) -> float:
        """Log likelihood as in Easley, Hvidkjaer, and O’Hara (2010)

        Args:
            params (np.ndarray): (alpha, delta, eB, eS, mu)

        Returns:
            float: negative log likelihood
        """
        alpha, delta, eB, eS, mu = params
        xs = eS / (mu + eS)
        xb = eB / (mu + eB)
        B = self.B
        S = self.S
        M: np.ndarray = np.minimum(B, S) + np.maximum(B, S) / 2
        # fmt: off
        l = np.sum(-eB-eS + M * (np.log(xb) + np.log(xs)) +
                B * np.log(mu + eB) + S * np.log(mu + eS)) + \
            np.sum(np.log(alpha * (1 - delta) * np.exp(-mu) * xs**(S - M) * xb**(-M) +
                        alpha * delta * np.exp(-mu) * xb**(B - M) * xs**(-M) +
                        (1 - alpha) * xs**(S - M) * xb**(B - M)))
        # fmt: on
        return -l

    def loglikelihood_LK2011(self, params: np.ndarray) -> float:
        """Log likelihood as in Lin and Ke (2011)

        Args:
            params (np.ndarray): (alpha, delta, eB, eS, mu)

        Returns:
            float: negative log likelihood
        """
        alpha, delta, eB, eS, mu = params
        B = self.B
        S = self.S
        e_1i = -mu - S * np.log(1 + mu / eS)
        e_2i = -mu - B * np.log(1 + mu / eB)
        e_3i = -B * np.log(1 + mu / eB) - S * np.log(1 + mu / eS)
        e_max_i = np.maximum.reduce([e_1i, e_2i, e_3i])
        # fmt: off
        l = np.sum(-eB-eS + B * np.log(mu + eB) + S * np.log(mu + eS) + e_max_i) + \
            np.sum(np.log(alpha * (1 - delta) * np.exp(e_1i - e_max_i) +
                        alpha * delta * np.exp(e_2i - e_max_i) +
                        (1 - alpha) * np.exp(e_3i - e_max_i)))
        # fmt: on
        return -l

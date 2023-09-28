import warnings
import itertools
from typing import List
import numpy as np
from scipy.optimize import minimize, OptimizeResult


class GARCHModel:
    """:doc:`/algorithms/garch` model with constant mean and Normal noise

    It estimates the model parameters only. No standard errors calculated.

    This code is heavily based on the `arch <https://arch.readthedocs.io/>`_.
    Modifications are made for easier understaing of the code flow.
    """

    def __init__(self, returns: np.ndarray) -> None:
        """__init__

        Args:
            returns (np.ndarray): ``(T,)`` array of ``T`` returns

        .. note:: ``returns`` is best to be percentage returns for optimization
        """
        self.returns = np.asarray(returns, dtype=np.float64)

    def fit(self) -> List[float]:
        """Estimates the GARCH(1,1) parameters via MLE

        Returns:
            List[float]: [mu, omega, alpha, beta, loglikelihood]
        """
        # fmt: off
        # Step 1. Compute a starting value for volatility process by backcasting
        resids = self.returns - np.mean(self.returns)
        backcast = self._backcast(resids)

        # Step 2. Compute the starting values for MLE
        # Starting values for the volatility process
        var_params = self._starting_values(resids)
        # Starting value for mu is the sample mean return
        initial_mu = self.returns.mean()
        # Starting values are [mu, omega, alpha, beta]
        starting_vals = [initial_mu, *var_params]

        # Step 3. Compute a loose bound for the volatility process
        # This is to avoid NaN in MLE by avoiding zero/negative variance,
        # as well as unreasonably large variance.
        var_bounds = self._variance_bounds(resids)

        # Step 4. Set bounds for parameters
        bounds = [
            (-np.inf, np.inf), # No bounds for mu
            (1e-6, np.inf), # Lower bound for omega
            (0.0, 1.0), # Bounds for alpha 
            (0.0, 1.0), # Boudns for beta
        ]

        # Step 5. Set constraint for stationarity
        def persistence_smaller_than_one(params: List[float]):
            _, _, alpha, beta = params
            return 1.0 - (alpha + beta)

        # Step 6. MLE via minimizing the negative log-likelihood
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
            )
            opt: OptimizeResult = minimize(
                self._loglikelihood_model,
                starting_vals,
                args=(backcast, var_bounds),
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", 
                             "fun": persistence_smaller_than_one},
            )
            loglikelihood = -opt.fun
            estimated_params = list(opt.x)
            return [*estimated_params, loglikelihood]

    def _loglikelihood_model(
        self, params: np.ndarray, backcast: float, var_bounds
    ) -> float:
        """Calculates the negative log-likelihood based on the current ``params``.

        Args:
            params (np.ndarray): [mu, omega, alpha, beta]
            backcast (float): backcast value
            var_bounds (np.ndarray): variance bounds

        Returns:
            float: negative log-likelihood
        """
        mu, omega, alpha, beta = params
        resids = self.returns - mu
        var_params = [omega, alpha, beta]
        sigma2 = self._compute_variance(var_params, resids, backcast, var_bounds)
        return -self._loglikelihood(resids, sigma2)

    @staticmethod
    def _loglikelihood(resids: np.ndarray, sigma2: np.ndarray) -> float:
        """Computes the log-likelihood assuming residuals are
        normally distributed conditional on the variance.

        Args:
            resids (np.ndarray): residuals to use in computing log-likelihood.
            sigma2 (np.ndarray): conditional variance of residuals.

        Returns:
            float: log-likelihood
        """
        l = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids**2.0 / sigma2)
        return np.sum(l)

    def _compute_variance(
        self,
        params: List[float],
        resids: np.ndarray,
        backcast: float,
        var_bounds: np.ndarray,
    ) -> np.ndarray:
        """Computes the variances conditional on given parameters

        Args:
            params (List[float]): [omega, alpha, beta]
            resids (np.ndarray): residuals from mean equation
            backcast (float): backcast value
            var_bounds (_type_): variance bounds

        Returns:
            np.ndarray: conditional variance
        """
        omega, alpha, beta = params
        sigma2 = np.zeros_like(resids)
        sigma2[0] = omega + (alpha + beta) * backcast
        for t in range(1, len(resids)):
            sigma2[t] = omega + alpha * (resids[t - 1] ** 2) + beta * sigma2[t - 1]
            sigma2[t] = self._bounds_check(sigma2[t], var_bounds[t])
        return sigma2

    def _starting_values(self, resids: np.ndarray) -> List[float]:
        """Finds the optimal initial values for the volatility model via a grid
        search. For varying target persistence and alpha values, return the
        combination of alpha and beta that gives the highest loglikelihood.

        Args:
            resids (np.ndarray): residuals from the mean model

        Returns:
            List[float]: [omega, alpha, beta]
        """
        # Candidate target persistence - we don't have a prior knowledge here
        persistence_grid = [0.5, 0.7, 0.9, 0.98]
        # Candidate alpha values
        alpha_grid = [0.01, 0.05, 0.1, 0.2]
        # Sample variance
        var = self.returns.var()
        # Backcast initial value for the volaility process
        var_bounds = self._variance_bounds(resids)
        backcast = self._backcast(resids)
        initial_params = []
        max_likelihood = -np.inf
        for alpha, p in itertools.product(*[alpha_grid, persistence_grid]):
            # Note that (alpha+beta) is the "persistence"
            beta = p - alpha
            # Note that the unconditional variance is omega/(1-persistence).
            # As we are guessing the initial value, we use the sample variance
            # as the unconditional variance to guess the omega value.
            omega = var * (1 - p)
            params = [omega, alpha, beta]
            sigma2 = self._compute_variance(params, resids, backcast, var_bounds)
            if -self._loglikelihood(resids, sigma2) > max_likelihood:
                initial_params = params
        return initial_params

    def _variance_bounds(self, resids: np.ndarray) -> np.ndarray:
        """Compute bounds for conditional variances using EWMA.

        This function calculates the lower and upper bounds for conditional variances
        based on the residuals provided. The bounds are computed to ensure numerical
        stability during the parameter estimation process of GARCH models. The function
        uses Exponentially Weighted Moving Average (EWMA) to estimate the initial variance
        and then adjusts these estimates to lie within global bounds.

        Args:
            resids (np.ndarray): residuals from the mean model.

        Returns:
            np.ndarray: an array where each row contains the lower and upper bounds for the conditional variance at each time point.
        """

        T = len(resids)
        tau = min(75, T)
        # Compute initial variance using EWMA
        decay_factor = 0.94
        weights = decay_factor ** np.arange(tau)
        weights /= weights.sum()
        initial_variance = np.dot(weights, resids[:tau] ** 2)
        # Compute var_bound using EWMA (assuming ewma_recursion is defined)
        var_bound = self._ewma(resids, initial_variance)
        # Compute global bounds
        global_lower_bound = resids.var() / 1e8
        global_upper_bound = 1e7 * (1 + (resids**2).max())
        # Adjust var_bound to ensure it lies within global bounds
        var_bound = np.clip(var_bound, global_lower_bound, global_upper_bound)
        # Create bounds matrix
        var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T

        return np.ascontiguousarray(var_bounds)

    @staticmethod
    def _bounds_check(sigma2: float, var_bounds: np.ndarray) -> float:
        """Adjust the conditional variance at time t based on its bounds

        Args:
            sigma2 (float): conditional variance
            var_bounds (np.ndarray): lower and upper bounds

        Returns:
            float: adjusted conditional variance
        """
        lower, upper = var_bounds[0], var_bounds[1]
        sigma2 = max(lower, sigma2)
        if sigma2 > upper:
            if not np.isinf(sigma2):
                sigma2 = upper + np.log(sigma2 / upper)
            else:
                sigma2 = upper + 1000
        return sigma2

    @staticmethod
    def _backcast(resids: np.ndarray) -> float:
        """Computes the starting value for estimating conditional variance.

        Args:
            resids (np.ndarray): residuals

        Returns:
            float: initial value from backcasting
        """
        # Limit to first tau observations to reduce computation
        tau = min(75, resids.shape[0])
        # Weights for Exponential Weighted Moving Average (EWMA)
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)  # Ensure weights add up to 1
        # Let the initial value to be the EWMA of first tau observations
        return float(np.sum((resids[:tau] ** 2) * w))

    @staticmethod
    def _ewma(resids: np.ndarray, initial_value: float, lam=0.94) -> np.ndarray:
        """Compute the conditional variance estimates using
        Exponentially Weighted Moving Average (EWMA).

        Args:
            resids (np.ndarray): Residuals from the model.
            initial_value (float): Initial value for the conditional variance.
            lam (float): Decay factor for the EWMA.

        Returns:
            np.ndarray: Array containing the conditional variance estimates.
        """
        T = len(resids)
        variance = np.empty(T)
        variance[0] = initial_value  # Set the initial value
        # Compute the squared residuals
        squared_resids = resids**2
        # Compute the EWMA using the decay factors and squared residuals
        for t in range(1, T):
            variance[t] = lam * variance[t - 1] + (1 - lam) * squared_resids[t - 1]
        return variance


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_stata(
        "https://www.stata-press.com/data/r18/stocks.dta", convert_dates=["date"]
    )
    df.set_index("date", inplace=True)
    # Scale returns to percentage returns for better optimization results
    returns = df["nissan"].to_numpy() * 100

    g = GARCHModel(returns)
    res = g.fit()
    print(res)

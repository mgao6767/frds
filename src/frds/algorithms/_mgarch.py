import warnings
import itertools
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult


class GARCHModel_CCC:
    """:doc:`/algorithms/garch-ccc` model with the following specification:

    - Bivariate
    - Constant mean
    - Normal noise

    It estimates the model parameters only. No standard errors calculated.
    """

    def __init__(self, returns1: np.ndarray, returns2: np.ndarray) -> None:
        """__init__

        Args:
            returns1 (np.ndarray): ``(T,)`` array of ``T`` returns of first asset
            returns2 (np.ndarray): ``(T,)`` array of ``T`` returns of second asset

        .. note:: ``returns`` is best to be percentage returns for optimization
        """
        self.returns1 = np.asarray(returns1, dtype=np.float64)
        self.returns2 = np.asarray(returns2, dtype=np.float64)

    def fit(self) -> List[float]:
        """Estimates the Multivariate GARCH(1,1) parameters via MLE

        Returns:
            List[float]: [mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho, log-likelihood]
        """
        # fmt: off
        # Step 1. Compute a starting value for volatility process by backcasting
        resids1 = self.returns1 - np.mean(self.returns1)
        resids2 = self.returns2 - np.mean(self.returns2)
        backcast1 = self._backcast(resids1)
        backcast2 = self._backcast(resids2)

        # Step 2. Compute the starting values for MLE
        # Starting values for the volatility process
        [omega1, alpha1, beta1, omega2, alpha2, beta2, rho]= self._starting_values(resids1, resids2)
        # Starting value for mu is the sample mean return
        initial_mu1 = self.returns1.mean()
        initial_mu2 = self.returns2.mean()
        starting_vals = [initial_mu1, omega1, alpha1, beta1, initial_mu2, omega2, alpha2, beta2, rho]
        print(f"{starting_vals=}")

        # Step 3. Compute a loose bound for the volatility process
        # This is to avoid NaN in MLE by avoiding zero/negative variance,
        # as well as unreasonably large variance.
        var_bounds1 = self._variance_bounds(resids1)
        var_bounds2 = self._variance_bounds(resids2)

        # Step 4. Set bounds for parameters
        bounds = [
            # For first returns
            (None, None), # No bounds for mu
            (1e-6, None), # Lower bound for omega
            (0.0, 1.0), # Bounds for alpha 
            (0.0, 1.0), # Boudns for beta
            # For second returns
            (None, None), # No bounds for mu
            (1e-6, None), # Lower bound for omega
            (0.0, 1.0), # Bounds for alpha 
            (0.0, 1.0), # Boudns for beta
            # Constant correlation
            (-0.9, 0.9) # Bounds for rho
        ]

        # Step 5. Set constraint for stationarity
        def persistence_smaller_than_one_1(params: List[float]):
            mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
            return 1.0 - (alpha1 + beta1)

        def persistence_smaller_than_one_2(params: List[float]):
            mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
            return 1.0 - (alpha2 + beta2)

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
                args=(backcast1, backcast2, var_bounds1, var_bounds2),
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": persistence_smaller_than_one_1},
                             {"type": "ineq", "fun": persistence_smaller_than_one_2}]
            )
            loglikelihood = -opt.fun
            estimated_params = list(opt.x)
            print(opt.success)
            return [*estimated_params, loglikelihood]

    def _loglikelihood_model(
        self,
        params: np.ndarray,
        backcast1: float,
        backcast2: float,
        var_bounds1: np.ndarray,
        var_bounds2: np.ndarray,
    ) -> float:
        """Calculates the negative log-likelihood based on the current ``params``.

        Args:
            params (np.ndarray): [mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho]
            backcast1 (float): Backcast value for initializing the first return series variance.
            backcast2 (float): Backcast value for initializing the second return series variance.
            var_bounds1 (np.ndarray): Array of variance bounds for the first return series.
            var_bounds2 (np.ndarray): Array of variance bounds for the second return series.

        Returns:
            float: negative log-likelihood
        """
        mu1, omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho = params
        resids1 = self.returns1 - mu1
        resids2 = self.returns2 - mu2
        var_params1 = [omega1, alpha1, beta1]
        var_params2 = [omega2, alpha2, beta2]
        sigma2_1, sigma2_2, sigma2_12 = self._compute_bivariate_variance(
            var_params1,
            var_params2,
            rho,
            resids1,
            resids2,
            backcast1,
            backcast2,
            var_bounds1,
            var_bounds2,
        )
        return -self._loglikelihood_bivariate(resids1, sigma2_1, resids2, sigma2_2, rho)

    def _loglikelihood_bivariate(
        self,
        resids1: np.ndarray,
        sigma2_1: np.ndarray,
        resids2: np.ndarray,
        sigma2_2: np.ndarray,
        rho: float,
    ) -> float:
        """
        Computes the log-likelihood for a bivariate GARCH(1,1) model with constant correlation.

        Args:
            resids1 (np.ndarray): Residuals for the first return series.
            sigma2_1 (np.ndarray): Array of conditional variances for the first return series.
            resids2 (np.ndarray): Residuals for the second return series.
            sigma2_2 (np.ndarray): Array of conditional variances for the second return series.
            rho (float): Constant correlation.

        Returns:
            float: The log-likelihood value for the bivariate model.

        """
        # z1 and z2 are standardized residuals
        z1 = resids1 / np.sqrt(sigma2_1)
        z2 = resids2 / np.sqrt(sigma2_2)
        # fmt: off
        log_likelihood_terms = -0.5 * (
            2 * np.log(2 * np.pi) 
            + np.log(sigma2_1 * sigma2_2 * (1 - rho ** 2))
            + (z1 ** 2 / sigma2_1 + z2 ** 2 / sigma2_2 - 2 * rho * z1 * z2 / np.sqrt(sigma2_1 * sigma2_2)) / (1 - rho ** 2)
        )
        
        log_likelihood = np.sum(log_likelihood_terms)
        return log_likelihood

    def _compute_bivariate_variance(
        self,
        params1: List[float],
        params2: List[float],
        rho: float,
        resids1: np.ndarray,
        resids2: np.ndarray,
        backcast1: float,
        backcast2: float,
        var_bounds1: np.ndarray,
        var_bounds2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the bivariate variances and covariances for two time series
        following the Constant Conditional Correlation (CCC) GARCH(1,1) model.

        Args:
            params1 (List[float]): Parameters (omega, alpha, beta) for the first return series.
            params2 (List[float]): Parameters (omega, alpha, beta) for the second return series.
            rho (float): Constant correlation coefficient between the two return series.
            resids1 (np.ndarray): Array of residuals for the first return series.
            resids2 (np.ndarray): Array of residuals for the second return series.
            backcast1 (float): Backcast value for initializing the first return series variance.
            backcast2 (float): Backcast value for initializing the second return series variance.
            var_bounds1 (np.ndarray): Array of variance bounds for the first return series.
            var_bounds2 (np.ndarray): Array of variance bounds for the second return series.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Conditional variances (sigma2_1, sigma2_2) and covariance (sigma2_12) arrays.
        """
        # fmt: off
        omega1, alpha1, beta1 = params1
        omega2, alpha2, beta2 = params2

        T = len(resids1)
        sigma2_1 = np.zeros_like(resids1)
        sigma2_2 = np.zeros_like(resids2)
        sigma2_12 = np.zeros_like(resids1)  # bivariate covariance
        
        sigma2_1[0] = omega1 + (alpha1 + beta1) * backcast1
        sigma2_2[0] = omega2 + (alpha2 + beta2) * backcast2
        sigma2_12[0] = rho * np.sqrt(sigma2_1[0] * sigma2_2[0])

        for t in range(1, T):
            # z_tm1 = resids1[t-1] / np.sqrt(sigma2_1[t-1])
            z_tm1 = resids1[t-1]
            sigma2_1[t] = omega1 + alpha1 * (z_tm1 ** 2) + beta1 * sigma2_1[t - 1]
            sigma2_1[t] = self._bounds_check(sigma2_1[t], var_bounds1[t])

            # z_tm1 = resids2[t-1] / np.sqrt(sigma2_2[t-1])
            z_tm1 = resids2[t-1]
            sigma2_2[t] = omega2 + alpha2 * (z_tm1 ** 2) + beta2 * sigma2_2[t - 1]
            sigma2_2[t] = self._bounds_check(sigma2_2[t], var_bounds2[t])

            sigma2_12[t] = rho * np.sqrt(sigma2_1[t] * sigma2_2[t])

        return sigma2_1, sigma2_2, sigma2_12

    def _starting_values(self, resids1: np.ndarray, resids2: np.ndarray) -> List[float]:
        """Finds the optimal initial values for the volatility model via a grid
        search. For varying target persistence and alpha values, return the
        combination of alpha and beta that gives the highest loglikelihood.

        Args:
            resids1 (np.ndarray): Array of residuals for the first return series.
            resids2 (np.ndarray): Array of residuals for the second return series.

        Returns:
            List[float]: [omega1, alpha1, beta1, mu2, omega2, alpha2, beta2, rho]
        """
        # Candidate target persistence - we don't have a prior knowledge here
        persistence_grid1 = [0.5, 0.7, 0.9]
        persistence_grid2 = persistence_grid1
        # Candidate alpha values
        alpha_grid1 = [0.01, 0.05, 0.1, 0.2]
        alpha_grid2 = alpha_grid1
        # Constant correlation
        rho_grid = [-0.3, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6]
        # Sample variance
        var1 = self.returns1.var()
        var2 = self.returns2.var()
        # Standarized residuals
        z1 = resids1 / np.sqrt(var1)
        z2 = resids2 / np.sqrt(var2)
        # Backcast initial value for the volaility process
        var_bounds1 = self._variance_bounds(z1)
        var_bounds2 = self._variance_bounds(z2)
        backcast1 = self._backcast(z1)
        backcast2 = self._backcast(z2)
        initial_params = []
        max_likelihood = -np.inf
        for alpha1, alpha2, p1, p2, rho in itertools.product(
            *[alpha_grid1, alpha_grid2, persistence_grid1, persistence_grid2, rho_grid]
        ):
            # Note that (alpha+beta) is the "persistence"
            beta1 = p1 - alpha1
            beta2 = p2 - alpha2
            # Note that the unconditional variance is omega/(1-persistence).
            # As we are guessing the initial value, we use the sample variance
            # as the unconditional variance to guess the omega value.
            omega1 = var1 * (1 - p1)
            omega2 = var2 * (1 - p2)
            params1 = [omega1, alpha1, beta1]
            params2 = [omega2, alpha2, beta2]

            sigma2_1, sigma2_2, sigma2_12 = self._compute_bivariate_variance(
                params1,
                params2,
                rho,
                z1,
                z2,
                backcast1,
                backcast2,
                var_bounds1,
                var_bounds2,
            )
            z1 = resids1 / np.sqrt(sigma2_1)
            z2 = resids2 / np.sqrt(sigma2_2)
            # fmt: off
            if -self._loglikelihood_bivariate(z1, sigma2_1, z2, sigma2_2, rho) > max_likelihood:
                initial_params = [*params1, *params2, rho]

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
    returns1 = df["toyota"].to_numpy() * 100
    returns1 = df["nissan"].to_numpy() * 100

    model = GARCHModel_CCC(returns1, returns1)
    res = model.fit()
    print(res)

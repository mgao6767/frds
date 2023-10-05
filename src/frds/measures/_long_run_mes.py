from typing import Tuple
import numpy as np
from frds.algorithms import GJRGARCHModel, GJRGARCHModel_DCC


class LongRunMarginalExpectedShortfall:
    """:doc:`/measures/long_run_mes`"""

    def __init__(self, firm_returns: np.ndarray, market_returns: np.ndarray) -> None:
        """__init__

        Args:
            firm_returns (np.ndarray): ``(n_days,)`` array of the firm raw returns.
            market_returns (np.ndarray): ``(n_days,)`` array of the market raw returns.

        .. note::

           Raw returns should be used! They are automatically converted to log returns.
           Do NOT use percentage returns.
        """
        # Convert raw returns to log returns
        self.firm_returns = np.log(1 + np.asarray(firm_returns, dtype=np.float64))
        self.market_returns = np.log(1 + np.asarray(market_returns, dtype=np.float64))
        assert self.firm_returns.shape == self.market_returns.shape
        # Scale to percentage (log) returns
        # This is for better (GJR)GARCH estimation
        self.firm_returns *= 100
        self.market_returns *= 100
        self.firm_model = GJRGARCHModel(self.firm_returns)
        self.market_model = GJRGARCHModel(self.market_returns)
        self.dcc_model = GJRGARCHModel_DCC(self.firm_model, self.market_model)

    def estimate(self, h=22, S=10_000, C=-0.1, random_seed=42) -> float:
        """h-step-ahead LRMES forecasts conditional on a systemic event of market decline C

        Args:
            h (int, optional): h-period-ahead prediction horizon. Defaults to 22.
            S (int, optional): sample size used in simulation to generate LRMES forecasts. Defaults to 10000.
            C (float, optional): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
            random_seed (int, optional): random seed. Defaults to 42.

        Returns:
            float: the firm's LRMES forecast
        """
        rng = np.random.default_rng(random_seed)

        # Estimate the (GJR)GARCH-DCC model
        self.dcc_model.fit()

        # Construct GJR-GARCH-DCC standardized innovations for the sample
        # See the step 1 of Computing LRMES section
        firm_variances = self.firm_model.sigma2
        firm_resids = self.firm_model.resids
        firm_mu = self.firm_model.parameters.mu
        market_variances = self.market_model.sigma2
        market_resids = self.market_model.resids
        market_mu = self.market_model.parameters.mu
        # Conditional correlations
        a, b = self.dcc_model.parameters.a, self.dcc_model.parameters.b
        rho = self.dcc_model.conditional_correlations(a, b)
        # Standarized residuals
        z_m = (market_resids - market_mu) / np.sqrt(market_variances)
        z_i = (firm_resids - firm_mu) / np.sqrt(firm_variances)
        # Firm shock orthogonal to market
        xi_i = (z_i - rho * z_m) / np.sqrt(1 - rho**2)
        sample = np.array([xi_i, z_m])

        # Sample with replacement S*h innovations
        sample = sample.T[rng.choice(sample.shape[1], (S, h), replace=True)]
        assert sample.shape == (S, h, 2)

        firm_var = self.firm_model.sigma2[-1]
        mkt_var = self.market_model.sigma2[-1]
        a, b = self.dcc_model.parameters.a, self.dcc_model.parameters.b
        rho = self.dcc_model.conditional_correlations(a, b)[-1]
        Q_bar = np.cov(z_i, z_m)

        firm_avg_return = 0.0
        n_systemic_event = 0
        for s in range(S):
            # Each simulation
            inv = sample[s, :, :]  # shape=(h,2)
            assert inv.shape == (h, 2)
            firm_return, systemic_event = self.simulation(
                inv, C, firm_var, mkt_var, a, b, rho, Q_bar
            )
            firm_avg_return += firm_return
            n_systemic_event += systemic_event

        if n_systemic_event == 0.0:
            return 0.0
        return firm_avg_return / n_systemic_event

    def simulation(
        self,
        innovation: np.ndarray,
        C: float,
        firm_var: float,
        mkt_var: float,
        a: float,
        b: float,
        rho: float,
        Q_bar: np.ndarray,
    ) -> Tuple[float, bool]:
        """A simulation to compute the firm's return given the parameters.
        This method should be used internally.

        Args:
            innovation (np.ndarray): ``(h,2)`` array of market and firm innovations
            C (float): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
            firm_var (float): the firm conditional variance at time :math:`T`, used as starting value in forecast
            mkt_var (float): the market conditional variance at time :math:`T`, used as starting value in forecast
            a (float): DCC parameter
            b (float): DCC parameter
            rho (float): the last conditional correlation at time :math:`T`, used as starting value in forecast
            Q_bar (np.ndarray): ``(2,2)`` array of sample correlation matrix of standarized residuals

        Returns:
            Tuple[float, bool]: tuple of the firm return and whether a systemic event occurs
        """
        q_i_bar = Q_bar[0, 0]
        q_m_bar = Q_bar[1, 1]
        q_im_bar = Q_bar[1, 0]
        q_i, q_m, q_im = 1.0, 1.0, rho

        pi = self.firm_model.parameters
        mu_i = pi.mu
        omega_i, alpha_i, gamma_i, beta_i = pi.omega, pi.alpha, pi.gamma, pi.beta

        pm = self.market_model.parameters
        mu_m = pm.mu
        omega_m, alpha_m, gamma_m, beta_m = pm.omega, pm.alpha, pm.gamma, pm.beta

        firm_return = np.empty(shape=len(innovation))
        mkt_return = np.empty(shape=len(innovation))

        for h in range(len(innovation)):
            # Each iteration is a one-step-ahead forecast
            firm_innov = innovation[h, 0]
            firm_var = omega_i + alpha_i * firm_innov**2 + beta_i * firm_var
            firm_var += gamma_i * firm_innov**2 if firm_innov < 0 else 0

            mkt_innov = innovation[h, 1]
            mkt_var = omega_m + alpha_m * mkt_innov**2 + beta_m * mkt_var
            mkt_var += gamma_m * mkt_innov**2 if mkt_innov < 0 else 0

            q_i = (1 - a - b) * q_i_bar + a * firm_innov**2 + b * q_i
            q_m = (1 - a - b) * q_m_bar + a * mkt_innov**2 + b * q_m
            q_im = (1 - a - b) * q_im_bar + a * firm_innov * mkt_innov + b * q_im

            rho = q_im / np.sqrt(q_i * q_m)

            mkt_return[h] = mu_m + rho * mkt_innov
            firm_return[h] = mu_i + rho * firm_innov

        # Convert back to original scale
        mkt_return /= 100
        firm_return /= 100

        # systemic event if over the prediction horizon,
        # the market falls by more than C
        systemic_event = np.exp(np.sum(mkt_return)) - 1 < C
        # no need to simulate firm returns if there is no systemic event
        if not systemic_event:
            return 0.0, False

        # firm return over the horizon
        firmret = np.exp(np.sum(firm_return)) - 1

        return firmret, True

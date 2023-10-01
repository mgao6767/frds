import numpy as np
from warnings import warn
from typing import Union
from functools import lru_cache
from arch import arch_model
from frds.algorithms.dcc import dcc, calc_Q_avg, calc_Q, calc_R


class SRISK:
    """:doc:`/measures/srisk` of firm(s) or market at a given time"""

    def __init__(
        self,
        firm_returns: np.ndarray,
        market_returns: np.ndarray,
        W: Union[float, np.ndarray],
        D: Union[float, np.ndarray],
    ) -> None:
        """__init__

        Args:
            firm_returns (np.ndarray): ``(n_days,)`` array of firm log returns.
                Can also be ``(n_day,n_firms)`` array of multiple firms' log returns.
            market_returns (np.ndarray): ``(n_days,)`` array of market log returns
            W (float | np.ndarray): market value of equity. It can be either a
                single float value for a firm or a ``(n_firms,)`` array for multiple firms.
            D (float | np.ndarray): book value of debt. It can be either a
                single float value for a firm or a ``(n_firms,)`` array for multiple firms.

        .. note::

            If ``firm_returns`` is a ``(n_day,n_firms)``,
            then ``W`` and ``D`` must be of shape ``(n_firms,)``.

        """
        warn(f"{type(self)} is not numerically stable! Do not use!")
        if len(firm_returns.shape) == 1:
            # Single firm
            n_firms = 1
            assert firm_returns.shape == market_returns.shape
            firm_returns = firm_returns.reshape((firm_returns.shape[0], 1))
            assert isinstance(D, float) and isinstance(W, float)
        else:
            # Multple firms
            n_days, n_firms = firm_returns.shape
            assert n_firms > 1
            assert market_returns.shape[0] == n_days
            assert isinstance(D, np.ndarray) and isinstance(W, np.ndarray)
            assert D.shape == W.shape
            assert D.shape == np.zeros((n_firms,)).shape

        self.firm_returns = firm_returns
        self.market_returns = market_returns
        self.W = W
        self.D = D
        self.n_firms = n_firms
        self.n_days = n_days

    @lru_cache()
    def lrmes(
        self,
        h=22,
        S=10_000,
        C=-0.1,
        random_seed=42,
    ) -> np.ndarray:
        """h-step-ahead LRMES forecasts conditional on a systemic event of market decline C

        Args:
            h (int, optional): h-period-ahead prediction horizon. Defaults to 22.
            S (int, optional): sample size used in simulation to generate LRMES forecasts. Defaults to 10000.
            C (float, optional): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
            random_seed (int, optional): random seed. Defaults to 42.

        Returns:
            np.ndarray: ``(n_firms,)`` array of LRMES forecasts
        """
        # Fit GJR-GARCH for the market returns
        mkt_am = arch_model(self.market_returns, p=1, o=1, q=1, rescale=True)
        mkt_res = mkt_am.fit(update_freq=0, disp=False)
        epsilon_mkt = self.market_returns / mkt_res.conditional_volatility
        # Forecasted volatility
        mkt_vol_hat = np.sqrt(
            np.squeeze(
                mkt_res.forecast(
                    mkt_res.params, horizon=h, reindex=False
                ).variance.T.to_numpy()
            )
        )  # (h,1) array of volatility forecasts

        lrmes = np.zeros((self.n_firms,))
        for i in range(self.n_firms):
            # Fit GJR-GARCH for each firm's returns
            firm_am = arch_model(self.firm_returns[:, i], p=1, o=1, q=1, rescale=True)
            firm_res = firm_am.fit(update_freq=0, disp=False)
            epsilon_firm = self.firm_returns[:, i] / firm_res.conditional_volatility
            # Forecasted volatility
            firm_vol_hat = np.sqrt(
                np.squeeze(
                    firm_res.forecast(
                        firm_res.params, horizon=h, reindex=False
                    ).variance.T.to_numpy()
                )
            )  # (h,1) array of volatility forecasts

            # Estimate DCC for each firm-market pair
            epsilon = np.array([epsilon_firm, epsilon_mkt])
            a, b = dcc(epsilon)  # params in DCC model
            Q_avg = calc_Q_avg(epsilon)
            Q = calc_Q(epsilon, a, b)  # Qt for training data
            R = calc_R(epsilon, a, b)  # Rt for training data

            # DCC predictions for correlations
            et = epsilon[:, -1]
            Q_Tplus1 = (1 - a - b) * Q_avg + a * np.outer(et, et) + b * Q[-1]

            diag_q = 1.0 / np.sqrt(np.abs(Q_Tplus1))
            diag_q = diag_q * np.eye(2)
            R_Tplus1 = np.dot(np.dot(diag_q, Q_Tplus1), diag_q)

            diag_q = 1.0 / np.sqrt(np.abs(Q_avg))
            diag_q = diag_q * np.eye(2)
            R_avg = np.dot(np.dot(diag_q, Q_avg), diag_q)

            Rhat = []
            for _h in range(1, h + 1):
                _ab = np.power(a + b, _h - 1)
                # R_Tplus_h is correlation matrix for T+h, symmetric 2*2 matrix
                R_Tplus_h = (1 - _ab) * R_avg + _ab * R_Tplus1
                # In case predicted correlation is larger than 1
                if abs(R_Tplus_h[0, 1]) >= 1:
                    R_Tplus_h[0, 1] = 0.9999 * (1 if R_Tplus_h[0, 1] >= 0 else -1)
                    R_Tplus_h[1, 0] = R_Tplus_h[0, 1]
                Rhat.append(R_Tplus_h)

            # Sample innovations
            innov = np.zeros((self.n_days,))
            for t in range(self.n_days):
                rho = R[t][0, 1]
                innov[t] = (epsilon_firm[t] - epsilon_mkt[t] * rho) / np.sqrt(
                    1 - rho**2
                )
            sample = np.array([epsilon_mkt, innov])  # shape=(2,n_days)

            # Sample S*h pairs of standardized innovations
            rng = np.random.RandomState(random_seed)
            # sample.shape=(S,h,2)
            sample = sample.T[rng.choice(sample.shape[1], (S, h), replace=True)]

            # list of simulated firm total returns when there're systemic events
            firm_total_return = []
            for s in range(S):
                # Each simulation
                inv = sample[s, :, :]  # (h,2)
                # mkt log return = mkt innovation * predicted mkt volatility
                mktrets = np.multiply(inv[:, 0], mkt_vol_hat)

                # systemic event if over the prediction horizon,
                # the market falls by more than C
                systemic_event = np.exp(np.sum(mktrets)) - 1 < C
                # no need to simulate firm returns if there is no systemic event
                if not systemic_event:
                    continue
                # when there is a systemic event
                firmrets = np.zeros((h,))
                for _h in range(h):
                    mktinv, firminv = inv[_h][0], inv[_h][1]
                    # Simulated firm return at T+h
                    rho = Rhat[_h][0, 1]
                    firmret = firminv * np.sqrt(1 - rho**2) + rho * mktinv
                    firmret = firm_vol_hat[_h] * firmret
                    firmrets[_h] = firmret
                # firm return over the horizon
                firmret = np.exp(np.sum(firmrets)) - 1
                firm_total_return.append(firmret)

            # Store result
            if len(firm_total_return):
                lrmes[i] = np.mean(firm_total_return)
            else:
                lrmes[i] = np.nan

        return lrmes

    def estimate(
        self,
        k=0.08,
        lrmes_h=22,
        lrmes_S=10000,
        lrmes_C=-0.1,
        lrmes_random_seed=42,
        aggregate_srisk=False,
    ) -> Union[np.ndarray, float]:
        """estimate

        Args:
            k (float, optional): prudential capital factor. Defaults to 8%.
            lrmes_h (int, optional): parameter used to estimate :func:`lrmes`. Prediction horizon. Defaults to 22.
            lrmes_S (int, optional): parameter used to estimate `LRMES`. The number of simulations. Defaults to 10_000.
            lrmes_C (float, optional): parameter used to estimate `LRMES`. The markdown decline that defines a systemic event. Defaults to -0.1.
            lrmes_random_seed (int, optional): random seed in estimating `LRMES`. Defaults to 42.
            aggregate_srisk (bool, optional): whether to compute the aggregate SRISK. Defaults to False.

        Returns:
            np.ndarray | float: If ``aggregate_srisk=False``, ``(n_firms,)`` array of firm-level SRISK measures. Otherwise, a single float value for aggregate SRISK.
        """
        # Firm-level LRMES
        LRMES = self.lrmes(
            h=lrmes_h,
            S=lrmes_S,
            C=lrmes_C,
            random_seed=lrmes_random_seed,
        )  # (n_firms,) array of LRMES estimates

        LVG = (self.D + self.W) / self.W

        SRISK = self.W * (k * LVG + (1 - k) * LRMES - 1)
        if not aggregate_srisk:
            return SRISK
        else:
            return np.sum(SRISK.clip(min=0.0))

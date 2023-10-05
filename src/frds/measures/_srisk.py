import numpy as np
from typing import Union

from frds.measures import LongRunMarginalExpectedShortfall


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
        if len(firm_returns.shape) == 1:
            # Single firm
            n_firms = 1
            n_days = len(firm_returns)
            assert firm_returns.shape == market_returns.shape
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
        market_returns = self.market_returns

        if self.n_firms == 1:
            lrmes = LongRunMarginalExpectedShortfall(
                self.firm_returns, market_returns
            ).estimate(lrmes_h, lrmes_S, lrmes_C, lrmes_random_seed)
        else:
            lrmes = np.empty(self.n_firms)
            for i in range(self.n_firms):
                firm_returns = self.firm_returns[:, i]
                lrmes[i] = LongRunMarginalExpectedShortfall(
                    firm_returns, market_returns
                ).estimate(lrmes_h, lrmes_S, lrmes_C, lrmes_random_seed)

        lvg = (self.D + self.W) / self.W
        srisk = self.W * (k * lvg + (1 - k) * lrmes - 1)
        if not aggregate_srisk:
            return srisk
        else:
            return np.sum(srisk.clip(min=0.0))

import numpy as np


class MarginalExpectedShortfall:
    """:doc:`/measures/marginal_expected_shortfall`"""

    def __init__(self, firm_returns: np.ndarray, market_returns: np.ndarray) -> None:
        """__init__

        Args:
            firm_returns (np.ndarray): ``(n_days,)`` array of the returns (equity or CDS) for the firm.
            market_returns (np.ndarray): ``(n_days,)`` array of the returns (equity or CDS) for the market as a whole.
        """
        assert firm_returns.shape == market_returns.shape
        self.firm_returns = firm_returns
        self.market_returns = market_returns

    def estimate(self, q: float = 0.05) -> float:
        """estiamte

        Args:
            q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.

        Returns:
            float: The marginal expected shortfall of firm :math:`i` at time :math:`t`.
        """
        assert 0 <= q <= 1
        low_threshold = np.percentile(self.market_returns, q * 100)
        worst_days = np.argwhere(self.market_returns < low_threshold)
        return np.mean(self.firm_returns[worst_days])

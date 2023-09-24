from decimal import Decimal, ROUND_HALF_UP
from functools import cached_property
import numpy as np


class AbsorptionRatio:
    """:doc:`/measures/absorption_ratio`"""

    def __init__(self, asset_returns: np.ndarray) -> None:
        """__init__

        Args:
            asset_returns (np.ndarray): ``(n_assets, n_days)`` arrays of asset returns.
        """
        self.asset_returns = asset_returns

    def estimate(self, fraction_eigenvectors: float = 0.2) -> float:
        """estimate
        Estimate

        Args:
            fraction_eigenvectors (float, optional): The fraction of eigenvectors used to calculate the absorption ratio. Defaults to 0.2 as in the paper.

        Returns:
            float: Absorption ratio for the market
        """
        n_assets, _ = self.asset_returns.shape
        eig_sorted = sorted(self.eigvals)
        # fmt: off
        num_eigenvalues = int(Decimal(fraction_eigenvectors * n_assets).to_integral_value(rounding=ROUND_HALF_UP))
        return sum(eig_sorted[len(eig_sorted) - num_eigenvalues :]) / np.trace(self.asset_covariance)

    @cached_property
    def asset_covariance(self) -> np.ndarray:
        """asset_covariance
        Asset returns covariance (cached)

        Returns:
            np.ndarray: covariance of asset returns
        """
        return np.cov(self.asset_returns)

    @cached_property
    def eigvals(self) -> np.ndarray:
        """eigvals
        Eigenvalues of :func:`asset_covariance` (cached)

        Returns:
            np.ndarray: eigenvalues
        """
        return np.linalg.eigvals(self.asset_covariance)

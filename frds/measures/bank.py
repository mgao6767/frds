from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def absorption_ratio(
    asset_returns: np.ndarray, fraction_eigenvectors: float = 0.2
) -> float:
    """Kritzman, Li, Page, and Rigobon (2010) propose to measure systemic risk via the Absorption Ratio (AR),\
    which they define as the fraction of the total variance of a set of asset returns explained or absorbed by \
    a fixed number of eigenvectors. The absorption ratio captures the extent to which markets are unified or tightly coupled. \
    When markets are tightly coupled, they become more fragile in the sense that negative shocks propagate more quickly \
    and broadly than when markets are loosely linked. The authors apply their AR analysis to several broad markets, \
    introduce a standardized measure of shifts in the AR, and analyze how these shifts relate to changes in asset prices and financial turbulence.

    A high value for the absorption ratio corresponds to a high level of systemic risk because it implies the sources of risk are more unified. \
    A low absorption ratio indicates less systemic risk because it implies the sources of risk are more disparate. \
    High systemic risk does not necessarily lead to asset depreciation or financial turbulence. \
    It is simply an indication of market fragility in the sense that a shock is more likely to propagate quickly and broadly when sources of risk are tightly coupled.

    The description above is from Bisias, Lo, and Valavanis.

    Args:
        asset_returns (np.ndarray): 2d arrays of asset returns such that `n_assets, n_days = asset_returns.shape`
        fraction_eigenvectors (float, optional): The fraction of eigenvectors used to calculate the absorption ratio. In the paper it is 0.2. Defaults to 0.2.

    Returns:
        float: absorption ratio
    """

    n_assets, _ = asset_returns.shape
    cov = np.cov(asset_returns)
    eig = np.linalg.eigvals(cov)
    eig_sorted = sorted(eig)
    num_eigenvalues = int(
        Decimal(fraction_eigenvectors * n_assets).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    return sum(eig_sorted[len(eig_sorted) - num_eigenvalues :]) / np.trace(cov)

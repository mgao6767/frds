from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def estimate(asset_returns: np.ndarray, fraction_eigenvectors: float = 0.2) -> float:
    """Absorption Ratio

    Args:
        asset_returns (np.ndarray): (n_assets, n_days) arrays of asset returns
        fraction_eigenvectors (float, optional): The fraction of eigenvectors used to calculate the absorption ratio. Defaults to 0.2 as in the paper.

    Returns:
        float: Absorption ratio for the market

    Examples:
        >>> import numpy as np
        >>> from frds.measures.absorption_ratio import estimate

        3 assets daily returns for 6 days
        >>> data = np.array(
        ...             [
        ...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
        ...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
        ...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
        ...             ]
        ...         )

        Calculate the absorption ratio.
        >>> estimate(data)
        0.7746543307660252

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

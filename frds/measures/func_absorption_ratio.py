from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def absorption_ratio(
    asset_returns: np.ndarray, fraction_eigenvectors: float = 0.2
) -> float:
    """Absorption Ratio
    
    A measure of systemic risk defined as the fraction of the total variance of a set of asset returns explained or absorbed by \
    a fixed number of eigenvectors.

    Proposed by [Kritzman, Li, Page, and Rigobon (2010)](https://doi.org/10.3905/jpm.2011.37.4.112), the absorption ratio captures 
    the extent to which markets are unified or tightly coupled. \
    When markets are tightly coupled, they become more fragile in the sense that negative shocks propagate more quickly \
    and broadly than when markets are loosely linked. The authors apply their AR analysis to several broad markets, \
    introduce a standardized measure of shifts in the AR, and analyze how these shifts relate to changes in asset prices and financial turbulence.

    A high value for the absorption ratio corresponds to a high level of systemic risk because it implies the sources of risk are more unified. \
    A low absorption ratio indicates less systemic risk because it implies the sources of risk are more disparate. \
    High systemic risk does not necessarily lead to asset depreciation or financial turbulence. \
    It is simply an indication of market fragility in the sense that a shock is more likely to propagate quickly and broadly when sources of risk are tightly coupled.

    Args:
        asset_returns (np.ndarray): (n_assets, n_days) arrays of asset returns
        fraction_eigenvectors (float, optional): The fraction of eigenvectors used to calculate the absorption ratio. Defaults to 0.2 as in the paper.

    Returns:
        float: Absorption ratio for the market

    Examples:
        >>> import numpy as np
        >>> from frds.measures import absorption_ratio

        3 assets daily returns for 6 days
        >>> data = np.array(
        ...             [
        ...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
        ...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
        ...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
        ...             ]
        ...         )

        Calculate the absorption ratio.
        >>> absorption_ratio(data)
        0.7746543307660252

    References:
        * [Kritzman, Li, Page, and Rigobon (2010)](https://doi.org/10.3905/jpm.2011.37.4.112), 
            Principal components as a measure of systemic risk, *Journal of Portfolio Management*, 37 (4) 112-126.
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Distress Insurance Premium](/measures/distress_insurance_premium/)
        * [Contingent Claim Analysis](/measures/cca/)
        * [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
        * [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)
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

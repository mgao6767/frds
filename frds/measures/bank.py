from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def absorption_ratio(
    asset_returns: np.ndarray, fraction_eigenvectors: float = 0.2
) -> float:
    """Absorption Ratio
    
    Kritzman, Li, Page, and Rigobon (2010) propose to measure systemic risk via the Absorption Ratio (AR),\
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


def marginal_expected_shortfall(
    firm_returns: np.ndarray, market_returns: np.ndarray, q: float = 0.05
) -> float:
    """Marginal Expected Shortfall (MES).

    Acharya, Pedersen, Philippon, and Richardson (2010) argue that each financial institutions contribution to \
    systemic risk can be measured as its systemic expected shortfall (SES), i.e., its propensity to be undercapitalized \
    when the system as a whole is undercapitalized. SES is a theoretical construct and the authors use the following 3 measures to proxy it:

    1. The outcome of stress tests performed by regulators. The SES metric of a firm here is defined as the recommended capital that \
        it was required to raise as a result of the stress test in February 2009.
    2. The decline in equity valuations of large financial firms during the crisis, as measured by their cumulative equity return \
        from July 2007 to December 2008.
    3. The widening of the credit default swap spreads of large financial firms as measured by their cumulative CDS spread increases \
        from July 2007 to December 2008.

    Given these proxies, the authors seek to develop leading indicators which "predict" an institutions SES; \
    these leading indicators are marginal expected shortfall (MES) and leverage (LVG).

    The description above is from Bisias, Lo, and Valavanis.

    Args:
        firm_returns (np.ndarray): The time series of returns (equity or CDS) for the firm.
        market_returns (np.ndarray): The time series of returns (equity or CDS) for the market as a whole. \
            It is necessary to have enough data points so that there exists a value.
        q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.

    Returns:
        float: MES
    """
    assert firm_returns.shape == market_returns.shape
    low_threshold = np.percentile(market_returns, q * 100)
    worst_days = np.argwhere(market_returns < low_threshold)
    return np.mean(firm_returns[worst_days])


def systemic_expected_shortfall(
    mes_training_sample: np.ndarray,
    lvg_training_sample: np.ndarray,
    ses_training_sample: np.ndarray,
    mes_firm: float,
    lvg_firm: float,
) -> float:
    """Systemic Expected Shortfall (SES)

    Acharya, Pedersen, Philippon, and Richardson (2010) argue that each financial institutions contribution to \
    systemic risk can be measured as its systemic expected shortfall (SES), i.e., its propensity to be undercapitalized \
    when the system as a whole is undercapitalized. SES is a theoretical construct and the authors use the following 3 measures to proxy it:

    1. The outcome of stress tests performed by regulators. The SES metric of a firm here is defined as the recommended capital that \
        it was required to raise as a result of the stress test in February 2009.
    2. The decline in equity valuations of large financial firms during the crisis, as measured by their cumulative equity return \
        from July 2007 to December 2008.
    3. The widening of the credit default swap spreads of large financial firms as measured by their cumulative CDS spread increases \
        from July 2007 to December 2008.

    Given these proxies, the authors seek to develop leading indicators which predict an institutions SES; \
    these leading indicators are marginal expected shortfall (MES) and leverage (LVG).

    Args:
        mes_training_sample (np.ndarray): MES or value per firm defined as avg equity return during 5% worst days for overall market during training period.
        lvg_training_sample (np.ndarray): Leverage per firm defined on the last day of the period of training data. \
            LVG defined as (book_assets - book_equity + market_equity)/market_equity.
        ses_training_sample (np.ndarray): Cumulative return per firm for date range after mes/lvg_training_sample.
        mes_firm (float): The current firm MES used to calculate the firm SES value.
        lvg_firm (float): The current firm leverage used to calculate the firm SES value.

    Returns:
        float: The systemic risk that firm i poses to the system at a future time t.
    """
    assert mes_training_sample.shape == lvg_training_sample.shape
    assert mes_training_sample.shape == ses_training_sample.shape

    n_firms = mes_training_sample.shape

    data = np.vstack([np.ones(n_firms), mes_training_sample, lvg_training_sample]).T
    betas = np.linalg.lstsq(data, ses_training_sample, rcond=None)[0]
    _, b, c = betas
    ses = (b * mes_firm + c * lvg_firm) / (b + c)
    return ses

import numpy as np


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

    The description above is from Bisias, Lo, and Valavanis.

    Returns:
        float: The systemic risk that firm i poses to the system at a future time t.

    References:
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Absorption Ratio](/measures/absorption_ratio/)
        * [Distress Insurance Premium](/measures/distress_insurance_premium/)
        * [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
    """
    assert mes_training_sample.shape == lvg_training_sample.shape
    assert mes_training_sample.shape == ses_training_sample.shape

    n_firms = mes_training_sample.shape

    data = np.vstack([np.ones(n_firms), mes_training_sample, lvg_training_sample]).T
    betas = np.linalg.lstsq(data, ses_training_sample, rcond=None)[0]
    _, b, c = betas
    ses = (b * mes_firm + c * lvg_firm) / (b + c)
    return ses

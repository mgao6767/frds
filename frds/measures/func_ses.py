import numpy as np


def systemic_expected_shortfall(
    mes_training_sample: np.ndarray,
    lvg_training_sample: np.ndarray,
    ses_training_sample: np.ndarray,
    mes_firm: float,
    lvg_firm: float,
) -> float:
    r"""Systemic Expected Shortfall (SES)

    A measure of a financial institution's contribution to a systemic crisis by
    [Acharya, Pedersen, Philippon, and Richardson (2017)](https://doi.org/10.1093/rfs/hhw088), which equals to
    the expected amount a bank is undercapitalized in a future systemic event in which the overall financial system is undercapitalized.

    SES increases in the bank’s expected losses during a crisis, and is related to the bank's
    [marginal expected shortfall (MES)](/measures/marginal_expected_shortfall/),
    i.e., its losses in the tail of the aggregate sector’s loss distribution, and leverage.

    SES is a theoretical construct and the authors use the following 3 measures to proxy it:

    1. The outcome of stress tests performed by regulators. The SES metric of a firm here is defined as the recommended capital that
        it was required to raise as a result of the stress test in February 2009.
    2. The decline in equity valuations of large financial firms during the crisis, as measured by their cumulative equity return
        from July 2007 to December 2008.
    3. The widening of the credit default swap spreads of large financial firms as measured by their cumulative CDS spread increases
        from July 2007 to December 2008.

    Given these proxies, the authors seek to develop leading indicators which “predict” an institution’s SES, including
    marginal expected shortfall (MES) and leverage (LVG).

    !!! note
        Since SES is a theoretical construct, this function estimates the **fitted SES** following Bisias, Flood, Lo, and Valavanis (2012).

        Specifically, the following model is estimated:

        $$
        \textit{realized SES}_{i,\textit{crisis}} = a + b MES_{i,\textit{pre-crisis}} + c LVG_{i,\textit{pre-crisis}} + \varepsilon_{i}
        $$

        where $\textit{realized SES}_{i,\textit{crisis}}$ is the stock return during the crisis, and $LVG_{i,\textit{pre-crisis}}$ is
        defined as $(\text{book assets - book equity + market equity}) / \text{market equity}$.

        The fitted SES is computed as

        $$
        \textit{fitted SES} = \frac{b}{b+c} MES + \frac{c}{b+c} LVG
        $$

    ??? note "Model in Acharya, Pedersen, Philippon, and Richardson (2017)"
        In Acharya, Pedersen, Philippon, and Richardson (2017), fitted SES is abtained via estimating the model:

        $$
        \textit{realized SES}_{i,\textit{crisis}} = a + b MES_{i,\textit{pre-crisis}} + c LVG_{i,\textit{pre-crisis}} + \text{industriy dummies} + \varepsilon_{i}
        $$

        and calculating the fitted value of $\textit{realized SES}_{i}$ directly, where
        the industry dummies inlcude indicators for whether the bank is a broker-dealer, an insurance company and other.

        See Model 6 in Table 4 (p.23) and Appendix C.

    Args:
        mes_training_sample (np.ndarray): (n_firms,) array of firm ex ante MES.
        lvg_training_sample (np.ndarray): (n_firms,) array of firm ex ante LVG (say, on the last day of the period of training data)
        ses_training_sample (np.ndarray): (n_firms,) array of firm ex post cumulative return for date range after `lvg_training_sample`.
        mes_firm (float): The current firm MES used to calculate the firm (fitted) SES value.
        lvg_firm (float): The current firm leverage used to calculate the firm (fitted) SES value.

    Returns:
        float: The systemic risk that firm $i$ poses to the system at a future time.

    Examples:
        >>> from frds.measures import systemic_expected_shortfall
        >>> import numpy as np
        >>> mes_training_sample = np.array([-0.023, -0.07, 0.01])
        >>> lvg_training_sample = np.array([1.8, 1.5, 2.2])
        >>> ses_training_sample = np.array([0.3, 0.4, -0.2])
        >>> mes_firm = 0.04
        >>> lvg_firm = 1.7
        >>> systemic_expected_shortfall(mes_training_sample, lvg_training_sample, ses_training_sample, mes_firm, lvg_firm)
        -0.33340757238306845

    References:
        * [Acharya, Pedersen, Philippon, and Richardson (2017)](https://doi.org/10.1093/rfs/hhw088),
            Measuring systemic risk, *The Review of Financial Studies*, 30, (1), 2-47.
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Absorption Ratio](/measures/absorption_ratio/)
        * [Contingent Claim Analysis](/measures/cca/)
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

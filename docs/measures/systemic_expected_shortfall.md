---
tags:
  - Banking
  - Systemic Risk
---

# Systemic Expected Shortfall (SES)

## Introduction

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
    In Acharya, Pedersen, Philippon, and Richardson (2017), fitted SES is obtained via estimating the model:

    $$
    \textit{realized SES}_{i,\textit{crisis}} = a + b MES_{i,\textit{pre-crisis}} + c LVG_{i,\textit{pre-crisis}} + \text{industriy dummies} + \varepsilon_{i}
    $$

    and calculating the fitted value of $\textit{realized SES}_{i}$ directly, where
    the industry dummies inlcude indicators for whether the bank is a broker-dealer, an insurance company and other.

    See Model 6 in Table 4 (p.23) and Appendix C.

## API

### ::: frds.measures.systemic_expected_shortfall

## References

* [Acharya, Pedersen, Philippon, and Richardson (2017)](https://doi.org/10.1093/rfs/hhw088),
    Measuring systemic risk, *The Review of Financial Studies*, 30, (1), 2-47.
* [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
        A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

## See Also

Systemic risk measures:

* [Absorption Ratio](/measures/absorption_ratio/)
* [Contingent Claim Analysis](/measures/contingent_claim_analysis/)
* [Distress Insurance Premium](/measures/distress_insurance_premium/)
* [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)

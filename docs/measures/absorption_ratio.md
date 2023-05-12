---
tags:
  - Banking
  - Systemic Risk
---

# Absorption Ratio

## Introduction

A measure of systemic risk defined as the fraction of the total variance of a set of asset returns explained or absorbed by
a fixed number of eigenvectors.

Proposed by [Kritzman, Li, Page, and Rigobon (2010)](https://doi.org/10.3905/jpm.2011.37.4.112), the absorption ratio captures the extent to which markets are unified or tightly coupled.
When markets are tightly coupled, they become more fragile in the sense that negative shocks propagate more quickly
and broadly than when markets are loosely linked. The authors apply their AR analysis to several broad markets,
introduce a standardized measure of shifts in the AR, and analyze how these shifts relate to changes in asset prices and financial turbulence.

A high value for the absorption ratio corresponds to a high level of systemic risk because it implies the sources of risk are more unified.
A low absorption ratio indicates less systemic risk because it implies the sources of risk are more disparate.
High systemic risk does not necessarily lead to asset depreciation or financial turbulence.
It is simply an indication of market fragility in the sense that a shock is more likely to propagate quickly and broadly when sources of risk are tightly coupled.

## API

### ::: frds.measures.absorption_ratio

## References

* [Kritzman, Li, Page, and Rigobon (2010)](https://doi.org/10.3905/jpm.2011.37.4.112),
    Principal components as a measure of systemic risk, *Journal of Portfolio Management*, 37 (4) 112-126.
* [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
    A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

## See Also

Systemic risk measures:

* [Distress Insurance Premium](/measures/distress_insurance_premium/)
* [Contingent Claim Analysis](/measures/contingent_claim_analysis/)
* [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
* [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
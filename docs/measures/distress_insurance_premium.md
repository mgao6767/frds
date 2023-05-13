---
tags:
  - Banking
  - Systemic Risk
---

# Distress Insurance Premium (DIP)

## Introduction

A systemic risk metric by [Huang, Zhou, and Zhu (2009)](https://doi.org/10.1016/j.jbankfin.2009.05.017) which represents a hypothetical insurance premium against a systemic financial distress, defined as total losses that exceed a given threshold, say 15%, of total bank liabilities.

The methodology is general and can apply to any pre-selected group of firms with publicly tradable equity and CDS contracts.
Each institutions marginal contribution to systemic risk is a function of its size, probability of default, and asset correlation.
The last two components need to be estimated from market data.

The general steps are:

1. Use simulated asset returns from a joint normal distribution (using the correlations) to compute the distribution of joint defaults.
2. The loss-given-default (LGD) is assumed to follow a symmetric triangular distribution with a mean of 0.55 and in the range of [0.1, 1].
    The mean LGD of 0.55 is taken down from the Basel II IRB formula.
3. Compute the probability of losses and the expected losses from the simulations.

## API

### :::frds.measures.distress_insurance_premium

## References

* [Huang, Zhou and Zhu (2009)](https://doi.org/10.1016/j.jbankfin.2009.05.017),
   A framework for assessing the systemic risk of major financial institutions, *Journal of Banking & Finance*, 33(11), 2036-2049.
* [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
   A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

## See Also

Systemic risk measures:

* [Absorption Ratio](/measures/absorption_ratio/)
* [Contingent Claim Analysis](/measures/contingent_claim_analysis/)
* [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
* [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
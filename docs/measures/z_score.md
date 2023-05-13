---
tags:
  - Banking
  - Insolvency Risk
---

# Z-score

## Introduction

A measure of bank insolvency risk, defined as:

$$
\text{Z-score} = \frac{\text{ROA}+\text{CAR}}{\sigma_{\text{ROA}}}
$$

where $\text{ROA}$ is the bank's ROA, $\text{CAR}$ is the bank's capital ratio and $\sigma_{\text{ROA}}$
is the standard deviation of bank ROA.

The rationale behind Z-score is simple. A bank is insolvent when its loss $-\pi$ exceeds equity $E$, i.e., $-\pi>E$.
The probability of insolvency is $P(-\pi>E)$.

If bank assets is $A$, then $P(-\pi>E)=P(-\frac{\pi}{A}>\frac{E}{A})=P(-ROA>CAR)$.

Assuming profits are normally distributed, then scaling $(\text{ROA}+\text{CAR})$ by $\sigma_{\text{ROA}}$ yields an
estimate of the distance to insolvency.

A higher Z-score implies that larger shocks to profitability are required to cause the losses to exceed bank equity.

## API

### ::: frds.measures.z_score

## References

* [Laeven and Levine (2009)](https://doi.org/10.1016/j.jfineco.2008.09.003),
    Bank governance, regulation and risk taking, *Journal of Financial Economics*, 93, 2, 259-275.
* [Houston, Lin, Lin and Ma (2010)](https://doi.org/10.1016/j.jfineco.2010.02.008),
    Creditor rights, information sharing, and bank risk taking, *Journal of Financial Economics*, 96, 3, 485-512.
* [Beck, De Jonghe, and Schepens (2013)](https://doi.org/10.1016/j.jfi.2012.07.001),
    Bank competition and stability: cross-country heterogeneity, *Journal of Financial Intermediation*, 22, 2, 218-244.
* [Delis, Hasan, and Tsionas (2014)](https://doi.org/10.1016/j.jbankfin.2014.03.024),
    The risk of financial intermediaries, *Journal of Banking & Finance*, 44, 1-12.
* [Fang, Hasan, and Marton (2014)](https://doi.org/10.1016/j.jbankfin.2013.11.003),
    Institutional development and bank stability: Evidence from transition countries, *Journal of Banking & Finance*, 39, 160-176.

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
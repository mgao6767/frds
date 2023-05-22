---
tags:
  - Banking
  - Systemic Risk
---

# SRISK

## Introduction

A conditional capital shortfall measure of systemic risk by [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060).

### Capital shortfall

Capital shortfall is a firm's required capital reserve minus the firm's equity. Specifically, capital shortfall of a firm $i$ on day $t$ is

$$
\begin{equation}
  CS_{it} = kA_{it} - W_{it} = k(D_{it}+W_{it}) - W_{it}
\end{equation}
$$
where,

* $W_{it}$ is the market value of equity
* $D_{it}$ is the book value of debt
* $A_{it} = W_{it} + D_{it}$ is the value of quasi assets
* $k$ is the prudential capital fraction, set to 8%

??? note
    A positive capital shortfall $CS$ means the firm is in distress, i.e., the capital reserve required is larger than the firm's equity value.

### Systemic event and SRISK

A systemic event is a market decline below a threshold $C$ over a time horizon $h$.

If the multiperiod arithmetic market return between $t+1$ to $t+h$ is $R_{mt+1:t+h}$, then the systemic event is $\{R_{mt+1:t+h}<C\}$.

!!! note
    $h=1$ month and $C=-10\%$ are chosen in [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060).

**SRISK** is the expected capital shortfall conditional on a systemic event.

$$
\begin{equation}
  SRISK_{it} = E_t(CS_{it+h} | R_{mt+1:t+h} < C)
\end{equation}
$$

The total amount of systemic risk in the financial system is measured as the sum of all firm-level SRISK of the $N$ institutions in the system with **positive** SRISK measures.

$$
\begin{equation}
  SRISK_{t} = \sum_{i=1}^{N} SRISK_{it}
\end{equation}
$$

!!! warning "Institutions with negative SRISK are ignored"
    In a crisis it is unlikely that surplus capital will be easily mobilized through mergers or loans to support failing firms.

### Computation of SRISK

First, we expand $CS_{it+h}$,

$$
\begin{align}
  SRISK_{it} &= E_t(CS_{it+h} | R_{mt+1:t+h} < C) \\
  &= k E_t(D_{it+h} | R_{mt+1:t+h} < C) + (1-k) E_t(W_{it+h} | R_{mt+1:t+h} < C)
\end{align}
$$

!!! note "Assumption"
    If debt cannot be renegotiated in case of systemic event, $E_t(D_{it+h} | R_{mt+1:t+h} < C)=D_{it}$

So we have,

$$
\begin{align}
  SRISK_{it} &= k D_{it} + (1-k) W_{it} (1 - LRMES_{it}) \\
  &= W_{it} [k LVG_{it} + (1-k) LRMES_{it} - 1]
\end{align}
$$

where,

* $LRMES_{it}$ is [Long-Run MES](/measures/long_run_mes), defined as $LRMES_{it}=-E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]$.
* $LVG_{it}$ is quasi leverage ratio $(D_{it}+W_{it})/W_{it}$.

!!! success "Estimating LRMES"
    The key step in computing SRISK is estimating the LRMES. Please refer to [Long-Run MES](/measures/long_run_mes) for details.

## API

### ::: frds.measures.srisk

## References

* [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060), SRISK: A Conditional Capital Shortfall Measure of Systemic Risk, *Review of Financial Studies*, 30 (1), 48–79.
* [Duan and Zhang (2015)](http://dx.doi.org/10.2139/ssrn.2675877), Non-Gaussian Bridge Sampling with an Application, *SSRN*.
* [Orskaug (2009)](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf), Multivariate dcc-garch model with various error distributions.

## See Also

Systemic risk measures:

* [Long-Run MES](/measures/long_run_mes/)
* [Absorption Ratio](/measures/absorption_ratio/)
* [Distress Insurance Premium](/measures/distress_insurance_premium/)
* [Contingent Claim Analysis](/measures/contingent_claim_analysis/)
* [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
* [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
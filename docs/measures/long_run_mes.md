---
tags:
  - Banking
  - Systemic Risk
---

# Long-Run Marginal Expected Shortfall (LRMES)

## Introduction

LRMES is used in estimating [SRISK](/measures/srisk) by [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060). It measures the expected firm return conditional on a systemic event.

A systemic event is a market decline below a threshold $C$ over a time horizon $h$.

If the multiperiod arithmetic market return between $t+1$ to $t+h$ is $R_{mt+1:t+h}$, then the systemic event is $\{R_{mt+1:t+h}<C\}$.

$LRMES_{it}$ for firm $i$ at time $t$ is then defined as

$$
LRMES_{it}=-E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]
$$

[Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060) use a GARCH-DCC model to construct the LRMES predictions.


### GARCH-DCC

Let firm and market log returns be $r_{it}=\log(1+R_{it})$ and $r_{mt}=\log(1+R_{mt})$.
Conditional on the information set $\mathcal{F}_{t−1}$ available at time $t−1$, the return pair has an (unspecified) distribution $\mathcal{D}$ with zero mean and time–varying covariance,

$$
\begin{bmatrix}r_{it} \\ r_{mt}\end{bmatrix} | \mathcal F_{t-1} \sim \mathcal D\left(\mathbf 0, \begin{bmatrix}\sigma_{it}^2 & \rho_{it}\sigma_{it}\sigma_{mt} \\ \rho_{it}\sigma_{it}\sigma_{mt} & \sigma_{mt}^2 \end{bmatrix}\right)
$$

To specify the evolution of the time varying volatilities and correlation, GJR-GARCH volatility model and the standard DCC correlation model are used. 

#### GARCH for volatility

Specifically, the GJR-GARCH model equations for the volatility dynamics are:

$$
\begin{align}
\sigma_{it}^2 &= \omega_{Vi} + \alpha_{Vi} r^2_{it-1} + \gamma_{Vi} r^2_{it-1} I^-_{it-1} + \beta_{Vi} \sigma^2_{it-1} \\ 
\sigma_{mt}^2 &= \omega_{Vm} + \alpha_{Vm} r^2_{mt-1} + \gamma_{Vm} r^2_{mt-1} I^-_{mt-1} + \beta_{Vm} \sigma^2_{mt-1}
\end{align}
$$

where $I^-_{it} = 1$ if $r_{it}<0$ and $I^-_{mt} = 1$ if $r_{mt}<0$.

#### DCC for correlation

The DCC models correlation through the volatility-adjusted returns $\epsilon=r/\sigma$.

$$
\text{Corr}
\begin{pmatrix}
  \epsilon_{it} \\ \epsilon_{mt}
\end{pmatrix}
= \begin{bmatrix}
  1 & \rho_{it} \\
  \rho_{it} & 1
\end{bmatrix}
= \text{diag}(Q_{it})^{-1/2} Q_{it} \text{diag}(Q_{it})^{-1/2}
$$

where $Q_{it}$ is the so-called pseudo correlation matrix.

The DCC then models the dynamics of $Q_{it}$ as

$$
Q_{it} = (1-\alpha_{Ci}-\beta_{Ci})S_i + \alpha_{Ci} + \begin{bmatrix}
  \epsilon_{it} \\ \epsilon_{mt}
\end{bmatrix}\begin{bmatrix}
  \epsilon_{it} \\ \epsilon_{mt}
\end{bmatrix}^{'} + \beta_{Ci} Q_{it-1}
$$

where $S_i$ is the unconditional correlation matrix of the firm and market adjusted returns.

### Estimating GARCH-DCC

The above model is typically estimated by a two–step QML estimation procedure. More extensive details on this modeling approach and estimation are provided in [Engle (2009)](http://www.jstor.org/stable/j.ctt7sb6w).

## API

### ::: frds.measures.long_run_mes

## References

* [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060), SRISK: A Conditional Capital Shortfall Measure of Systemic Risk, *Review of Financial Studies*, 30 (1), 48–79.

## See Also

---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
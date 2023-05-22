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
\sigma_{it}^2 &= \omega_{i} + \alpha_{i} r^2_{it-1} + \gamma_{i} r^2_{it-1} I^-_{it-1} + \beta_{i} \sigma^2_{it-1} \\ 
\sigma_{mt}^2 &= \omega_{m} + \alpha_{m} r^2_{mt-1} + \gamma_{m} r^2_{mt-1} I^-_{mt-1} + \beta_{m} \sigma^2_{mt-1}
\end{align}
$$

where $I^-_{it} = 1$ if $r_{it}<0$ and $I^-_{mt} = 1$ if $r_{mt}<0$.

#### DCC for correlation

The DCC models correlation through the volatility-adjusted returns $\epsilon=r/\sigma$.

$$
\mathbf R_t =
\text{Corr}
\begin{pmatrix}
  \epsilon_{it} \\ \epsilon_{mt}
\end{pmatrix}
= \begin{bmatrix}
  1 & \rho_{it} \\
  \rho_{it} & 1
\end{bmatrix}
= \text{diag}(\mathbf Q_{it})^{-1/2} \mathbf Q_{it} \text{diag}(\mathbf Q_{it})^{-1/2}
$$

where $\mathbf Q_{it}$ is the so-called pseudo correlation matrix.

The DCC then models the dynamics of $\mathbf Q_{it}$ as

$$
\mathbf Q_{it} = (1-a-b) \bar{\mathbf Q}_i + a \mathbf{e}_{t-1} \mathbf{e}'_{t-1} + b \mathbf Q_{it-1}
$$

where,

- $\bar{\mathbf{Q}}_i$ is the unconditional correlation matrix of the firm and market adjusted returns, and
- $\mathbf e_{t-1}=\begin{bmatrix}
      \epsilon_{it-1} \\ \epsilon_{mt-1}
    \end{bmatrix}$

### Estimating GARCH-DCC

The above model is typically estimated by a two–step QML estimation procedure. More extensive details on this modeling approach and estimation are provided in [Engle (2009)](http://www.jstor.org/stable/j.ctt7sb6w).

??? note "Note by Mingze"
    Equation 4.33 in [Engle (2009)](http://www.jstor.org/stable/j.ctt7sb6w) shows that the log likelihood can be additively divided into two parts, one concerns the variance and the other concerns correlation. Therefore, we can solve for the variance and correlation parameters in two separate steps, hence "two-step" QML.

Specifically, we first estimate a GJR-GARCH(1,1) for each firm (and market)'s log return series to obtain the conditional volatilities $\sigma$ and hence $\epsilon=r/\sigma$. In the second step, we use the estimated coefficients to estimate the DCC model for $\epsilon$ for each pair of firm returns and market returns.

!!! note
    Refer to [algorithms/dcc_garch](/algorithms/dcc_garch) for the estimation process.

### Predicting LRMES

**Appendix A. Simulation Algorithm for LRMES** in [Engle (2009)](http://www.jstor.org/stable/j.ctt7sb6w) describes the exact steps to construct LRMES forecasts.

!!! tip "Summary by Mingze"
    The general idea is to simulate market returns and use the estimated GARCH-DCC model to derive the corresponding firm returns. We then use the distribution of returns to estimate LRMES.

- Step 1. Construct GARCH-DCC standardized innovations for the training sample $t=1,...,T$, where $\xi_{it}$ is standardized, linearly orthogonal shocks of the
firm to the market on day $t$,

$$
\epsilon_{mt} = \frac{r_{mt}}{\sigma_{mt}} \text{ and } \xi_{it} = \left(\frac{r_{it}}{\sigma_{it}} - \rho_{it} \epsilon_{mt}\right) / \sqrt{1-\rho^2_{it}}
$$

- Step 2. Sample with replacement $S\times h$ pairs of $[\xi_{it}, \epsilon_{mt}]'$, which are used as the simulated innovations from time $T+1$ to $T+h$. Notice that in the algorithm, the innovations are simulated by resampling the standardized residuals of the GARCH-DCC rather than relying on parametric assumptions.

- Step 3. Use the pseudo sample of innovations as inputs of the DCC and GARCH filters, respectively. Initial conditions are the last values of the conditional correlation $\rho_{iT}$ and variances $\sigma^2_{iT}$ and $\sigma^2_{mT}$. This step delivers $S$ pseudo samples of GARCH-DCC logarithmic returns from period $T+1$ to period $T+h$, conditional on the realized process up to time $T$, that is

$$
\begin{bmatrix}
  r^s_{iT+t} \\ r^s_{mT+t}
\end{bmatrix}_{t=1,...,h} | \mathcal{F}_{T}
$$

!!! note "Note by Mingze"
    Suppose we have a simulated volatility-adjusted market return $\epsilon^s_{mT+h}$ at time $T+h$, then the corresponding volatility-adjusted firm return at $T+h$ is computed as

    $$
    \left(\sqrt{1-\rho^2_{iT+h}} \times \xi^s_{iT+h} + \rho_{iT+h} \epsilon^s_{mt}\right)
    $$
    
    Therefore, we need to predict $\rho_{iT+h}$, i.e., the off-diagnal element of $\mathbf R_{T+h} = \begin{bmatrix}
      1 & \rho_{iT+h} \\
      \rho_{iT+h} & 1
    \end{bmatrix}$.

    Note that $\mathbf R_{T+h} = \text{diag}(\mathbf Q_{iT+h})^{-1/2} \mathbf Q_{iT+h} \text{diag}(\mathbf Q_{iT+h})^{-1/2}$, and

    $$
    E_T[\mathbf Q_{iT+h}] = (1-a-b)\bar{\mathbf Q}_i + a E_T[\mathbf{e}_{T+h-1} \mathbf{e}'_{T+h-1}] + b E_T[\mathbf Q_{iT+h-1}]
    $$

    We therefore need to make assumptions about $E_T[\mathbf{e}_{T+h-1} \mathbf{e}'_{T+h-1}]$ because these are the volatility-adjusted returns in the future, but we don't have future returns.

    Since, $E_T[\mathbf e_{T+h-1} \mathbf e'_{T+h-1}]=E_T[R_{T+h-1}]$, **the assumption we make here** is that 
    
    - $\bar{\mathbf R} \approx \bar{\mathbf Q}$, and 
    - $E_T[\mathbf R_{T+h}] \approx E_T[\mathbf Q_{T+h}]$


    > According to [Engle and Sheppard (2001)](https://www.nber.org/papers/w8554), this assumption seems to provide better bias properties.
    
    So,

    $$
    \begin{align}
    E_T[\mathbf{R}_{T+h}] &\approx E_T[\mathbf Q_{T+h}] \\
     &= (1-a-b)\bar{\mathbf Q}_i + a E_T[\mathbf R_{T+h-1}] + b E_T[\mathbf Q_{iT+h-1}] \\
     &\approx (1-a-b)\bar{\mathbf R}_i + (a + b) \mathbf R_{T+h-1} \\ 
     &= \dots \\
     &= (1-(a+b)^{h-1}) \bar{\mathbf R}_i + (a+b)^{h-1} E_T[\mathbf R_{T+1}]
    \end{align}
    $$

    where,
    
    - $\bar{\mathbf R}_i = \text{diag}(\bar{\mathbf{Q}}_{i})^{-1/2} \bar{\mathbf{Q}}_{i} \text{diag}(\bar{\mathbf{Q}}_{i})^{-1/2}$
    - $E_T[\mathbf R_{T+1}]= \text{diag}(\hat{\mathbf{Q}}_{iT+1})^{-1/2} \hat{\mathbf{Q}}_{iT+1} \text{diag}(\hat{\mathbf{Q}}_{iT+1})^{-1/2}$, 
    
    Further, 
   
    $$
    \hat{\mathbf{Q}}_{T+1} = (1-a-b)\bar{\mathbf Q}_i + a \mathbf{e}_T \mathbf{e}'_T+b \mathbf Q_{iT}
    $$

    and that $\mathbf{e}_T = [\epsilon_{iT}, \epsilon_{mT}]'$ and $\mathbf Q_{iT}$ are known.

- Step 4. Construct the multiperiod arithmetic firm (market) return of each pseudo sample,

$$
R^s_{iT+1:T+h} = \exp \left(\sum_{t=1}^{h} r^s_{iT+t} \right) -1
$$

- Step 5. Compute LRMES as the Monte Carlo average of the simulated multiperiod arithmetic returns conditional on the systemic event,

$$
LRMES_{iT} = - \frac{\sum_{s=1}^S R^s_{iT+1:T+h} I(R^s_{mT+1:T+h}<C)}{\sum_{s=1}^S I(R^s_{mT+1:T+h}<C)}
$$

## API

### ::: frds.measures.long_run_mes

## TODO

- [ ] Use `multiprocessing` for execution speed.
- [ ] Use C++ to rewrite for execution speed.

## References

* [Brownlees and Engle (2017)](https://doi.org/10.1093/rfs/hhw060), SRISK: A Conditional Capital Shortfall Measure of Systemic Risk, *Review of Financial Studies*, 30 (1), 48–79.
* [Duan and Zhang (2015)](http://dx.doi.org/10.2139/ssrn.2675877), Non-Gaussian Bridge Sampling with an Application, *SSRN*.
* [Orskaug (2009)](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf), Multivariate dcc-garch model with various error distributions.

## See Also

* [SRISK](/measures/srisk/)
* [Absorption Ratio](/measures/absorption_ratio/)
* [Contingent Claim Analysis](/measures/contingent_claim_analysis/)
* [Distress Insurance Premium](/measures/distress_insurance_premium/)
* [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
* [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)


---

[:octicons-bug-24: Bug report](https://github.com/mgao6767/frds/issues/new?assignees=mgao6767&labels=&template=bug_report.md&title=%5BBUG%5D) | [:octicons-heart-24: Sponsor me](https://github.com/sponsors/mgao6767)
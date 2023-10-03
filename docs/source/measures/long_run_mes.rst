##############################################
 Long-Run Marginal Expected Shortfall (LRMES)
##############################################

**************
 Introduction
**************

Long-run Marginal Exepected Shortfall (LRMES) measures the expected firm
return conditional on a systemic event, which is a market decline below
a threshold :math:`C` over a time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to
:math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is
:math:`\{R_{mt+1:t+h}<C\}`.

:math:`LRMES_{it}` for firm :math:`i` at time :math:`t` is then defined
as

.. math::

   LRMES_{it} = -E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]

`Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_ use a
:doc:`/algorithms/gjr-garch-dcc` model to construct the LRMES
predictions.

.. tip::

   If you're new to econommetrics, I suggest the following sequence of
   readings.

   Univariate GARCH models:

   #. :doc:`/algorithms/garch`, Generalized Autoregressive Conditional
      Heteroskedasticity modelling.
   #. :doc:`/algorithms/gjr-garch` extends GARCH to take into account
      asymmetric effect of negative news.

   Multivariate GARCH models:

   #. :doc:`/algorithms/garch-ccc` captures both conditional variances
      and conditioal covariances, assuming constant correlation.
   #. :doc:`/algorithms/garch-dcc` further allows for time-varying
      correlations.

   Finally,

   #. :doc:`/algorithms/gjr-garch-dcc` combines the above.

*************************
 Bivariate GJR-GARCH-DCC
*************************

Here, I dicuss the bivariate case in the context of computing
:doc:`/measures/srisk`, where :math:`i` index the firm and :math:`m`
refers to the market.

Return series
=============

Let firm and market log returns be :math:`r_{it} = \log(1 + R_{it})` and
:math:`r_{mt} = \log(1 + R_{mt})`. Conditional on the information set
:math:`\mathcal{F}_{t-1}` available at time :math:`t-1`, the return pair
has an (unspecified) distribution :math:`\mathcal{D}` with zero mean and
time-varying covariance,

.. math::
   :label: eq:garch_dcc

    \begin{bmatrix}r_{it} \\ r_{mt}\end{bmatrix} | \mathcal F_{t-1} \sim \mathcal D\left(\mathbf 0, \begin{bmatrix}\sigma_{it}^2 & \rho_{it}\sigma_{it}\sigma_{mt} \\\\ \rho_{it}\sigma_{it}\sigma_{mt} & \sigma_{mt}^2 \end{bmatrix}\right)

Alternatively, I present below the constant mean form of return series,

.. math::
   :label: return_eqn_1

   r_{it} = \mu_i + \epsilon_{it}

.. math::
   :label: return_eqn_2

   r_{mt} = \mu_m + \epsilon_{mt}

where,

-  :math:`\mu=0` assumes zero-mean. One can also assume a constant
   non-zero mean or even other structure.

-  :math:`\epsilon_{it} = \sigma_{it} z_{it}`, :math:`\epsilon_{mt} =
   \sigma_{mt} z_{mt}`, where :math:`\sigma_{it}` is the conditional
   volatility at time :math:`t`.

-  :math:`z_{it}, z_{mt}` follows an unknown bivariate distribution with
   zero mean and some covariance structure.

However, generally we can assume :math:`z_{it}, z_{mt}\sim N(0,1)`.

.. important::

   In a DCC-GARCH model, assuming that the shock innovation :math:`z` is
   standard normally distributed only implies that the conditional
   innovations are normal given the past information. It does **not**
   imply that the returns themselves are normally distributed.

   The returns :math:`r` in a GARCH framework are often modeled as:

   .. math::

      r_t = \mu_t + \sigma_t \cdot z_t

   where :math:`\mu_t` is the conditional mean, :math:`\sigma_t` is the
   conditional volatility, and :math:`z_t` is the innovation term.

   -  :math:`\mu_t` could be modeled in many ways, perhaps as a function
      of past returns or other variables, and this will affect the
      distribution of :math:`r_t`.

   -  :math:`\sigma_t` is also a function of past volatilities and
      innovations, and hence varies over time, even if :math:`z_t` is
      standard normal.

   So, while :math:`z_t` might be standard normal, :math:`\sigma_t \cdot
   z_t` is normal but not standard normal, and :math:`\mu_t + \sigma_t
   \cdot z_t` may not be normal at all depending on the form of
   :math:`\mu_t`. Therefore, the assumption about :math:`z` being
   standard normal is about the standardized returns or innovations, not
   the raw returns :math:`r`. **This is the reason why the return pair
   has an (unspecified) distribution** :math:`\mathcal{D}`.

Conditional variance
====================

Specifically, the GJR-GARCH models the conditional variances as:

.. math::
   :label: eq:gjr_garch

   \begin{align*}
   \sigma_{it}^2 &= \omega_{i} + \alpha_{i} \epsilon^2_{it-1} + \gamma_{i} \epsilon^2_{it-1} I^-_{it-1} + \beta_{i} \sigma^2_{it-1}, \\\\
   \sigma_{mt}^2 &= \omega_{m} + \alpha_{m} \epsilon^2_{mt-1} + \gamma_{m} \epsilon^2_{mt-1} I^-_{mt-1} + \beta_{m} \sigma^2_{mt-1}
   \end{align*}

where :math:`I^-_{it} = 1` if :math:`r_{it} < 0` and :math:`I^-_{mt} =
1` if :math:`r_{mt} < 0`.

Dynamic correlation
===================

With these individual variances, the **Dynamic Conditional Correlation
(DCC)** models the covariance matrix :math:`\mathbf{H}_t` as:

.. math::
   :label: conditional_cov_matrix_modified

       \mathbf{H}_t = \begin{pmatrix}
       \sigma^2_{it} & \rho_{it}\sigma_{it} \sigma_{mt} \\\\
       \rho_{it}\sigma_{it} \sigma_{mt} & \sigma^2_{mt}
       \end{pmatrix} = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t

where :math:`\mathbf{D}_t=\begin{bmatrix}\sigma_{it} & 0 \\\\ 0 &
\sigma_{mt}\end{bmatrix}`, and :math:`\mathbf{R_t}=\begin{bmatrix}1 &
\rho_t \\\\ \rho_t & 1\end{bmatrix}` is the correlation matrix of the
volatility-adjusted returns :math:`z = (r-\mu)/\sigma` (standardized
residuals from the return model).

A proxy process for R
=====================

As discussed in :doc:`/algorithms/gjr-garch-dcc`, Engle (2002) models
the time-varying :math:`\mathbf{R}_t` via a proxy process
:math:`\mathbf{Q}_t`.

.. math::
   :label: eq:dcc_corr

   \mathbf{R}_t = \text{Corr}\begin{pmatrix}
   \epsilon_{it} \\\\
   \epsilon_{mt}
   \end{pmatrix}
   = \begin{bmatrix}
   1 & \rho_{it} \\\\
   \rho_{it} & 1
   \end{bmatrix}
   = \text{diag}(\mathbf{Q}_{it})^{-1/2} \mathbf{Q}_{it} \text{diag}(\mathbf{Q}_{it})^{-1/2}

where :math:`\mathbf{Q}_{it}` is the so-called pseudo correlation
matrix.

.. math::
   :label: eq:q_matrix

   \mathbf{Q}_{it} = \begin{bmatrix}
   q_{it} & q_{imt} \\\\
   q_{imt} & q_{mt}
   \end{bmatrix}

The DCC then models the dynamics of :math:`\mathbf{Q}_{it}` as

.. math::
   :label: eq:dcc_dynamics

   \mathbf{Q}_{it} = (1-a-b) \bar{\mathbf{Q}}_i + a \mathbf{z}_{t-1} \mathbf{z}'_{t-1} + b \mathbf{Q}_{it-1}

where,

-  :math:`\bar{\mathbf{Q}}_i` is the unconditional correlation matrix of
   the firm and market adjusted returns (standardized residuals), and

-  :math:`\mathbf{z}_{t-1} = \begin{bmatrix}\epsilon_{it-1}/\sigma_{it}
   \\\\ \epsilon_{mt-1}/\sigma_{mt} \end{bmatrix} =
   \begin{bmatrix}(r_{it-1}-\mu_i)/\sigma_{it} \\\\
   (r_{mt-1}-\mu_m)/\sigma_{mt} \end{bmatrix}`

The Q process is updated as follows:

.. math::
   :label: q11

   q_{it} = (1 - a - b) \overline{q}_{i} + a z^2_{1,t-1} + b q_{i,t-1}

.. math::
   :label: q22

   q_{mt} = (1 - a - b) \overline{q}_{m} + a z^2_{m,t-1} + b q_{m,t-1}

.. math::
   :label: q12

   q_{imt} = (1 - a - b) \overline{q}_{im} + a z_{i,t-1} z_{m,t-1} + b q_{im,t-1}

The dynamic conditional correlation :math:`\rho_t` is given by:

.. math::
   :label: rho_t

   \rho_{it} = \frac{q_{imt}}{\sqrt{q_{i} q_{mt}}}

Estimating GJR-GARCH-DCC
========================

The above model is typically estimated by a two-step QML estimation
procedure. More extensive details on this modeling approach and
estimation are provided in `Engle (2009)
<http://www.jstor.org/stable/j.ctt7sb6w>`_.

.. note::

   Equation 4.33 in `Engle (2009)
   <http://www.jstor.org/stable/j.ctt7sb6w>`_ shows that the log
   likelihood can be additively divided into two parts, one concerns the
   variance and the other concerns correlation. Therefore, we can solve
   for the variance and correlation parameters in two separate steps,
   hence "two-step" QML.

   See the loglikelihood function section of
   :doc:`/algorithms/gjr-garch-dcc` for more.

Specifically, we first estimate a GJR-GARCH(1,1) for each firm (and
market)'s log return series to obtain the conditional volatilities
:math:`\sigma` and hence :math:`z = (r-\mu)/\sigma`. In the second step,
we use the estimated coefficients to estimate the DCC model for each
pair of firm returns and market returns.

This is done via :class:`frds.algorithms.GJRGARCHModel_DCC`.

*****************
 Computing LRMES
*****************

LRMES is computed via a simulation approach. Appendix A. Simulation
Algorithm for LRMES in `Engle (2009)
<http://www.jstor.org/stable/j.ctt7sb6w>`_ describes the exact steps to
construct LRMES forecasts.

.. tip::

   The general idea is to simulate market returns and use the estimated
   GJR-GARCH-DCC model to derive the corresponding firm returns. We then
   use the distribution of returns to estimate the firm's LRMES.

**Step 1**. Construct GJR-GARCH-DCC standardized innovations for the
training sample :math:`t=1,...,T`, where :math:`\xi_{it}` is
standardized, linearly orthogonal shocks of the firm to the market on
day :math:`t`,

.. math::
   :label: eq:step1

   \begin{align}
    z_{mt} &= \frac{r_{mt}-\mu_m}{\sigma_{mt}} \\\\
    \xi_{it} &= \left(\frac{r_{it}-\mu_{i}}{\sigma_{it}} - \rho_{it} z_{mt}\right) / \sqrt{1-\rho^2_{it}}
   \end{align}

.. note::

   Given the conditional correlation :math:`\rho` between the
   standardized residuals :math:`z_i` and :math:`z_m` from a bivariate
   (GJR)GARCH model under Dynamic Conditional Correlation (DCC), we can
   compute :math:`z_i` given :math:`z_m` using the following formula:

   .. math::

      z_i = \rho \times z_m + \sqrt{1 - \rho^2} \times \xi

   Here, :math:`\xi` is a standard normal random variable that is
   independent of :math:`z_m`, 

   .. math::

      \xi = (z_i - \rho z_m) / \sqrt{1 - \rho^2}

**Step 2**. Sample with replacement :math:`S\times h` pairs of
:math:`[\xi_{it}, z_{mt}]'`, which are used as the simulated
innovations from time :math:`T+1` to :math:`T+h`. Notice that in the
algorithm, the innovations are simulated by resampling the standardized
residuals of the GARCH-DCC rather than relying on parametric
assumptions.

.. note::
   
   Basically, we need innovations :math:`z_{it}` and :math:`z_{mt}` in order to forecast.
   To do so, after estimating the GJR-GARCH-DCC model, we use the standardized residuals
   from the market time series and the estimated conditinal correlations to compute the "innovation" 
   to the firm series *such that* its standarized residuals conforms the conditional correlations.


**Step 3**. Use the pseudo sample of innovations as inputs of the DCC
and (GJR)GARCH filters, respectively. Initial conditions are the last values
of the conditional correlation :math:`\rho_{iT}` and variances
:math:`\sigma^2_{iT}` and :math:`\sigma^2_{mT}`. This step delivers
:math:`S` pseudo samples of (GJR)GARCH-DCC logarithmic returns from period
:math:`T+1` to period :math:`T+h`, conditional on the realized process
up to time :math:`T`, that is

.. math::
   :label: eq:step3

   \begin{bmatrix}
      r^s_{iT+t} \\\\
      r^s_{mT+t}
   \end{bmatrix}_{t=1,...,h} | \mathcal{F}_{T}


**Step 4**. Construct the multiperiod arithmetic firm (market) return of
each pseudo sample,

.. math::
   :label: eq:returns_sim

   R^s_{iT+1:T+h} = \exp \left(\sum_{t=1}^{h} r^s_{iT+t} \right) -1

**Step 5**. Compute LRMES as the Monte Carlo average of the simulated
multiperiod arithmetic returns conditional on the systemic event,

.. math::
   :label: eq:lrmes_est

   LRMES_{iT} = - \frac{\sum_{s=1}^S R^s_{iT+1:T+h} I(R^s_{mT+1:T+h}<C)}{\sum_{s=1}^S I(R^s_{mT+1:T+h}<C)}

.. tip::
   This LRMES estimate can then be used to compute :doc:`/measures/srisk`.

************
 References
************

-  `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_,
   *SRISK: A Conditional Capital Shortfall Measure of Systemic Risk*,
   Review of Financial Studies, 30 (1), 48–79.

-  `Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993)
   <https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>`_, "On the
   Relation Between the Expected Value and the Volatility of the Nominal
   Excess Return on Stocks." *The Journal of Finance*, 48(5), 1779-1801.

-  `Engle, R. (2002) <https://www.jstor.org/stable/1392121>`_, "Dynamic
   Conditional Correlation: A Simple Class of Multivariate Generalized
   Autoregressive Conditional Heteroskedasticity Models." *Journal of
   Business & Economic Statistics*, 20(3), 339-350.

*****
 API
*****

.. autoclass:: frds.measures.LRMES

**********
 Examples
**********
=====
SRISK
=====

Introduction
============

A conditional capital shortfall measure of systemic risk by `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

Capital Shortfall
-----------------

Capital shortfall is a firm's required capital reserve minus the firm's equity. Specifically, capital shortfall of a firm :math:`i` on day :math:`t` is

.. math::
   :label: eq:capital_shortfall

   CS_{it} = kA_{it} - W_{it} = k(D_{it}+W_{it}) - W_{it}

where,

* :math:`W_{it}` is the market value of equity
* :math:`D_{it}` is the book value of debt
* :math:`A_{it} = W_{it} + D_{it}` is the value of quasi assets
* :math:`k` is the prudential capital fraction, set to 8%

.. note::
   A positive capital shortfall :math:`CS` means the firm is in distress, i.e., the capital reserve required is larger than the firm's equity value.

Systemic Event and SRISK
------------------------

A systemic event is a market decline below a threshold :math:`C` over a time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to :math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is :math:`\{R_{mt+1:t+h}<C\}`.

.. note::
   :math:`h=1` month and :math:`C=-10\%` are chosen in `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

**SRISK** is the expected capital shortfall conditional on a systemic event.

.. math::
   :label: eq:srisk

   SRISK_{it} = E_t(CS_{it+h} | R_{mt+1:t+h} < C)

The total amount of systemic risk in the financial system is measured as the sum of all firm-level SRISK of the :math:`N` institutions in the system with **positive** SRISK measures.

.. math::
   :label: eq:total_srisk

   SRISK_{t} = \sum_{i=1}^{N} SRISK_{it}

.. important:: 
   Institutions with negative SRISK are ignored. In a crisis it is unlikely that surplus capital will be easily mobilized through mergers or loans to support failing firms.

Computation of SRISK
--------------------

First, we expand :math:`CS_{it+h}`,

.. math::
   :label: eq:expand_srisk

   \begin{align*}
   SRISK_{it} &= E_t(CS_{it+h} | R_{mt+1:t+h} < C) \\\\
   &= k E_t(D_{it+h} | R_{mt+1:t+h} < C) + (1-k) E_t(W_{it+h} | R_{mt+1:t+h} < C)
   \end{align*}

.. note::
   If debt cannot be renegotiated in case of systemic event, :math:`E_t(D_{it+h} | R_{mt+1:t+h} < C)=D_{it}`

So we have,

.. math::
   :label: eq:final_srisk

   \begin{align*}
   SRISK_{it} &= k D_{it} + (1-k) W_{it} (1 - LRMES_{it}) \\\\
   &= W_{it} [k LVG_{it} + (1-k) LRMES_{it} - 1]
   \end{align*}

where,

* :math:`LRMES_{it}` is **Long-Run MES**, defined as :math:`LRMES_{it}=-E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]`.
* :math:`LVG_{it}` is quasi leverage ratio :math:`(D_{it}+W_{it})/W_{it}`.

.. tip::
   The key step in computing SRISK is estimating the LRMES. 

   The section below details how LRMES can be estimated.

Estimating LRMES
----------------

Long-run Marginal Exepected Shortfall (LRMES) measures the expected firm return conditional on a systemic event,
which is a market decline below a threshold :math:`C` over a time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to :math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is :math:`\{R_{mt+1:t+h}<C\}`.

:math:`LRMES_{it}` for firm :math:`i` at time :math:`t` is then defined as

.. math::

   LRMES_{it} = -E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]


`Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_ use a GARCH-DCC model to construct the LRMES predictions.

GARCH-DCC
^^^^^^^^^

Let firm and market log returns be :math:`r_{it} = \log(1 + R_{it})` and :math:`r_{mt} = \log(1 + R_{mt})`.
Conditional on the information set :math:`\mathcal{F}_{t-1}` available at time :math:`t-1`, 
the return pair has an (unspecified) distribution :math:`\mathcal{D}` with zero mean and time-varying covariance,

.. math::
   :label: eq:garch_dcc
    
    \begin{bmatrix}r_{it} \\ r_{mt}\end{bmatrix} | \mathcal F_{t-1} \sim \mathcal D\left(\mathbf 0, \begin{bmatrix}\sigma_{it}^2 & \rho_{it}\sigma_{it}\sigma_{mt} \\\\ \rho_{it}\sigma_{it}\sigma_{mt} & \sigma_{mt}^2 \end{bmatrix}\right)

To specify the evolution of the time-varying volatilities and correlation, 
GJR-GARCH volatility model and the standard DCC correlation model are used.

GARCH for Volatility
^^^^^^^^^^^^^^^^^^^^

Specifically, the GJR-GARCH model equations for the volatility dynamics are:

.. math::
   :label: eq:gjr_garch

   \begin{align*}
   \sigma_{it}^2 &= \omega_{i} + \alpha_{i} r^2_{it-1} + \gamma_{i} r^2_{it-1} I^-_{it-1} + \beta_{i} \sigma^2_{it-1}, \\\\
   \sigma_{mt}^2 &= \omega_{m} + \alpha_{m} r^2_{mt-1} + \gamma_{m} r^2_{mt-1} I^-_{mt-1} + \beta_{m} \sigma^2_{mt-1}
   \end{align*}

where :math:`I^-_{it} = 1` if :math:`r_{it} < 0` and :math:`I^-_{mt} = 1` if :math:`r_{mt} < 0`.

DCC for Correlation
^^^^^^^^^^^^^^^^^^^

The DCC models correlation through the volatility-adjusted returns :math:`\epsilon = r/\sigma`.

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

where :math:`\mathbf{Q}_{it}` is the so-called pseudo correlation matrix.

The DCC then models the dynamics of :math:`\mathbf{Q}_{it}` as

.. math::
   :label: eq:dcc_dynamics

   \mathbf{Q}_{it} = (1-a-b) \bar{\mathbf{Q}}_i + a \mathbf{e}_{t-1} \mathbf{e}'_{t-1} + b \mathbf{Q}_{it-1}

where,

- :math:`\bar{\mathbf{Q}}_i` is the unconditional correlation matrix of the firm and market adjusted returns, and
- :math:`\mathbf{e}_{t-1} = \begin{bmatrix}\epsilon_{it-1} \\\\ \epsilon_{mt-1} \end{bmatrix}`

Estimating GARCH-DCC
^^^^^^^^^^^^^^^^^^^^


The above model is typically estimated by a two-step QML estimation procedure. More extensive details on this modeling approach and estimation are provided in `Engle (2009) <http://www.jstor.org/stable/j.ctt7sb6w>`_.

.. note::
   Equation 4.33 in `Engle (2009) <http://www.jstor.org/stable/j.ctt7sb6w>`_ shows that the log likelihood can be additively divided into two parts, one concerns the variance and the other concerns correlation. Therefore, we can solve for the variance and correlation parameters in two separate steps, hence "two-step" QML.

Specifically, we first estimate a GJR-GARCH(1,1) for each firm (and market)'s log return series to obtain the conditional volatilities :math:`\sigma` and hence :math:`\epsilon = r/\sigma`. In the second step, we use the estimated coefficients to estimate the DCC model for :math:`\epsilon` for each pair of firm returns and market returns.

Predicting LRMES
^^^^^^^^^^^^^^^^

Appendix A. Simulation Algorithm for LRMES in `Engle (2009) <http://www.jstor.org/stable/j.ctt7sb6w>`_ describes the exact steps to construct LRMES forecasts.

.. tip::
   The general idea is to simulate market returns and use the estimated GARCH-DCC model to derive the corresponding firm returns. We then use the distribution of returns to estimate LRMES.

- Step 1. Construct GARCH-DCC standardized innovations for the training sample :math:`t=1,...,T`, where :math:`\xi_{it}` is standardized, linearly orthogonal shocks of the firm to the market on day :math:`t`,

  .. math::
     :label: eq:step1

     \epsilon_{mt} = \frac{r_{mt}}{\sigma_{mt}} \text{ and } \xi_{it} = \left(\frac{r_{it}}{\sigma_{it}} - \rho_{it} \epsilon_{mt}\right) / \sqrt{1-\rho^2_{it}}

- Step 2. Sample with replacement :math:`S\times h` pairs of :math:`[\xi_{it}, \epsilon_{mt}]'`, which are used as the simulated innovations from time :math:`T+1` to :math:`T+h`.
  Notice that in the algorithm, the innovations are simulated by resampling the standardized residuals of the GARCH-DCC rather than relying on parametric assumptions.

- Step 3. Use the pseudo sample of innovations as inputs of the DCC and GARCH filters, respectively. Initial conditions are the last values of the conditional correlation :math:`\rho_{iT}` and variances :math:`\sigma^2_{iT}` and :math:`\sigma^2_{mT}`.
  This step delivers :math:`S` pseudo samples of GARCH-DCC logarithmic returns from period :math:`T+1` to period :math:`T+h`, conditional on the realized process up to time :math:`T`, that is

  .. math::
     :label: eq:step3

     \begin{bmatrix}
       r^s_{iT+t} \\\\
       r^s_{mT+t}
     \end{bmatrix}_{t=1,...,h} | \mathcal{F}_{T}

  .. note:: 

      Suppose we have a simulated volatility-adjusted market return :math:`\epsilon^s_{mT+h}` at time :math:`T+h`, then the corresponding volatility-adjusted firm return at :math:`T+h` is computed as

      .. math::

         \left(\sqrt{1-\rho^2_{iT+h}} \times \xi^s_{iT+h} + \rho_{iT+h} \epsilon^s_{mt}\right)

      Therefore, we need to predict :math:`\rho_{iT+h}`, i.e., the off-diagonal element of :math:`\mathbf R_{T+h} = \begin{bmatrix} 1 & \rho_{iT+h} \\\\ \rho_{iT+h} & 1 \end{bmatrix}`.

      Note that 

      .. math::

         \mathbf R_{T+h} = \text{diag}(\mathbf Q_{iT+h})^{-1/2} \mathbf Q_{iT+h} \text{diag}(\mathbf Q_{iT+h})^{-1/2}

      and

      .. math::

         E_T[\mathbf Q_{iT+h}] = (1-a-b)\bar{\mathbf Q}_i + a E_T[\mathbf{e}_{T+h-1} \mathbf{e}'_{T+h-1}] + b E_T[\mathbf Q_{iT+h-1}]

      We therefore need to make assumptions about :math:`E_T[\mathbf{e}_{T+h-1} \mathbf{e}'_{T+h-1}]` because these are the volatility-adjusted returns in the future, but we don't have future returns.

      Since, :math:`E_T[\mathbf e_{T+h-1} \mathbf e'_{T+h-1}]=E_T[R_{T+h-1}]`, **the assumption we make here** is that 

      - :math:`\bar{\mathbf R} \approx \bar{\mathbf Q}`
      - :math:`E_T[\mathbf R_{T+h}] \approx E_T[\mathbf Q_{T+h}]`

      According to `Engle and Sheppard (2001) <https://www.nber.org/papers/w8554>`_, this assumption seems to provide better bias properties.

      So,

      .. math::

         \begin{align*}
         E_T[\mathbf{R}_{T+h}] &\approx E_T[\mathbf Q_{T+h}] \\\\
          &= (1-a-b)\bar{\mathbf Q}_i + a E_T[\mathbf R_{T+h-1}] + b E_T[\mathbf Q_{iT+h-1}] \\\\
          &\approx (1-a-b)\bar{\mathbf R}_i + (a + b) \mathbf R_{T+h-1} \\\\
          &= \dots \\\\
          &= (1-(a+b)^{h-1}) \bar{\mathbf R}_i + (a+b)^{h-1} E_T[\mathbf R_{T+1}]
         \end{align*}

      where,

      - :math:`\bar{\mathbf R}_i = \text{diag}(\bar{\mathbf{Q}}_{i})^{-1/2} \bar{\mathbf{Q}}_{i} \text{diag}(\bar{\mathbf{Q}}_{i})^{-1/2}`
      - :math:`E_T[\mathbf R_{T+1}]= \text{diag}(\hat{\mathbf{Q}}_{iT+1})^{-1/2} \hat{\mathbf{Q}}_{iT+1} \text{diag}(\hat{\mathbf{Q}}_{iT+1})^{-1/2}`

      Further,

      .. math::

         \hat{\mathbf{Q}}_{T+1} = (1-a-b)\bar{\mathbf Q}_i + a \mathbf{e}_T \mathbf{e}'_T+b \mathbf Q_{iT}

      and that :math:`\mathbf{e}_T = [\epsilon_{iT}, \epsilon_{mT}]'` and :math:`\mathbf Q_{iT}` are known.

- Step 4. Construct the multiperiod arithmetic firm (market) return of each pseudo sample,

.. math::
    :label: eq:returns_sim

    R^s_{iT+1:T+h} = \exp \left(\sum_{t=1}^{h} r^s_{iT+t} \right) -1

- Step 5. Compute LRMES as the Monte Carlo average of the simulated multiperiod arithmetic returns conditional on the systemic event,

.. math::
    :label: eq:lrmes_est

    LRMES_{iT} = - \frac{\sum_{s=1}^S R^s_{iT+1:T+h} I(R^s_{mT+1:T+h}<C)}{\sum_{s=1}^S I(R^s_{mT+1:T+h}<C)}


References
==========

- `Brownlees and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_, *SRISK: A Conditional Capital Shortfall Measure of Systemic Risk*, Review of Financial Studies, 30 (1), 48–79.
- `Duan and Zhang (2015) <http://dx.doi.org/10.2139/ssrn.2675877>`_, *Non-Gaussian Bridge Sampling with an Application*, SSRN.
- `Orskaug (2009) <https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/259296/724505_FULLTEXT01.pdf>`_, *Multivariate DCC-GARCH model with various error distributions*.

API
===

.. autoclass:: frds.measures.SRISK

Examples
========

Let's simulate daily returns for two firms and the market.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(1)

    # Parameters
    n_days = 252 * 3 # Number of trading days
    mean_returns = [0.0005, 0.0007, 0.0006]  # Mean daily returns for Stock1, Stock2, and Market
    volatilities = [0.02, 0.025, 0.015]  # Daily volatilities for Stock1, Stock2, and Market

    # Correlation matrix
    corr_matrix = np.array([
        [1.0, 0.75, 0.6],  # Correlation between Stock1 and Stock1, Stock2, Market
        [0.75, 1.0, 0.5],  # Correlation between Stock2 and Stock1, Stock2, Market
        [0.6, 0.5, 1.0]    # Correlation between Market and Stock1, Stock2, Market
    ])

    # Cholesky decomposition to get a lower triangular matrix
    chol_decomp = np.linalg.cholesky(corr_matrix)

    # Simulate daily returns
    random_returns = np.random.normal(0, 1, (n_days, 3))
    correlated_returns = np.dot(random_returns, chol_decomp.T)

    # Apply mean and volatility
    final_returns = mean_returns + np.multiply(correlated_returns, volatilities)

    # Simulate stock prices, assuming initial price is 100 for all
    initial_price = 100
    prices = np.exp(np.cumsum(final_returns, axis=0)) * initial_price

    # Plotting the simulated stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(prices)
    plt.title('Simulated Stock Prices')
    plt.xlabel('Trading Days')
    plt.ylabel('Price')
    plt.legend(['Stock1', 'Stock2', 'Market'])
    plt.show()

.. image:: /images/srisk_example_prices.png

Let's further assume for the two firms:

- Firm 1 (less capitalized): 100 equity and 900 debt
- Firm 2 (better capitalized): 80 equity and 250 debt


.. code-block:: python

    from frds.measures import SRISK

    firm_returns = final_returns[:,:-1]
    mkt_returns = final_returns[:,-1]
    srisk = SRISK(
        firm_returns, mkt_returns, W=np.array([100, 80]), D=np.array([900, 250])
    )
    srisk_firm1, srisk_firm2 = srisk.estimate(aggregate_srisk=False)
    print(srisk_firm1, srisk_firm2) # 8.2911043  -31.10596391

It turns out, Firm 1 has larger SRISK (8.29) than Firm 2 (-31<0), although it 
appears that Firm 2's stock price is more volatile. The better capitalization of
Firm 2 makes it less vulnerable to capital shortfall.


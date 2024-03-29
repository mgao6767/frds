##################
 GARCH(1,1) - DCC
##################

**************
 Introduction
**************

The Multivariate GARCH(1,1) model generalizes the univariate
:doc:`/algorithms/garch` framework to multiple time series, capturing
not only the conditional variances but also the conditional covariances
between the series. One common form is the Constant Conditional
Correlation (CCC) model proposed by Bollerslev (1990), discussed in
:doc:`/algorithms/garch-ccc`.

However, CCC model is limited by the assumption of a constant
correlation. Engle (2002) and Tse and Tsui (2002) address this by
proposing the **Dynamic Conditional Correlation (DCC)** model, which
allows for time-varying conditional correlation.

Here, the GARCH-DCC model by Engle (2002) is discussed.

.. tip::

   Check Examples_ section for code guide and comparison to Stata and R.

Return equation
===============

The return equation for a :math:`N`-dimensional time series is:

.. math::
   :label: mv_return_eq

   \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\epsilon}_t

Here, :math:`\mathbf{r}_t` is a :math:`N \times 1` vector of returns,
and :math:`\boldsymbol{\mu}` is a :math:`N \times 1` vector of mean
returns. :math:`\boldsymbol{\epsilon}_t` is the :math:`N \times 1`
vector of shock terms.

Shock equation
==============

The shock term is modelled as:

.. math::
   :label: mv_shock_eq

   \boldsymbol{\epsilon}_t = \mathbf{H}_t^{1/2} \mathbf{z}_t

Here, :math:`\mathbf{H}_t` is a :math:`N \times N` conditional
covariance matrix, :math:`\mathbf{H}_t^{1/2}` is a :math:`N \times N`
positive definite matrix, and :math:`\mathbf{z}_t` is a :math:`N \times
1` vector of i.i.d. standard normal innovations.

.. math::
   :label: shock_dcc

   \mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I}_N)

Conditional covariance matrix
=============================

In the DCC-GARCH(1,1) model, the conditional covariance matrix
:math:`\mathbf{H}_t` is constructed as:

.. math::
   :label: mv_volatility_eq

   \mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t

where :math:`\mathbf{D}_t=\text{diag}(\mathbf{h}_t)^{1/2}`, and
:math:`\mathbf{h}_t` is a :math:`N \times 1` vector whose elements are
univariate GARCH(1,1) variances for each time series.
:math:`\mathbf{R}_t` is the :math:`N \times N` conditional correlation
matrix which is time-varying in DCC-GARCH.

.. caution::

   The log-likelihood function for the :math:`N`-dimensional
   multivariate GARCH-DCC model is:

   .. math::

      \ell = -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]

   The formulation of dynamic conditional covariance above implies that
   the time-varying :math:`\mathbf{R}_t` must be inverted at each time
   :math:`t`, which has to be positive definite as well. These
   constraints would make estimation extremely slow.

Engle (2002) achieves these constraints by modelling
:math:`\mathbf{R}_t` via a proxy process :math:`\mathbf{Q}_t`.
Specifically, the conditional correlation matrix :math:`\mathbf{R}_t`
can be obtained as:

.. math::
   :label: conditional_correlation_dcc

   \mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \text{diag}(\mathbf{Q}_t)^{-1/2}

and the proxy process :math:`\mathbf{Q}_t` is

.. math::
   :label: dcc_equation

   \mathbf{Q}_t = (1 - a - b) \mathbf{\bar{Q}} + a (\mathbf{z}_{t-1}\mathbf{z}_{t-1}') + b \mathbf{Q}_{t-1}

Here, :math:`a` and :math:`b` are DCC parameters, and
:math:`\mathbf{\bar{Q}}` is the unconditional correlation matrix of
standardized residuals :math:`\mathbf{z}_t`.

Log-likelihood function
=======================

The log-likelihood function for the :math:`N`-dimensional multivariate
GARCH DCC model is:

.. math::
   :label: mv_log_likelihood

   \ell = -\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]

where :math:`\mathbf{z}_t=\mathbf{D}_t^{-1}\mathbf{\epsilon}_t` is the
vector of standardized residuals. We can rewrite and decompose the
log-likelihood function by adding and subtracting
:math:`\mathbf{\epsilon}_t' \mathbf{D}_t^{-1} \mathbf{D}_t^{-1}
\mathbf{\epsilon}_t = \mathbf{z}_t'\mathbf{z}_t`,

.. math::
   :label: mv_log_likelihood_decomposed

   \ell = \underbrace{-\frac{1}{2} \sum_{t=1}^T \left[ N\ln(2\pi) + 2 \ln(|\mathbf{D}_t|) + \mathbf{\epsilon}_t' \mathbf{D}_t^{-1} \mathbf{D}_t^{-1}\mathbf{\epsilon}_t \right]}_{\ell_{V}(\Theta_1)\text{ volatility component}} \underbrace{-\frac{1}{2} \sum_{t=1}^T \left[ -\mathbf{z}_t'\mathbf{z}_t + \ln(|\mathbf{R}_t|)+ \mathbf{z}_t' \mathbf{R}_t^{-1} \mathbf{z}_t \right]}_{\ell_{C}(\Theta_1, \Theta_2)\text{ correlation component}}

This decomposition reveals an interesting fact. We can view the
loglikelihood as sum of two components.

#. :math:`\ell_{V}(\Theta_1)` is about the conditioal variances of the
   returns.
#. :math:`\ell_{C}(\Theta_1, \Theta_2)` is about the conditional
   correlation.

*****************************************
 Two-step quasi-maximum likelihood (QML)
*****************************************

The above loglikelihood decomposition suggests a twp-step approach in
MLE.

Specifically, given the assumption of multivariate normal, the
volatility component :math:`\ell_{V}(\Theta_1)` is the sum of individual
GARCH loglikelihood. It can be maximized by separately maximizing each
univariate model. So, we can separately estimate for each returns a
GARCH model via MLE, and add up the loglikelihoods. This is the first
step.

After the first step, we have the parameters
:math:`\Theta_1=(\mu,\omega,\alpha,\beta)` for the GARCH models, and we
can then estimate the remaining parameters :math:`\Theta_2=(a, b)`.

****************
 Bivariate case
****************

The return equations for the two time series at time :math:`t` are:

.. math::
   :label: return_eqn_1

   r_{1t} = \mu_1 + \epsilon_{1t}

.. math::
   :label: return_eqn_2

   r_{2t} = \mu_2 + \epsilon_{2t}

Here, :math:`\epsilon_{1t} = \sqrt{h_{1t}} z_{1t}` and
:math:`\epsilon_{2t} = \sqrt{h_{2t}} z_{2t}`, with :math:`z_{1t}, z_{2t}
\sim N(0,1)`.

The conditional variances :math:`h_{1t}` and :math:`h_{2t}` are
specified as:

.. math::
   :label: h1

   h_{1t} = \omega_1 + \alpha_1 \epsilon_{1,t-1}^2 + \beta_1 h_{1,t-1}

.. math::
   :label: h2

   h_{2t} = \omega_2 + \alpha_2 \epsilon_{2,t-1}^2 + \beta_2 h_{2,t-1}

The dynamic conditional correlation :math:`\rho_t` is given by:

.. math::
   :label: rho_t

   \rho_t = \frac{q_{12t}}{\sqrt{q_{11t} q_{22t}}}

The Q process is updated as follows:

.. math::
   :label: q11

   q_{11t} = (1 - a - b) \overline{q}_{11} + a z_{1,t-1} z_{1,t-1} + b q_{11,t-1}

.. math::
   :label: q12

   q_{12t} = (1 - a - b) \overline{q}_{12} + a z_{1,t-1} z_{2,t-1} + b q_{12,t-1}

.. math::
   :label: q22

   q_{22t} = (1 - a - b) \overline{q}_{22} + a z_{2,t-1} z_{2,t-1} + b q_{22,t-1}

The log-likelihood function :math:`\ell` can be decomposed into two
components:

.. math::
   :label: log_likelihood_total

   \ell = \ell_{V}(\Theta_1) + \ell_{C}(\Theta_1, \Theta_2)

The first part, :math:`\ell_{V}(\Theta_1)`, is the sum of individual
GARCH log-likelihoods and is given by:

.. math::
   :label: ll_volatility

   \ell_{V}(\Theta_1) = -\frac{1}{2} \sum_{t=1}^T \left[ 2\ln(2\pi) + \ln(h_{1t}) + \ln(h_{2t}) + \frac{\epsilon_{1t}^2}{h_{1t}} + \frac{\epsilon_{2t}^2}{h_{2t}} \right]

The second part, :math:`\ell_{C}(\Theta_1, \Theta_2)`, focuses on the
correlation and is given by:

.. math::
   :label: ll_correlation

   \ell_{C}(\Theta_1, \Theta_2) = -\frac{1}{2} \sum_{t=1}^T \left[ -\left(z_{1t}^2 + z_{2t}^2\right) + \ln(1 - \rho_t^2) + \frac{z_{1t}^2 + z_{2t}^2 - 2\rho_t z_{1t} z_{2t}}{1 - \rho_t^2} \right]

Here,

-  :math:`z_{1t}` and :math:`z_{2t}` are the standardized residuals.

-  :math:`\rho_t` is the dynamic conditional correlation, derived from
   :math:`q_{11t}`, :math:`q_{12t}`, and :math:`q_{22t}`.

-  :math:`\Theta_1` includes the parameters for the individual GARCH
   models: :math:`\mu_1, \omega_1, \alpha_1, \beta_1, \mu_2, \omega_2,
   \alpha_2, \beta_2`.

-  :math:`\Theta_2` includes the parameters for the DCC model:
   :math:`\alpha, \beta`.

**********************
 Esimation techniques
**********************

My implementation of :class:`frds.algorithms.GARCHModel_DCC` fits the
GARCH-DCC model by a two-step quasi-maximum likelihood (QML) method.

Step 1. Use :class:`frds.algorithms.GARCHModel` to estimate the
:doc:`/algorithms/garch` model for each of the returns. This step yields
the estimates :math:`\hat{\Theta}_1`, including parameters for the
individual GARCH models: :math:`\mu_1, \omega_1, \alpha_1, \beta_1,
\mu_2, \omega_2, \alpha_2, \beta_2`. We obtain also the maximized
log-likelihood :math:`\ell(\hat{\Theta}_1)`.

Step 2. Use the estimated parameters from Step 1 to maximize
:math:`\ell_{C}(\hat{\Theta}_1, \Theta_2)` with respect to
:math:`\Theta_2=(a,b)`. A grid search is performed to find the starting values 
of :math:`(a,b)` based on loglikelihood.

************
 References
************

-  `Engle, R. F. (1982) <https://doi.org/10.2307/1912773>`_,
   "Autoregressive Conditional Heteroskedasticity with Estimates of the
   Variance of United Kingdom Inflation." *Econometrica*, 50(4),
   987-1007.

-  `Bollerslev, T. (1990) <https://doi.org/10.2307/2109358>`_,
   "Modelling the Coherence in Short-Run Nominal Exchange Rates: A
   Multivariate Generalized ARCH Model." *Review of Economics and
   Statistics*, 72(3), 498-505.

-  `Engle, R. (2002) <https://www.jstor.org/stable/1392121>`_, "Dynamic
   Conditional Correlation: A Simple Class of Multivariate Generalized
   Autoregressive Conditional Heteroskedasticity Models." *Journal of
   Business & Economic Statistics*, 20(3), 339-350.

-  `Tse, Y.K. and Tsui, A.K.C. (2002) <https://www.jstor.org/stable/1392122>`_,
   "A Multivariate Generalized Autoregressive Conditional Heteroskedasticity Model with Time-Varying Correlations." 
   *Journal of Business & Economic Statistics*, 20(3), 351-362.

*****
 API
*****

.. autoclass:: frds.algorithms.GARCHModel_DCC
   :exclude-members: Parameters

.. autoclass:: frds.algorithms.GARCHModel_DCC.Parameters
   :exclude-members: __init__
   :no-undoc-members:

**********
 Examples
**********

Let's import the dataset.

>>> import pandas as pd
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])

Scale returns to percentage returns for better optimization results

>>> returns1 = df["toyota"].to_numpy() * 100
>>> returns2 = df["nissan"].to_numpy() * 100

frds
======

Use :class:`frds.algorithms.GARCHModel_DCC` to estimate a GARCH(1,1)-DCC. 

>>> from frds.algorithms import GARCHModel_DCC
>>> model_dcc = GARCHModel_DCC(returns1, returns2)
>>> res = model_dcc.fit()
>>> from pprint import pprint
>>> pprint(res)
Parameters(mu1=0.039598837827953585,
           omega1=0.027895534722110118,
           alpha1=0.06942955278530698,
           beta1=0.9216715294923623,
           mu2=0.019315543596552513,
           omega2=0.05701047522984261,
           alpha2=0.0904653253307871,
           beta2=0.8983752570013462,
           a=0.04305972552559641,
           b=0.894147940765443,
           loglikelihood=-7256.572183143142)

These results are slighly different from the ones obtained in Stata, but with 
marginally better loglikelihood overall.

Stata
=======

In Stata, we can estimate the same model as below:

.. code-block:: stata

    webuse stocks, clear
    replace toyota = toyota * 100
    replace nissan = nissan * 100
    mgarch dcc (toyota nissan = ), arch(1) garch(1)
   
The Stata results are:

.. code-block:: stata

      Dynamic conditional correlation MGARCH model
      Sample: 1 thru 2015                                      Number of obs = 2,015
      Distribution: Gaussian                                   Wald chi2(.)  =     .
      Log likelihood = -7258.856                               Prob > chi2   =     .

      -------------------------------------------------------------------------------------
                          | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
      --------------------+----------------------------------------------------------------
      toyota              |
                  _cons   |   .0327834   .0303756     1.08   0.280    -.0267516    .0923185
      --------------------+----------------------------------------------------------------
      ARCH_toyota         |
                     arch |
                     L1.  |   .0686004   .0104037     6.59   0.000     .0482095    .0889913
                          |
                    garch |
                     L1.  |   .9183872   .0124013    74.06   0.000     .8940811    .9426934
                          |
                    _cons |   .0374049   .0115084     3.25   0.001     .0148489    .0599609
      --------------------+----------------------------------------------------------------
      nissan              |
                    _cons |    .001907   .0349832     0.05   0.957    -.0666588    .0704728
      --------------------+----------------------------------------------------------------
      ARCH_nissan         |
                     arch |
                     L1.  |   .0960886   .0138067     6.96   0.000      .069028    .1231493
                          |
                    garch |
                     L1.  |   .8914556   .0151421    58.87   0.000     .8617778    .9211335
                          |
                    _cons |   .0665498   .0185192     3.59   0.000     .0302528    .1028468
      --------------------+----------------------------------------------------------------
      corr(toyota,nissan) |   .6653973   .0189345    35.14   0.000     .6282863    .7025082
      --------------------+----------------------------------------------------------------
      /Adjustment         |
                  lambda1 |   .0468196   .0121601     3.85   0.000     .0229862    .0706529
                  lambda2 |   .8659869   .0474458    18.25   0.000     .7729948     .958979
      -------------------------------------------------------------------------------------

.. note::
   The difference is because Stata does not use 2-stage QML. Instead, it simultaneously
   esimate all parameters via MLE. According to Stata's manual, "The initial optimization step is performed in the unconstrained space. Once the maximum is found,
   we impose the constraints :math:`\lambda1\ge 0`, :math:`\lambda_2\ge 0`, :math:`\lambda_1+\lambda_2<1`, and maximize the log likelihood
   in the constrained space."

See `Stata's reference manual <https://www.stata.com/manuals/ts.pdf>`_ for its
estimation techniques.

R
===

In R, we can estimate the DCC-GARCH(1,1) as

.. code-block:: R

   library(rmgarch)
   library(haven)
   stocks <- read_dta("https://www.stata-press.com/data/r18/stocks.dta")
   data <- data.frame(toyota=stocks$toyota*100, nissan=stocks$nissan*100)
   uspec <- multispec(replicate(2, ugarchspec(mean.model=list(armaOrder=c(0, 0)),
                                             variance.model=list(model="sGARCH", garchOrder=c(1, 1)))))
   dccspec <- dccspec(uspec=uspec, dccOrder=c(1, 1), distribution="mvnorm")
   dcc_fit <- dccfit(dccspec, data=data)
   dcc_fit

The results are:

.. code-block:: R

   *---------------------------------*
   *          DCC GARCH Fit          *
   *---------------------------------*

   Distribution         :  mvnorm
   Model                :  DCC(1,1)
   No. Parameters       :  11
   [VAR GARCH DCC UncQ] : [0+8+2+1]
   No. Series           :  2
   No. Obs.             :  2015
   Log-Likelihood       :  -7258.016
   Av.Log-Likelihood    :  -3.6 

   Optimal Parameters
   -----------------------------------
                  Estimate  Std. Error  t value Pr(>|t|)
   [toyota].mu      0.040368    0.030579  1.32013 0.186790
   [toyota].omega   0.028452    0.014592  1.94984 0.051195
   [toyota].alpha1  0.070391    0.015048  4.67780 0.000003
   [toyota].beta1   0.920455    0.017295 53.22156 0.000000
   [nissan].mu      0.018490    0.036034  0.51313 0.607860
   [nissan].omega   0.058844    0.029039  2.02640 0.042724
   [nissan].alpha1  0.092924    0.027716  3.35268 0.000800
   [nissan].beta1   0.895593    0.029815 30.03821 0.000000
   [Joint]dcca1     0.043275    0.010592  4.08551 0.000044
   [Joint]dccb1     0.894212    0.032218 27.75480 0.000000

These are more comparable to the results of :class:`frds.algorithms.GARCHModel_CCC`,
because the package ``rmgarch`` also uses a 2-stage approach.

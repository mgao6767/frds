================
GARCH(1,1) - CCC
================

Introduction
============

The Multivariate GARCH(1,1) model generalizes the univariate :doc:`/algorithms/garch` 
framework to multiple time series, capturing not only the conditional variances 
but also the conditional covariances between the series. One common form is the 
**Constant Conditional Correlation (CCC) model** proposed by Robert Engle (1991).

Return equation
---------------

The return equation for a :math:`N`-dimensional time series is:

.. math::
   :label: mv_return_eq

   \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\epsilon}_t

Here, :math:`\mathbf{r}_t` is a :math:`N \times 1` vector of returns, and :math:`\boldsymbol{\mu}` is a :math:`N \times 1` vector of mean returns. :math:`\boldsymbol{\epsilon}_t` is the :math:`N \times 1` vector of shock terms.

Shock equation
--------------

The shock term is decomposed as:

.. math::
   :label: mv_shock_eq

   \boldsymbol{\epsilon}_t = \mathbf{H}_t^{1/2} \mathbf{z}_t

Here, :math:`\mathbf{H}_t` is a :math:`N \times N` conditional covariance matrix, and :math:`\mathbf{z}_t` is a :math:`N \times 1` vector of standard normal innovations.

Conditional covariance matrix
-----------------------------

In the CCC-GARCH(1,1) model, the conditional covariance matrix :math:`\mathbf{H}_t` is constructed as:

.. math::
   :label: mv_volatility_eq

   \mathbf{H}_t = \text{diag}(\mathbf{h}_t)^{1/2} \mathbf{R} \text{diag}(\mathbf{h}_t)^{1/2}

Where :math:`\mathbf{h}_t` is a :math:`N \times 1` vector whose elements are univariate GARCH(1,1) variances for each time series, and :math:`\mathbf{R}` is a constant correlation matrix.

.. admonition:: A bivarite example
   :class: note

   In a bivariate GARCH(1,1) setting, we have two univariate GARCH(1,1) processes, one for each return series. Note that the ARCH term in each of these univariate GARCH models is typically based on the standardized residuals :math:`z_{t-1}`, not the original residuals from the return series :math:`\epsilon_{t-1}`.

   Specifically, the GARCH(1,1) equations for the conditional variances :math:`h_{1t}` and :math:`h_{2t}` can be written as:

   .. math::
      :label: garch1_modified

      h_{1t} = \omega_1 + \alpha_1 \epsilon_{1,t-1}^2 + \beta_1 h_{1,t-1}

   .. math::
      :label: garch2_modified

      h_{2t} = \omega_2 + \alpha_2 \epsilon_{2,t-1}^2 + \beta_2 h_{2,t-1}

   where,

   - :math:`z_{i, t-1}` is the standardized residual at time :math:`t-1`, calculated as :math:`{\epsilon_{i, t-1}}/{\sqrt{h_{i, t-1}}}`.
   
   - :math:`\epsilon_{1,t-1}` and :math:`\epsilon_{2,t-1}` are past shock terms from their respective time series. 
   - The parameters :math:`\omega_1, \alpha_1, \beta_1, \omega_2, \alpha_2, \beta_2` are to be estimated.

   Using standardized residuals in this manner ensures that the innovations \( z_t \) have a standard normal distribution, which is a key assumption underlying GARCH models. This also allows for a more accurate modeling of the volatility dynamics.

   With these individual variances, the conditional covariance matrix :math:`\mathbf{H}_t` is:

   .. math::
      :label: conditional_cov_matrix_modified

      \mathbf{H}_t = \begin{pmatrix}
      h_{1t} & \sqrt{h_{1t} h_{2t}} \rho \\\\
      \sqrt{h_{1t} h_{2t}} \rho & h_{2t}
      \end{pmatrix}

   Here, :math:`\rho` is the correlation between the two time series. 
   It is assumed to be constant over time in the CCC-GARCH framework.

   The constant correlation matrix :math:`\mathbf{R}` simplifies to a :math:`2 \times 2` matrix:

   .. math::
      :label: bivariate_constant_corr_matrix
      
      \mathbf{R} = 
      \begin{pmatrix}
         1      & \rho \\\\
         \rho & 1
      \end{pmatrix}

Log-likelihood function
-----------------------

The log-likelihood function for the Multivariate GARCH(1,1) model is:

.. math::
   :label: mv_log_likelihood

   \ell = -\frac{1}{2} \sum_{t=1}^T \left[ N \ln(2\pi) + \ln(|\mathbf{H}_t|) + \mathbf{z}_t' \mathbf{H}_t^{-1} \mathbf{z}_t \right]

This function is maximized to estimate the model parameters.

.. admonition:: A bivariate example
   :class: note

   In the bivariate case, the log-likelihood function can be specifically written as a function of all parameters.

   The log-likelihood function :math:`\ell` for the bivariate case with all parameters :math:`\Theta = (\omega_1, \alpha_1, \beta_1, \omega_2, \alpha_2, \beta_2, \rho)` is:

   .. math::
      :label: log_likelihood_bivariate

      \ell(\Theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ 2 \ln(2\pi) + \ln(|\mathbf{H}_t|) + \mathbf{z}_t' \mathbf{H}_t^{-1} \mathbf{z}_t \right]

   Here, :math:`\mathbf{z}_t'` is the transpose of the vector of standardized residuals :math:`\mathbf{z}_t`, 

   .. math::

      \mathbf{z}_t = \begin{pmatrix}
        z_{1,t} \\
        z_{2,t}
      \end{pmatrix}

   and :math:`|\mathbf{H}_t|` is the determinant of :math:`\mathbf{H}_t`:

   .. math::

      |\mathbf{H}_t| = h_{1,t} \cdot h_{2,t} - \left(\rho \sqrt{h_{1,t} h_{2,t}}\right)^2
                 = h_{1,t} \cdot h_{2,t} (1 - \rho^2)


   The inverse :math:`\mathbf{H}_t^{-1}` is:

   .. math::
      
      \mathbf{H}_t^{-1} = \frac{1}{|\mathbf{H}_t|} \begin{pmatrix}
      h_{2,t} & -\rho \sqrt{h_{1,t} h_{2,t}} \\\\
      -\rho \sqrt{h_{1,t} h_{2,t}} & h_{1,t}
      \end{pmatrix}
   

   Further, the term :math:`\mathbf{z}_t' \mathbf{H}_t^{-1} \mathbf{z}_t` can be expanded as:

   .. math::

      \mathbf{z}_t' \mathbf{H}_t^{-1} \mathbf{z}_t = \frac{1}{|\mathbf{H}_t|} \left( h_{2,t} z_{1,t}^2 + h_{1,t} z_{2,t}^2 - 2 \rho \sqrt{h_{1,t} h_{2,t}} z_{1,t} z_{2,t}  \right)

   Dividing each term in the bracket by :math:`|\mathbf{H}_t|`, we get:

   .. math::

      \begin{align}
      \mathbf{z}_t' \mathbf{H}_t^{-1} \mathbf{z}_t &= \frac{h_{2,t} z_{1,t}^2 + h_{1,t} z_{2,t}^2 - 2\rho \sqrt{h_{1,t} h_{2,t}} z_{1,t} z_{2,t}}{h_{1,t} h_{2,t} (1 - \rho^2)} \\\\
      &= \frac{1}{1 - \rho^2} \left[\frac{z_{1,t}^2}{h_{1,t}}  + \frac{z_{2,t}^2}{h_{2,t}}  - \frac{2\rho z_{1,t} z_{2,t}}{\sqrt{h_{1,t} h_{2,t}}} \right]
      \end{align}


   Inserting all of these into :math:`\ell(\Theta)`:

   .. math::
      :label: log_likelihood_bivariate_nonmatrix

      \ell(\Theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ 2 \ln(2\pi) + \ln(h_{1,t} h_{2,t} \cdot (1 - \rho^2)) + \frac{1}{1 - \rho^2} \left( \frac{z_{1,t}^2}{h_{1,t}}  + \frac{z_{2,t}^2}{h_{2,t}}  - \frac{2\rho z_{1,t} z_{2,t}}{\sqrt{h_{1,t} h_{2,t}}} \right) \right]



API
===

.. autoclass:: frds.algorithms.GARCHModel_CCC
   :private-members:


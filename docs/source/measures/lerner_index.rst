====================
Lerner Index (Banks)
====================

Introduction
============

The Lerner Index is used to measure the market power of banks. It is defined as 
the markup of price over marginal cost, relative to the price. Mathematically, the Lerner Index :math:`L` is given by:

.. math::
   :label: lerner-index

   L = \frac{P - MC}{P}

where:

- :math:`P` is the price, often calculated as total bank revenue over assets.
- :math:`MC` is the marginal cost, which is usually estimated using econometric techniques.

.. note::

   `Elzinga and Mills (2011) <https://doi.org/10.1257/aer.101.3.558>`_ 
   "The Lerner Index of Monopoly Power: Origins and Uses" is a good read.

The marginal cost (:math:`MC`) is often estimated using advanced econometric models, such as the translog cost function. The translog cost function allows for more flexibility in capturing the complexities of banking operations compared to simpler functional forms.

The translog cost function :math:`\ln C(Q, W)` is often used to model the cost :math:`C`, e.g., total operating expenses, as a function of output :math:`Q` and a vector of input prices :math:`W`. The translog cost function can be represented as:

.. math::
   :label: lerner-index-translog

   \begin{align}
   \ln C(Q, W) &= \beta_0 + \beta_1 \ln Q + \frac{1}{2} \beta_2 (\ln Q)^2 \\\\
   &+ \sum_{i=1}^{n} \gamma_i \ln W_i + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \theta_{ij} \ln W_i \ln W_j \\\\
   &+ \sum_{i=1}^{n} \phi_i \ln Q \ln W_i
   \end{align}

.. important:: 
   
   When estimating :math:numref:`lerner-index-translog`, year fixed effects can also be introduced 
   with robust standard errors by bank to capture the specificities of each firm
   (e.g., `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_,
   `Beck, De Jonghe, and Schepens (2013) <https://doi.org/10.1016/j.jfi.2012.07.001>`_, among others)

   See the section `About Translog Functions`_  below to understand why the cost function is like this.

   See the section `Estimation of Translog Cost Function`_  below for more on the OLS vs SFA, as well as
   the homogeneity constraint on input prices.

After estimating this cost function, the marginal cost :math:`MC` with respect to the output :math:`Q` can be obtained by differentiating the cost function :math:`C(Q, W)` with respect to :math:`Q`:

.. math::
   :label: lerner-index-mc

   MC = \frac{\partial C(Q, W)}{\partial Q} = \frac{C}{Q} \left( \beta_1 + \beta_2 \ln Q + \sum_{i=1}^{n} \phi_i \ln W_i \right)

The output :math:`Q` is typically the total assets. [1]_

.. [1] `Shaffer and Spierdijk (2020) <https://doi.org/10.1016/j.jbankfin.2020.105859>`_ 
       Table 1 lists recent papers' measures of :math:`Q`, most of which are total assets. 

In the banking literature (e.g., `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_),
it is common to have three input factors: labour, funds, and fixed capital, so that we have

#. :math:`W_1` is the input price of labour, measured as the ratios of personnel expenses to total assets
#. :math:`W_2` is the input price of funds, measured as the interest expenses to total deposits 
#. :math:`W_3` is the input price of fixed capital, measured as other operating and administrative expenses to total assets

Improvements and Adjustments
============================

The above specification allows only one output :math:`Q`. Some studies allow banks to be 
multi-product and hence have several outputs. Here I do not discuss them.
A suggested reading is `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_.

Adjusted Lerner Index
---------------------

Moreover, `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_
propose the **adjusted Lerner Index**.

.. math::
   :label: efficiency-adjusted-lerner-index

   \text{adjusted } L = \frac{\pi + TC - MC \times Q}{\pi + TC}

where:

- :math:`\pi` is the bank's profit before tax
- :math:`TC` is the bank's total operating costs 

About this adjusted Lerner Index, `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_ comment

  Koetter, Kolari and Spierdijk (2012) have argued that the conventional approach 
  of computing the Lerner index assumes both profit efficiency (optimal choice 
  of prices) and cost efficiency (optimal choice of inputs by firms). As a result,
  the estimated price-cost margins do not correctly measure the true extent of market power. 
  The argument reflects a distinction pointed out by Lerner (1934) himself, who states that 
  “for practical purposes we must read monopoly power not as *potential* monopoly, but as monopoly *in force*” 
  (p. 170; italics as in the original). In other words, the Lerner index measures 
  actual (exercised) market power, while Koetter, Kolari, and Spierdijk (2012) 
  are interested in measuring potential market power.

Alternative parameteriation
---------------------------

Of course, one we choose a different functional form for the cost function and obtain margainal cost :math:`MC`
accordingly. For example, `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_ and  
`Deli, Delis, Hasan, and Liu (2019) <https://doi.org/10.1016/j.jbankfin.2019.01.016>`_ use 
a log-linear cost function model.

About Translog Functions
========================

This section discusses the use of translog function, from production to cost.

Translog Production Function
----------------------------

Let's start from the most general **production function**,
CES (Constant Elasticity of Substitution). 

The CES production function with two factors :math:`X_1` and :math:`X_2` is given by:

.. math::
   :label: ces-two-factor

   Q = A \left( \alpha X_1^{-\rho} + (1-\alpha) X_2^{-\rho} \right)^{-\frac{1}{\rho}}

Here, :math:`A` is a scale parameter, :math:`\alpha` is the distribution parameter, and :math:`\rho` is the substitution parameter.

.. note::

   The **Cobb-Douglas function** is a special case of the CES function when :math:`\rho \to 0`:

   .. math::

      Q = A X_1^{\alpha} X_2^{1-\alpha}

Taking the natural logarithm of both sides, we get:

.. math::
   :label: ln-ces-two-factor-taylor

   \ln(Q) = \ln(A) - \frac{1}{\rho} \ln \left( \alpha e^{-\rho \ln(X_1)} + (1-\alpha) e^{-\rho \ln(X_2)} \right)

We can expand the term inside the logarithm using a Taylor series around :math:`\rho = 0`. 
The first-order and second-order terms of the Taylor series give us the Translog approximation.

The Taylor expansion of :math:`\ln \left( \alpha e^{-\rho \ln(X_1)} + (1-\alpha) e^{-\rho \ln(X_2)} \right)` around :math:`\rho=0` can be expressed as:

.. math::
   :label: taylor-expansion

   \ln(\alpha) + \ln(X_1) - \rho \ln(X_1) + \frac{\rho^2 (\ln(X_1))^2}{2} + \ln(1-\alpha) + \ln(X_2) - \rho \ln(X_2) + \frac{\rho^2 (\ln(X_2))^2}{2}

Substituting the Taylor expansion into the logarithmic form of the CES function, we get the **Translog (Transcendental Logarithmic) production function**:

.. math::
   :label: translog-two-factor-taylor
   
   \begin{align}
   \ln(Q) &= a_0 + a_1 \ln(X_1) + a_2 \ln(X_2) \\\\
          &+ b_{11} (\ln(X_1))^2 + b_{22} (\ln(X_2))^2 + b_{12} \ln(X_1) \ln(X_2)
   \end{align}

Here, :math:`a_1` and :math:`a_2` are coefficients that capture the first-order effects, and :math:`b_{11}`, :math:`b_{22}`, and :math:`b_{12}` are coefficients that capture the second-order effects.

.. tip::

    In summary, the Translog function serves as a second-order Taylor 
    approximation of the CES production function, providing a more flexible form
    to capture the relationships between inputs and output. 
    This can be particularly important when we are modelling the production of
    banks, which are very complex corporations.


Translog Cost Function
----------------------

.. warning:: This part is incorrec and needs to be fixed.

The Translog cost function can be derived from the Translog production function through the duality theory in economics, specifically using Shepard's Lemma.

Shepard's Lemma
^^^^^^^^^^^^^^^

In economic theory, the cost function is the dual of the production function. Shepard's Lemma states that the derivative of the cost function with respect to output prices gives the input demands. Mathematically, if :math:`C(Q, P_1, P_2)` is the cost function, then:

.. math::
   :label: shepards-lemma

   X_i = \frac{\partial C(Q, P_1, P_2)}{\partial P_i}

Derivation
^^^^^^^^^^

To derive the Translog cost function, we start by expressing the cost as:

.. math::
   :label: cost-expression

   C = P_1 X_1 + P_2 X_2

Taking the natural logarithm of both sides, we get:

.. math::
   :label: ln-cost-expression

   \ln(C) = \ln(P_1 X_1 + P_2 X_2)

Now, using Shepard's Lemma, we can express :math:`X_1` and :math:`X_2` in terms of the derivatives of the Translog production function:

.. math::
   :label: shepard-x1-x2

   X_1 = \frac{\partial Q}{\partial X_1} = Q \left( a_1 + 2 b_{11} \ln(X_1) + b_{12} \ln(X_2) \right)

   X_2 = \frac{\partial Q}{\partial X_2} = Q \left( a_2 + 2 b_{22} \ln(X_2) + b_{12} \ln(X_1) \right)

Substitute these into the cost expression, and you get the Translog cost function:

.. math::
   :label: translog-cost-shepard
  
   \ln(C) = \ln \left[ P_1 Q \left( a_1 + 2 b_{11} \ln(X_1) + b_{12} \ln(X_2) \right) + P_2 Q \left( a_2 + 2 b_{22} \ln(X_2) + b_{12} \ln(X_1) \right) \right]


The above expression can be simplified and re-parameterized to get a more standard form of the Translog cost function, which will include terms involving :math:`\ln(Q)`, :math:`\ln(P_1)`, :math:`\ln(P_2)`, and their interactions.


Reparameterization
^^^^^^^^^^^^^^^^^^

Reparameterizing the Translog cost function involves simplifying the terms and introducing new parameters to capture the effects of output and input prices.

We can simplify this by separating terms involving :math:`\ln(Q)`, :math:`\ln(P_1)`, and :math:`\ln(P_2)`:

.. math::
   :label: separated-translog-cost

   \begin{align}
   \ln(C) &= \ln(Q) \\\\
   &+ \ln \left[ P_1 \left( a_1 + 2 b_{11} \ln(X_1) + b_{12} \ln(X_2) \right) + P_2 \left( a_2 + 2 b_{22} \ln(X_2) + b_{12} \ln(X_1) \right) \right]
   \end{align}

We can reparameterize by introducing new parameters to capture the effects of output and input prices:

.. math::
   :label: reparameterized-translog-cost

   \begin{align}
   \ln(C) &= a_0' + a_Q' \ln(Q) + a_1' \ln(P_1) + a_2' \ln(P_2) \\\\
   &+ b_{QQ}' (\ln(Q))^2 + b_{11}' (\ln(P_1))^2 + b_{22}' (\ln(P_2))^2 \\\\
   &+ b_{Q1}' \ln(Q) \ln(P_1) + b_{Q2}' \ln(Q) \ln(P_2) + b_{12}' \ln(P_1) \ln(P_2)
   \end{align}

Here, :math:`a_0', a_Q', a_1', a_2', b_{QQ}', b_{11}', b_{22}', b_{Q1}', b_{Q2}', b_{12}'` are new parameters that can be estimated from the data. These new parameters are functions of the original parameters :math:`a_0, a_1, a_2, b_{11}, b_{22}, b_{12}` from the Translog production function.

More generally, if we have :math:`n` production factors, we can rewrite it as

.. math::
   :label: translog-cost-from-production

   \begin{align}
   \ln(C) &= a_0' + a_Q' \ln(Q) + a_{QQ}' (\ln(Q))^2 \\\\
   &+ \sum_{i=1}^{n} a_i' \ln(P_i) + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} b_{ij}' \ln(P_i) \ln(P_j) \\\\
   &+ \sum_{i=1}^n b_{Qi}'  \ln(Q)\ln(P_i)
   \end{align}

Here, :math:`P_i` is the price of input :math:`i`, and :math:`a_0', a_Q', a_i', b_{ij}'` are parameters to be estimated. These parameters are related to the parameters of the Translog production function through duality.

.. tip::

   If we use :math:`W` for the input price and different greek letters for coefficients, this is the same as :math:numref:`lerner-index-translog`.

Estimation of Translog Cost Function
================================================

The Translog cost function can be estimated using either Ordinary Least Squares 
(OLS) or Stochastic Frontier Analysis (SFA), depending on the assumptions about the error term.

In OLS, the Translog cost function is estimated by minimizing the sum of squared residuals. The error term is assumed to be normally distributed and captures random noise.

In SFA, the error term is decomposed into two parts: one capturing random noise and another capturing inefficiency.

- **OLS** is simpler but does not separate inefficiency from random noise.
- **SFA** is more complex but provides additional insights into inefficiency.

.. important::

   In this current implementation, I use OLS to estimate the translog cost 
   function with time fixed effects and standard error clustering by firm. 
   Support for a SFA estimation via MLE may be added in the future.

Another important note is that we need to impose the **homogeneity of degree 1**
constraint, which basically states that if all input prices are scaled by a scalar 
:math:`s`, the cost would scale by the same :math:`s`. 

This is achieved by normalizing the total cost :math:`C` and input prices by :math:`W_1` 
(we can choose :math:`W_2` or :math:`W_3` instead). After normalization, we will 
have scaled :math:`\tilde{C}=\frac{C}{W_1}`, :math:`\tilde{W}_2=\frac{W_2}{W_1}` 
and :math:`\tilde{W}_3=\frac{W_3}{W_1}`.

With three inputs and one output, the model to estimate is then

.. math::
   :label: lerner-index-translog-ols-model

   \begin{align}
   \ln \tilde{C}_{it} &= \beta_0 + \beta_1 \ln Q_{it} + \frac{1}{2} \beta_2 (\ln Q_{it})^2 \\\\
   &+ \gamma_2 \ln\tilde{W}_{2,it} + \gamma_3 \ln\tilde{W}_{3,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{2,it} + \phi_3 \ln Q_{it} \ln\tilde{W}_{3,it} + d_{t} + \varepsilon_{it}
   \end{align}

where :math:`d_t` is the time fixed effects.

The marginal cost for bank :math:`i` at time :math:`t` is then obtained as 

.. math::
   :label: lerner-index-mc-from-estimates

   MC_{it} = \frac{\partial C_{it}}{\partial Q_{it}} = \frac{C_{it}}{Q_{it}} \left( \hat{\beta}_1 + \hat{\beta}_2 \ln Q + \hat{\phi}_2 \ln \tilde{W}_2 + \hat{\phi}_3 \ln \tilde{W}_3\right)

This :math:`MC_{it}` is used to compute the Lerner Index.


References
==========

- `Beck, De Jonghe, and Schepens (2013) <https://doi.org/10.1016/j.jfi.2012.07.001>`_,
  Bank competition and stability: Cross-country heterogeneity,
  *Journal of Financial Intermediation*, 22 (2), 218–244.

- `Berger, Klapper, and Turk-Ariss (2009) <https://doi.org/10.1007/s10693-008-0050-7>`_,
  Bank Competition and Financial Stability,
  *Journal of Financial Services Research*, 35, 99–118.
   
- `Clerides, Delis, and Kokas (2015) <https://doi.org/10.1111/fmii.12030>`_,
  A New Data Set On Competition In National Banking Markets,
  *Financial Markets, Institutions & Instruments*, 24, 267–311.

- `Deli, Delis, Hasan, and Liu (2019) <https://doi.org/10.1016/j.jbankfin.2019.01.016>`_,
  Enforcement of banking regulation and the cost of borrowing,
  *Journal of Banking & Finance*, 101, 147–160.

- `Elzinga and Mills (2011) <https://doi.org/10.1257/aer.101.3.558>`_,
  The Lerner Index of Monopoly Power: Origins and Uses,
  *American Economic Review*, 101(3), 558–564.

- `Koetter, Kolari, and Spierdijk (2012) <https://doi.org/10.1162/REST_a_00155>`_,
  Enjoying the Quiet Life under Deregulation? Evidence from Adjusted Lerner Indices for U.S. Banks,
  *Review of Economics and Statistics*, 94 (2), 462–480.

- `Shaffer and Spierdijk (2020) <https://doi.org/10.1016/j.jbankfin.2020.105859>`_,
  Measuring multi-product banks' market power using the Lerner index,
  *Journal of Banking & Finance*, Volume 117, August 2020, 105859.

API
===

.. autoclass:: frds.measures.LernerIndex

Examples
========

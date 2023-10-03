#######
 SRISK
#######

**************
 Introduction
**************

A conditional capital shortfall measure of systemic risk by `Brownlees
and Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

Capital Shortfall
=================

Capital shortfall is a firm's required capital reserve minus the firm's
equity. Specifically, capital shortfall of a firm :math:`i` on day
:math:`t` is

.. math::
   :label: eq:capital_shortfall

   CS_{it} = kA_{it} - W_{it} = k(D_{it}+W_{it}) - W_{it}

where,

-  :math:`W_{it}` is the market value of equity
-  :math:`D_{it}` is the book value of debt
-  :math:`A_{it} = W_{it} + D_{it}` is the value of quasi assets
-  :math:`k` is the prudential capital fraction, set to 8%

A positive capital shortfall :math:`CS` means the firm is in distress,
i.e., the capital reserve required is larger than the firm's equity
value.

Systemic Event and SRISK
========================

A systemic event is a market decline below a threshold :math:`C` over a
time horizon :math:`h`.

If the multiperiod arithmetic market return between :math:`t+1` to
:math:`t+h` is :math:`R_{mt+1:t+h}`, then the systemic event is
:math:`\{R_{mt+1:t+h}<C\}`.

.. note::

   :math:`h=1` month and :math:`C=-10\%` are chosen in `Brownlees and
   Engle (2017) <https://doi.org/10.1093/rfs/hhw060>`_.

**SRISK** of a firm :math:`i` is its expected capital shortfall
conditional on a systemic event.

.. math::
   :label: eq:srisk

   SRISK_{it} = E_t(CS_{it+h} | R_{mt+1:t+h} < C)

The total amount of systemic risk in the financial system is measured as
the sum of all firm-level SRISK of the :math:`N` institutions in the
system with **positive** SRISK measures.

.. math::
   :label: eq:total_srisk

   SRISK_{t} = \sum_{i=1}^{N} SRISK_{it}

.. note::

   Institutions with negative SRISK are ignored. In a crisis it is
   unlikely that surplus capital will be easily mobilized through
   mergers or loans to support failing firms.

Computation of SRISK
====================

First, we expand :math:`CS_{it+h}`,

.. math::
   :label: eq:expand_srisk

   \begin{align*}
   SRISK_{it} &= E_t(CS_{it+h} | R_{mt+1:t+h} < C) \\\\
   &= k E_t(D_{it+h} | R_{mt+1:t+h} < C) + (1-k) E_t(W_{it+h} | R_{mt+1:t+h} < C)
   \end{align*}

If debt cannot be renegotiated in case of systemic event,

.. math::
   :label: eq:expected_debt

   E_t(D_{it+h} | R_{mt+1:t+h} < C)=D_{it}

So we have,

.. math::
   :label: eq:final_srisk

   \begin{align*}
   SRISK_{it} &= k D_{it} + (1-k) W_{it} (1 - LRMES_{it}) \\\\
   &= W_{it} [k LVG_{it} + (1-k) LRMES_{it} - 1]
   \end{align*}

where,

-  :math:`LVG_{it}` is quasi leverage ratio
   :math:`LVG_{it}=(D_{it}+W_{it})/W_{it}`.
-  :math:`LRMES_{it}` is :doc:`/measures/long_run_mes`, which captures
   the expected firm return conditional on a systemic event.

.. important::

   The key step in computing SRISK is estimating the
   :doc:`/measures/long_run_mes`.

   :math:`LRMES_{it}` for firm :math:`i` at time :math:`t` is then
   defined as

   .. math::

      LRMES_{it} = -E_i[R_{it+1:t+h} | R_{mt+1:t+h} < C]

   Refer to :doc:`/measures/long_run_mes` for the steps of estimating
   LRMES using :doc:`/algorithms/gjr-garch-dcc`.

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


**********
 Examples
**********

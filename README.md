![frds](https://github.com/mgao6767/frds/raw/main/images/frds_logo.png)

# FRDS - Financial Research Data Services

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![DOWNLOADS](https://img.shields.io/pypi/dm/frds?label=PyPI%20downloads) [![Test](https://github.com/mgao6767/frds/actions/workflows/test.yml/badge.svg)](https://github.com/mgao6767/frds/actions/workflows/test.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[frds](https://github.com/mgao6767/frds/) is an open-sourced Python package for computing [a collection of major academic measures](https://frds.io/measures/) used in the finance literature in a simple and straightforward way.

## Installation

```bash
pip install frds
```

## Note

This library is still under development and breaking changes may be expected.

If there's any issue (likely), please contact me at [mingze.gao@sydney.edu.au](mailto:mingze.gao@sydney.edu.au)

## Supported measures

More to be added. For a complete list of supported built-in measures, please check [frds.io/measures/](https://frds.io/measures/).

* [Absorption Ratio](https://frds.io/measures/absorption_ratio/)
* [Contingent Claim Analysis](https://frds.io/measures/contingent_claim_analysis/)
* [Distress Insurance Premium](https://frds.io/measures/distress_insurance_premium/)
* [Long-Run MES](https://frds.io/measures/long_run_mes/)
* [Marginal Expected Shortfall (MES)](https://frds.io/measures/marginal_expected_shortfall/)
* [SRISK](https://frds.io/measures/srisk/)
* [Systemic Expected Shortfall (SES)](https://frds.io/measures/systemic_expected_shortfall/)
* [Z-score](https://frds.io/measures/z_score)


## Examples

The primary purpose of `frds` is to offer ready-to-use functions.

For example, Kritzman, Li, Page, and Rigobon (2010) propose an [Absorption Ratio](https://frds.io/measures/absorption_ratio/) that measures the fraction of the total variance of a set of asset returns explained or absorbed by a fixed number of eigenvectors. It captures the extent to which markets are unified or tightly coupled.

``` python
>>> import numpy as np
>>> from frds.measures import absorption_ratio
>>> data = np.array( # Hypothetical 6 daily returns of 3 assets.
...             [
...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
...             ]
...         )
>>> absorption_ratio.estimate(data, fraction_eigenvectors=0.2)
0.7746543307660252
```

Another example, [Distress Insurance Premium (DIP)](https://frds.io/measures/distress_insurance_premium/) proposed by Huang, Zhou, and Zhu (2009) as a systemic risk measure of a hypothetical insurance premium against a systemic financial distress, defined as total losses that exceed a given threshold, say 15%, of total bank liabilities.

``` python
>>> from frds.measures import distress_insurance_premium
>>> # hypothetical implied default probabilities of 6 banks
>>> default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15] 
>>> correlations = np.array(
...     [
...         [ 1.000, -0.126, -0.637, 0.174,  0.469,  0.283],
...         [-0.126,  1.000,  0.294, 0.674,  0.150,  0.053],
...         [-0.637,  0.294,  1.000, 0.073, -0.658, -0.085],
...         [ 0.174,  0.674,  0.073, 1.000,  0.248,  0.508],
...         [ 0.469,  0.150, -0.658, 0.248,  1.000, -0.370],
...         [ 0.283,  0.053, -0.085, 0.508, -0.370,  1.000],
...     ]
... )
>>> distress_insurance_premium.estimate(default_probabilities, correlations)       
0.28661995758
```

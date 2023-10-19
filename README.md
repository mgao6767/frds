![frds](https://github.com/mgao6767/frds/raw/main/docs/source/images/frds_logo.png)

# FRDS - Financial Research Data Services

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![DOWNLOADS](https://img.shields.io/pypi/dm/frds?label=PyPI%20downloads) [![Test](https://github.com/mgao6767/frds/actions/workflows/test.yml/badge.svg)](https://github.com/mgao6767/frds/actions/workflows/test.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[frds](https://github.com/mgao6767/frds/) is a Python library to simplify the complexities often encountered in financial research. It provides a collection of ready-to-use methods for computing a wide array of measures in the literature. 

It is developed by Dr. [Mingze Gao](https://mingze-gao.com) from the University of Sydney, as a personal project during his postdoctoral research fellowship.

## Installation

```bash
pip install frds
```

## Note

This library is still under development and breaking changes may be expected.

If there's any issue (likely), please contact me at [mingze.gao@sydney.edu.au](mailto:mingze.gao@sydney.edu.au)

## Supported measures and algorithms

For a complete list of supported built-in measures, please check [frds.io/measures/](https://frds.io/measures/) and [frds.io/algorithms](https://frds.io/algorithms/).

[Supported Measures](https://frds.io/measures/)

- Absorption Ratio
- Contingent Claim Analysis
- Distress Insurance Premium
- Lerner Index (Banks)
- Long-Run Marginal Expected Shortfall (LRMES)
- Marginal Expected Shortfall
- Option Prices
- SRISK
- Systemic Expected Shortfall
- Z-score

[Algorithms](https://frds.io/algorithems/)

- GARCH(1,1)
- GARCH(1,1) - CCC
- GARCH(1,1) - DCC
- GJR-GARCH(1,1)
- GJR-GARCH(1,1) - DCC

## Examples

Some simple examples.

### Absorption Ratio

For example, Kritzman, Li, Page, and Rigobon (2010) propose an [Absorption Ratio](https://frds.io/measures/absorption_ratio/) that measures the fraction of the total variance of a set of asset returns explained or absorbed by a fixed number of eigenvectors. It captures the extent to which markets are unified or tightly coupled.

``` python
>>> import numpy as np
from frds.measures import AbsorptionRatio
>>> data = np.array( # Hypothetical 6 daily returns of 3 assets.
...             [
...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
...             ]
...         )
ar = AbsorptionRatio(data)
ar.estimate()
0.7746543307660252
```

### Bivariate GARCH-CCC

Use [`frds.algorithms.GARCHModel_CCC`](https://frds.io/algorithms/garch-ccc) to estimate a bivariate Constant Conditional Correlation (CCC) GARCH model. The results are as good as those obtained in Stata, marginally better based on log-likelihood.

``` python 
>>> import pandas as pd
>>> from pprint import pprint
>>> from frds.algorithms import GARCHModel_CCC
>>> data_url = "https://www.stata-press.com/data/r18/stocks.dta"
>>> df = pd.read_stata(data_url, convert_dates=["date"])
>>> nissan = df["nissan"].to_numpy() * 100
>>> toyota = df["toyota"].to_numpy() * 100
>>> model_ccc = GARCHModel_CCC(toyota, nissan)
>>> res = model_ccc.fit()
>>> pprint(res)
Parameters(mu1=0.02745814255283541,
           omega1=0.03401400758840226,
           alpha1=0.06593379740524756,
           beta1=0.9219575443861723,
           mu2=0.009390068254041505,
           omega2=0.058694325049554734,
           alpha2=0.0830561828957614,
           beta2=0.9040961791372522,
           rho=0.6506770477876749,
           loglikelihood=-7281.321453218112)
```

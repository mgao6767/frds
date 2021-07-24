# FRDS - *for better and easier finance research*

## What is FRDS?

[frds](https://github.com/mgao6767/frds/) is an open-sourced Python package for computing [a collection of major academic measures](/measures/) used in the finance literature in a simple and straightforward way.

It is developed and maintained by [Dr. Mingze Gao](https://mingze-gao.com) from the University of Sydney, as a personal project during postdoctoral research fellowship.

![frds](https://github.com/mgao6767/frds/raw/master/images/frds_logo.png)

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![DOWNLOADS](https://img.shields.io/pypi/dm/frds?label=PyPI%20downloads)

## Installation

=== "Install via `pip`"

    frds is available on PyPI and can be installed via `pip`.

    ```bash
    pip install frds -U
    ```

=== "Install from source"
    
    Sometimes new measures are added and available on GitHub but not yet published to PyPI.

    In this case, it may be useful to install directly from source.

    ``` bash
    git clone https://github.com/mgao6767/frds.git
    ```

    Build and install the package locally.

    ``` bash
    cd frds
    python setup.py build_ext --inplace
    pip install -e .
    ```

!!! note
    On Windows, [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) may need to be installed so that the C/C++ extensions in the package can be compiled. 

## Example uses

### Built-in measures

The primary purpose of `frds` is to offer ready-to-use functions used in researches.

For example, Kritzman, Li, Page, and Rigobon (2010) propose an [Absorption Ratio](https://frds.io/measures/absorption_ratio/) that measures the fraction of the total variance of a set of asset returns explained or absorbed by a fixed number of eigenvectors. It captures the extent to which markets are unified or tightly coupled.

``` python
>>> import numpy as np
>>> from frds.measures import absorption_ratio # (1)
>>> data = np.array( # (2)
...             [
...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
...             ]
...         )
>>> absorption_ratio(data, fraction_eigenvectors=0.2)
0.7746543307660252
```

1. `#!python absorption_ratio` function can also be imported using:
   
    ``` python
    from frds.measures.bank import absorption_ratio
    ```
    
    :octicons-light-bulb-16: Tip: You can use ++tab++ to navigate annotations.

2. Hypothetical 6 daily returns of 3 assets.

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
>>> distress_insurance_premium(default_probabilities, correlations)       
0.29787743892
```

For a complete list of supported built-in measures, please check [frds.io/measures/](https://frds.io/measures/).

### Data source integration

Additionally, `frds` provides an interface to load data from common data sources such as the [Wharton Research Data Services (WRDS)](https://wrds-web.wharton.upenn.edu/wrds/).

As an example, let's say we want to download the Compustat Fundamentals Annual dataset.

``` python
>>> from frds.data.wrds.comp import Funda
>>> from frds.io.wrds import load # (1)
>>> FUNDA = load(Funda, use_cache=True, obs=100) # (2)
>>> FUNDA.data.head()
                                    FYEAR INDFMT CONSOL POPSRC DATAFMT   TIC      CUSIP                   CONM  ... PRCL_F   ADJEX_F RANK    AU  AUOP  AUOPIC CEOSO CFOSO
GVKEY  DATADATE                                                                                                 ...
001000 1961-12-31 00:00:00.000000  1961.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.341831  NaN  None  None    None  None  None
       1962-12-31 00:00:00.000000  1962.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.341831  NaN  None  None    None  None  None
       1963-12-31 00:00:00.000000  1963.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.244497  NaN  None  None    None  None  None
       1964-12-31 00:00:00.000000  1964.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.089999  NaN  None  None    None  None  None
       1965-12-31 00:00:00.000000  1965.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.089999  NaN  None  None    None  None  None

[5 rows x 946 columns]
```

1.  Here it skips the setup of WRDS login credentials. To do so, run the following script. 

    ``` python
    from frds.io.wrds import setup
    setup(username="username", password="password", save_credentials=True)
    ```

    If `save_credentials=True`, the username and password will be saved locally in `credentials.json` in the `frds` folder. Then in later uses, no more setup is required (no just current session).

    The `frds` folder is created under the user's home directory to store downloaded data upon installation.

2. `#!python use_cache=True` attempts to load the data from local cache instead of downloading it again.

We can then compute some measures on the go:

``` python
>>> tangibility = FUNDA.PPENT / FUNDA.AT # (1)
>>> type(tangibility)
<class 'pandas.core.series.Series'>
>>> tangibility.sample(10).sort_index()
GVKEY   DATADATE
001000  1965-12-31 00:00:00.000000    0.604762
        1967-12-31 00:00:00.000000    0.539495
        1968-12-31 00:00:00.000000    0.654171
        1977-12-31 00:00:00.000000    0.452402
001001  1985-12-31 00:00:00.000000    0.567439
001003  1980-12-31 00:00:00.000000         NaN
        1988-01-31 00:00:00.000000    0.073495
001004  1967-05-31 00:00:00.000000    0.175518
        1980-05-31 00:00:00.000000    0.183682
        1982-05-31 00:00:00.000000    0.286231
dtype: float64
```

1. The `frds.data.wrds.comp.Funda` class has all the variables in the Fundamental Annual dataset as attributes with proper docstrings. 

    Hence, we can write much simpler expressions whenever possible. 

## Next step

- [x] Introduction of `frds`
- [ ] Check out the [notes to get started](https://frds.io/../../getting-started/)
- [ ] Check out the [documentation of built-in measures](https://frds.io/measures/)
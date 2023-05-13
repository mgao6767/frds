# `frds` - *for better and easier finance research*

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![DOWNLOADS](https://img.shields.io/pypi/dm/frds?label=PyPI%20downloads) [![Test](https://github.com/mgao6767/frds/actions/workflows/test.yml/badge.svg)](https://github.com/mgao6767/frds/actions/workflows/test.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[`frds`](https://github.com/mgao6767/frds/) is a Python package for computing [a collection of academic measures](/measures/) used in the finance literature. It is developed by [Dr. Mingze Gao](https://mingze-gao.com) from the University of Sydney, as a personal project during his postdoctoral research fellowship.

![frds](https://github.com/mgao6767/frds/raw/main/images/frds_logo.png)

## Installation

=== "Install via `pip`"

    frds is available on PyPI and can be installed via `pip`.

    ```bash
    pip install frds
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
    pip install -e .
    ```

## Built-in measures

The primary purpose of `frds` is to offer ready-to-use functions.

=== "Example: Absorption Ratio"

    For example, Kritzman, Li, Page, and Rigobon (2010) propose an [Absorption Ratio](/measures/absorption_ratio/) that measures the fraction of the total variance of a set of asset returns explained or absorbed by a fixed number of eigenvectors. It captures the extent to which markets are unified or tightly coupled.

    ``` python title="Example: Absorption Ratio"
    >>> import numpy as np
    >>> from frds.measures import absorption_ratio # (1)
    >>> data = np.array( # (2)
    ...             [
    ...                 [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
    ...                 [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
    ...                 [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
    ...             ]
    ...         )
    >>> absorption_ratio.estimate(data, fraction_eigenvectors=0.2)
    0.7746543307660252
    ```

    1. We could also import directly the `estimate` function in `#!python absorption_ratio`. 
    
        ``` python
        from frds.measures.absorption_ratio import estimate
        ```
        
        :octicons-light-bulb-16: Tip: You can use ++tab++ to navigate annotations.

    2. Hypothetical 6 daily returns of 3 assets.

=== "Example: Distress Insurance Premium"

    Another example, [Distress Insurance Premium (DIP)](/measures/distress_insurance_premium/) proposed by Huang, Zhou, and Zhu (2009) as a systemic risk measure of a hypothetical insurance premium against a systemic financial distress, defined as total losses that exceed a given threshold, say 15%, of total bank liabilities.

    ``` python title="Example: Distress Insurance Premium"
    >>> from frds.measures import distress_insurance_premium as dip
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
    >>> dip.estimate(default_probabilities, correlations)       
    0.28661995758
    ```

For a complete list of supported built-in measures, please check [frds.io/measures/](/measures/).

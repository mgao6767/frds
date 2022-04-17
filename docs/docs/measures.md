# Using Built-In Measures

The real time-saver is the built-in measures in `frds.measures`.

=== "Example 1"

    For example, Kritzman, Li, Page, and Rigobon (2010) propose an [Absorption Ratio](https://frds.io/measures/absorption_ratio/) that measures the fraction of the total variance of a set of asset returns explained or absorbed by a fixed number of eigenvectors. It captures the extent to which markets are unified or tightly coupled.

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
    >>> absorption_ratio(data, fraction_eigenvectors=0.2)
    0.7746543307660252
    ```

    1. `#!python absorption_ratio` function can also be imported using:
    
        ``` python
        from frds.measures.bank import absorption_ratio
        ```
        
        :octicons-light-bulb-16: Tip: You can use ++tab++ to navigate annotations.

    2. Hypothetical 6 daily returns of 3 assets.

=== "Example 2"

    Another example, [Distress Insurance Premium (DIP)](https://frds.io/measures/distress_insurance_premium/) proposed by Huang, Zhou, and Zhu (2009) as a systemic risk measure of a hypothetical insurance premium against a systemic financial distress, defined as total losses that exceed a given threshold, say 15%, of total bank liabilities.

    ``` python title="Example: Distress Insurance Premium"
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
    0.28661995758
    ```

For a complete list of supported built-in measures, please check [frds.io/measures/](https://frds.io/measures/).


# MIT License

# Copyright (c) 2021 Kairong Xiao
# Copyright (c) 2021 Mingze Gao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union, Tuple
import numpy as np
from numpy.core.fromnumeric import cumsum
from numpy.lib import interp, polyval
from numpy.lib.scimath import sqrt


def histcounts(x: np.ndarray, edge: np.ndarray):
    """Returns the same as the Matlab code:

    ``` matlab
    r = histcounts(X,edge,'Normalization','probability')';
    ```

    Normalization: probability
    Specify 'Normalization' as 'probability' to normalize the bin counts so that sum(N) is 1.
    That is, each bin count represents the probability that an observation falls within that bin.

    Args:
        X (np.ndarray): input data.
        edge (np.ndarray): edges used to determine the bins.

    Returns:
        np.ndarray: Array of probabilities that an observation falls within the i-th bin.
    """
    probs = np.zeros(len(edge))
    map_to_bins = np.digitize(x, edge)
    for i in map_to_bins:
        probs[i - 1] += 1
    # Normalization
    probs = probs / sum(probs)
    # The last one is not meaningful since #bins will be one less than len(edges)
    return probs[:-1]


def fuzzy_bunching(
    X: np.ndarray,
    X0: Union[np.ndarray, None],
    x_l: float,
    x_min: float,
    x_max: float,
    degree: int,
    friction=True,
    noise=True,
    fig=False,
) -> Tuple[float, float]:
    r"""Fuzzy bunching

    See e.g., [Kleven and Waseem (2013)](https://doi.org/10.1093/qje/qjt004) for bunching estimation,
    and [Alvero and Xiao (2020)](https://dx.doi.org/10.2139/ssrn.3611447) for fuzzy bunching.

    Note:
        This code is adapted from Xiao's Matlab code, available at [his website](https://sites.google.com/site/kairongxiao/).

    Args:
        X (np.ndarray): bunching sample
        X0 (Union[np.ndarray, None]): non-bunching sample if exists, skip otherwise by setting it to `None`.
        x_l (float): threshold
        x_min (float): excluded range lower bound
        x_max (float): excluded range upper bound
        degree (int): degree of polynomial for conterfactual density
        friction (bool, optional): whether to allow optimization friction. Defaults to True.
        noise (bool, optional): whether to allow noise in the data. Defaults to True.
        fig (bool, optional): whether to create a figure of CDF. Defaults to False.

    Returns:
        Tuple[float, float]: (dx_hat, alpha_hat), where `dx_hat` is bunching range, $\Delta q$, and `alpha_hat` is non-optimizing share, $\alpha$.

    Examples:
        >>> from frds.algorithms.bunching import fuzzy_bunching
        >>> import numpy as np

        Generate a sample.
        >>> N = 1000
        >>> x_l = 0.5
        >>> dx = 0.1
        >>> alpha = 0
        >>> rng = np.random.RandomState(0)
        >>> Z0 = rng.rand(N)
        >>> Z = Z0.copy()
        >>> Z[(Z0 > x_l) & (Z0 <= x_l + dx)] = x_l
        >>> Z[: round(alpha * N)] = Z0[: round(alpha * N)]
        >>> u = 0.0
        >>> U = u * np.random.randn(N)
        >>> X = np.add(Z, U)
        >>> X0 = np.add(Z0, U)
        >>> x_max = x_l + 1.1 * dx
        >>> x_min = x_l * 0.99

        Fuzzy bunching estimation
        >>> fuzzy_bunching(X, X0, x_l, x_min, x_max, degree=5, friction=True, noise=True, fig=False)
        (0.032179950901143374, 0.0)

    References:
        * [Alvero and Xiao (2020)](https://dx.doi.org/10.2139/ssrn.3611447), Fuzzy bunching, *SSRN*.
        * [Kleven and Waseem (2013)](https://doi.org/10.1093/qje/qjt004),
            Using notches to uncover optimization frictions and structural elasticities: Theory and evidence from pakistan, *The Quarterly Journal of Ecnomics*, 128(2), 669-723.

    Todo:
        - [ ] Option to specify output plot path.
        - [ ] Plot styling.
    """
    G = 10 ** 4
    edge = np.linspace(x_min, x_max, G + 1)
    grid = edge[:-1]
    f = histcounts(X, edge)
    if X0 is None:
        N = len(X)
        J = round(N / 10)  # J bins, with 10 obs in each bin

        # y: unconditional probability for the whole sample.
        #
        # Try the best to reproduce the same behaviour as in Xiao's Matlab code:
        # ``` matlab
        # y, x = histcounts(X, J, 'Normalizaton', 'probability');
        # ```
        # This variant of Matlab `histcounts` produces J bins with equal sizes.
        # The computation of bins is automatic and no details provided.
        # So, here I use numpy `histogram` function to compute the optimal bins (edges).
        _, edges = np.histogram(X, J)
        # Then , use the my `histcounts` with calculated bin edges.
        y = histcounts(X, edges)
        # Because ('Normalization','probability') is specified in the Matlab code,
        # the returned `x`, i.e. edges, from the Matlab code above is always the same as
        # `linspace(0,1,J+1)`, since it's probability.
        x = np.linspace(0, 1, J + 1)
        x = x[:-1]

        w = (x >= x_min) & (x <= x_max)
        p = np.polyfit(x[~w], y[~w], degree)
        f0 = np.polyval(p, grid) / sum(np.polyval(p, grid))

    else:
        N = len(X0)
        J = round(N / 10)
        # Again, I use numpy `histogram` to compute the optimal bins (edges).
        _, edges = np.histogram(X0, J)
        y = histcounts(X0, edges)
        x = np.linspace(0, 1, J + 1)
        x = x[:-1]
        p = np.polyfit(x, y, degree)
        f0 = histcounts(X0, edge)

    # In Numpy, np.trapz(y, x); in Matlab, trapz(x, y).
    # Special attention to the order of parameters.
    A = np.trapz(cumsum(f) - cumsum(f0), grid)
    f0_x_l = 1 / (x_max - x_min)

    # Warning: `2*A/f0_x_l` may be negative, yielding imaginary roots.
    # Have to use `numpy.lib.scimath.sqrt` to handle it.
    if not friction:
        dx_hat = sqrt(2 * A / f0_x_l)
        alpha_hat = 0

    if friction and noise:
        dx_hat = sqrt(2 * A / f0_x_l)
        dF_alpha = interp(x_l + dx_hat, grid, cumsum(f)) - interp(x_l, grid, cumsum(f))
        dF0_alpha = interp(x_l + dx_hat, grid, cumsum(f0)) - interp(
            x_l, grid, cumsum(f0)
        )
        alpha_hat = dF_alpha / dF0_alpha
        dx_hat = sqrt(2 * A / ((1 - alpha_hat) * f0_x_l))

    if friction and noise:
        F_x_l = interp(x_l, grid, cumsum(f))
        alpha_vec = np.linspace(0, 0.99, 100)
        dx_vec = sqrt((2 * A) / ((1 - alpha_vec) * f0_x_l))
        alpha_vec2 = (
            2 * (interp(x_l + dx_vec, grid, cumsum(f)) - F_x_l) / (f0_x_l * dx_vec) - 1
        )
        index_values = np.argmin(np.abs(alpha_vec - alpha_vec2))
        alpha_hat = alpha_vec[index_values]
        dx_hat = sqrt((2 * A) / ((1 - alpha_hat) * f0_x_l))

    # Make plot
    if fig:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Require matplotlib to generate the plot!")
        else:
            plt.plot(grid, cumsum(f))
            plt.plot(grid, cumsum(f0))
            plt.savefig("fig.png")

    return dx_hat, alpha_hat


def sharp_bunching(
    X: np.ndarray,
    X0: Union[np.ndarray, None],
    x_l: float,
    x_min: float,
    x_max: float,
    degree: int,
    friction=True,
    noise=True,
    fig=False,
) -> Tuple[float, float]:
    r"""Sharp bunching

    See e.g., [Kleven and Waseem (2013)](https://doi.org/10.1093/qje/qjt004) for bunching estimation,
    and [Alvero and Xiao (2020)](https://dx.doi.org/10.2139/ssrn.3611447) for fuzzy bunching.

    Note:
        This code is adapted from Xiao's Matlab code, available at [his website](https://sites.google.com/site/kairongxiao/).

    Args:
        X (np.ndarray): bunching sample
        X0 (Union[np.ndarray, None]): non-bunching sample if exists, skip otherwise by setting it to `None`.
        x_l (float): threshold
        x_min (float): excluded range lower bound
        x_max (float): excluded range upper bound
        degree (int): degree of polynomial for conterfactual density
        friction (bool, optional): whether to allow optimization friction. Defaults to True.
        noise (bool, optional): whether to allow noise in the data. Defaults to True.
        fig (bool, optional): whether to create a figure of CDF. Defaults to False.

    Returns:
        Tuple[float, float]: (dx_hat, alpha_hat), where `dx_hat` is bunching range, $\Delta q$, and `alpha_hat` is non-optimizing share, $\alpha$.

    Examples:
        >>> from frds.algorithms.bunching import sharp_bunching
        >>> import numpy as np

        Generate a sample.
        >>> N = 1000
        >>> x_l = 0.5
        >>> dx = 0.1
        >>> alpha = 0
        >>> rng = np.random.RandomState(0)
        >>> Z0 = rng.rand(N)
        >>> Z = Z0.copy()
        >>> Z[(Z0 > x_l) & (Z0 <= x_l + dx)] = x_l
        >>> Z[: round(alpha * N)] = Z0[: round(alpha * N)]
        >>> u = 0.0
        >>> U = u * np.random.randn(N)
        >>> X = np.add(Z, U)
        >>> X0 = np.add(Z0, U)
        >>> x_max = x_l + 1.1 * dx
        >>> x_min = x_l * 0.99

        Sharpe bunching estimation
        >>> sharp_bunching(X, X0, x_l, x_min, x_max, degree=5, friction=True, noise=True, fig=False)
        (0.00984977111886289, -1.0)

    References:
        * [Alvero and Xiao (2020)](https://dx.doi.org/10.2139/ssrn.3611447), Fuzzy bunching, *SSRN*.
        * [Kleven and Waseem (2013)](https://doi.org/10.1093/qje/qjt004),
            Using notches to uncover optimization frictions and structural elasticities: Theory and evidence from pakistan, *The Quarterly Journal of Ecnomics*, 128(2), 669-723.

    Todo:
        - [ ] Option to specify output plot path.
        - [ ] Plot styling.
    """
    G = 10 ** 4
    edge = np.linspace(x_min, x_max, G + 1)
    grid = edge[:-1]
    f = histcounts(X, edge)
    if X0 is None:
        N = len(X)
        J = round(N / 10)  # J bins, with 10 obs in each bin
        _, edges = np.histogram(X, J)
        y = histcounts(X, edges)
        x = np.linspace(0, 1, J + 1)
        x = x[:-1]

        w = (x >= x_min) & (x <= x_max)
        p = np.polyfit(x[~w], y[~w], degree)
        f0 = np.polyval(p, grid) / sum(np.polyval(p, grid))

    else:
        N = len(X0)
        J = round(N / 10)
        _, edges = np.histogram(X0, J)
        y = histcounts(X0, edges)
        x = np.linspace(0, 1, J + 1)
        x = x[:-1]
        p = np.polyfit(x, y, degree)
        f0 = histcounts(X0, edge)

    dF = interp(x_l + (x_l - x_min), grid, cumsum(f)) - interp(x_min, grid, cumsum(f))
    dF0 = interp(x_l + (x_l - x_min), grid, cumsum(f0)) - interp(
        x_min, grid, cumsum(f0)
    )
    # Bunching mass
    B = dF - dF0
    f0_x_l = polyval(p, x_l) / np.sum(polyval(p, grid)) / (grid[1] - grid[0])

    if not friction:
        dx_hat = B / f0_x_l
        alpha_hat = 0

    if friction and noise:
        dx_hat = B / f0_x_l

        dF_alpha = interp(x_l + dx_hat, grid, cumsum(f)) - interp(
            x_l + (x_l - x_min), grid, cumsum(f)
        )
        dF0_alpha = interp(x_l + dx_hat, grid, cumsum(f0)) - interp(
            x_l + (x_l - x_min), grid, cumsum(f0)
        )
        alpha_hat = dF_alpha / dF0_alpha
        dx_hat = B / ((1 - alpha_hat) * f0_x_l)

    if friction and noise:
        F_x_l = interp(x_l, grid, cumsum(f))
        alpha_vec = np.linspace(0, 0.99, 100)
        dx_vec = B / ((1 - alpha_vec) * f0_x_l)
        alpha_vec2 = (
            2 * (interp(x_l + dx_vec, grid, cumsum(f)) - F_x_l) / (f0_x_l * dx_vec) - 1
        )
        index_values = np.argmin(np.abs(alpha_vec - alpha_vec2))
        alpha_hat = alpha_vec[index_values]
        dx_hat = B / ((1 - alpha_hat) * f0_x_l)
        alpha_hat = (
            2 * (interp(x_l + dx_hat, grid, cumsum(f)) - F_x_l) / (f0_x_l * dx_hat) - 1
        )  # one more iteration

    # Make plot
    if fig:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Require matplotlib to generate the plot!")
        else:
            plt.plot(grid, cumsum(f))
            plt.plot(grid, cumsum(f0))
            plt.savefig("fig.png")

    return dx_hat, alpha_hat

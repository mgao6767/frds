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


def fuzzy_bunching(X, X0, x_l, x_min, x_max, degree, friction, noise, fig=False):
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
    if friction == 0:
        dx_hat = sqrt(2 * A / f0_x_l)
        alpha_hat = 0

    if friction == 1 and noise == 0:
        dx_hat = sqrt(2 * A / f0_x_l)
        dF_alpha = interp(x_l + dx_hat, grid, cumsum(f)) - interp(x_l, grid, cumsum(f))
        dF0_alpha = interp(x_l + dx_hat, grid, cumsum(f0)) - interp(
            x_l, grid, cumsum(f0)
        )
        alpha_hat = dF_alpha / dF0_alpha
        dx_hat = sqrt(2 * A / ((1 - alpha_hat) * f0_x_l))

    if friction == 1 and noise == 1:
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


def sharp_bunching(X, X0, x_l, x_min, x_max, degree, friction, noise, fig=False):
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

    if friction == 0:
        dx_hat = B / f0_x_l
        alpha_hat = 0

    if friction == 1 and noise == 0:
        dx_hat = B / f0_x_l

        dF_alpha = interp(x_l + dx_hat, grid, cumsum(f)) - interp(
            x_l + (x_l - x_min), grid, cumsum(f)
        )
        dF0_alpha = interp(x_l + dx_hat, grid, cumsum(f0)) - interp(
            x_l + (x_l - x_min), grid, cumsum(f0)
        )
        alpha_hat = dF_alpha / dF0_alpha
        dx_hat = B / ((1 - alpha_hat) * f0_x_l)

    if friction == 1 and noise == 1:
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

import numpy as np


def hhi_index(firm_sizes: np.ndarray, weights: np.ndarray = None) -> float:
    r"""Herfindahl–Hirschman Index

    A common measure of market concentration, defined as

    $$
    H = \sum_{i=1}^{N} s_i^2
    $$

    where $s_i$ is firm $i$'s market share in the industry of $N$ firms.

    Args:
        firm_sizes (np.ndarray): (n_firms,) array of firm sizes suchs as sales, used to compute market shares
        weights (np.ndarray): (n_firms,) array of the weights given to each firm's market share. Defaults to equal weights.

    !!! note
        If `weights` are provided, the HHI index is computed as:

        $$
        H = \sum_{i=1}^{N} s_i^2 \times w_i
        $$

        where $w_i$ is the weight given to the market share of firm $i$.

    Returns:
        float: HHI-index for the industry

    Examples:
        >>> import numpy as np
        >>> from frds.measures import hhi_index

        7 firms with equal sales
        >>> firm_sales = np.array([1,1,1,1,1,1,1])
        >>> hhi_index(firm_sales)
        0.14285714285714285

        6 firms, of which 1 has much larger sales
        >>> firm_sales = np.array([100,1,1,1,1,1])
        >>> hhi_index(firm_sales)
        0.9074829931972791

    References:
        - [Wikipedia](https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_Index)

    Todo:
        - [ ] Allow `firm_sizes` to be multidimensional.
        - [ ] Check validity of input data (no negative firm sizes, etc.).
        - [x] Allow market shares to be weighted.
    """
    if weights is None:
        weights = np.ones(firm_sizes.shape)
    mkt_shares = firm_sizes / np.sum(firm_sizes)
    return np.sum(np.square(mkt_shares) * weights)

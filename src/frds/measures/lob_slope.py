import numpy as np


def NS2006_bid_slope() -> float:
    raise NotImplementedError


def NS2006_ask_slope() -> float:
    raise NotImplementedError


def GN2019_slope() -> float:
    raise NotImplementedError


def HS2001_quote_slope() -> float:
    raise NotImplementedError


def HS2001_log_quote_slope() -> float:
    raise NotImplementedError


def DGGW_ask_slope(
    ask_size: np.ndarray, ask_price_highest_level: np.ndarray, mid_point: np.ndarray
) -> float:
    """Ask Slope from Della Vedova, Gao, Grant and Westerholm (working paper)

    Args:
        ask_size (np.ndarray): ``(n,k)`` array of LOB ask sizes, where ``n`` is number of quotes and ``k`` is number of levels.
        ask_price_highest_level (np.ndarray): ``(n,)`` array of ask prices at the highest level ``k``.
        mid_point (np.ndarray): ``(n,)`` array of bid-ask midpoints.

    Returns:
        float: ask slope

    Examples:
        >>> from frds.measures.lob_slope import DGGW_ask_slope
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> mid_point = np.ones((1000,)) # Assume bid-ask midpoint stays at 1
        >>> lv1 = rng.uniform(low=100, high=120, size=(1000,)) # Simulated level 1 ask sizes
        >>> lv2 = rng.uniform(low=90, high=110, size=(1000,)) # Simulated level 2 ask sizes
        >>> lv3 = rng.uniform(low=80, high=100, size=(1000,))
        >>> lv4 = rng.uniform(low=70, high=90, size=(1000,))
        >>> lv5 = rng.uniform(low=50, high=80, size=(1000,))
        >>> ask_size = np.array([lv1, lv2, lv3, lv4, lv5]).T
        >>> ask_size.shape
        (1000, 5)
        >>> ask_price_highest_level = rng.uniform(low=1.1, high=2.0, size=(1000,)) # Simulated ask price at level 5
        >>> DGGW_ask_slope(ask_size, ask_price_highest_level, mid_point)
        1136.8718207500128
    """
    assert len(ask_size.shape) > 1
    n, k = ask_size.shape
    assert k > 1  # at least two levels
    assert len(mid_point.shape) == 1 and n == mid_point.shape[0]

    slope = np.sum(ask_size, axis=1) / (ask_price_highest_level - mid_point)
    return np.nanmean(slope)


def DGGW_bid_slope(
    bid_size: np.ndarray, bid_price_highest_level: np.ndarray, mid_point: np.ndarray
) -> float:
    """Bid Slope from Della Vedova, Gao, Grant and Westerholm (working paper)

    Args:
        bid_size (np.ndarray): ``(n,k)`` array of LOB bid sizes, where ``n`` is number of quotes and ``k`` is number of levels.
        bid_price_highest_level (np.ndarray): ``(n,)`` array of bid prices at the highest level ``k``.
        mid_point (np.ndarray): ``(n,)`` array of bid-ask midpoints.

    Returns:
        float: bid slope

    Examples:
        >>> from frds.measures.lob_slope import DGGW_bid_slope
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> mid_point = np.ones((1000,)) # Assume bid-ask midpoint stays at 1
        >>> lv1 = rng.uniform(low=100, high=120, size=(1000,)) # Simulated level 1 bid sizes
        >>> lv2 = rng.uniform(low=90, high=110, size=(1000,)) # Simulated level 2 bid sizes
        >>> lv3 = rng.uniform(low=80, high=100, size=(1000,))
        >>> lv4 = rng.uniform(low=70, high=90, size=(1000,))
        >>> lv5 = rng.uniform(low=50, high=80, size=(1000,))
        >>> bid_size = np.array([lv1, lv2, lv3, lv4, lv5]).T
        >>> bid_size.shape
        (1000, 5)
        >>> bid_price_highest_level = rng.uniform(low=.7, high=.99, size=(1000,)) # Simulated bid price at level 5
        >>> DGGW_bid_slope(bid_size, bid_price_highest_level, mid_point)
        5316.539811566988
    """
    assert len(bid_size.shape) > 1
    n, k = bid_size.shape
    assert k > 1  # at least two levels
    assert len(mid_point.shape) == 1 and n == mid_point.shape[0]

    slope = np.sum(bid_size, axis=1) / np.abs((bid_price_highest_level - mid_point))
    return np.nanmean(slope)


def DGGW_bid_side_slope_difference(
    bid_size: np.ndarray, bid_price: np.ndarray
) -> float:
    """Bid-side Slope Difference from Della Vedova, Gao, Grant and Westerholm (working paper)

    Args:
        bid_size (np.ndarray): ``(n,k)`` array of LOB bid sizes, where ``n`` is number of quotes and ``k`` is number of levels.
        bid_price (np.ndarray): ``(n,k)`` array of LOB bid prices, where ``n`` is number of quotes and ``k`` is number of levels.

    Returns:
        float: bid-side slope difference
    """
    assert len(bid_size.shape) > 1
    n, k = bid_size.shape
    assert k > 1  # at least two levels
    # fmt: off
    slope_h = (bid_size[:,4] - bid_size[:,2]) / np.abs(bid_price[:,4] - bid_price[:,2])
    slope_l = (bid_size[:,2] - bid_size[:,0]) / np.abs(bid_price[:,2] - bid_price[:,0])
    return np.nanmean(slope_h - slope_l)


def DGGW_ask_side_slope_difference(
    ask_size: np.ndarray, ask_price: np.ndarray
) -> float:
    """Ask-side Slope Difference from Della Vedova, Gao, Grant and Westerholm (working paper)

    Args:
        ask_size (np.ndarray): ``(n,k)`` array of LOB ask sizes, where ``n`` is number of quotes and ``k`` is number of levels.
        ask_price (np.ndarray): ``(n,k)`` array of LOB ask prices, where ``n`` is number of quotes and ``k`` is number of levels.

    Returns:
        float: ask-side slope difference
    """
    return DGGW_bid_side_slope_difference(ask_size, ask_price)


def DGGW_scaled_depth_difference(bid_size: np.ndarray, ask_size: np.ndarray) -> float:
    """SDD from Della Vedova, Gao, Grant and Westerholm (working paper)

    Args:
        bid_size (np.ndarray): ``(n,k)`` array of LOB bid sizes, where ``n`` is number of quotes and ``k`` is number of levels.
        ask_size (np.ndarray): ``(n,k)`` array of LOB ask sizes, where ``n`` is number of quotes and ``k`` is number of levels.

    Returns:
        float: scaled depth difference (simple weighted over ``n`` quotes)

    Examples:
        >>> from frds.measures.lob_slope import DGGW_scaled_depth_difference
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> lv1 = rng.uniform(low=100, high=120, size=(1000,)) # Simulated level 1 bid sizes
        >>> lv2 = rng.uniform(low=90, high=110, size=(1000,)) # Simulated level 2 bid sizes
        >>> lv3 = rng.uniform(low=80, high=100, size=(1000,))
        >>> lv4 = rng.uniform(low=70, high=90, size=(1000,))
        >>> lv5 = rng.uniform(low=50, high=80, size=(1000,))
        >>> bid_size = np.array([lv1, lv2, lv3, lv4, lv5]).T
        >>> lv1 = rng.uniform(low=200, high=300, size=(1000,)) # Simulated level 1 ask sizes
        >>> lv2 = rng.uniform(low=90, high=110, size=(1000,)) # Simulated level 2 ask sizes
        >>> lv3 = rng.uniform(low=80, high=100, size=(1000,))
        >>> lv4 = rng.uniform(low=70, high=90, size=(1000,))
        >>> lv5 = rng.uniform(low=50, high=80, size=(1000,))
        >>> ask_size = np.array([lv1, lv2, lv3, lv4, lv5]).T
        >>> DGGW_scaled_depth_difference(bid_size, ask_size)
        >>> 0.2697934067000831
    """
    assert bid_size.shape == ask_size.shape
    bid_depth = np.sum(bid_size, axis=1)
    ask_depth = np.sum(ask_size, axis=1)
    sdd = 2 * (ask_depth - bid_depth) / (ask_depth + bid_depth)
    return np.nanmean(sdd)

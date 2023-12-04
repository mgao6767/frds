import numpy as np


def simple_price_impact(
    price: np.ndarray,
    midpoint_later: np.ndarray,
    midpoint: np.ndarray,
    volume: np.ndarray,
    trade_direction: np.ndarray = None,
    pct_spread=True,
) -> float:
    """Simple Price Impact (dollar volume weighted)

    Args:
        price (np.ndarray): ``(N,)`` array of ``N`` trade prices
        midpoint_later (np.ndarray): ``(N,)`` array of ``N`` bid-ask midpoints some time (e.g., 5min) after corresponding trade
        midpoint (np.ndarray): ``(N,)`` array of ``N`` bid-ask midpoints at trade
        volume (np.ndarray): ``(N,)`` array of ``N`` trade sizes
        trade_direction (np.ndarray, optional): ``(N,)`` array of ``N`` trade directions. Defaults to None.
            If None, use equation :math:numref:`simple-price-impact` or :math:numref:`simple-price-impact-log`.
            If set, use equation :math:numref:`simple-price-impact-with-direction` or :math:numref:`simple-price-impact-log-with-direction`.
        pct_spread (bool, optional): whether to return percentage spread.
            Defaults to True. If False, return log spread.

    Returns:
        float: simple price impact
    """
    midpoint_later = np.asarray(midpoint_later)
    midpoint = np.asarray(midpoint)
    volume = np.asarray(volume)
    assert midpoint_later.shape == midpoint.shape == volume.shape
    if not isinstance(trade_direction, np.ndarray):
        # No trade direction
        if pct_spread:
            spread = 2 * np.absolute(midpoint_later - midpoint) / midpoint * 100
        else:
            spread = 2 * np.absolute(np.log(midpoint_later) - np.log(midpoint))
    else:
        # Use trade direction
        assert midpoint.shape == trade_direction.shape
        if pct_spread:
            spread = 2 * trade_direction * (midpoint_later - midpoint) / midpoint * 100
        else:
            spread = 2 * trade_direction * (np.log(midpoint_later) - np.log(midpoint))

    return np.average(spread, weights=volume * price)

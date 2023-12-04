import numpy as np


def quoted_spread(bid: np.ndarray, ask: np.ndarray, pct_spread=True) -> float:
    """Quoted bid-ask spread (simple weighted)

    Args:
        bid (np.ndarray): ``(N,)`` array of ``N`` bids
        ask (np.ndarray): ``(N,)`` array of ``N`` asks
        pct_spread (bool, optional): whether to return percentage spread.
            Defaults to True. If False, return log spread.

    Returns:
        float: quoted spread
    """
    bid = np.asarray(bid)
    ask = np.asarray(ask)
    assert bid.shape == ask.shape
    if pct_spread:
        midpt = (bid + ask) / 2
        spread = (ask - bid) / midpt
        return np.average(spread) * 100
    else:
        return np.average(np.log(ask) - np.log(bid))


def effective_spread(
    price: np.ndarray,
    midpoint: np.ndarray,
    volume: np.ndarray,
    trade_direction: np.ndarray = None,
    pct_spread=True,
) -> float:
    """Effective spread (dollar volume weighted)

    Args:
        price (np.ndarray): ``(N,)`` array of ``N`` trade prices
        midpoint (np.ndarray): ``(N,)`` array of ``N`` bid-ask midpoints
        volume (np.ndarray): ``(N,)`` array of ``N`` trade sizes
        trade_direction (np.ndarray, optional): ``(N,)`` array of ``N`` trade directions. Defaults to None.
            If None, use equation :math:numref:`effective-spread` or :math:numref:`effective-spread-log`.
            If set, use equation :math:numref:`effective-spread-with-direction` or :math:numref:`effective-spread-log-with-direction`.
        pct_spread (bool, optional): whether to return percentage spread.
            Defaults to True. If False, return log spread.

    Returns:
        float: effective spread
    """
    price = np.asarray(price)
    midpoint = np.asarray(midpoint)
    volume = np.asarray(volume)
    assert price.shape == midpoint.shape == volume.shape
    if not isinstance(trade_direction, np.ndarray):
        # No trade direction
        if pct_spread:
            spread = 2 * np.absolute(price - midpoint) / midpoint * 100
        else:
            spread = 2 * np.absolute(np.log(price) - np.log(midpoint))
    else:
        # Use trade direction
        assert price.shape == trade_direction.shape
        if pct_spread:
            spread = 2 * trade_direction * (price - midpoint) / midpoint * 100
        else:
            spread = 2 * trade_direction * (np.log(price) - np.log(midpoint))
    return np.average(spread, weights=volume * price)


def realized_spread(
    price: np.ndarray,
    midpoint_later: np.ndarray,
    midpoint: np.ndarray,
    volume: np.ndarray,
    trade_direction: np.ndarray = None,
    pct_spread=True,
) -> float:
    """Realized spread (dollar volume weighted)

    Args:
        price (np.ndarray): ``(N,)`` array of ``N`` trade prices
        midpoint_later (np.ndarray): ``(N,)`` array of ``N`` bid-ask midpoints some time (e.g., 5min) after corresponding trade
        midpoint (np.ndarray): ``(N,)`` array of ``N`` bid-ask midpoints at trade
        volume (np.ndarray): ``(N,)`` array of ``N`` trade sizes
        trade_direction (np.ndarray, optional): ``(N,)`` array of ``N`` trade directions. Defaults to None.
            If None, use equation :math:numref:`realized-spread` or :math:numref:`realized-spread-log`.
            If set, use equation :math:numref:`realized-spread-with-direction` or :math:numref:`realized-spread-log-with-direction`.
        pct_spread (bool, optional): whether to return percentage spread.
            Defaults to True. If False, return log spread.

    Returns:
        float: realized spread
    """
    price = np.asarray(price)
    midpoint_later = np.asarray(midpoint_later)
    midpoint = np.asarray(midpoint)
    volume = np.asarray(volume)
    assert price.shape == midpoint_later.shape == midpoint.shape == volume.shape
    if not isinstance(trade_direction, np.ndarray):
        # No trade direction
        if pct_spread:
            spread = 2 * np.absolute(price - midpoint_later) / midpoint * 100
        else:
            spread = 2 * np.absolute(np.log(price) - np.log(midpoint_later))
    else:
        # Use trade direction
        assert price.shape == trade_direction.shape
        if pct_spread:
            spread = 2 * trade_direction * (price - midpoint_later) / midpoint * 100
        else:
            spread = 2 * trade_direction * (np.log(price) - np.log(midpoint_later))

    return np.average(spread, weights=volume * price)

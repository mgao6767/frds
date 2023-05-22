import numpy as np

from .long_run_mes import estimate as lrmes


def estimate(
    firm_returns: np.ndarray,
    market_returns: np.ndarray,
    W: float | np.ndarray,
    D: float | np.ndarray,
    k=0.08,
    lrmes_h=22,
    lrmes_S=10000,
    lrmes_C=-0.1,
    lrmes_random_seed=42,
    aggregate_srisk=False,
) -> np.ndarray | float:
    """SRISK of firm(s) or market at a given time

    Args:
        firm_returns (np.ndarray): (n_days,) array of firm log returns. Can also be (n_day,n_firms) array of multiple firms' log returns.
        market_returns (np.ndarray): (n_days,) array of market log returns
        W (float | np.ndarray): market value of equity. It can be either a single float value for a firm or a (n_firms,) array for multiple firms.
        D (float | np.ndarray): book value of debt. It can be either a single float value for a firm or a (n_firms,) array for multiple firms.
        k (float, optional): prudential capital factor. Defaults to 8%.
        lrmes_h (int, optional): parameter used to estimate `LRMES`. Prediction horizon. Defaults to 22.
        lrmes_S (int, optional): parameter used to estimate `LRMES`. The number of simulations. Defaults to 10_000.
        lrmes_C (float, optional): parameter used to estimate `LRMES`. The markdown decline that defines a systemic event. Defaults to -0.1.
        lrmes_random_seed (int, optional): random seed in estimating `LRMES`. Defaults to 42.
        aggregate_srisk (bool, optional): whether to compute the aggregate SRISK. Defaults to False.

    Returns:
        np.ndarray | float: If `aggregate_srisk=False`, (n_firms,) array of firm-level SRISK measures. Otherwise, a single float value for aggregate SRISK.

    Examples:
        >>> from frds.measures.srisk import estimate as srisk
        >>> import yfinance as yf
        >>> import numpy as np
        >>> df = yf.download(tickers="SPY JPM GS", start="2017-01-01", end="2022-12-31",
        ...         progress=False, rounding=True)
        >>> df = df[['Adj Close']]
        >>> df.columns = df.columns.droplevel(0)
        >>> df
                        GS     JPM     SPY
        Date
        2017-01-03  214.04   72.67  202.09
        2017-01-04  215.43   72.80  203.29
        2017-01-05  213.82   72.13  203.13
        2017-01-06  216.99   72.14  203.85
        2017-01-09  215.21   72.19  203.18
        ...            ...     ...     ...
        2022-12-23  343.05  129.30  381.45
        2022-12-27  339.54  129.76  379.95
        2022-12-28  338.45  130.46  375.23
        2022-12-29  340.99  131.21  381.98
        2022-12-30  340.94  132.08  380.98
        [1510 rows x 3 columns]
        >>> mkt_returns = np.log(df.SPY.pct_change()+1).dropna().to_numpy()
        >>> firm_returns = np.array([np.log(df.JPM.pct_change()+1).dropna().to_numpy(),
        ...                          np.log(df.GS.pct_change()+1).dropna().to_numpy()]).T
        >>> srisk(firm_returns, mkt_returns,
        ...         W=np.array([100,80]), D=np.array([900,250])) # (1)
        array([  5.79157017, -64.68651904])
        >>> srisk(firm_returns, mkt_returns,
        ...         W=np.array([100,80]), D=np.array([900,250]),
        ...         aggregate_srisk=True) # (2)
        5.7915701669051245

        1. Hypothetical market value of equity and book value of debt.

        2. Only positive SRISK estimates are summed.

        !!! note
            `yfinance.download` may lead to different data at each run, so the results can vary.
            However, given a fixed input dataset, the function will yield the same estimate conditional on the same random seed.

    """
    if len(firm_returns.shape) == 1:
        # Single firm
        n_firms = 1
        assert firm_returns.shape == market_returns.shape
        assert isinstance(D, float) and isinstance(W, float)
    else:
        # Multple firms
        n_days, n_firms = firm_returns.shape
        assert n_firms > 1
        assert market_returns.shape[0] == n_days
        assert isinstance(D, np.ndarray) and isinstance(W, np.ndarray)
        assert D.shape == W.shape
        assert D.shape == np.zeros((n_firms,)).shape

    # Firm-level LRMES
    LRMES = lrmes(
        firm_returns,
        market_returns,
        h=lrmes_h,
        S=lrmes_S,
        C=lrmes_C,
        random_seed=lrmes_random_seed,
    )  # (n_firms,) array of LRMES estimates

    LVG = (D + W) / W

    SRISK = W * (k * LVG + (1 - k) * LRMES - 1)
    if not aggregate_srisk:
        return SRISK
    else:
        return np.sum(SRISK.clip(min=0.0))

import numpy as np
from arch import arch_model
from frds.algorithms.dcc import dcc, calc_Q_avg, calc_Q, calc_R


def estimate(
    firm_returns: np.ndarray,
    market_returns: np.ndarray,
    h=22,
    S=10_000,
    C=-0.1,
    random_seed=42,
) -> np.ndarray:
    """h-step-ahead LRMES forecasts conditional on a systemic event of market decline C

    Args:
        firm_returns (np.ndarray): (n_days,n_firms) array of firm log returns.
        market_returns (np.ndarray): (n_days,) array of market log returns.
        h (int, optional): h-period-ahead prediction horizon. Defaults to 22.
        S (int, optional): sample size used in simulation to generate LRMES forecasts. Defaults to 10000.
        C (float, optional): market decline used to define systemic event. Defaults to -0.1, i.e. -10%.
        random_seed (int, optional): random seed. Defaults to 42.

    Returns:
        np.ndarray: (n_firms,) array of LRMES forecasts

    Examples:
        >>> import frds.measures.long_run_mes as lrmes
        >>> import yfinance as yf
        >>> import numpy as np
        >>> df = yf.download(tickers="SPY JPM GS", start="2017-01-01", end="2022-12-31", progress=False)
        >>> df = df[['Adj Close']]
        >>> df.columns = df.columns.droplevel(0)
        >>> df
                            GS         JPM         SPY
        Date
        2017-01-03  214.042953   72.665413  202.085266
        2017-01-04  215.425156   72.799454  203.287506
        2017-01-05  213.821426   72.129333  203.126007
        2017-01-06  216.993469   72.137695  203.852783
        2017-01-09  215.212524   72.187935  203.179840
        ...                ...         ...         ...
        2022-12-23  343.053650  129.302628  381.454193
        2022-12-27  339.538818  129.755707  379.949921
        2022-12-28  338.446625  130.464844  375.227936
        2022-12-29  340.988434  131.213409  381.982178
        2022-12-30  340.938812  132.080154  380.975983
        [1510 rows x 3 columns]
        >>> mkt_returns = np.log(df.SPY.pct_change()+1).dropna().to_numpy()
        >>> firm_returns = np.array([np.log(df.JPM.pct_change()+1).dropna().to_numpy(),
        ...                          np.log(df.GS.pct_change()+1).dropna().to_numpy()]).T
        >>> lrmes.estimate(firm_returns, mkt_returns)
        array([ 0.12958814, -0.09460028])

        !!! note
            `yfinance.download` may lead to different data at each run, so the results can vary.
            However, given a fixed input dataset, the function will yield the same estimate conditional on the same random seed.

        Hence, the 22-day LRMES for JPM is 12.96% and -9.5% for GS.
        The latter would be ignored in computing SRISK, though.
    """
    if len(firm_returns.shape) == 1:
        firm_returns = firm_returns.reshape((firm_returns.shape[0], 1))
    n_days, n_firms = firm_returns.shape
    (_n_days,) = market_returns.shape
    assert n_days == _n_days

    # Fit GJR-GARCH for the market returns
    mkt_am = arch_model(market_returns, p=1, o=1, q=1, rescale=True)
    mkt_res = mkt_am.fit(update_freq=0, disp=False)
    epsilon_mkt = market_returns / mkt_res.conditional_volatility
    # Forecasted volatility
    mkt_vol_hat = np.sqrt(
        np.squeeze(
            mkt_res.forecast(
                mkt_res.params, horizon=h, reindex=False
            ).variance.T.to_numpy()
        )
    )  # (h,1) array of volatility forecasts

    lrmes = np.zeros((n_firms,))
    for i in range(n_firms):
        # Fit GJR-GARCH for each firm's returns
        firm_am = arch_model(firm_returns[:, i], p=1, o=1, q=1, rescale=True)
        firm_res = firm_am.fit(update_freq=0, disp=False)
        epsilon_firm = firm_returns[:, i] / firm_res.conditional_volatility
        # Forecasted volatility
        firm_vol_hat = np.sqrt(
            np.squeeze(
                firm_res.forecast(
                    firm_res.params, horizon=h, reindex=False
                ).variance.T.to_numpy()
            )
        )  # (h,1) array of volatility forecasts

        # Estimate DCC for each firm-market pair
        epsilon = np.array([epsilon_firm, epsilon_mkt])
        a, b = dcc(epsilon)  # params in DCC model
        Q_avg = calc_Q_avg(epsilon)
        Q = calc_Q(epsilon, a, b)  # Qt for training data
        R = calc_R(epsilon, a, b)  # Rt for training data

        # DCC predictions for correlations
        et = epsilon[:, -1]
        Q_Tplus1 = (1 - a - b) * Q_avg + a * np.outer(et, et) + b * Q[-1]

        diag_q = 1.0 / np.sqrt(np.abs(Q_Tplus1))
        diag_q = diag_q * np.eye(2)
        R_Tplus1 = np.dot(np.dot(diag_q, Q_Tplus1), diag_q)

        diag_q = 1.0 / np.sqrt(np.abs(Q_avg))
        diag_q = diag_q * np.eye(2)
        R_avg = np.dot(np.dot(diag_q, Q_avg), diag_q)

        Rhat = []
        for _h in range(1, h + 1):
            _ab = np.power(a + b, _h - 1)
            # R_Tplus_h is correlation matrix for T+h, symmetric 2*2 matrix
            R_Tplus_h = (1 - _ab) * R_avg + _ab * R_Tplus1
            # In case predicted correlation is larger than 1
            if abs(R_Tplus_h[0, 1]) >= 1:
                R_Tplus_h[0, 1] = 0.9999 * (1 if R_Tplus_h[0, 1] >= 0 else -1)
                R_Tplus_h[1, 0] = R_Tplus_h[0, 1]
            Rhat.append(R_Tplus_h)

        # Sample innovations
        innov = np.zeros((n_days,))
        for t in range(n_days):
            rho = R[t][0, 1]
            innov[t] = (epsilon_firm[t] - epsilon_mkt[t] * rho) / np.sqrt(1 - rho**2)
        sample = np.array([epsilon_mkt, innov])  # shape=(2,n_days)

        # Sample S*h pairs of standardized innovations
        rng = np.random.RandomState(random_seed)
        # sample.shape=(S,h,2)
        sample = sample.T[rng.choice(sample.shape[1], (S, h), replace=True)]

        # list of simulated firm total returns when there're systemic events
        firm_total_return = []
        for s in range(S):
            # Each simulation
            inv = sample[s, :, :]  # (h,2)
            # mkt log return = mkt innovation * predicted mkt volatility
            mktrets = np.multiply(inv[:, 0], mkt_vol_hat)

            # systemic event if over the prediction horizon,
            # the market falls by more than C
            systemic_event = np.exp(np.sum(mktrets)) - 1 < C
            # no need to simulate firm returns if there is no systemic event
            if not systemic_event:
                continue
            # when there is a systemic event
            firmrets = np.zeros((h,))
            for _h in range(h):
                mktinv, firminv = inv[_h][0], inv[_h][1]
                # Simulated firm return at T+h
                rho = Rhat[_h][0, 1]
                firmret = firminv * np.sqrt(1 - rho**2) + rho * mktinv
                firmret = firm_vol_hat[_h] * firmret
                firmrets[_h] = firmret
            # firm return over the horizon
            firmret = np.exp(np.sum(firmrets)) - 1
            firm_total_return.append(firmret)

        # Store result
        if len(firm_total_return):
            lrmes[i] = np.mean(firm_total_return)
        else:
            lrmes[i] = np.nan

    return lrmes

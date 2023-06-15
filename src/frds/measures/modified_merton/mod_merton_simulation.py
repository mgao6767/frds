import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import griddata
from scipy.stats import norm

from frds.measures.option_price import blsprice
from frds.measures.modified_merton.mod_merton_computation import mod_merton_computation
from frds.measures.modified_merton.mod_merton_create_lookup import (
    mod_merton_create_lookup,
)


def simulate():
    # fmt: off
    fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
    fs = fs.reshape(-1, 1)

    N = 10        # number of loan cohorts
    Nsim2 = 1000 # number of simulated factor realization paths (10,000 works well)

    r = 0.01      # risk-free rate
    d = 0.005     # depreciation rate of borrower assets
    y = 0.002     # bank payout rate as a percent of face value of loan portfolio
    T = 10        # loan maturity
    bookD = 0.63
    H = 5         # bank debt maturity
    D = bookD * np.exp(r * H) * np.mean(np.exp(r * np.arange(1, T+1)))   # face value of bank debt: as if bookD issued at start of each cohort, so it grows just like RN asset value
    rho = 0.5
    sig = 0.2
    ltv = 0.66  # initial ltv
    g = 0.5     # prob of govt bailout (only used for bailout valuation analysis)

    j = 3
    bookF = 0.6+j*0.02 # initial loan amount = book value of assets (w/ coupon-bearing loans the initial amount would be book value) 


    param = [r, T, bookF,  H, D, rho, ltv, sig, d, y, g]

    result = mod_merton_computation(fs, param, N, Nsim2)

    Lt, Bt, Et, LH, BH, EH, sigEt, mFt, default, mdef, face, FH, Gt, mu, F, sigLt = result

    # fmt: on

    _tmp = np.abs(mFt - np.exp(r * T / 2))
    j0 = np.argmin(_tmp)
    # Or in Matlab: j0 = find(abs(mFt-1) < 0.01);
    M = _tmp[j0]
    mface = np.max(face[:, j0]) * np.exp(-r * ((T - H) / 2))  # approx.

    K = sigEt.shape[0]
    mertdd = np.zeros(K)
    mertdef = np.zeros(K)
    mertGG = np.zeros(K)
    mertsp = np.zeros(K)
    mertA = np.zeros(K)
    merts = np.zeros(K)
    xmertdd = np.zeros(K)
    xmertdef = np.zeros(K)
    mdefsingle1 = np.zeros(K)
    mFtsingle1 = np.zeros(K)
    Ltsingle1 = np.zeros(K)
    Ltsingle2 = np.zeros(K)
    bookFsingle2 = np.zeros(K)
    mdefsingle2 = np.zeros(K)
    mFtsingle2 = np.zeros(K)
    bookFsingle3 = np.zeros(K)
    mdefsingle3 = np.zeros(K)
    mFtsingle3 = np.zeros(K)
    Ltsingle3 = np.zeros(K)

    Et1 = []
    sigEt1 = []
    fs1 = []
    bookF1 = []

    # Merton model fit to equity value and volatility from our model
    for k in range(K):
        initb = (Et[k] + D * np.exp(-r * H), sigEt[k] / 2)

        def MertonSolution(b):
            A, s = b
            return [
                (np.log(A) - np.log(D) + (r - y - s**2 / 2) * H) / (s * np.sqrt(H)),
                norm.cdf(-mertdd[k]) - mertdef[k],
            ]

        bout = fsolve(MertonSolution, initb)

        A = bout[0]
        s = max(bout[1], 0.0001)

        mertdd[k] = (np.log(A) - np.log(D) + (r - y - s**2 / 2) * H) / (
            s * np.sqrt(H)
        )
        mertdef[k] = norm.cdf(-mertdd[k])
        C, P = blsprice(A, D, r, H, s, y)
        mertGG[k] = g * P
        mertsp[k] = (1 / H) * np.log((D * np.exp(-r * H)) / ((D * np.exp(-r * H)) - P))
        mertA[k] = A
        merts[k] = s

    mktCR = Et / (Et + Bt)
    sp = (1 / H) * np.log((D * np.exp(-r * H)) / Bt)

    # Alternative Model (1):
    # Perfectly correlated borrowers with overlapping cohorts
    rho = 0.99  # approx to rho = 1, code can't handle rho = 1
    sig = 0.20 * np.sqrt(0.5)
    sfs = np.arange(-2.6, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))

    # we have D = bookD*exp(r*H)*mean(exp(r*([1:T])')) here
    # and D = bookD*exp(xr(1,j,k,q)*H); in ModMertonCreateLookup,
    # so adjust bookD1 here
    bookD1 = bookD * np.mean(np.exp(r * np.arange(1, T + 1)))

    # FIXME: problem is here, xfs,etc.dims don't match Matlab ndgrid's output
    xfs, xsig, xr, xF = np.meshgrid(sfs, sig, r, np.arange(0.4, 1.45, 0.01))

    xLt, xBt, xEt, xFt, xmdef, xsigEt = mod_merton_create_lookup(
        d, y, T, H, bookD1, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
    )
    xxsigEt = xsigEt.squeeze()
    xxEt = xEt.squeeze()
    xxmdef = xmdef.squeeze()
    xxFt = xFt.squeeze()

    ymdef = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxmdef.ravel(), (Et, sigEt), method="linear"
    )
    yFt = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxFt.ravel(), (Et, sigEt), method="linear"
    )

    mdefsingle1 = ymdef
    mFtsingle1 = yFt


if __name__ == "__main__":
    import time

    start_time = time.time()

    simulate()

    print(time.time() - start_time, "seconds")

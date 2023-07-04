import warnings  # fsolve may yield runtime warnings about invalid values
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import griddata
from scipy.stats import norm

import matplotlib.pyplot as plt

from frds.measures.option_price import blsprice
from frds.measures.modified_merton.merton_solution import merton_solution
from frds.measures.modified_merton.mod_merton_computation import mod_merton_computation
from frds.measures.modified_merton.mod_merton_create_lookup import (
    mod_merton_create_lookup,
)
from frds.measures.modified_merton.mod_single_merton_create_lookup import (
    mod_single_merton_create_lookup,
)


def simulate(Nsim=10000, direc=None):
    """
    This function is translated from Matlab code `ModMertonSimulation.m` in Nagel and Purnanandam (2020).

    It produces the Table 1 Summary of simulation results and Figures 2 to 5.

    Args:
        Nsim (int): number of simulated factor realization paths. Defaults to 10,000 (about 1hr).
        direc (str): path to store the generated figures. Defaults to current working directory.

    Examples:
        >>> from frds.measures.modified_merton import mod_merton_simulation
        >>> mod_merton_simulation.simulate()
        ----------------------------------------------------------------------------
                                                Borrower asset value
                                            ----------------------------------------
                                            No shock        +ve shock       -ve shock
        ----------------------------------------------------------------------------
        A. True properties
        Agg. Borrower Asset Value           1.06            1.33            0.85
        Bank Asset Value                    0.74            0.79            0.67
        Bank Market Equity/Market Assets    0.12            0.16            0.07
        Bank 5Y RN Default Prob.            0.23            0.11            0.49
        Bank Credit Spread (%)              0.50            0.19            1.37
        ----------------------------------------------------------------------------
        B. Misspecified estimates based on standard Merton model
        Merton 5Y RN Default Prob.          0.13            0.01            0.58
        Merton Credit Spread (%)            0.12            0.00            1.54
        ----------------------------------------------------------------------------

        These results are largely the same as Table 1 in Nagel and Purnanandam (2020).

        Additionally, several plots will be saved in the working directory.

    """
    if direc is None:
        direc = os.getcwd()

    # fmt: off
    fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
    fs = fs.reshape(-1, 1)

    N = 10        # number of loan cohorts
    Nsim2 = Nsim  # number of simulated factor realization paths (10,000 works well)

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

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Merton model fit to equity value and volatility from our model
    for k in range(K):
        initb = (Et[k] + D * np.exp(-r * H), sigEt[k] / 2)

        bout = fsolve(lambda b: merton_solution(b, Et[k], D, r, y, H, sigEt[k]), initb)

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

    warnings.filterwarnings("default")

    mktCR = Et / (Et + Bt)
    sp = (1 / H) * np.log((D * np.exp(-r * H)) / Bt)

    ###########################################################################
    # Alternative Model (1):
    ###########################################################################
    # Perfectly correlated borrowers with overlapping cohorts
    rho = 0.99  # approx to rho = 1, code can't handle rho = 1
    sig = 0.20 * np.sqrt(0.5)
    sfs = np.arange(-2.6, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))

    # we have D = bookD*exp(r*H)*mean(exp(r*([1:T])')) here
    # and D = bookD*exp(xr(1,j,k,q)*H); in ModMertonCreateLookup,
    # so adjust bookD1 here
    bookD1 = bookD * np.mean(np.exp(r * np.arange(1, T + 1)))

    xfs, xsig, xr, xF = np.meshgrid(
        sfs, sig, r, np.arange(0.4, 1.45, 0.01), indexing="ij"
    )

    xLt, xBt, xEt, xFt, xmdef, xsigEt = mod_merton_create_lookup(
        d, y, T, H, bookD1, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
    )
    xxsigEt = xsigEt.squeeze()
    xxEt = xEt.squeeze()
    xxmdef = xmdef.squeeze()
    xxFt = xFt.squeeze()

    ymdef = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxmdef.ravel(), (Et, sigEt), method="cubic"
    )
    yFt = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxFt.ravel(), (Et, sigEt), method="cubic"
    )

    mdefsingle1 = ymdef.copy()
    mFtsingle1 = yFt.copy()
    # recomputing Et, sigEt based on fs1, bookF1 gets back Et, sigEt to
    # a very good approximation

    ###########################################################################
    # Model (2) Single cohort of borrowers
    ###########################################################################
    T = 5
    rho = 0.5
    sig = 0.2

    xfs, xsig, xr, xF = np.meshgrid(
        sfs, sig, r, np.arange(0.4, 1.45, 0.01), indexing="ij"
    )

    xLt, xBt, xEt, xFt, xmdef, xsigEt = mod_single_merton_create_lookup(
        d, y, T, H, bookD1, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
    )

    xxsigEt = xsigEt.squeeze()
    xxEt = xEt.squeeze()
    xxmdef = xmdef.squeeze()
    xxFt = xFt.squeeze()

    ymdef = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxmdef.ravel(), (Et, sigEt), method="cubic"
    )
    yFt = griddata(
        (xxEt.ravel(), xxsigEt.ravel()), xxFt.ravel(), (Et, sigEt), method="cubic"
    )

    mdefsingle2 = ymdef.copy()
    mFtsingle2 = yFt.copy()

    ###########################################################################
    # Model (3) Single (or perfectly correlated) borrower model
    ###########################################################################
    T = 5
    rho = 0.99
    sig = 0.2 * np.sqrt(0.5)

    xfs, xsig, xr, xF = np.meshgrid(
        sfs, sig, r, np.arange(0.4, 1.45, 0.01), indexing="ij"
    )

    xLt, xBt, xEt, xFt, xmdef, xsigEt = mod_single_merton_create_lookup(
        d, y, T, H, bookD1, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
    )

    xxsigEt = np.squeeze(xsigEt)
    xxEt = np.squeeze(xEt)
    xxmdef = np.squeeze(xmdef)
    xxFt = np.squeeze(xFt)

    points = np.column_stack((xxEt.ravel(), xxsigEt.ravel()))
    values_mdef = xxmdef.ravel()
    values_Ft = xxFt.ravel()

    ymdef = griddata(points, values_mdef, (Et, sigEt), method="cubic")
    yFt = griddata(points, values_Ft, (Et, sigEt), method="cubic")

    mdefsingle3 = ymdef.copy()
    mFtsingle3 = yFt.copy()

    ###########################################################################
    # Model (4) Merton model with asset value and asset volatility implied
    # from modified model
    ###########################################################################

    for k in range(K):
        A = Lt[k]
        s = sigLt[k]
        xmertdd[k] = (np.log(A) - np.log(D) + (r - y - s**2 / 2) * H) / (
            s * np.sqrt(H)
        )
        xmertdef[k] = norm.cdf(-xmertdd[k])

    ###########################################################################
    # Some comparisons and figures for main part of the paper
    ###########################################################################
    jup = j0 + 8
    jdn = j0 - 8
    # print("shock")
    # print([fs[j0], fs[jup]])
    # print("Agg Borrower Asset Value")
    # print([mFt[j0], mFt[jup]])
    # print("Market equity ratio")
    # print([mktCR[j0], mktCR[jup]])
    # print("Mod RN def prob")
    # print([mdef[j0], mdef[jup]])
    # print("Merton RN def prob")
    # print([mertdef[j0], mertdef[jup]])
    # print("Mod Credit spread")
    # print([sp[j0], sp[jup]])
    # print("Merton Credit spread")
    # print([mertsp[j0], mertsp[jup]])

    # Table 1. Summary of simulation results
    print(
        f"""
    ----------------------------------------------------------------------------
                                            Borrower asset value
                                        ----------------------------------------
                                        No shock\t+ve shock\t-ve shock
    ----------------------------------------------------------------------------
    A. True properties
    Agg. Borrower Asset Value           {mFt[j0]:.2f}\t\t{mFt[jup]:.2f}\t\t{mFt[jdn]:.2f}
    Bank Asset Value                    {Lt[j0]:.2f}\t\t{Lt[jup]:.2f}\t\t{Lt[jdn]:.2f}
    Bank Market Equity/Market Assets    {mktCR[j0]:.2f}\t\t{mktCR[jup]:.2f}\t\t{mktCR[jdn]:.2f}
    Bank 5Y RN Default Prob.            {mdef[j0]:.2f}\t\t{mdef[jup]:.2f}\t\t{mdef[jdn]:.2f}
    Bank Credit Spread (%)              {sp[j0]*100:.2f}\t\t{sp[jup]*100:.2f}\t\t{sp[jdn]*100:.2f}
    ----------------------------------------------------------------------------
    B. Misspecified estimates based on standard Merton model
    Merton 5Y RN Default Prob.          {mertdef[j0]:.2f}\t\t{mertdef[jup]:.2f}\t\t{mertdef[jdn]:.2f}
    Merton Credit Spread (%)            {mertsp[j0]*100:.2f}\t\t{mertsp[jup]*100:.2f}\t\t{mertsp[jdn]*100:.2f}
    ----------------------------------------------------------------------------
    """
    )

    # Figure 2
    f = plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(FH[:, j0], LH[:, j0] + face[:, j0] * (np.exp(y * H) - 1), s=5)
    plt.plot([0, mface], [0, mface], linewidth=2, color="r", linestyle="--")
    plt.plot([mface, 2.5], [mface, mface], linewidth=2, color="r", linestyle="--")
    plt.xlim(0, 2.5)
    plt.ylim(0, 1.2)
    plt.title("(a) Bank asset value")
    plt.ylabel("Bank asset value", fontsize=12)

    plt.subplot(3, 1, 2)
    plt.scatter(FH[:, j0], EH[:, j0], s=6)
    plt.plot([0, D], [0, 0], linewidth=2, color="r", linestyle="--")
    plt.plot([D, mface], [0, mface - D], linewidth=2, color="r", linestyle="--")
    plt.plot(
        [mface, 2.5], [mface - D, mface - D], linewidth=2, color="r", linestyle="--"
    )
    plt.xlim(0, 2.5)
    plt.title("(b) Bank equity value")
    plt.ylabel("Bank equity value", fontsize=12)

    plt.subplot(3, 1, 3)
    plt.scatter(FH[:, j0], BH[:, j0], s=6)
    plt.plot([0, D], [0, D], linewidth=2, color="r", linestyle="--")
    plt.plot([D, 2.5], [D, D], linewidth=2, color="r", linestyle="--")
    plt.xlim(0, 2.5)
    plt.title("(c) Bank debt value")
    plt.ylabel("Bank debt value", fontsize=12)
    plt.xlabel("Aggregate borrower asset value", fontsize=12)

    plt.gcf().set_size_inches(6, 10)
    plt.savefig(os.path.join(direc, "figure2_PayoffsAtDMat.pdf"), format="pdf")

    # Plotting mFt vs. mktCR
    plt.figure()
    plt.scatter(mFt, mktCR)

    # Figure 3 Panel (a)
    plt.figure()
    plt.scatter(mFt, Et)
    plt.ylabel("Bank asset value", fontsize=12)
    plt.xlabel("Aggregate borrower asset value", fontsize=12)
    # Save the figure as a PDF
    plt.savefig(os.path.join(direc, "figure3_mVe.pdf"), bbox_inches="tight")

    # Figure 4
    f = plt.figure()
    plt.scatter(mFt, mdef, c="b", edgecolors="b")
    plt.scatter(mFt, mertdef, c="r", marker="+")
    plt.legend(["Actual", "Merton Model"])
    plt.ylabel("RN bank default probability", fontsize=12)
    plt.xlabel("Aggregate borrower asset value", fontsize=12)
    plt.savefig(os.path.join(direc, "figure4_mdef.pdf"), bbox_inches="tight")

    # Figure 5 Panel (a)
    plt.figure()
    plt.scatter(mFt, mdef, marker="o", c="b")
    plt.plot(mFt, mdefsingle2, "--k")
    plt.plot(mFt, mdefsingle3, "--r")
    plt.scatter(mFt, mertdef, marker="+", c="r")
    plt.legend(
        ["Actual", "Single cohort", "Single borrower", "Merton Model"], fontsize=12
    )
    plt.ylabel("RN bank default probability", fontsize=12)
    plt.xlabel("Aggregate borrower asset value", fontsize=12)
    plt.savefig(
        os.path.join(direc, "figure5_panel_a_mdefsingles.pdf"), bbox_inches="tight"
    )

    # Figure 5 Panel (b)
    plt.figure()
    plt.scatter(mFt, mdef, marker="o", c="b")
    plt.plot(mFt, xmertdef, "--k")
    plt.scatter(mFt, mertdef, marker="+", c="r")
    plt.legend(
        ["Actual", "Merton w/ actual asset value and volatility", "Merton Model"],
        fontsize=12,
    )
    plt.ylabel("RN bank default probability", fontsize=12)
    plt.xlabel("Aggregate borrower asset value", fontsize=12)
    plt.savefig(os.path.join(direc, "figure5_panel_b_mertalt.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    import time

    start_time = time.time()

    simulate()

    print(time.time() - start_time, "seconds")

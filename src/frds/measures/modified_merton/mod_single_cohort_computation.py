import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from frds.measures.modified_merton.fftsmooth import fftsmooth
from frds.measures.modified_merton.loan_payoff import loan_payoff
from frds.measures.modified_merton.find_face_value_indiv import find_face_value_indiv


# TODO: complete and test this function
def mod_single_cohort_computation(fs, param, N, Nsim2, w=None, random_seed=1):
    # fmt: off
    r = param[0]  # log risk-free rate
    T = param[1]  # original maturity of bank loans
    bookF = param[2]  # cash amount of loan issued = book value for a coupon-bearing loan issued at par
    H = param[3]  # bank debt maturity
    D = param[4]  # face value of bank debt
    rho = param[5]  # borrower asset value correlation
    ltv = param[6]  # initial LTV
    sig = param[7]  # borrower asset value volatility
    d = param[8]  # depreciation rate of borrower assets
    y = param[9]  # bank payout rate

    # With single borrower cohort, N just determines discretization horizon of
    # shocks, not the number of cohorts 

    # optional: calculate value of govt guarantee, with prob g, govt absorbs
    # Loss given default (bank and equity values not adjusted for presence of
    # govt guarantee) 
    if len(param) > 10:
        g = param[10]  # value of government guarantee
    else:
        g = 0

    # optional: provide simulated factor shocks (for faster repeated computations) 
    # if not provided as input, then generate here 
    if w is None:
        rng = np.random.RandomState(random_seed)
        # w = rng.normal(0, 1, (Nsim2, N))
        w = norm.ppf(rng.rand(N, Nsim2).T, 0, 1)

    # initial log asset value of borrower at origination
    ival = np.log(bookF) - np.log(ltv)
    sigf = np.sqrt(rho) * sig
    # Mingze's note: `HN` must be an integer because it's used in array indexing
    # This problem also presents in Nagel's original Matlab code
    HN = H * (N / T) # maturity in N time
    if isinstance(HN, float):
        assert HN.is_integer()
        HN = int(HN)
    szfs = fs.shape[0]
    fs = np.concatenate((fs, fs, fs), axis=0)
    Nsim1 = fs.shape[0]

    # Euler discretization 
    f = np.concatenate(
        (np.zeros((Nsim2, 1)), 
         np.cumsum((r - d - 0.5 * sig ** 2) * (T / N) + sigf * np.sqrt(T / N) * w, axis=1)),
        axis=1,
    ) 
    # from start of first loan to maturity of first loan w/ shocks until t remove
    f1 = f[:, N]

    # add fs shocks after loan origination
    fsa = (fs * sigf * np.sqrt(T)).reshape((1,Nsim1))
    dstep = 10
    df = sigf / dstep
    fsa = fsa + df * np.concatenate(
        (np.zeros((Nsim2, szfs)), np.ones((Nsim2, szfs)), -np.ones((Nsim2, szfs))),
        axis=1,
    )
    f1j = np.tile(f1.reshape(Nsim2, 1), (1, Nsim1)) + fsa

    initmu = r + 0.01
    muout = fsolve(lambda mu: find_face_value_indiv(mu, bookF * np.exp(mu * T), ival, sig, T, r, d), initmu)
    mu = muout[0]
    F = bookF * np.exp(mu * T)

    L1 = loan_payoff(F, f1j, ival, rho, sig, T)
    face1 = np.ones_like(f1j) * F

    ft = fsa + ival
    fH1 = f[:, HN].copy()
    fH1j = np.tile(fH1.reshape(Nsim2, 1), (1, Nsim1)) + fsa + ival
    FH1j = np.exp(fH1j) * np.exp(0.5 * (1 - rho) * H * sig ** 2)
    Ft = np.exp(ft)

    FHr1 = FH1j.copy()
    Lr1 = L1.copy()

    LHj = np.zeros((Nsim2, Nsim1))
    FHr = FHr1.flatten(order='F')
    Lr = Lr1.flatten(order='F')
    sort_indices = np.argsort(np.round(FHr, 9), kind="stable")
    sortF = FHr[sort_indices]
    sortL = Lr[sort_indices]
    win = (Nsim2 * Nsim1) // 20
    LHs = fftsmooth(sortL, win)
    new_indices = np.zeros_like(sort_indices)
    new_indices[sort_indices] = np.arange(len(FHr))
    LH1j = LHs[new_indices].reshape(Nsim2, Nsim1, order="F")

    LH = LH1j * np.exp(-r * (T - H))
    FH = FHr1
    face = face1 * np.exp(-r * (T - H))

    BH = np.minimum(D, LH * np.exp(-y * H))
    EHex = LH * np.exp(-y * H) - BH
    EH = LH - BH
    GH = g * np.maximum(D - LH * np.exp(-y * H), 0)

    Lt = np.mean(LH, axis=0) * np.exp(-r * H)
    Bt = np.mean(BH, axis=0) * np.exp(-r * H)
    Et = np.mean(EH, axis=0) * np.exp(-r * H)
    Gt = np.mean(GH, axis=0) * np.exp(-r * H)
    mFt = np.mean(Ft, axis=0)
    def_val = np.ones_like(EHex)
    ldef = EHex > 0
    def_val[ldef] = 0
    mdef = np.mean(def_val, axis=0)
    sigEt = (dstep / 2) * (np.log(Et[szfs:2 * szfs]) - np.log(Et[2 * szfs:3 * szfs]))
    sigLt = (dstep / 2) * (np.log(Lt[szfs:2 * szfs]) - np.log(Lt[2 * szfs:3 * szfs]))

    Lt = Lt[:szfs]
    Bt = Bt[:szfs]
    Et = Et[:szfs]
    LH = LH[:, :szfs]
    BH = BH[:, :szfs]
    EH = EH[:, :szfs]
    FH = FH[:, :szfs]
    mFt = mFt[:szfs]
    def_val = def_val[:, :szfs]
    mdef = mdef[:szfs]
    face = face[:, :szfs]
    Gt = Gt[:szfs]

    return Lt, Bt, Et, LH, BH, EH, sigEt, mFt, def_val, mdef, face, FH, Gt, mu, F, sigLt

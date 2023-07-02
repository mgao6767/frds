import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from frds.measures.modified_merton.fftsmooth import fftsmooth
from frds.measures.modified_merton.loan_payoff import loan_payoff
from frds.measures.modified_merton.find_face_value_indiv import find_face_value_indiv


def mod_merton_computation(fs, param, N, Nsim2, w=None, random_seed=1):
    """Compute modified Merton model

    This function is translated from Nagel's Matlab code `ModMertonComputation.m`
    """
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

    # Optional: calculate the value of government guarantee, with prob g, the government absorbs
    # Loss given default (bank and equity values not adjusted for the presence of the government guarantee)
    if len(param) > 10:
        g = param[10]
    else:
        g = 0

    # Optional: provide simulated factor shocks (for faster repeated computations)
    # If not provided as input, generate them here
    if w is None:
        rng = np.random.RandomState(random_seed)
        # w = rng.normal(0, 1, (Nsim2, 3 * N))
        w = norm.ppf(rng.rand(3*N, Nsim2).T, 0, 1)

    # initial log asset value of borrower at origination
    ival = np.log(bookF) - np.log(ltv)
    sigf = np.sqrt(rho) * sig
    # Mingze's note: `HN` must be an integer because it's used in array indexing
    # This problem also presents in Nagel's original Matlab code
    HN = H * (N / T)  # maturity in N time
    if isinstance(HN, float):
        assert HN.is_integer()
        HN = int(HN)
    szfs = fs.shape[0]
    # use second and third blocks for numerical derivative
    fs = np.concatenate((fs, fs, fs), axis=0)
    Nsim1 = fs.shape[0]

    # Remaining maturity of first loans at t
    rmat = np.arange(N)
    # Mingze's note: this line below is to replicate Matlab's
    # `rmat = repmat(rmat, [Nsim2,1, Nsim1]);`
    # transforming (N,) array `rmat` to  (Nsim2, N, Nsim1) array with correct
    # replications of elements. It is equivalent to:
    # `rmat = np.dstack((np.matlib.repmat(rmat, Nsim2, 1),) * Nsim1)`
    # To test this,
    # import numpy as np
    # import numpy.matlib
    # from numpy.testing import assert_equal
    # N = 10
    # Nsim2 = 100
    # Nsim1 = 99
    # rmat = np.arange(N)
    # rmat1 = np.tile(rmat.reshape(N, 1), (Nsim2, 1, Nsim1))
    # rmat2 = np.dstack((np.matlib.repmat(rmat, Nsim2, 1),) * Nsim1)
    # assert_equal(rmat1, rmat2)
    rmat = np.tile(rmat.reshape(N, 1), (Nsim2, 1, Nsim1))
    ind1 = rmat >= HN
    ind2 = rmat < HN

    # Fractional accumulated loan life time at t+H
    aHmat = np.concatenate((np.arange(HN, -1, -1) / N, np.arange(N - 1, HN, -1) / N))
    aHmat = np.tile(aHmat.reshape(N, 1), (Nsim2, 1, Nsim1))

    # Fractional accumulated loan life time at t
    atmat = (N - np.arange(N)) / N
    atmat = np.tile(atmat.reshape(N, 1), (Nsim2, 1, Nsim1))

    # Euler discretization for log value has a Jensen's term
    # which depends on total, not just factor volatility,
    # since f here is the average log asset value of borrowers, the idiosyncratic
    # shock averaged out, but to have the expected return (gross of depreciation)
    # for each individual borrower equal to exp(r), the drift needs to adjust
    # for total volatility.
    # Further below:
    # To get the average level of asset value, we take E[exp(f + 0.5*idiovar)]
    # because only volatility from factors generates variation in our
    # simulations (which then raises E[exp(.)] by averaging convex function,
    # this 0.5*idiovar part combined with the total volatility drift
    # adjustment here then only leaves the factor volatility drift adjustment
    f = np.concatenate(
        (
            np.zeros((Nsim2, 1)),
            np.cumsum((r - d - 0.5 * sig**2) * (T / N) + sigf * np.sqrt(T / N) * w, axis=1),
        ),
        axis=1,
    )

    # fw is what we need to remove from f (until t) to let the LEVEL of aggregate
    # borrower asset value grow at the expected path exp(r-d)
    # (after we also add 0.5*idiovar further below)
    fw = np.concatenate(
        (
            np.zeros((Nsim2, 1)),
            np.cumsum(-0.5 * rho * sig**2 * (T / N) + sigf * np.sqrt(T / N) * w, axis=1),
        ),
        axis=1,
    )

    # Factor realizations at relevant points for staggered loan cohorts (#sim x cohort)
    # t = Time we do valuation
    # First loan of the first cohort starts at 0 and matures at t = N, i.e., accumulated factor shocks
    # are those in 1, 2, ..., N. First loan of the N-th cohort starts at N-1 and
    # matures at 2*N-1
    # Second loan of the first cohort starts at N and accumulated factor shocks are those
    # in N+1, ..., 2*N. Second loan of the N-th cohort starts at 2*N-1 and matures
    # at 3*N-1
    # Note that the first element of f is zero, so f(1) is time 0 and f(N+1) is time t
    # Mingze's note: f(1) in Matlab is f[0] in Python

    # From start of the first loan to the maturity of the first loan, first cohort only
    # Mingze's note: `xf1` unused, same in original Matlab code
    # xf1 = f[:, N] - f[:, 0]
    # Factor shocks from the start of the first loan shocks until t
    f0w = np.tile(fw[:, N].reshape(fw[:, N].shape[0], 1), (1, N)) - fw[:, 0:N]
    # From the start of the first loan to the maturity of the first loan w/ shocks until t removed
    f1 = f[:, N : 2 * N] - f0w - f[:, 0:N]
    # From the start of the second loan to the maturity of the second loan
    f2 = f[:, 2 * N : 3 * N] - f[:, N : 2 * N]

    # Add fs shocks after loan origination
    # Dimension of f1tj, f2tj becomes (#sim x cohort x #fs-values)
    # Do not apply maturity weighting to df increments for numerical
    # derivatives used for volatility calculation
    fsa = (fs[np.newaxis, np.newaxis, :] * sigf * np.sqrt(T)).reshape((1, 1, Nsim1))
    dstep = 10  # 1/dstep of SD step to evaluate numerical derivative
    df = sigf / dstep
    fsa = np.repeat(fsa, N, axis=1) * atmat + df * np.concatenate(
        (
            np.zeros((Nsim2, N, szfs)),
            np.ones((Nsim2, N, szfs)),
            -np.ones((Nsim2, N, szfs)),
        ),
        axis=2,
    )
    # Mingze's note: translated from `f1j = repmat(f1,[1,1,Nsim1]) + fsa;`
    f1j = np.dstack((f1,) * Nsim1) + fsa
    # fs shock not here because they occurred before the origination of the second loan
    f2j = np.dstack((f2,) * Nsim1)

    # Solve for the promised yield on loan (a fixed point)
    # Do not remove factor shocks until t in this calculation
    initmu = r + 0.01
    muout = fsolve(
        lambda mu: find_face_value_indiv(mu, bookF * np.exp(mu * T), ival, sig, T, r, d),
        initmu,
    )
    mu = muout[0]  # Output is promised total yield
    F = bookF * np.exp(mu * T)  # Loan face value

    # Payoffs at loan maturities
    # T is the correct "maturity" below because idiosyncratic risk accumulated
    # since issuance of the loan
    # Loan portfolio payoff at the first maturity date (#sim  x cohort x fs shocks)
    # Newmu coming out here is not exactly the same as mu because f1j doesn't
    # have shocks until time t
    L1 = loan_payoff(F, f1j, ival, rho, sig, T)
    face1 = np.ones_like(f1j) * F

    # For the second generation of loans, use ival2 which accounts for collateral
    # adjustment at loan roll-over: same LTV as first-generation loans, hence
    # same risk, same promised yield
    # Here newmu comes out identical to mu initially calculated above
    ival2 = np.log(L1 / ltv)
    F2 = L1 * np.exp(mu * T)
    L2 = loan_payoff(F2, f2j, ival2, rho, sig, T)
    face2 = F2

    face2[ind1] = 0
    face1[ind2] = 0

    # Factor realizations at t and t+H
    # First by loan cohort, then average across cohorts
    # The relevant number is the asset value of the loan portfolio in the whole portfolio,
    # hence the Jensen's inequality term for the idio variance part

    # From start of first loan to time t  
    # CHECK: no need for ival here because it's constant, makes no diff in conditioning below
    # including shock fs since origination of loan
    # Mingze's note: the three lines below are translated from
    # `ft = repmat( repmat(f(:,N+1),1,N) - f0w - f(:,1:N), [1,1,Nsim1]) + fsa + ival;`
    _newshape = tuple((*f[:, N].shape, 1))
    ft = np.tile(f[:, N].reshape(_newshape), (1, N)) - f0w - f[:, 0:N]
    ft = np.dstack((ft,) * Nsim1) + fsa + ival

    # From start of the first loan to to time t+H
    _newshape = tuple((*f[:, N + HN].shape, 1))
    _tmp = np.tile(f[:, N + HN].reshape(_newshape), (1, N))
    fH1 = _tmp - f0w - f[:, 0:N]
    # From start of the second loan to time t+H
    fH2 = _tmp - f[:, N : 2 * N]

    fH1j = np.dstack((fH1,) * Nsim1) + fsa + ival
    fH2j = np.dstack((fH2,) * Nsim1) + ival2  # fs shock here goes into ival2

    # Idio part is uncertain over the whole maturity (past and future) when conditioning on common factor
    FH1j = np.exp(fH1j) * np.exp(0.5 * (1 - rho) * (T * aHmat) * sig**2)
    # But only accumulated dispersion until valuation point matters for Jensen's adj.
    FH2j = np.exp(fH2j) * np.exp(0.5 * (1 - rho) * (T * aHmat) * sig**2)

    # average across cohorts
    Ft = np.mean(np.exp(ft) * np.exp(0.5 * (1 - rho) * (T * atmat) * sig**2), axis=1)

    # Get conditional payoff distribution based on t+H information
    # i.e., conditional on factor path since initial origination
    # including any collateral amounts added or subtracted in the course of
    # roll over into the second generation of loans
    # Use a (fast) smoother to compute this conditional expectation through local averaging
    # Conditional expectation of loan payoff for rolled-over loans is a function of both
    # (a) log factor shocks since roll over plus log collateral replenishment
    # (b) the face value of the rolled-over loan (which depends on factor realizations up to rollover date)
    # However, (b) scales both loan value and face value, so by descaling we can reduce conditional expectation
    # to one that depends only on (a)
    sc = L1 / bookF
    sc[ind1] = 1
    FHr1 = FH1j.copy()
    FHr1[ind2] = 0
    FHr2 = FH2j / sc
    FHr2[ind1] = 0
    Lr1 = L1.copy()
    Lr1[ind2] = 0
    Lr2 = L2 / sc
    Lr2[ind1] = 0

    LHj = np.zeros((Nsim2, N, Nsim1))
    for j in range(N):
        FHr = np.reshape(FHr1[:, j, :] + FHr2[:, j, :], (Nsim2 * Nsim1, 1), order="F")
        Lr = np.reshape(Lr1[:, j, :] + Lr2[:, j, :], (Nsim2 * Nsim1, 1), order="F")
        ind = np.round(FHr.flatten(), 9).argsort(kind="stable")
        sortL = Lr[ind].copy()
        win = int(Nsim2 * Nsim1 / 20)  # / 10 seems to give about sufficient smoothness
        LHs = fftsmooth(sortL.flatten(), win)
        newInd = np.zeros(ind.shape, dtype=np.int64)
        newInd[ind] = np.arange(len(FHr))
        LHsn = np.reshape(LHs[newInd], (Nsim2, Nsim1), order="F")
        LHsn = LHsn * sc[:, j, :]
        LHj[:, j, :] = LHsn.copy()

    # Integrate over cohorts and discount to get portfolio payoff distribution at t+H
    LH1j = LHj.copy()
    LH1j[ind2] = 0
    LH2j = LHj.copy()
    LH2j[ind1] = 0

    LH = np.mean(
        LH1j * np.exp(-r * (rmat - HN) * (T / N)) + LH2j * np.exp(-r * (rmat - HN + N) * (T / N)),
        axis=1,
    )
    FH = np.mean(FHr1 + FHr2, axis=1)

    face = np.mean(
        face1 * np.exp(-r * (rmat - HN) * (T / N)) + face2 * np.exp(-r * (rmat - HN + N) * (T / N)),
        axis=1,
    )

    BH = np.minimum(D, LH * np.exp(-y * H))
    EHex = LH * np.exp(-y * H) - BH
    EH = LH - BH
    GH = g * np.maximum(D - LH * np.exp(-y * H), 0)

    # Now integrate conditional on f_t over factor distribution at t+H 
    # simply taking mean here works because no factor shocks until t 
    # so factor paths are all the same until t 
    _tmp = np.exp(-r *H)
    Lt = np.mean(LH, axis=0) * _tmp
    Bt = np.mean(BH, axis=0) * _tmp
    Et = np.mean(EH, axis=0) * _tmp
    Gt = np.mean(GH, axis=0) * _tmp
    mFt = np.mean(Ft, axis=0)

    # RN default indicator 
    default = np.ones_like(EHex)
    ldef = EHex > 0
    default[ldef] = 0

    mdef = np.mean(default, axis=0)


    # Take numerical first derivative to get instantaneous equity vol
    # consider SD/dstep move of f 
    # this is ok even though future replenishment/removal of f as collateral 
    # here just SD/dstep move in only source of stochastic shocks
    # of interest here is how this feeds through to Et (including through dampening collateral replenishment/removal)
    # (if one looked at an SD move in the dampened asset value, it would have a correspondingly  
    # bigger derivative) 
    sigEt = (dstep / 2) * (np.log(Et[szfs : 2 * szfs]) - np.log(Et[2 * szfs : 3 * szfs]))
    sigLt = (dstep / 2) * (np.log(Lt[szfs : 2 * szfs]) - np.log(Lt[2 * szfs : 3 * szfs]))

    Lt = Lt[:szfs]
    Bt = Bt[:szfs]
    Et = Et[:szfs]
    LH = LH[:, :szfs]
    BH = BH[:, :szfs]
    EH = EH[:, :szfs]
    FH = FH[:, :szfs]
    mFt = mFt[:szfs]
    default = default[:, :szfs]
    mdef = mdef[:szfs]
    face = face[:, :szfs]
    Gt = Gt[:szfs]


    return Lt, Bt, Et, LH, BH, EH, sigEt, mFt, default, mdef, face, FH, Gt, mu, F, sigLt
    # fmt: on


if __name__ == "__main__":
    # This following example is based on the `ModMertonSimulation.m`
    import time

    start_time = time.time()

    # fmt: off
    fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
    fs = fs.reshape(-1, 1)

    N = 10        # number of loan cohorts
    Nsim2 = 10000 # number of simulated factor realization paths (10,000 works well)

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

    print(time.time() - start_time, "seconds")

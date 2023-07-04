import numpy as np
from scipy.stats import norm
from numpy.random import RandomState

from frds.measures.modified_merton.mod_single_cohort_computation import (
    mod_single_cohort_computation,
)


def mod_single_merton_create_lookup(
    d, y, T, H, bookD, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
):
    """
    same as `mod_merton_create_lookup` except that the function called differs
    """
    # fmt: off
    rng = RandomState(1)
    w = norm.ppf(rng.rand(3*N,Nsim2)).T
    
    J = xsig.shape[1]
    K = xr.shape[2]
    Q = xF.shape[3]
    G = xfs.shape[0]
    
    fs = xfs[:, 0, 0, 0]
    
    xLt = np.zeros((G, J, K, Q))
    xBt = np.zeros((G, J, K, Q))
    xEt = np.zeros((G, J, K, Q))
    xFt = np.zeros((G, J, K, Q))
    xmdef = np.zeros((G, J, K, Q))
    xsigEt = np.zeros((G, J, K, Q))
    
    for j in range(J):
        for k in range(K):
            for q in range(Q):
                param = [xr[0, j, k, q], T, xF[0, j, k, q], H, bookD * np.exp(xr[0, j, k, q] * H), rho, ltv, xsig[0, j, k, q], d, y]
                
                Lt, Bt, Et, _, _, _, sigEt, mFt, _, mdef, *_ = mod_single_cohort_computation(fs, param, N, Nsim2, w)

                xLt[:, j, k, q] = Lt
                xBt[:, j, k, q] = Bt
                xEt[:, j, k, q] = Et
                xFt[:, j, k, q] = mFt
                xmdef[:, j, k, q] = mdef
                xsigEt[:, j, k, q] = sigEt

    return xLt, xBt, xEt, xFt, xmdef, xsigEt 

    # fmt: on

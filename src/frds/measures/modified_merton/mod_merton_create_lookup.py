import numpy as np
from numpy.random import RandomState

from frds.measures.modified_merton.mod_merton_computation import mod_merton_computation


def mod_merton_create_lookup(d, y, T, H, bookD, rho, ltv, xfs, xr, xF, xsig, N, Nsim2):
    # fmt: off
    rng = RandomState(1)
    w = rng.standard_normal((Nsim2, 3 * N))
    
    J = xsig.shape[1]
    K = xr.shape[2]
    Q = xF.shape[3]
    G = xfs.shape[0]
    
    fs = xfs[:, 0, 0, 0] # (1,69,1,105), but in Matlab is (69,1,1,105)
    
    xLt = np.zeros((G, J, K, Q))
    xBt = np.zeros((G, J, K, Q))
    xEt = np.zeros((G, J, K, Q))
    xFt = np.zeros((G, J, K, Q))
    xmdef = np.zeros((G, J, K, Q))
    xsigEt = np.zeros((G, J, K, Q))
    
    last_param = []
    for j in range(J):
        for k in range(K):
            for q in range(Q):
                param = [xr[0, j, k, q], T, xF[0, j, k, q], H, bookD * np.exp(xr[0, j, k, q] * H), rho, ltv, xsig[0, j, k, q], d, y]
                
                # FIXME: outputs dimenstions incorret: only 1*1, should be 69*1
                Lt, Bt, Et, _, _, _, sigEt, mFt, _, mdef, *_ = mod_merton_computation(fs, param, N, Nsim2, w)
                if param!=last_param:
                    print(j,k,q, param, Et)
                    last_param = param
                xLt[:, j, k, q] = Lt.copy()
                xBt[:, j, k, q] = Bt.copy()
                xEt[:, j, k, q] = Et.copy() 
                xFt[:, j, k, q] = mFt.copy()
                xmdef[:, j, k, q] = mdef.copy()
                xsigEt[:, j, k, q] = sigEt.copy()

    return xLt, xBt, xEt, xFt, xmdef, xsigEt # FIXME: somehow all Lt, Bt, Et... remain same for all q

    # fmt: on

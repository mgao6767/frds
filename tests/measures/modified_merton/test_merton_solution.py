import unittest
import pathlib
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.optimize import fsolve

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")


from frds.measures.modified_merton.merton_solution import merton_solution
from frds.measures.modified_merton.mod_merton_computation import mod_merton_computation


class MertonSolutionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        mp = pathlib.Path(__file__).parent.joinpath("matlab").as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(mp, nargout=0)

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

        K = sigEt.shape[0]

        self.K = K
        self.Et = Et
        self.sigEt = sigEt
        self.r = r
        self.D = D
        self.H = H
        self.y = y

    def test_merton_solution(self):
        # Check if `merton_solution` in Python provides the same result
        # as `MertonSolution` in Matlab.
        K = self.K
        Et = self.Et
        sigEt = self.sigEt
        r = self.r
        D = self.D
        H = self.H
        y = self.y

        # fmt: off
        for k in range(K):
            # Result from Python code
            initb = [Et[k] + D * np.exp(-r * H), sigEt[k] / 2]
            _a = merton_solution(initb, Et[k], D, r, y, H, sigEt[k])

            # Result from Matlab code
            initb = matlab.double(initb)
            _b = self.eng.MertonSolution(initb, Et[k], D, r, y, float(H), sigEt[k])

            # Up to 9 decimals
            assert_array_almost_equal(np.array(_a), np.array(_b).ravel(), decimal=9)

        # fmt: on

    def test_fsolve_merton_solution(self):
        K = self.K
        Et = self.Et
        sigEt = self.sigEt
        r = self.r
        D = self.D
        H = self.H
        y = self.y

        # fmt: off
        # for k in range(K):
        for k in [2]:
            # Result from Python code
            initb = [Et[k] + D * np.exp(-r * H), sigEt[k] / 2]
            bout_py = fsolve(lambda b: merton_solution(b, Et[k], D, r, y, H, sigEt[k]), initb)

            # Result from Matlab code
            bout_matlab = fsolve(lambda b: 
                                 np.array(self.eng.MertonSolution(matlab.double(b), Et[k], D, r, y, float(H), sigEt[k])).ravel(), 
                                 initb)

            # Up to 9 decimals
            assert_array_almost_equal(np.array(bout_py), np.array(bout_matlab).ravel(), decimal=9)

        # fmt: on

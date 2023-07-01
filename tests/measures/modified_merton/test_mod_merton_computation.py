import unittest
import pathlib
import numpy as np
from numpy.testing import assert_array_almost_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")

from frds.measures.modified_merton.mod_merton_computation import mod_merton_computation


class ModMertonComputationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        mp = pathlib.Path(__file__).parent.joinpath("matlab").as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(mp, nargout=0)

    def test_mod_merton_computation(self):
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

        # Matlab results
        result_mt = self.eng.ModMertonComputation(fs, matlab.double(param, size=[11,1]), matlab.double(N), matlab.double(Nsim2), nargout=16)
        Lt_mt, Bt_mt, Et_mt, LH_mt, BH_mt, EH_mt, sigEt_mt, mFt_mt, default_mt, mdef_mt, face_mt, FH_mt, Gt_mt, mu_mt, F_mt, sigLt_mt = result_mt

        # Python results
        result = mod_merton_computation(fs, param, N, Nsim2)
        Lt, Bt, Et, LH, BH, EH, sigEt, mFt, default, mdef, face, FH, Gt, mu, F, sigLt = result

        n_precision = 9

        # NOTE: This test used to fail as results from Matlab are all zero.
        # After debugging, it turns out the `g` was 0 but should have been param(11).
        # This is because `matlab.double` assumes 11*1 dim for param,
        # but the `ModMertonComputation` assumes 1*11 dim for param.
        # This is fixed by setting `size` in `matlab.double`.
        assert_array_almost_equal(Gt, np.asarray(Gt_mt).ravel(), n_precision) 

        assert_array_almost_equal(Lt, np.asarray(Lt_mt).ravel(), n_precision)
        assert_array_almost_equal(Bt, np.asarray(Bt_mt).ravel(), n_precision)
        assert_array_almost_equal(Et, np.asarray(Et_mt).ravel(), n_precision)
        assert_array_almost_equal(mFt, np.asarray(mFt_mt).ravel(), n_precision)
        assert_array_almost_equal(face, np.asarray(face_mt), n_precision)
        assert_array_almost_equal(FH, np.asarray(FH_mt), n_precision)
        assert_array_almost_equal(mu, np.asarray(mu_mt).ravel(), n_precision)
        assert_array_almost_equal(F, np.asarray(F_mt).ravel(), n_precision)
        assert_array_almost_equal(LH, np.asarray(LH_mt), n_precision)
        assert_array_almost_equal(BH, np.asarray(BH_mt), n_precision)
        assert_array_almost_equal(EH, np.asarray(EH_mt), n_precision)
        assert_array_almost_equal(sigEt, np.asarray(sigEt_mt).ravel(), n_precision)
        assert_array_almost_equal(default, np.asarray(default_mt), n_precision) 
        assert_array_almost_equal(mdef, np.asarray(mdef_mt).ravel(), n_precision)
        assert_array_almost_equal(sigLt, np.asarray(sigLt_mt).ravel(), n_precision) 

        # fmt: on


if __name__ == "__main__":
    unittest.main()

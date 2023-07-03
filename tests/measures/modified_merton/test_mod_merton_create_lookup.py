import unittest
import os
import pathlib
import numpy as np
from scipy.interpolate import griddata
from numpy.testing import assert_array_almost_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")


from frds.measures.modified_merton.mod_merton_create_lookup import (
    mod_merton_create_lookup,
)


class ModMertonCreateLookupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mp = pathlib.Path(__file__).parent.joinpath("matlab").as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.mp, nargout=0)

    def make_matlab_code(self, func_name: str, func_string: str):
        with open(pathlib.Path(self.mp).joinpath(f"{func_name}.m"), "w+") as f:
            f.write(func_string)

    def tearDown(self) -> None:
        for root, dirs, files in os.walk(self.mp):
            for f in files:
                if f.startswith("FRDSTest"):
                    os.remove(os.path.join(root, f))

    def test_mod_merton_create_lookup(self):
        Nsim2 = 1000
        N = 10
        ltv = 0.66
        r = 0.01  # risk-free rate
        d = 0.005  # depreciation rate of borrower assets
        y = 0.002  # bank payout rate as a percent of face value of loan portfolio
        T = 10  # loan maturity
        bookD = 0.63
        H = 5  # bank debt maturity
        rho = 0.99
        sig = 0.20 * np.sqrt(0.5)
        sfs = np.arange(-2.6, 0.8, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
        bookD1 = bookD * np.mean(np.exp(r * np.arange(1, T + 1)))
        g = 0.5  # prob of govt bailout (only used for bailout valuation analysis)

        D = (
            bookD * np.exp(r * H) * np.mean(np.exp(r * np.arange(1, T + 1)))
        )  # face value of bank debt: as if bookD issued at start of each cohort, so it grows just like RN asset value

        j = 3
        bookF = (
            0.6 + j * 0.02
        )  # initial loan amount = book value of assets (w/ coupon-bearing loans the initial amount would be book value)

        fs = np.arange(-0.8, 0.85, 0.05) / (0.2 * np.sqrt(0.5) * np.sqrt(10))
        fs = fs.reshape(-1, 1)

        param = [r, float(T), bookF, float(H), D, rho, ltv, sig, d, y, g]

        # Use Matlab version of modmertoncomputation
        # result = mod_merton_computation(fs, param, N, Nsim2)
        result = self.eng.ModMertonComputation(
            fs,
            matlab.double(param, size=[11, 1]),
            matlab.double(N),
            matlab.double(Nsim2),
            nargout=16,
        )
        # fmt: off
        Lt, Bt, Et, LH, BH, EH, sigEt, mFt, default, mdef, face, FH, Gt, mu, F, sigLt = result
        # fmt: on

        xfs, xsig, xr, xF = np.meshgrid(
            sfs, sig, r, np.arange(0.4, 1.45, 0.01), indexing="ij"
        )

        # fmt: off
        
        xLt, xBt, xEt, xFt, xmdef, xsigEt = mod_merton_create_lookup(
            d, y, T, H, bookD1, rho, ltv, xfs, xr, xF, xsig, N, Nsim2
        )

        xLt_mt, xBt_mt, xEt_mt, xFt_mt, xmdef_mt, xsigEt_mt = self.eng.ModMertonCreateLookup(
            d, y, matlab.double(T), matlab.double(H), bookD1, rho, ltv, xfs, xr, xF, xsig, matlab.double(N), matlab.double(Nsim2),
            nargout=6
        )
        # fmt: on
        n_precision = 9
        assert_array_almost_equal(xLt, np.asarray(xLt_mt), n_precision)
        assert_array_almost_equal(xBt, np.asarray(xBt_mt), n_precision)
        assert_array_almost_equal(xEt, np.asarray(xEt_mt), n_precision)
        assert_array_almost_equal(xFt, np.asarray(xFt_mt), n_precision)
        assert_array_almost_equal(xmdef, np.asarray(xmdef_mt), n_precision)
        assert_array_almost_equal(xsigEt, np.asarray(xsigEt_mt), n_precision)

        # Additional tests for replicating scatteredInterpolant in Matlab
        self.make_matlab_code(
            "FRDSTestInterp",
            """function [mdefsingle1, mFtsingle1] = FRDSTestInterp(Et, sigEt, xxEt, xxsigEt, xxmdef, xxFt)
            ymdef = scatteredInterpolant(xxEt(:),xxsigEt(:),xxmdef(:),'natural','none');
            yFt = scatteredInterpolant(xxEt(:),xxsigEt(:),xxFt(:),'natural','none');
            mdefsingle1 = ymdef(Et,sigEt);
            mFtsingle1 = yFt(Et,sigEt);
            """,
        )

        xxsigEt = xsigEt.squeeze()
        xxEt = xEt.squeeze()
        xxmdef = xmdef.squeeze()
        xxFt = xFt.squeeze()

        ymdef = griddata(
            (xxEt.flatten(), xxsigEt.flatten()),
            xxmdef.flatten(),
            (Et, sigEt),
            method="cubic",
        )
        yFt = griddata(
            (xxEt.flatten(), xxsigEt.flatten()),
            xxFt.flatten(),
            (Et, sigEt),
            method="cubic",
        )

        mdefsingle1 = ymdef
        mFtsingle1 = yFt

        res_mt = self.eng.FRDSTestInterp(
            Et, sigEt, xxEt, xxsigEt, xxmdef, xxFt, nargout=2
        )
        mdefsingle1_mt, mFtsingle1_mt = res_mt

        n_precision = 1
        # NOTE: almost surely to fail for higher precision due to interpolation!
        # 12% differ on second decimal.
        assert_array_almost_equal(mdefsingle1, mdefsingle1_mt, n_precision)
        assert_array_almost_equal(mFtsingle1, mFtsingle1_mt, n_precision)


if __name__ == "__main__":
    unittest.main()

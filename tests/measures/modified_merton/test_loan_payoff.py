import unittest
import pathlib
import numpy as np
from numpy.testing import assert_array_almost_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")

from frds.measures.modified_merton.loan_payoff import loan_payoff


class LoanPayoffTestCase(unittest.TestCase):
    def setUp(self) -> None:
        mp = pathlib.Path(__file__).parent.joinpath("matlab").as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(mp, nargout=0)

    def test_loan_payoff(self):
        result = self.eng.GenTestDataLoanPayoff(nargout=6)

        F, _, ival, rho, sig, T = result
        f1j = np.asarray(result[1])

        # Result from Python code
        L1 = loan_payoff(F, f1j, ival, rho, sig, T)
        # Result from Matlab code
        L2 = np.asarray(self.eng.LoanPayoff(F, f1j, ival, rho, sig, T))

        assert_array_almost_equal(L1, L2, decimal=9)


if __name__ == "__main__":
    unittest.main()

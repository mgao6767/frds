import unittest
import pathlib
import numpy as np

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")

from frds.measures.modified_merton.find_face_value_indiv import find_face_value_indiv


class FindFaceValueIndivTestCase(unittest.TestCase):
    def setUp(self) -> None:
        mp = pathlib.Path(__file__).parent.joinpath("matlab").as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(mp, nargout=0)

    def test_find_face_value_indiv(self):
        r = 0.01
        j = 3.0
        bookF = 0.6 + j * 0.02
        sig = 0.2
        T = 10.0
        ltv = 0.66
        ival = np.log(bookF) - np.log(ltv)
        d = 0.005

        # fmt: off
        for mu in np.linspace(r+0.01, r+0.5, 20):
            _a = find_face_value_indiv(mu, bookF * np.exp(mu*T), ival, sig, T, r, d)
            _b = self.eng.FindFaceValueIndiv(mu, bookF * np.exp(mu*T), ival, sig, T, r, d)
            self.assertAlmostEqual(_a, _b, 9)
        # fmt: on


if __name__ == "__main__":
    unittest.main()

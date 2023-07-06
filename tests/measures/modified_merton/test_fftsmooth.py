import unittest
import pathlib
import numpy as np
from numpy.testing import assert_array_almost_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")


from frds.measures.modified_merton.fftsmooth import fftsmooth


class FFTSmoothTestCase(unittest.TestCase):
    def setUp(self) -> None:
        frds_path = [
            i for i in pathlib.Path(__file__).parents if i.as_posix().endswith("frds")
        ].pop()
        self.mp = frds_path.joinpath(
            "src", "frds", "measures", "modified_merton", "matlab"
        ).as_posix()
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.mp, nargout=0)

    def test_fftsmooth(self):
        Nsim1 = 99
        Nsim2 = 1000
        np.random.seed(1)
        x = np.random.randn(Nsim1 * Nsim2).reshape((Nsim1 * Nsim2, 1))
        w = (Nsim1 * Nsim2) / 20

        result_python = fftsmooth(x.ravel(), int(w))
        result_matlab = np.asarray(self.eng.fftsmooth(x, w)).ravel()

        self.assertEqual(len(result_python), len(result_matlab))
        assert_array_almost_equal(result_python, result_matlab, decimal=9)


if __name__ == "__main__":
    unittest.main()

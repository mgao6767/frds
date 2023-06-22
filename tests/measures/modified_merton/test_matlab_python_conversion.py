import unittest
import os
import pathlib
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_array_almost_equal

try:
    import matlab.engine
    import matlab
except ImportError:
    raise unittest.SkipTest("MATLAB Engine API not available.")


class MatlabPythonConversionTestCase(unittest.TestCase):
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

    def test_random_normal(self):
        np.random.seed(1)
        d0 = 20
        d1 = 1000
        normrand_py = norm.ppf(np.random.rand(d1, d0).T, 0, 1)

        self.make_matlab_code(
            "FRDSTestRRNG",
            """function [w] = FRDSTestRRNG()
            rng(1,'twister');
            d0 = 20; d1 = 1000;
            w = norminv(rand(d0, d1),0,1);""",
        )

        normrand_mt = self.eng.FRDSTestRRNG()
        assert_array_almost_equal(normrand_py, normrand_mt, 9)


if __name__ == "__main__":
    unittest.main()

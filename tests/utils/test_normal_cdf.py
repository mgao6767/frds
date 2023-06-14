import unittest
import numpy as np
from scipy.stats import norm
from frds.utils import normal_cdf


class NormalCDFCase(unittest.TestCase):
    def test_cdf(self):
        for _ in range(1000):
            a = np.random.normal()
            self.assertAlmostEqual(normal_cdf(a), norm.cdf(a), 10)

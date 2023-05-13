import unittest
from frds.measures import z_score
import numpy as np


class BankZScoreCase(unittest.TestCase):
    def test_z_score(self):
        z = z_score.estimate(0.1, 0.3, np.array([0.14, 0.15, 0.12, 0.13]))
        self.assertAlmostEqual(z, 35.777088, 4)

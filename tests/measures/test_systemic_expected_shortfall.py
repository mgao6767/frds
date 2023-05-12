import unittest
from frds.measures import systemic_expected_shortfall as ses
import numpy as np


class SystemicExpectedShortfallCase(unittest.TestCase):
    def test_ses(self) -> None:
        # Data from and results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis

        mes_training_sample = np.array([-0.023, -0.07, 0.01])
        lvg_training_sample = np.array([1.8, 1.5, 2.2])
        ses_training_sample = np.array([0.3, 0.4, -0.2])
        mes_firm = 0.04
        lvg_firm = 1.7

        ses_estimate = ses.estimate(
            mes_training_sample,
            lvg_training_sample,
            ses_training_sample,
            mes_firm,
            lvg_firm,
        )

        self.assertAlmostEqual(ses_estimate, -0.333407572383073, 6)

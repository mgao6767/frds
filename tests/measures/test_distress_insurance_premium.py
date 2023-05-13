import unittest
from frds.measures import distress_insurance_premium as dip
import numpy as np


class DistressInsurancePremiumCase(unittest.TestCase):
    def test_dip(self):
        # Data from and results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis

        default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15])
        correlations = np.array(
            [
                [1, -0.1260125, -0.6366762, 0.1744837, 0.4689378, 0.2831761],
                [-0.1260125, 1, 0.294223, 0.673963, 0.1499695, 0.05250343],
                [-0.6366762, 0.294223, 1, 0.07259309, -0.6579669, -0.0848825],
                [0.1744837, 0.673963, 0.07259309, 1, 0.2483188, 0.5078022],
                [0.4689378, 0.1499695, -0.6579669, 0.2483188, 1, -0.3703121],
                [0.2831761, 0.05250343, -0.0848825, 0.5078022, -0.3703121, 1],
            ]
        )
        dip_result = dip.estimate(default_probabilities, correlations)
        self.assertAlmostEqual(dip_result, 0.29, 2)

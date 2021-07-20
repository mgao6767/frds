import unittest
from frds.measures import bank
import numpy as np


class AbsorptionRatioCase(unittest.TestCase):
    def setUp(self) -> None:
        # The data in the doc by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        self.data = np.array(
            [
                [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
                [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
                [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
            ]
        )
        np.random.seed(0)
        self.more_data = np.random.normal(0, 1, (50, 10))
        self.more_data = np.round(self.more_data / 100, 3).T

    def test_original_data(self):
        # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.1), 0.0, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.2), 0.7747, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.3), 0.7747, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.4), 0.7747, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.5), 0.9435, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.6), 0.9435, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.7), 0.9435, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.8), 0.9435, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 0.9), 1, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.data, 1), 1, 4)

    def test_more_data(self):
        # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.1), 0.1851, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.2), 0.3234, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.3), 0.4594, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.4), 0.5752, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.5), 0.6743, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.6), 0.7596, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.7), 0.8405, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.8), 0.9103, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 0.9), 0.9740, 4)
        self.assertAlmostEqual(bank.absorption_ratio(self.more_data, 1), 1, 4)

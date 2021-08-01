import unittest
from frds.measures import general
import numpy as np


class KyleLambdaCase(unittest.TestCase):
    def test_lamdba(self):
        volumes = np.array(
            [
                [100, 180, 900, 970, 430, 110],
                [300, 250, 400, 590, 260, 600],
                [200, 700, 220, 110, 290, 310],
            ]
        )
        prices = np.array(
            [
                [44, 39, 36, 28, 23, 18],
                [82, 81, 79, 40, 26, 13],
                [55, 67, 13, 72, 10, 65],
            ]
        )
        mil = np.mean(general.kyle_lambda(prices, volumes))
        self.assertAlmostEqual(mil, 0.0035, 4)


class HHIIndexCase(unittest.TestCase):
    def test_hhi_index(self):
        self.assertEqual(general.hhi_index(np.array([1, 1])), 0.5)
        self.assertAlmostEqual(
            general.hhi_index(np.array([1, 1, 100, 1, 1, 1, 1])), 0.8905, 4
        )

    def test_weighted_hhi_index(self):
        self.assertEqual(
            general.hhi_index(np.array([1, 1]), weights=np.array([1, 0])), 0.25
        )

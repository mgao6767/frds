import unittest
from frds.measures import general
import numpy as np


class KyleLambdaCase(unittest.TestCase):
    def test_lamdba(self):
        volumes = np.array(
            [
                [180, 900, 970, 430, 110],
                [250, 400, 590, 260, 600],
                [700, 220, 110, 290, 310],
            ]
        )
        price_raw = np.array(
            [
                [44, 39, 36, 28, 23, 18],
                [82, 81, 79, 40, 26, 13],
                [55, 67, 13, 72, 10, 65],
            ]
        )
        returns = np.diff(price_raw, axis=1) / np.roll(price_raw, 1)[:, 1:]
        prices = price_raw[:, 1:]
        mil = np.mean(general.kyle_lambda(returns, prices, volumes))
        self.assertAlmostEqual(mil, 0.0035, 4)

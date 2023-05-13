import unittest
from frds.measures import marginal_expected_shortfall as mes
import numpy as np


class MarginalExpectedShortfallCase(unittest.TestCase):
    def setUp(self):
        # The data in the doc by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        self.firm_returns = np.array(
            [
                -0.1595,
                0.1211,
                -0.0806,
                -0.0291,
                0.0897,
                -0.0254,
                0.1210,
                0.0132,
                -0.1214,
                0.1901,
                0.0243,
            ]
        )
        self.market_returns = np.array(
            [
                -0.0205,
                -0.0510,
                0.0438,
                0.0914,
                0.0243,
                -0.1051,
                0.0121,
                0.0221,
                -0.0401,
                -0.0111,
                -0.0253,
            ]
        )
        # simulated data
        np.random.seed(0)
        self.sim_firm_returns = np.round(np.random.normal(0, 1, (100,)) / 100, 3)
        self.sim_market_returns = np.round(np.random.normal(0, 1, (100,)) / 100, 3)

    def test_mes(self):
        # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        res = mes.estimate(self.firm_returns, self.market_returns, q=0.05)
        self.assertAlmostEqual(res, -0.0254, 4)

    def test_simulated_data(self):
        # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        res = mes.estimate(self.sim_firm_returns, self.sim_market_returns, q=0.01)
        self.assertAlmostEqual(res, -0.015, 4)

        res = mes.estimate(self.sim_firm_returns, self.sim_market_returns, q=0.05)
        self.assertAlmostEqual(res, 0.0016, 4)

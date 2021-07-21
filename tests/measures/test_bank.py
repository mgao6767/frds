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
        mes = bank.marginal_expected_shortfall(
            self.firm_returns, self.market_returns, q=0.05
        )
        self.assertAlmostEqual(mes, -0.0254, 4)

    def test_simulated_data(self):
        # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
        mes = bank.marginal_expected_shortfall(
            self.sim_firm_returns, self.sim_market_returns, q=0.01
        )
        self.assertAlmostEqual(mes, -0.015, 4)

        mes = bank.marginal_expected_shortfall(
            self.sim_firm_returns, self.sim_market_returns, q=0.05
        )
        self.assertAlmostEqual(mes, 0.0016, 4)


class SystemicExpectedShortfallCase(unittest.TestCase):
    def test_ses(self) -> None:
        # Data from and results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis

        mes_training_sample = np.array([-0.023, -0.07, 0.01])
        lvg_training_sample = np.array([1.8, 1.5, 2.2])
        ses_training_sample = np.array([0.3, 0.4, -0.2])
        mes_firm = 0.04
        lvg_firm = 1.7

        ses = bank.systemic_expected_shortfall(
            mes_training_sample,
            lvg_training_sample,
            ses_training_sample,
            mes_firm,
            lvg_firm,
        )

        self.assertAlmostEqual(ses, -0.333407572383073, 6)

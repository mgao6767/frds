import unittest
from frds.data.wrds.comp import Funda, Fundq
from frds.io.wrds import load
from frds.measures.corporate import *


class FundaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        obs = 30_000
        self.data: Funda = load(Funda, use_cache=True, save=False, obs=obs)
        self.assertIsInstance(self.data, Funda)
        # Filters may reduce the sample size
        self.assertLessEqual(len(self.data.data), obs)

    def test_roa(self):
        roa(self.data)
        roa(self.data, use_lagged_total_asset=True)

    def test_roe(self):
        roe(self.data)
        roe(self.data, use_lagged_ceq=True)

    def test_tangibility(self):
        tangibility(self.data)

    def test_book_leverage(self):
        book_leverage(self.data)

    def test_market_to_book(self):
        market_to_book(self.data)

    def test_firm_size(self):
        size = firm_size(self.data)
        query = (self.data.AT <= 0) | (np.isnan(self.data.AT) == True)
        idx = self.data.data[query].index
        # Firm size should be NaN if total assets is negative
        self.assertTrue(all(np.isnan(size[idx])))

    def test_capital_expenditure(self):
        capital_expenditure(self.data)


class FundqTestCase(unittest.TestCase):
    def setUp(self) -> None:
        obs = 10_000
        self.data = load(Fundq, use_cache=True, save=False, obs=obs)
        self.assertIsInstance(self.data, Fundq)
        # Filters may reduce the sample size
        self.assertLessEqual(len(self.data.data), obs)

    def test_roa(self):
        roa(self.data)
        roa(self.data, use_lagged_total_asset=True)

    def test_roe(self):
        roe(self.data)
        roe(self.data, use_lagged_ceq=True)

    def test_tangibility(self):
        tangibility(self.data)

    def test_book_leverage(self):
        book_leverage(self.data)

    def test_market_to_book(self):
        market_to_book(self.data)

    def test_firm_size(self):
        self.data.data["Test_FirmSize"] = firm_size(self.data)
        query = (self.data.ATQ <= 0) | (np.isnan(self.data.ATQ) == True)
        # Firm size should be NaN if total assets is negative
        # TODO: This test is written this way as the default filter (for now) on FUNDQ seems not to
        # guarantee unique gvkey-datadate, hence index-based test doesn't work as expected.
        self.assertTrue(all(np.isnan(self.data.data[query]["Test_FirmSize"])))

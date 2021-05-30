import unittest
from frds.data.wrds.comp import Funda, Fundq
from frds.io.wrds import load
from frds.measures.corporate import *


class FundaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        obs = 30_000
        self.data = load(Funda, use_cache=True, save=False, obs=obs)
        self.assertIsInstance(self.data, Funda)
        self.assertEqual(len(self.data.data), obs)

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


class FundqTestCase(unittest.TestCase):
    def setUp(self) -> None:
        obs = 10_000
        self.data = load(Fundq, use_cache=True, save=False, obs=obs)
        self.assertIsInstance(self.data, Fundq)
        self.assertEqual(len(self.data.data), obs)

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

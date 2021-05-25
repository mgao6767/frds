import unittest
from frds.data.wrds.comp import funda, fundq
from frds.io.wrds import load
from frds.measures import ROE


class ROETestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDA = load(funda, use_cache=True, save=False, obs=100)
        self.FUNDQ = load(fundq, use_cache=True, save=False, obs=100)

    def test_roe_default(self):
        ROE(self.FUNDA)
        ROE(self.FUNDQ)

    def test_roe_v2(self):
        ROE.v2(self.FUNDA)
        ROE.v2(self.FUNDQ)

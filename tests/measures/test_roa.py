import unittest
from frds.data.wrds.comp import funda, fundq
from frds.io.wrds import load
from frds.measures import ROA


class ROATestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDA = load(funda, use_cache=True, save=False, obs=100)
        self.FUNDQ = load(fundq, use_cache=True, save=False, obs=100)

    def test_roa_default(self):
        ROA(self.FUNDA)
        ROA(self.FUNDQ)

    def test_roa_v2(self):
        ROA.v2(self.FUNDA)
        ROA.v2(self.FUNDQ)

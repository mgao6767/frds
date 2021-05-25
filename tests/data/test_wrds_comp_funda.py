import unittest
import pandas as pd
from frds.data.wrds.comp import funda
from frds.io.wrds import load


class FundaLoadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDA = load(funda, use_cache=False, save=False, obs=100)

    def test_load_funda(self):
        self.assertIsInstance(self.FUNDA, funda)

    def test_save_funda(self):
        load(funda, use_cache=True, save=True, obs=100)


class FundaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDA = load(funda, use_cache=True, save=False, obs=100)

    def test_access_attrs(self):
        attrs = [
            varname
            for varname, prop in vars(funda).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            v = self.FUNDA.__getattribute__(attr)
            self.assertIsInstance(v, pd.Series)

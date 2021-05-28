import unittest
import pandas as pd
from frds.data.wrds.comp import Fundq
from frds.io.wrds import load


class FundqLoadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDQ = load(Fundq, use_cache=False, save=False, obs=100)

    def test_load_fundq(self):
        self.assertIsInstance(self.FUNDQ, Fundq)

    def test_save_fundq(self):
        load(Fundq, use_cache=True, save=True, obs=100)


class FundqTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.FUNDQ = load(Fundq, use_cache=True, save=False, obs=100)

    def test_access_attrs(self):
        attrs = [
            varname
            for varname, prop in vars(Fundq).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            v = self.FUNDQ.__getattribute__(attr)
            self.assertIsInstance(v, pd.Series)

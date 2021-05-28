import unittest
import pandas as pd
from frds.data.wrds.execucomp import Anncomp
from frds.io.wrds import load


class AnncompLoadTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ANNCOMP = load(Anncomp, use_cache=False, save=False, obs=100)

    def test_load_anncomp(self):
        self.assertIsInstance(self.ANNCOMP, Anncomp)

    def test_save_anncomp(self):
        load(Anncomp, use_cache=True, save=True, obs=100)


class AnncompTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ANNCOMP = load(Anncomp, use_cache=True, save=False, obs=100)

    def test_access_attrs(self):
        attrs = [
            varname
            for varname, prop in vars(Anncomp).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            v = self.ANNCOMP.__getattribute__(attr)
            self.assertIsInstance(v, pd.Series)

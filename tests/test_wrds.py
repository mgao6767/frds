import unittest
import numpy as np
from frds.data import wrds, DataManager
from frds.measures import sample_measure


class Login(unittest.TestCase):

    def test_login_with_empty_username_and_password(self):
        with self.assertRaises(Exception):
            wrds.Connection(usr=None, pwd=None)


class GetData(unittest.TestCase):

    def test_get_data(self):
        with DataManager() as dm:
            # Should be able to create np.recarry from shared memory
            shm, shape, dtype = dm.get_dataset(sample_measure.dataset)
            nparray = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
            self.assertIsInstance(nparray, np.recarray)
            self.assertGreater(np.nanmean(nparray.at), 0)
            # The same dataset should have the same shared memory name
            shm2, _, _ = dm.get_dataset(sample_measure.dataset)
            self.assertEqual(shm.name, shm2.name)
            self.assertEqual(len(dm._datasets), 1)

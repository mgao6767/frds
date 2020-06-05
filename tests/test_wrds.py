import unittest
from frds.data import wrds


class Login(unittest.TestCase):

    def test_login_with_empty_username_and_password(self):
        with self.assertRaises(Exception):
            wrds.Connection(usr=None, pwd=None)


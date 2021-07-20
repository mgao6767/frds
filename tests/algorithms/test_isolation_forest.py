import unittest
from frds.algorithms import isolation_forest
import pandas as pd
import numpy as np


class IsolationForestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_obs = 1_000
        self.n_attrs = 100
        np.random.seed(0)
        self.data = pd.DataFrame(
            {
                f"Attr{i}": np.random.normal(0, 1, self.n_obs)
                for i in range(self.n_attrs)
            },
            index=[f"obs.{i}" for i in range(self.n_obs)],
        )

    def test_anomaly_scores(self):
        ascores = isolation_forest.anomaly_scores(self.data)
        self.assertIsInstance(ascores, pd.DataFrame)
        self.assertTrue(len(ascores), self.n_obs)

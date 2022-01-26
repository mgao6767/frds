import unittest
from frds.algorithms import isolation_forest
import pandas as pd
import numpy as np


class IsolationForestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_obs = 1_000
        self.n_attrs = 100
        self.n_char_attrs = 5
        np.random.seed(0)

    def test_anomaly_scores_numeric_only(self):
        data = pd.DataFrame(
            {
                f"Attr{i}": np.random.normal(0, 1, self.n_obs)
                for i in range(self.n_attrs)
            },
            index=[f"obs.{i}" for i in range(self.n_obs)],
        )
        ascores = isolation_forest.anomaly_scores(data)
        self.assertIsInstance(ascores, pd.DataFrame)
        self.assertTrue(len(ascores), self.n_obs)

    def test_anomaly_scores_numeric_and_char(self):
        data = pd.DataFrame(
            {
                f"Attr{i}": np.random.normal(0, 1, self.n_obs)
                if i < self.n_attrs - self.n_char_attrs
                else str(np.random.randint(0, 100000000))
                for i in range(self.n_attrs)
            },
            index=[f"obs.{i}" for i in range(self.n_obs)],
        )
        ascores = isolation_forest.anomaly_scores(data)
        self.assertIsInstance(ascores, pd.DataFrame)
        self.assertTrue(len(ascores), self.n_obs)
        print(ascores)


if __name__ == "__main__":
    unittest.main()

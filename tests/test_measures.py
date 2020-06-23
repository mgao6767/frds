import unittest
from datetime import datetime
import numpy as np
import pandas as pd
from frds.measures import BoardIndependence
from .test_data import (
    MOCK_WRDS_BOARDEX_NA_WRDS_COMPANY_PROFILE,
    MOCK_WRDS_BOARDEX_NA_WRDS_ORG_COMPOSITION,
    MOCK_WRDS_COMP_FUNDA,
)


DATA = [
    MOCK_WRDS_BOARDEX_NA_WRDS_COMPANY_PROFILE,
    MOCK_WRDS_BOARDEX_NA_WRDS_ORG_COMPOSITION,
    MOCK_WRDS_COMP_FUNDA,
]


class TestBoardIndependence(unittest.TestCase):
    def test_estimate_board_independence(self):
        measure = BoardIndependence()
        result, variable_labels = measure.estimate(DATA)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(variable_labels, dict)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.BoardSize.tolist(), [5, 4, 2, 2])
        self.assertEqual(result.IndependentMembers.tolist(), [2, 2, 1, 1])

    def test_estimate_board_independence_with_params(self):
        measure = BoardIndependence(
            missing_independent_board_members_as_zero=False
        )
        result, variable_labels = measure.estimate(DATA)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(variable_labels, dict)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.BoardSize.tolist(), [5, 4, 2, 2])
        self.assertEqual(result.IndependentMembers.tolist(), [2, 2, 1, 1])


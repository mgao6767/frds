import pytest
import numpy as np
from frds.measures import DistressInsurancePremium


def test_dip():
    default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15])
    correlations = np.array(
        [
            [1.000, -0.126, -0.637, 0.174, 0.469, 0.283],
            [-0.126, 1.000, 0.294, 0.674, 0.150, 0.053],
            [-0.637, 0.294, 1.000, 0.073, -0.658, -0.085],
            [0.174, 0.674, 0.073, 1.000, 0.248, 0.508],
            [0.469, 0.150, -0.658, 0.248, 1.000, -0.370],
            [0.283, 0.053, -0.085, 0.508, -0.370, 1.000],
        ]
    )
    dip = DistressInsurancePremium(default_probabilities, correlations)
    res = dip.estimate()
    assert res == pytest.approx(0.28657335507)

import pytest
import numpy as np
from frds.measures import AbsorptionRatio


def test_ar():
    # 3 assets daily returns for 6 days
    data = np.array(
        [
            [0.015, 0.031, 0.007, 0.034, 0.014, 0.011],
            [0.012, 0.063, 0.027, 0.023, 0.073, 0.055],
            [0.072, 0.043, 0.097, 0.078, 0.036, 0.083],
        ]
    )
    # Calculate the absorption ratio.
    ar = AbsorptionRatio(data)
    res = ar.estimate()
    assert res == pytest.approx(0.7746543307660252)

    # Results computed using the Matlab code by Dimitrios Bisias, Andrew W. Lo, and Stavros Valavanis
    assert ar.estimate(0.1) == pytest.approx(0.0, rel=0.0001)
    assert ar.estimate(0.2) == pytest.approx(0.7747, rel=0.0001)
    assert ar.estimate(0.3) == pytest.approx(0.7747, rel=0.0001)
    assert ar.estimate(0.4) == pytest.approx(0.7747, rel=0.0001)
    assert ar.estimate(0.5) == pytest.approx(0.9435, rel=0.0001)
    assert ar.estimate(0.6) == pytest.approx(0.9435, rel=0.0001)
    assert ar.estimate(0.7) == pytest.approx(0.9435, rel=0.0001)
    assert ar.estimate(0.8) == pytest.approx(0.9435, rel=0.0001)
    assert ar.estimate(0.9) == pytest.approx(1, rel=0.0001)

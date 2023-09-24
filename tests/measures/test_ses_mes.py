import pytest
import numpy as np
from numpy.random import RandomState
from frds.measures import SystemicExpectedShortfall, MarginalExpectedShortfall


def test_ses():
    mes_training_sample = np.array([-0.023, -0.07, 0.01])
    lvg_training_sample = np.array([1.8, 1.5, 2.2])
    ses_training_sample = np.array([0.3, 0.4, -0.2])
    mes_firm = 0.04
    lvg_firm = 1.7
    ses = SystemicExpectedShortfall(
        mes_training_sample,
        lvg_training_sample,
        ses_training_sample,
        mes_firm,
        lvg_firm,
    )
    res = ses.estimate()
    assert res == pytest.approx(-0.33340757238306845)


def test_mes():
    rng = RandomState(0)
    firm_returns = rng.normal(0, 1, 100)
    mkt_returns = rng.normal(0, 1, 100)
    mes = MarginalExpectedShortfall(firm_returns, mkt_returns)
    res = mes.estimate()
    assert res == pytest.approx(0.13494025343324562)

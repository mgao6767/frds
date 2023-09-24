import pytest
from frds.measures import ContingentClaimAnalysis


def test_cca():
    cca = ContingentClaimAnalysis()
    put_price, cca_risk = cca.estimate(
        equity=5,
        volatility=1.2,
        risk_free_rate=0.02,
        default_barrier=10,
        time_to_maturity=20,
        cds_spread=1.5,
    )
    assert put_price == pytest.approx(6.659378336338627)
    assert cca_risk == pytest.approx(3.3467523905471133)

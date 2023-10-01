import pytest
from frds.algorithms import GARCHModel_CCC
from frds.datasets import StockReturns


def test_garch_ccc():
    returns = StockReturns.stocks_us

    sp500 = returns["^GSPC"].to_numpy() * 100
    googl = returns["GOOGL"].to_numpy() * 100

    model_ccc = GARCHModel_CCC(sp500, googl)
    res = model_ccc.fit()

    tol = 0.01
    assert res.mu1 == pytest.approx(0.0699378, rel=tol)
    assert res.omega1 == pytest.approx(0.0585878, rel=tol)
    assert res.alpha1 == pytest.approx(0.1477404, rel=tol)
    assert res.beta1 == pytest.approx(0.7866691, rel=tol)
    assert res.mu2 == pytest.approx(0.0940275, rel=tol)
    assert res.omega2 == pytest.approx(0.4842512, rel=tol)
    assert res.alpha2 == pytest.approx(0.12166, rel=tol)
    assert res.beta2 == pytest.approx(0.7113389, rel=tol)
    assert res.rho == pytest.approx(0.6646705, rel=tol)


if __name__ == "__main__":
    pytest.main([__file__])

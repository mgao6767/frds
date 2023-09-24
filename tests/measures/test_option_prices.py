import pytest
from frds.measures import blsprice


def test_blsprice():
    c, p = blsprice(0.67, 0.7, 0.01, 5.0, 0.33, 0.002)
    assert c == pytest.approx(0.19003370474049647)
    assert p == pytest.approx(0.1925609132790535)

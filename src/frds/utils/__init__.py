import math


def normal_cdf(x: float) -> float:
    """CDF of standard normal"""
    q = math.erf(x / 1.4142135623730951)  # sqrt(2)=1.4142135623730951
    return (1.0 + q) / 2.0

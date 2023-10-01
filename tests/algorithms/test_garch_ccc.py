import pytest
import numpy as np

from frds.algorithms import GARCHModel_CCC


def test_garch_ccc():
    rng = np.random.default_rng(42)

    # Initialize parameters
    T = 500  # Number of observations
    omega1, alpha1, beta1 = 0.1, 0.2, 0.7  # Parameters for first series
    omega2, alpha2, beta2 = 0.1, 0.3, 0.6  # Parameters for second series
    rho = 0.5  # Constant correlation

    # Initialize variables
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    h1 = np.zeros(T)
    h2 = np.zeros(T)
    z1 = rng.normal(size=T)
    z2 = rng.normal(size=T)

    # Simulate conditional variances and returns
    for t in range(1, T):
        h1[t] = omega1 + alpha1 * y1[t - 1] ** 2 + beta1 * h1[t - 1]
        h2[t] = omega2 + alpha2 * y2[t - 1] ** 2 + beta2 * h2[t - 1]

        # Apply constant correlation
        e1 = np.sqrt(h1[t]) * z1[t]
        e2 = np.sqrt(h2[t]) * (rho * z1[t] + np.sqrt(1 - rho**2) * z2[t])

        y1[t] = e1
        y2[t] = e2

    model = GARCHModel_CCC(y1, y2)
    res = model.fit()
    print(res)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_garch_ccc()

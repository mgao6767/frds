import pytest
import numpy as np
from frds.measures import SRISK


@pytest.mark.skip(reason="SRISK is not numerically stable")
def test_srisk():
    # fmt: off
    np.random.seed(1)

    # Parameters
    n_days = 252 * 3  # Number of trading days 
    mean_returns = [0.0005, 0.0007, 0.0006]  # Mean daily returns for Stock1, Stock2, and Market
    volatilities = [0.02, 0.025, 0.015]  # Daily volatilities for Stock1, Stock2, and Market

    # Correlation matrix
    corr_matrix = np.array([
        [1.0, 0.75, 0.6],  # Correlation between Stock1 and Stock1, Stock2, Market
        [0.75, 1.0, 0.5],  # Correlation between Stock2 and Stock1, Stock2, Market
        [0.6, 0.5, 1.0]    # Correlation between Market and Stock1, Stock2, Market
    ])

    # Cholesky decomposition to get a lower triangular matrix
    chol_decomp = np.linalg.cholesky(corr_matrix)

    # Simulate daily returns
    random_returns = np.random.normal(0, 1, (n_days, 3))
    correlated_returns = np.dot(random_returns, chol_decomp.T)

    # Apply mean and volatility
    final_returns = mean_returns + np.multiply(correlated_returns, volatilities)

    firm_returns = final_returns[:,:-1]
    mkt_returns = final_returns[:,-1]

    srisk = SRISK(
        firm_returns, mkt_returns, W=np.array([100, 80]), D=np.array([900, 250])
    )

    srisk1, srisk2 = srisk.estimate(aggregate_srisk=False)
    assert srisk1 == pytest.approx(8.2911)
    assert srisk2 == pytest.approx(-31.10596)

    srisk_agg = srisk.estimate(aggregate_srisk=True)
    assert srisk_agg == pytest.approx(8.2911)


if __name__ == "__main__":
    pytest.main([__file__])

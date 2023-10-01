import itertools
import pytest
import numpy as np
from frds.algorithms import GJRGARCHModel
from arch import arch_model


def generate_gjrgarch_returns(mu, omega, alpha, gamma, beta, rng: np.random.Generator):
    T = 1000
    # fmt: off
    epsilon = rng.normal(size=T)
    sigma_squared = np.zeros(T)
    r = np.zeros(T)
    
    for t in range(1, T):
        sigma_squared[t] = omega + alpha * (epsilon[t-1]**2) + beta * sigma_squared[t-1]
        sigma_squared[t] += gamma * (epsilon[t-1]**2) if epsilon[t-1] < 0 else 0
        epsilon[t] = np.sqrt(sigma_squared[t]) * rng.normal()
        r[t] = mu + epsilon[t]
        # Let returns be bounded to -99% to 200%
        r[t] = np.clip(r[t], -0.99, 2.0)
        
    return r * 100


def test_gjrgarch_estimate():
    rng = np.random.default_rng(42)
    # Test against many different GARCH processes
    mu = 0.0
    omega_values = [0.01, 0.05]
    alpha_values = [0.1, 0.15]
    gamma_values = alpha_values
    beta_values = [0.7, 0.8]
    # fmt: off
    for omega, alpha, gamma, beta in itertools.product(*[omega_values, alpha_values, gamma_values, beta_values]):
        if alpha+beta+gamma/2>=1:
            continue
        print(f"Parameters to simulate GJR-GARCH(1,1): {mu=}, {omega=}, {alpha=}, {gamma=}, {beta=}")
        returns = generate_gjrgarch_returns(mu, omega, alpha, gamma, beta, rng)
        __test_gjrgarch_estimation(returns)


# Define the test function
def __test_gjrgarch_estimation(returns):
    # Fit the GJR-GARCH(1,1) model using frds
    frds_gjrgarch = GJRGARCHModel(returns)
    frds_result = frds_gjrgarch.fit()

    # Fit the GARCH(1,1) model using arch
    arch_garch = arch_model(returns, vol="Garch", p=1, o=1, q=1)
    arch_result = arch_garch.fit(disp="off")
    arch_mu = arch_result.params["mu"]
    arch_omega = arch_result.params["omega"]
    arch_alpha = arch_result.params["alpha[1]"]
    arch_gamma = arch_result.params["gamma[1]"]
    arch_beta = arch_result.params["beta[1]"]
    arch_loglikelihood = arch_result.loglikelihood

    # Define a tolerance level for the parameter estimates
    tol = 0.1  # 10% relative difference
    # When the frds estimates are not as good, check if they are close.
    # Even though loglikelihoods are very close, the parameters can vary widely.
    # GARCH models are notorously difficult to estimate?
    if frds_result.loglikelihood < arch_loglikelihood:
        # Compare the estimates
        assert np.isclose(frds_result.mu, arch_mu, rtol=tol)
        assert np.isclose(frds_result.omega, arch_omega, rtol=tol)
        assert np.isclose(frds_result.alpha, arch_alpha, rtol=tol)
        assert np.isclose(frds_result.gamma, arch_gamma, rtol=tol)
        assert np.isclose(frds_result.beta, arch_beta, rtol=tol)


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])

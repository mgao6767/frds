import math
from statistics import NormalDist
import numpy as np
from numpy.random import RandomState


def distress_insurance_premium(
    default_prob: np.ndarray,
    correlations: np.ndarray,
    default_threshold: float = 0.15,
    random_seed: int = 0,
    n_simulated_returns: int = 500_000,
    n_simulations: int = 1_000,
) -> float:
    """Distress Insurance Preimum (DIP)

    A systemic risk metric by [Huang, Zhou, and Zhu (2009)](https://doi.org/10.1016/j.jbankfin.2009.05.017) \
    which represents a hypothetical insurance premium against a systemic financial distress, defined as total losses that \
    exceed a given threshold, say 15%, of total bank liabilities.

    The methodology is general and can apply to any pre-selected group of firms with publicly tradable equity and CDS contracts. \
    Each institutions marginal contribution to systemic risk is a function of its size, probability of default, and asset correlation. \
    The last two components need to be estimated from market data.

    The general steps are:

    1. Use simulated asset returns from a joint normal distribution (using the correlations) to compute the distribution of joint defaults.
    2. The loss-given-default (LGD) is assumed to follow a symmetric triangular distribution with a mean of 0.55 and in the range of [0.1, 1]. \
        The mean LGD of 0.55 is taken down from the Basel II IRB formula.
    3. Compute the probability of losses and the expected losses from the simulations.

    Args:
        default_prob (np.ndarray): (n_banks,) array of the bank risk-neutral default probabilities.
        correlations (np.ndarray): (n_banks, n_banks) array of the correlation matrix of the banks' asset returns.
        default_threshold (float, optional): the threshold used to calculate the total losses to total liabilities. Defaults to 0.15.
        random_seed (int, optional): the random seed used in Monte Carlo simulation for reproducibility. Defaults to 0.
        n_simulated_returns (int, optional): the number of simulations to compute the distrituion of joint defaults. Defaults to 500,000.
        n_simulations (int, optional): the number of simulations to compute the probability of losses. Defaults to 1,000.

    Returns:
        float: The distress insurance premium against a systemic financial distress.

    Examples:
        >>> import numpy as np
        >>> from frds.measures import distress_insurance_premium

        Arbitrary default probabilities for 6 banks.
        >>> default_probabilities = np.array([0.02, 0.10, 0.03, 0.20, 0.50, 0.15])

        Hypothetical correlations of the banks' asset returns.
        >>> correlations = np.array(
        ...     [
        ...         [1, -0.1260125, -0.6366762, 0.1744837, 0.4689378, 0.2831761],
        ...         [-0.1260125, 1, 0.294223, 0.673963, 0.1499695, 0.05250343],
        ...         [-0.6366762, 0.294223, 1, 0.07259309, -0.6579669, -0.0848825],
        ...         [0.1744837, 0.673963, 0.07259309, 1, 0.2483188, 0.5078022],
        ...         [0.4689378, 0.1499695, -0.6579669, 0.2483188, 1, -0.3703121],
        ...         [0.2831761, 0.05250343, -0.0848825, 0.5078022, -0.3703121, 1],
        ...     ]
        ... )

        Calculate the distress insurance premium.
        >>> distress_insurance_premium(default_probabilities, correlations)
        0.28661995758
        >>> distress_insurance_premium(default_probabilities, correlations, n_simulations=10_000, n_simulated_returns=1_000_000)
        0.2935815484909995

    References:
        * [Huang, Zhou and Zhu (2009)](https://doi.org/10.1016/j.jbankfin.2009.05.017),
            A framework for assessing the systemic risk of major financial institutions, *Journal of Banking & Finance*, 33(11), 2036-2049.
        * [Bisias, Flood, Lo, and Valavanis (2012)](https://doi.org/10.1146/annurev-financial-110311-101754),
            A survey of systemic risk analytics, *Annual Review of Financial Economics*, 4, 255-296.

    See Also:
        Systemic risk measures:

        * [Absorption Ratio](/measures/absorption_ratio/)
        * [Contingent Claim Analysis](/measures/cca/)
        * [Marginal Expected Shortfall (MES)](/measures/marginal_expected_shortfall/)
        * [Systemic Expected Shortfall (SES)](/measures/systemic_expected_shortfall/)
    """
    # Use the class to avoid impacting the global numpy state
    rng = RandomState(random_seed)
    n_banks = len(default_prob)
    # Simulate correlated normal distributions
    norm = NormalDist()
    default_thresholds = np.fromiter(
        (norm.inv_cdf(i) for i in default_prob),
        default_prob.dtype,
        count=n_banks,
    )
    R = np.linalg.cholesky(correlations).T
    z = np.dot(rng.normal(0, 1, size=(n_simulated_returns, n_banks)), R)

    default_dist = np.sum(z < default_thresholds, axis=1)

    # an array where the i-th element is the frequency of i banks jointly default
    # where len(frequency_of_join_defaults) is n_banks+1
    frequency_of_join_defaults = np.bincount(default_dist, minlength=n_banks + 1)
    dist_joint_defaults = frequency_of_join_defaults / n_simulated_returns

    loss_given_default = np.empty(shape=(n_banks, n_simulations))
    for i in range(n_banks):
        lgd = np.sum(rng.triangular(0.1, 0.55, 1, size=(i + 1, n_simulations)), axis=0)
        loss_given_default[i:] = lgd

    # Maximum losses are N. Divide this into N*100 intervals.
    # Find the probability distribution of total losses in the default case
    intervals = 100
    loss_given_default *= intervals

    prob_losses = np.zeros(n_banks * intervals)
    for i in range(n_banks):
        for j in range(n_simulations):
            # Multiply losses_given_default(i,j) by intervals to find the right slot
            # in the prob_losses. Then we increment this by probability of i defaults
            idx = math.ceil(loss_given_default[i, j])
            prob_losses[idx] += dist_joint_defaults[i + 1]

    # Convert to probabilities
    prob_losses = prob_losses / n_simulations
    pct_threshold = int(default_threshold * 100)

    # Find the probability that the losses are great than 0.15 the total liabilities i.e. > 0.15*N
    prob_great_losses = np.sum(prob_losses[pct_threshold * n_banks :])

    exp_losses = (
        np.dot(
            np.array(range(pct_threshold * n_banks, intervals * n_banks)),
            prob_losses[pct_threshold * n_banks :],
        )
        / (100 * prob_great_losses)
    )

    return exp_losses * prob_great_losses

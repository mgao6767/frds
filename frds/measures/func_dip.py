import math
from statistics import NormalDist
import numpy as np


def distress_insurance_premium(default_prob: np.ndarray, corr: np.ndarray) -> float:
    """Distress Insurance Preimum (DIP)

    The Distress Insurance Premium (DIP) is proposed as an ex ante systemic risk metric by Huang, Zhou, and Zhu (2009b) \
    and it represents a hypothetical insurance premium against a systemic financial distress, defined as total losses that \
    exceed a given threshold, say 15%, of total bank liabilities. The methodology is general and can apply to any pre-selected \
    group of firms with publicly tradable equity and CDS contracts. Each institutions marginalcontribution to systemic risk is \
    a function of its size, probability of default (PoD), and asset correlation. The last two components need to be estimated from market data.

    Args:
        default_prob (np.ndarray): The default probabilities of the banks.
        corr (np.ndarray): The correlation matrix of the assets' returns of the banks.

    Returns:
        float: The distress insurance premium against a systemic financial distress.
    """
    n_repetitions = 500_000
    n_banks = len(default_prob)
    norm = NormalDist()
    default_threshold = np.fromiter(
        (norm.inv_cdf(i) for i in default_prob),
        default_prob.dtype,
        count=n_banks,
    )
    R = np.linalg.cholesky(corr).T
    z = np.dot(np.random.normal(0, 1, size=(n_repetitions, n_banks)), R)

    default_dist = np.sum(z < default_threshold, axis=1)

    # an array where the i-th element is the frequency of i banks jointly default
    # where len(frequency_of_join_defaults) is n_banks+1
    frequency_of_join_defaults = np.bincount(default_dist, minlength=n_banks + 1)
    dist_joint_defaults = frequency_of_join_defaults / n_repetitions

    n_sims = 1_000
    loss_given_default = np.empty(shape=(n_banks, n_sims))
    for i in range(n_banks):
        lgd = np.sum(np.random.triangular(0.1, 0.55, 1, size=(i + 1, n_sims)), axis=0)
        loss_given_default[i:] = lgd

    intervals = 100
    loss_given_default *= intervals

    prob_losses = np.zeros(n_banks * intervals)
    for i in range(n_banks):
        for j in range(1000):
            idx = math.ceil(loss_given_default[i, j])
            prob_losses[idx] += dist_joint_defaults[i + 1]

    prob_losses = prob_losses / n_sims
    prob_great_losses = np.sum(prob_losses[15 * n_banks :])

    exp_losses = np.dot(
        np.array(range(15 * n_banks, 100 * n_banks)), prob_losses[15 * n_banks :]
    ) / (100 * prob_great_losses)

    return exp_losses * prob_great_losses

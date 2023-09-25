import math
from statistics import NormalDist
import numpy as np
from numpy.random import RandomState


class DistressInsurancePremium:
    """:doc:`/measures/distress_insurance_premium`"""

    def __init__(self, default_prob: np.ndarray, correlations: np.ndarray) -> None:
        """__init__

        Args:
            default_prob (np.ndarray): ``(n_banks,)`` array of the bank risk-neutral default probabilities.
            correlations (np.ndarray): ``(n_banks, n_banks)`` array of the correlation matrix of the banks' asset returns.
        """
        self.default_prob = default_prob
        self.correlations = correlations

    def estimate(
        self,
        default_threshold: float = 0.15,
        random_seed: int = 0,
        n_simulated_returns: int = 500_000,
        n_simulations: int = 1_000,
    ) -> float:
        """estimate

        Args:
            default_threshold (float, optional): the threshold used to calculate the total losses to total liabilities. Defaults to 0.15.
            random_seed (int, optional): the random seed used in Monte Carlo simulation for reproducibility. Defaults to 0.
            n_simulated_returns (int, optional): the number of simulations to compute the distrituion of joint defaults. Defaults to 500,000.
            n_simulations (int, optional): the number of simulations to compute the probability of losses. Defaults to 1,000.

        Returns:
            float: The distress insurance premium against a systemic financial distress.
        """
        # Use the class to avoid impacting the global numpy state
        rng = RandomState(random_seed)
        n_banks = len(self.default_prob)
        # Simulate correlated normal distributions
        norm = NormalDist()
        default_thresholds = np.fromiter(
            (norm.inv_cdf(i) for i in self.default_prob),
            self.default_prob.dtype,
            count=n_banks,
        )
        R = np.linalg.cholesky(self.correlations).T
        z = np.dot(rng.normal(0, 1, size=(n_simulated_returns, n_banks)), R)

        default_dist = np.sum(z < default_thresholds, axis=1)

        # an array where the i-th element is the frequency of i banks jointly default
        # where len(frequency_of_join_defaults) is n_banks+1
        frequency_of_join_defaults = np.bincount(default_dist, minlength=n_banks + 1)
        dist_joint_defaults = frequency_of_join_defaults / n_simulated_returns

        loss_given_default = np.empty(shape=(n_banks, n_simulations))
        for i in range(n_banks):
            # fmt: off
            lgd = np.sum(rng.triangular(0.1, 0.55, 1, size=(i + 1, n_simulations)), axis=0)
            loss_given_default[i:] = lgd
            # fmt: on

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

        exp_losses = np.dot(
            np.array(range(pct_threshold * n_banks, intervals * n_banks)),
            prob_losses[pct_threshold * n_banks :],
        ) / (100 * prob_great_losses)

        return exp_losses * prob_great_losses

import numpy as np


def estimate(
    mes_training_sample: np.ndarray,
    lvg_training_sample: np.ndarray,
    ses_training_sample: np.ndarray,
    mes_firm: float,
    lvg_firm: float,
) -> float:
    """Systemic Expected Shortfall

    Args:
        mes_training_sample (np.ndarray): (n_firms,) array of firm ex ante MES.
        lvg_training_sample (np.ndarray): (n_firms,) array of firm ex ante LVG (say, on the last day of the period of training data)
        ses_training_sample (np.ndarray): (n_firms,) array of firm ex post cumulative return for date range after `lvg_training_sample`.
        mes_firm (float): The current firm MES used to calculate the firm (fitted) SES value.
        lvg_firm (float): The current firm leverage used to calculate the firm (fitted) SES value.

    Returns:
        float: The systemic risk that firm $i$ poses to the system at a future time.

    Examples:
        >>> from frds.measures import systemic_expected_shortfall as ses
        >>> import numpy as np
        >>> mes_training_sample = np.array([-0.023, -0.07, 0.01])
        >>> lvg_training_sample = np.array([1.8, 1.5, 2.2])
        >>> ses_training_sample = np.array([0.3, 0.4, -0.2])
        >>> mes_firm = 0.04
        >>> lvg_firm = 1.7
        >>> ses.estimate(mes_training_sample, lvg_training_sample, ses_training_sample, mes_firm, lvg_firm)
        -0.33340757238306845

    """
    assert mes_training_sample.shape == lvg_training_sample.shape
    assert mes_training_sample.shape == ses_training_sample.shape

    n_firms = mes_training_sample.shape

    data = np.vstack([np.ones(n_firms), mes_training_sample, lvg_training_sample]).T
    betas = np.linalg.lstsq(data, ses_training_sample, rcond=None)[0]
    _, b, c = betas
    ses = (b * mes_firm + c * lvg_firm) / (b + c)
    return ses

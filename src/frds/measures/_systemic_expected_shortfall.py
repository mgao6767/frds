import numpy as np


class SystemicExpectedShortfall:
    """:doc:`/measures/systemic_expected_shortfall`"""

    def __init__(
        self,
        mes_training_sample: np.ndarray,
        lvg_training_sample: np.ndarray,
        ses_training_sample: np.ndarray,
        mes_firm: float,
        lvg_firm: float,
    ) -> None:
        """__init__

        Args:
            mes_training_sample (np.ndarray): ``(n_firms,)`` array of firm ex ante MES.
            lvg_training_sample (np.ndarray): ``(n_firms,)`` array of firm ex ante LVG (say, on the last day of the period of training data)
            ses_training_sample (np.ndarray): ``(n_firms,)`` array of firm ex post cumulative return for date range after `lvg_training_sample`.
            mes_firm (float): The current firm MES used to calculate the firm (fitted) SES value.
            lvg_firm (float): The current firm leverage used to calculate the firm (fitted) SES value.
        """
        assert mes_training_sample.shape == lvg_training_sample.shape
        assert mes_training_sample.shape == ses_training_sample.shape

        self.mes = mes_training_sample
        self.lvg = lvg_training_sample
        self.ses = ses_training_sample
        self.mes_firm = mes_firm
        self.lvg_firm = lvg_firm

    def estimate(self, version="BFLV2012") -> float:
        """estimate

        Args:
            version (str, optional): version of methods. Any of ["BFVL2012", "APPR2017"]. Defaults to "BFLV2012".

        Returns:
            float: The systemic risk that firm :math:`i` poses to the system at a future time.
        """
        assert version in [
            "BFLV2012",
            "APPR2017",
        ]
        if version == "BFLV2012":
            return self._bflv2012()
        if version == "APPR2017":
            raise NotImplementedError

    def _bflv2012(self) -> float:
        n_firms = self.mes.shape

        data = np.vstack([np.ones(n_firms), self.mes, self.lvg]).T
        betas = np.linalg.lstsq(data, self.ses, rcond=None)[0]
        _, b, c = betas
        ses = (b * self.mes_firm + c * self.lvg_firm) / (b + c)
        return ses

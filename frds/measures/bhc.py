import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure

DATASETS = [
    Dataset(
        source="frb_chicago",
        library="bhc",
        table="bhcf",
        vars=[
            "RSSD9001",  # RSSD ID
            "RSSD9999",  # Reporting date
            "BHCK2170",  # Total assets
            "BHCK4059",  # Fee and interest income from loans in foreign offices
            "BHCK4107",  # Total interest income
            "BHCK4340",  # Net income
            "BHCK4460",  # Cash dividends on common stock
            "BHCK3792",  # Total qualifying capital allowable under the risk-based capital guidelines
            "BHCKA223",  # Risk-weighted assets
            "BHCK8274",  # Tier 1 capital allowable under the risk-based capital guidelines
        ],
        date_vars=["RSSD9999"],
    )
]
VARIABLE_LABELS = {
    "BHCSize": "Natural logarithm of total assets (BHCK2170)",
    "BHCFxExposure": "BHCK4059/BHCK4107",
    "RSSD9001": "RSSD ID",
    "RSSD9999": "Reporting date",
    "BHCK2170": "Total assets",
    "BHCK4340": "Net income",
    "BHCK4460": "Cash dividends on common stock",
    "BHCK3792": "Total qualifying capital allowable under the risk-based capital guidelines",
    "BHCK8274": "Tier 1 capital allowable under the risk-based capital guidelines",
    "BHCKA223": "Risk-weighted assets",
}
KEY_VARS = ["RSSD9001", "RSSD9999"]


class BHCSize(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany Size", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = np.log(bhcf["BHCK2170"])
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCFxExposure(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany FX Exposure", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4059 / bhcf.BHCK4107
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCNetIncomeToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany NetIncomeToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4340 / bhcf.BHCK2170
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCDividendToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany DividendToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4460 / bhcf.BHCK2170
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCRegCapToAssets(Measure):
    def __init__(self):
        super().__init__(
            "BankHoldingCompany RegulatoryCapitalToAssets", DATASETS
        )

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK3792 / bhcf.BHCKA223
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCTier1CapToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany Tier1CapitalToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK8274 / bhcf.BHCKA223
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS

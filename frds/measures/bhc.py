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
}
KEY_VARS = ["RSSD9001", "RSSD9999"]


class BHCSize(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany Size", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf["BHCSize"] = np.log(bhcf["BHCK2170"])
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCFxExposure(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany FX Exposure", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf["BHCFxExposure"] = bhcf.BHCK4059 / bhcf.BHCK4107
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS

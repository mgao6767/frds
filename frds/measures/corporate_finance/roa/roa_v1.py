import pandas as pd
from frds.data import Dataset

name = "ROA"
frequency = "Firm-Year"
source = "Compustat"
description = "Net income scaled by total assets"
datasets = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "fyear",
            "at",
            "ib",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    )
]
labels = {name: description}


def estimate(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROA

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset

    Returns
    -------
    pd.DataFrame
        Output dataset with four columns `datadate`, `gvkey`, `fyear` and `ROA`
    """
    data[name] = data["ib"] / data["at"]
    return data[["datadate", "gvkey", "fyear", name]]

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import CorporateFinanceMeasure
from frds.data.utils import filter_funda

NAME = "ExecutiveTenure"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="execcomp",
        table="anncomp",
        vars=["gvkey", "year", "execid", "co_per_rol", "ceoann"],
        date_vars=[],
    ),
]

VARIABLE_LABELS: Dict[str, str] = {
    "execid": "Executive ID from Execucomp",
    "tenure": "Executive tenure",
}


class ExecutiveTenure(CorporateFinanceMeasure):

    url_docs = "https://frds.io/measuers/executive_tenure/"

    def __init__(self):
        super().__init__("Executive Tenure", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        anncomp = pd.DataFrame.from_records(nparrays[0])
        anncomp.sort_values(["co_per_rol", "year"], inplace=True)
        anncomp.drop_duplicates()  # no duplicates found
        anncomp["one"] = 1
        anncomp["tenure"] = anncomp.groupby(["co_per_rol"]).cumsum()["one"]
        anncomp.sort_values(["co_per_rol", "year"], inplace=True)
        return anncomp.drop(columns=["one"]), VARIABLE_LABELS

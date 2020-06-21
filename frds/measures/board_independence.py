from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="boardex",
        table="na_wrds_company_profile",
        vars=["cikcode", "boardid"],
        date_vars=[],
    ),
    Dataset(
        source="wrds",
        library="boardex",
        table="na_wrds_org_composition",
        vars=[
            "companyid",
            "datestartrole",
            "dateendrole",
            "rolename",
            "directorid",
            "seniority",
        ],
        date_vars=["datestartrole", "dateendrole"],
    ),
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "fyear",
            "cik",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    ),
]


VARIABLE_LABELS: Dict[str, str] = {
    "BoardSize": "Number of directors",
    "IndependentMembers": "Number of independent members on board",
    "BoardIndependence": "Ratio of independent board members to board size (%)",
}


class BoardIndependence(Measure):
    def __init__(self, missing_independent_board_members_as_zero=True):
        super().__init__("Board Size and Independence", DATASETS_REQUIRED)
        self._missing_as_zero = missing_independent_board_members_as_zero

    def estimate(
        self, nparrays: List[np.recarray]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        company = pd.DataFrame.from_records(nparrays[0])
        compositon = pd.DataFrame.from_records(nparrays[1])
        funda = pd.DataFrame.from_records(
            filter_funda(nparrays[2]),
            columns=["gvkey", "datadate", "cik", "fyear"],
        )
        tmp = (
            company.merge(
                compositon, left_on=["boardid"], right_on=["companyid"]
            )
            .merge(funda, left_on=["cikcode"], right_on=["cik"])
            .drop_duplicates()
        )
        cond = (
            (
                (tmp.datestartrole <= tmp.datadate)
                & (tmp.datadate <= tmp.dateendrole)
            )
            | (pd.isnull(tmp.dateendrole) & (tmp.datestartrole <= tmp.datadate))
            | (pd.isnull(tmp.datestartrole) & (tmp.datadate <= tmp.dateendrole))
        ) & np.in1d(
            tmp.seniority, ("Executive Director", "Supervisory Director")
        )
        tmp = tmp.where(cond)
        tmp.fillna({"rolename": ""}, inplace=True)
        keys = ["gvkey", "datadate"]
        board_size = (
            tmp.groupby(keys, as_index=False)["directorid"]
            .count()
            .rename(columns={"directorid": "BoardSize"})
        )
        independent_dirs = (
            tmp[tmp.rolename.str.lower().str.contains("independent")]
            .groupby(keys, as_index=False)["directorid"]
            .count()
            .rename(columns={"directorid": "IndependentMembers"})
        )

        result = board_size.merge(independent_dirs, on=keys, how="left")

        if self._missing_as_zero:
            result.fillna({"IndependentMembers": 0}, inplace=True)

        result["BoardIndependence"] = (
            result.IndependentMembers / result.BoardSize * 100
        )

        cols = [*keys, *(VARIABLE_LABELS.keys())]
        result = result[cols].drop_duplicates()

        return result, VARIABLE_LABELS

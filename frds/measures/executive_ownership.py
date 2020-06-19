from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

NAME = "ExecutiveOwnership"
DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "datadate",
            "gvkey",
            "fyear",
            "indfmt",
            "datafmt",
            "popsrc",
            "consol",
            "csho",
        ],
        date_vars=["datadate"],
    ),
    Dataset(
        source="wrds",
        library="execcomp",
        table="anncomp",
        vars=[
            "gvkey",
            "year",
            "execid",
            "shrown_tot",
            "shrown_excl_opts",
            "opt_exer_num",
            "opt_exer_val",
            "shrs_vest_num",
            "shrs_vest_val",
            "tdc1",
        ],
        date_vars=[],
    ),
]

VARIABLE_LABELS: Dict[str, str] = {
    "execid": "Executive ID from Execucomp",
    "ExecSharePct": "Executive share ownership (%)",
    "ExecSharePctExclOpt": "Executive share ownership (%) excluding options",
    "ExecOptPct": "Executive share ownership (%) based on shares acquired on option exercise",
    "ExecShareVestPct": "Executive share ownership (%) based on shared acquired on vesting",
    "ExecIncentivePct": "Value realized on option exercise and vesting scaled by total compensation",
}


class ExecutiveOwnership(Measure):
    """Compute various executive ownership measures"""

    def __init__(self):
        super().__init__("Executive Ownership", DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):

        funda = pd.DataFrame.from_records(filter_funda(nparrays[0]))
        anncomp = pd.DataFrame.from_records(nparrays[1])

        work = anncomp.merge(
            funda, left_on=["gvkey", "year"], right_on=["gvkey", "fyear"]
        )

        # CSHO is in millions and SHROWN_TOT is in thousands
        work["ExecSharePct"] = work.shrown_tot / work.csho / 10
        work["ExecSharePctExclOpt"] = work.shrown_excl_opts / work.csho / 10
        work["ExecOptPct"] = work.opt_exer_num / work.csho / 10
        work["ExecShareVestPct"] = work.shrs_vest_num / work.csho / 10
        work["ExecIncentivePct"] = (
            (work.opt_exer_val + work.shrs_vest_num) / work.tdc1 / 10
        )
        # Replace infinity with nan
        work["ExecSharePctExclOpt"].replace(np.inf, np.nan, inplace=True)
        work["ExecOptPct"].replace(np.inf, np.nan, inplace=True)
        work["ExecShareVestPct"].replace(np.inf, np.nan, inplace=True)
        work["ExecIncentivePct"].replace(np.inf, np.nan, inplace=True)
        keys = ["gvkey", "datadate"]
        cols = [*keys, *(VARIABLE_LABELS.keys())]
        result = work[cols].drop_duplicates().sort_values(by=keys)
        return result, VARIABLE_LABELS

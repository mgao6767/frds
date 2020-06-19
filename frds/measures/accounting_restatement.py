"""
This measure counts the numer of restatements during the past fiscal year(s).

Data Souces
-----------
WRDS Compustat Fundamentals Annual, AuditAnalytics Non-Reliance Restatement

Author
------
Mingze Gao, 16 June 2020
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure
from frds.data.utils import filter_funda

DATASETS_REQUIRED: List[Dataset] = [
    Dataset(
        source="wrds",
        library="audit",
        table="auditnonreli",
        vars=[
            "company_fkey",  # EDGAR CIK
            "file_date",  # Filing date
            "res_notif_key",  # Restatement notification key
            "res_accounting",  # Restatement accounting
            "res_adverse",  # Restatement adverse
            "res_fraud",  # Restatement fraud
            "res_cler_err",  # Restatement clerical errors
            "res_sec_invest",  # Restatement SEC investigation
        ],
        date_vars=["file_date"],
    ),
    Dataset(
        source="wrds",
        library="comp",
        table="funda",
        vars=[
            "gvkey",
            "datadate",
            "cik",
            "datafmt",
            "indfmt",
            "popsrc",
            "consol",
        ],
        date_vars=["datadate"],
    ),
]
"""List[Dataset]: list of `frds.data.Dataset` required for this measure"""

VARIABLE_LABELS: Dict[str, str] = {
    "NumResAcct": "Numer of restatements (accounting) in the fiscal year",
    "NumResFraud": "Numer of restatements (fraud) in the fiscal year",
    "NumResAdver": "Numer of restatements (adverse) in the fiscal year",
    "NumResClerErr": "Numer of restatements (clerical errors) in the fiscal year",
    "NumResSECInvest": "Numer of restatements (SEC investigation) in the fiscal year",
}
"""Dict[str, str]: dict of the variable labels for the output Stata file"""


class AccountingRestatement(Measure):
    """Counts the numer of restatements during the past fiscal year"""

    def __init__(self, years=1):
        """Set numer of fiscal years to consider

        Parameters
        ----------
        years : int, optional
            Number of past fiscal years to consider, by default 1
        """
        super().__init__(
            name="Accounting Restatement", datasets_required=DATASETS_REQUIRED
        )
        self._years = int(years)

    def __str__(self):
        """Return the measure's full name, including the parameters used

        Returns
        -------
        str
            Measure's full name, including the parameters (years used)
        """
        return f"{self.name} years={self._years}"

    def estimate(
        self, nparrays: List[np.recarray]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Counts the numer of restatements during the past fiscal year(s)

        Parameters
        ----------
        nparrays : List[np.recarray]
            Input datasets, same order as specified in `DATASETS_REQUIRED`

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, str]]
            Output dataset and the `VARIABLE_LABELS`
        """
        # Note that the order of dataset is preseved
        rests = pd.DataFrame.from_records(nparrays[0])
        funda = pd.DataFrame.from_records(filter_funda(nparrays[1]))
        # Inner join funda and restatements
        work = funda.merge(rests, left_on="cik", right_on="company_fkey")
        # Filing date must be in the past year(s) relative the datadate
        time_delta = work.datadate - work.file_date
        years = np.timedelta64(self._years, "Y")
        cond = (time_delta < years) & (work.datadate >= work.file_date)
        # Forget about the non-matched
        cond = cond & (work.cik != "")
        # Apply the filtering condition
        work.where(cond, inplace=True)
        # Count by gvkey and datadate/fyear
        keys = ["gvkey", "datadate"]
        work = work.groupby((keys), as_index=False).sum()
        # Rename columns to match output variable labels
        work.rename(
            columns={
                "res_accounting": "NumResAcct",
                "res_fraud": "NumResFraud",
                "res_adverse": "NumResAdver",
                "res_cler_err": "NumResClerErr",
                "res_sec_invest": "NumResSECInvest",
            },
            inplace=True,
        )
        # Left join with funda so to retain the missing values
        result = funda[keys].merge(work, how="left", on=keys, copy=False)
        # Keep only useful columns
        cols = [*keys, *(VARIABLE_LABELS.keys())]
        # Some cosmetic issues
        result = result[cols].drop_duplicates().sort_values(by=keys)

        return result, VARIABLE_LABELS

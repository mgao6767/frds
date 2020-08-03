import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure

DATASETS = [
    Dataset(
        source="wrds",
        library="ciq",
        table="wrds_erating",
        vars=["company_id", "rdate", "rtime", "rating", "rtype"],
        date_vars=["rdate"],
    ),
    Dataset(
        source="wrds",
        library="ciq",
        table="wrds_gvkey",
        vars=["gvkey", "companyid", "startdate", "enddate"],
        date_vars=["startdate", "enddate"],
    ),
]
VARIABLE_LABELS = {
    "rdate": "Rating date",
    "rating_rank": "1 represents a AAA rating and 22 reflects a D rating.",
}


class CreditRating(Measure):
    def __init__(self):
        super().__init__("CreditRating", DATASETS)

    def estimate(self, nparrays):
        ratings = pd.DataFrame.from_records(nparrays[0])
        gvkeys = pd.DataFrame.from_records(nparrays[1])

        tmp = ratings.merge(
            gvkeys, left_on=["company_id"], right_on=["companyid"]
        ).drop_duplicates()
        cond = (
            (pd.isnull(tmp.enddate) | (tmp.rdate <= tmp.enddate))
            & (pd.isnull(tmp.startdate) | (tmp.rdate >= tmp.startdate))
        ) & np.in1d(tmp.rtype, ("Local Currency LT"))
        tmp = tmp.where(cond)

        tmp["maxrtime"] = tmp.groupby(["company_id", "rdate"])[
            "rtime"
        ].transform(max)
        tmp = tmp.where(tmp.maxrtime == tmp.rtime)
        tmp = tmp[["gvkey", "companyid", "rdate", "rtype", "rating"]]

        ratings_to_num = {
            "AAA": 1,
            "AA+": 2,
            "AA": 3,
            "AA-": 4,
            "A+": 5,
            "A": 6,
            "A-": 7,
            "BBB+": 8,
            "BBB": 9,
            "BBB-": 10,
            "BB+": 11,
            "BB": 12,
            "BB-": 13,
            "B+": 14,
            "B": 15,
            "B-": 16,
            "CCC+": 17,
            "CCC": 18,
            "CCC-": 19,
            "CC": 20,
            "C": 21,
            "D": 22,
        }
        tmp["rating_rank"] = tmp.rating.map(ratings_to_num)
        return tmp.dropna(subset=["gvkey"]), VARIABLE_LABELS

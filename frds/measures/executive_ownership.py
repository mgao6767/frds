import pandas as pd
import numpy as np
from typing import List
from ..data import Dataset

name = 'ExecutiveOwnership'
description = "Executive ownership"
datasets = [
    Dataset(source='wrds', library='comp', table='funda',
            vars=[
                'datadate', 'gvkey', 'fyear', 'indfmt', 'datafmt', 'popsrc',
                'consol', 'csho'],
            date_vars=['datadate']),
    Dataset(source='wrds', library='execcomp', table='anncomp',
            vars=["gvkey", "year", "execid", "shrown_tot", "shrown_excl_opts",
                  "opt_exer_num", "opt_exer_val", "shrs_vest_num",
                  "shrs_vest_val", "tdc1"],
            date_vars=[])
]
variable_labels = {
    'execid': 'Executive ID from Execucomp',
    'ExecSharePct': 'Executive share ownership (%)',
    'ExecSharePctExclOpt': 'Executive share ownership (%) excluding options',
    'ExecOptPct': 'Executive share ownership (%) based on shares acquired on option exercise',
    'ExecShareVestPct': 'Executive share ownership (%) based on shared acquired on vesting',
    'ExecIncentivePct': 'Value realized on option exercise and vesting scaled by total compensation'
}


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x): return x[
        np.in1d(x.datafmt, ('STD')) &
        np.in1d(x.indfmt, ('INDL')) &
        np.in1d(x.popsrc, ('D')) &
        np.in1d(x.consol, ('C'))
    ]

    funda = pd.DataFrame.from_records(filter_funda(nparrays[0]))
    anncomp = pd.DataFrame.from_records(nparrays[1])

    work = anncomp.merge(funda, left_on=['gvkey', 'year'],
                         right_on=['gvkey', 'fyear'])

    # CSHO is in millions and SHROWN_TOT is in thousands
    work['ExecSharePct'] = work.shrown_tot / work.csho / 10
    work['ExecSharePctExclOpt'] = work.shrown_excl_opts / work.csho / 10
    work['ExecOptPct'] = work.opt_exer_num / work.csho / 10
    work['ExecShareVestPct'] = work.shrs_vest_num / work.csho / 10
    work['ExecIncentivePct'] = (
        work.opt_exer_val + work.shrs_vest_num) / work.tdc1 / 10
    # Replace infinity with nan
    work['ExecSharePctExclOpt'].replace(np.inf, np.nan, inplace=True)
    work['ExecOptPct'].replace(np.inf, np.nan, inplace=True)
    work['ExecShareVestPct'].replace(np.inf, np.nan, inplace=True)
    work['ExecIncentivePct'].replace(np.inf, np.nan, inplace=True)
    keys = ['gvkey', 'datadate']
    cols = [*keys, *(variable_labels.keys())]
    result = work[cols].drop_duplicates().sort_values(by=keys)
    return result, variable_labels

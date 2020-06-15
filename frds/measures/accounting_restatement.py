import pandas as pd
import numpy as np
from typing import List
from ..data import Dataset

name = 'AccountingRestatement'
description = """Number of restatements in the past fiscal year by merging WRDS
AuditAnalytics Non-Reliance Restatement and Compustat Fundamentals Annual"""
datasets = [
    Dataset(source='wrds', library='audit',
            table='auditnonreli',
            vars=[
                'company_fkey',         # EDGAR CIK
                'file_date',            # Filing date
                'res_notif_key',        # Restatement notification key
                'res_accounting',       # Restatement accounting
                'res_adverse',          # Restatement adverse
                'res_fraud',            # Restatement fraud
                'res_cler_err',         # Restatement clerical errors
                'res_sec_invest',       # Restatement SEC investigation
            ],
            date_vars=['file_date']),
    Dataset(source='wrds', library='comp',
            table='funda',
            vars=[
                'gvkey', 'datadate', 'cik', 'datafmt', 'indfmt',
                'popsrc', 'consol'
            ],
            date_vars=['datadate'])
]
variable_labels = {
    'NumResAcct': 'Numer of restatements (accounting) in the fiscal year',
    'NumResFraud': 'Numer of restatements (fraud) in the fiscal year',
    'NumResAdver': 'Numer of restatements (adverse) in the fiscal year',
    'NumResClerErr': 'Numer of restatements (clerical errors) in the fiscal year',
    'NumResSECInvest': 'Numer of restatements (SEC investigation) in the fiscal year',
}
note = """
    This meathod is equivalent to the following SAS code:
    proc sql;
    create table AccountingRestatement as
		select distinct a.gvkey, a.datadate, 
			sum(b.res_accounting)   as NumResAcct,
			sum(b.res_fraud)        as NumResFraud,
			sum(b.res_adverse)      as NumResAdver,
            sum(b.res_cler_err)     as NumResClerErr,
			sum(b.res_sec_invest)   as NumResSECInvest 
		from 
            comp.funda(where=(indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C')) as a 
			left join audit.auditnonreli as b
		on a.cik=b.company_fkey
			and b.file_date>intnx('month', a.datadate, -12, 'S') and b.file_date<=a.datadate
		group by a.gvkey, a.datadate
		order by a.gvkey, a.datadate;
    quit;
"""


def estimate(nparrays: List[np.recarray]):

    def filter_funda(x):
        return x[
            np.in1d(x.datafmt, ('STD')) &
            np.in1d(x.indfmt, ('INDL')) &
            np.in1d(x.popsrc, ('D')) &
            np.in1d(x.consol, ('C'))
        ]

    # Note that the order of dataset is preseved
    rests = pd.DataFrame.from_records(nparrays[0])
    funda = pd.DataFrame.from_records(filter_funda(nparrays[1]))
    # Inner join funda and restatements
    work = funda.merge(rests, left_on='cik', right_on='company_fkey')
    # Filing date must be in the past year relative the datadate
    time_delta = work.datadate - work.file_date
    one_year = np.timedelta64(1, 'Y')
    cond = (time_delta < one_year) & (work.datadate >= work.file_date)
    # Forget about the non-matched
    cond = cond & (work.cik != '')
    # Apply the filtering condition
    work.where(cond, inplace=True)
    # Count by gvkey and datadate/fyear
    keys = ['gvkey', 'datadate']
    work = work.groupby(keys, as_index=False).sum()
    work.rename(columns={
        'res_accounting': 'NumResAcct',
        'res_fraud': 'NumResFraud',
        'res_adverse': 'NumResAdver',
        'res_cler_err': 'NumResClerErr',
        'res_sec_invest': 'NumResSECInvest'
    }, inplace=True)
    # Left join with funda so to retain the missing values
    result = funda[keys].merge(work, how='left', on=keys, copy=False)
    # Keep only useful columns
    cols = [*keys, *(variable_labels.keys())]
    # Some cosmetic issues
    result = result[cols].drop_duplicates().sort_values(by=keys)
    return result, variable_labels

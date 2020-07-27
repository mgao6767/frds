---
path: tree/master/frds
source: measures/accounting_restatement.py
---

# Accounting Restatement

## Definition

Restating previous financial statements is a signal worth attention and investigation. This measure counts the number of restatements in the past fiscal year (configurable to arbitrary number of years) due to various reasons based on the WRDS AuditAnalytics Non-Reliance Restatement dataset.

Specifically, the following variables from `WRDS.AUDIT.AUDITNONRELI` are used:

| Variable         | Description                               |
|------------------|-------------------------------------------|
| `company_fkey`   | EDGAR CIK                                 |
| `file_date`      | Filing date                               |
| `res_notif_key`  | Restatement notification key              |
| `res_accounting` | Restatement due to accounting issues      |
| `res_adverse`    | Restatement had an adverse impact         |
| `res_fraud`      | Restatement related to fraud              |
| `res_cler_err`   | Restatement due to clerical errors        |
| `res_sec_invest` | Restatement followed by SEC investigation |

The dataset is merged with `WRDS.COMP.FUNDA` on `CIK`, grouped by `gvkey` and `datadate`. Then the frequency of each restatement type is counted.

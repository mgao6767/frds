---
path: tree/master/frds
source: measures/bhc.py
---

# BHC Loan Growth

## Definition

The natural logarithm of a bank's total loans in the current quarter, divided by its total loans in the previous quarter.[^zheng] Here the total loans are measures by `BHCK2122` in FR Y-9C reports. This item represents the proportion of a bank's total loans that do not exclude the allowance for loans and lease losses.

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

$$
\text{LoanGrowth}_{i,t} = \ln \left( \frac{\text{BHCK2122}_{i,t}}{\text{BHCK2122}_{i,t-1}} \right)
$$

where `BHCK2122` is from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

!!! warning
    A caveat is that if $\text{BHCK2122}_{i,t}$ is 0, then this measure would be invalid as $\ln(0)$ is undefined. In this case, I replace it with -1 which means the total loans is reduced by 100% to 0.

An alternative measure of loan growth, the percentage change in the total loans, is not affected by this issue.

$$
\text{LoanGrowthPct}_{i,t} = \left( \frac{\text{BHCK2122}_{i,t}}{\text{BHCK2122}_{i,t-1}} -1\right) \times 100
$$

Their correlation is over 99%.

## Equivalent Stata Code

```stata
use "~/frds/result/BankHoldingCompany LoanGrowth.dta", clear
gen qtr = qofd(RSSD9999)
format qtr %tq
xtset RSSD9001 qtr, quarterly
gen BHCLoanGrowthPct = (BHCK2122 / L.BHCK2122 - 1) * 100
```


## Reference

[^zheng]: This definition is from [Zheng (2020 JBF)](https://doi.org/10.1016/j.jbankfin.2020.105900).
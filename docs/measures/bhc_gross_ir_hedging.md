---
path: tree/master/frds
source: measures/bhc.py
---

# BHC Gross Interest Rate Hedging

## Definition

Total gross notional amount of interest rate derivatives held for purposes other than trading `BHCK8725` over total assets `BHCK2170`; for the period 1995 to 2000, contracts not marked to market `BHCK8729` are added.

If $t \in [1995, 2000]$:

$$
\text{Gross IR Hedging}_{i,t}=\frac{\text{BHCK8725}_{i,t}+\text{BHCK8729}_{i,t}}{\text{BHCK2170}_{i,t}}
$$

Otherwise:

$$
\text{Gross IR Hedging}_{i,t}=\frac{\text{BHCK8725}_{i,t}}{\text{BHCK2170}_{i,t}}
$$

where `BHCK8725`, `BHCK8729` and `BHCK4107` are from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

## Reference

[Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
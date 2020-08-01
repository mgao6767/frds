---
path: tree/master/frds
source: measures/bhc.py
---

# BHC Regulatory Capital / Assets

## Definition

Total qualifying capital allowable under the risk-based capital guidelines `BHCK3792` normalized by risk-weighted assets `BHCKA223`.

$$
\text{RegCap/Assets}_{i,t}=\frac{\text{BHCK3792}_{i,t}}{\text{BHCKA223}_{i,t}}
$$

where `BHCK3792` and `BHCKA223` are from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

## Reference

[Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
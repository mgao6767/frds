---
path: tree/master/frds
source: measures/bhc.py
---

# BHC FX Exposure

## Definition

Fee and interest income from loans in foreign offices `BHCK4059` scaled by total interest income `BHCK4107`.

$$
\text{FX Exposure}_{i,t}=\frac{\text{BHCK4059}_{i,t}}{\text{BHCK4107}_{i,t}}
$$

where `BHCK4059` and `BHCK4107` are from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

## Reference

[Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
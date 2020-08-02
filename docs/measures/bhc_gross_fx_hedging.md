---
path: tree/master/frds
source: measures/bhc.py
---

# BHC Gross Foreign Exchange Rate Hedging

## Definition

Total gross notional amount of foreign exchange rate derivatives held for purposes other than trading `BHCK8726` over total assets `BHCK2170`; for the period 1995 to 2000, contracts not marked to market `BHCK8730` are added.

If $t \in [1995, 2000]$:

$$
\text{Gross FX Hedging}_{i,t}=\frac{\text{BHCK8726}_{i,t}+\text{BHCK8730}_{i,t}}{\text{BHCK2170}_{i,t}}
$$

Otherwise:

$$
\text{Gross FX Hedging}_{i,t}=\frac{\text{BHCK8726}_{i,t}}{\text{BHCK2170}_{i,t}}
$$

where `BHCK8726`, `BHCK8730` and `BHCK4107` are from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

## Reference

[Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
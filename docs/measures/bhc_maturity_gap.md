---
path: tree/master/frds
source: measures/bhc.py
---

# BHC Maturity Gap

## Definition

### Maturity Gap

Earning assets that are repriceable or mature within one year `BHCK3197` minus interest-bearing deposits that mature or reprice within one year `BHCK3296` minus long-term debt that reprices or matures within one year `BHCK3298 + BHCK3409` minus variable rate preferred stock `BHCK3408` minus other borrowed money with a maturity of one year or less `BHCK2332` minus commercial paper `BHCK2309` minus federal funds and repo liabilities `BHDMB993 + BHCKB995`, normalized by total assets `BHCK2170`, where all variables are from the Bank Holding Company data by the Federal Reserve Bank of Chicago.[^1] 

### Narrow Maturity Gap

Narrow maturity gap is defined similarly to Maturity Gap, except that it does not subtract interest-bearing deposits that mature or reprice within one year `BHCK3296`.

[^1]: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data.

## Reference

[Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
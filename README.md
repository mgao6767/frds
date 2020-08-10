![frds](https://github.com/mgao6767/frds/raw/master/images/frds_logo.png)

# FRDS - Financial Research Data Services
![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![PyPIDownloads](https://img.shields.io/pypi/dm/frds) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[`frds`](https://github.com/mgao6767/frds/) is a Python framework that aims to provide the simplest way to compute [a collection of major academic measures](#supported-measures) used in the finance literature, one-click with a Graphical User Interface (GUI).

![GUI](https://github.com/mgao6767/frds/raw/master/images/frds_demo.gif)

## Supported Measures

The built-in measures currently supported by `frds` are as below. New measures will be added to `frds` gradually and can be easily developed following a template shown in [Developing New Measures](#developing-new-measures).

### Firm Characteristics

* [Accounting Restatements](https://frds.io/measures/accounting_restatement)
    * Number of various accounting restatements during the past (*n*) fiscal year.
    * Source: `wrds.comp.funda`, `wrds.audit.auditnonreli`. 
* [Asset Tangibility](https://frds.io/measures/asset_tangibility) 
    * Property, Plant and Equipment (Net) scaled by total assets.
    * Source: `wrds.comp.funda`.
* [Board Independence](https://frds.io/measures/board_independence)
    * Board size and independence measured as the ratio of independent board members to board size.
    * Source: `wrds.funda`, `wrds.boardex.na_wrds_company_profile`, `wrds.boardex.na_wrds_org_composition`.
* [Book Leverage](https://frds.io/measures/book_leverage)
    * Amount of debts scaled by the firm's total debts plus common equity.
    * Source: `wrds.comp.funda`.
* [Capital Expenditure](https://frds.io/measures/capital_expenditure)
    * Capital expenditures scaled by total assets.
    * Source: `wrds.comp.funda`.
* [Credit Rating](https://frds.io/measures/credit_rating)
    * S&P credit rating.
    * Source: `wrds.ciq.erating`, `wrds.ciq.gvkey`.
* [Executive Ownership](https://frds.io/measures/executive_ownership)
    * Various measures of executive stock ownership.
    * Source: `wrds.comp.funda`, `wrds.execcomp.anncomp`.
* [Firm Size](https://frds.io/measures/firm_size)
    * Natural logarithm of total assets.
    * Source: `wrds.comp.funda`.
* [Market-to-Book Ratio](https://frds.io/measures/market_to_book)
    * Market value of common equity to book value of common equity.
    * Source: `wrds.comp.funda`.
* [ROA](https://frds.io/measures/roa)
    * Income before extraordinary items scaled by total assets.
    * Source: `wrds.comp.funda`.
* [ROE](https://frds.io/measures/roe)
    * Income before extraordinary items scaled by common equity.
    * Source: `wrds.comp.funda`.
* [Stock Delisting](https://frds.io/measures/stock_delisting)
    * Stocks delisted due to financial troubles or as a result of being merged.
    * Source: `wrds.crsp.dse`.

### Bank Holding Company (BHC) Characteristics

* [BHC Size](https://frds.io/measures/bhc_size)
    * Natural logarithm of total assets.
    * Source: `frb_chicago.bhc.bhcf`.
* [BHC Loan Growth](https://frds.io/measures/bhc_loan_growth)
    * Natural logarithm of total loans in the current quarter divided by the total loans in the previous quarter.
    * Source: `frb_chicago.bhc.bhcf`.
    * Referece: [Zheng (2020 JBF)](https://doi.org/10.1016/j.jbankfin.2020.105900).
* [BHC FX Exposure](https://frds.io/measures/bhc_fx_exposure)
    * Fee and interest income from loans in foreign offices (BHCK4059) scaled by total interest income (BHCK4107).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC NetIncome/Assets](https://frds.io/measures/bhc_netincome_to_assets)
    * Net income (BHCK4340) / total assets (BHCK2170).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Dividend/Assets](https://frds.io/measures/bhc_dividend_to_assets)
    * Cash dividends on common stock (BHCK4460) / total assets (BHCK2170).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC RegulatoryCapital/Assets](https://frds.io/measures/bhc_regcap_to_assets)
    * Total qualifying capital allowable under the risk-based capital guidelines (BHCK3792) normalized by risk-weighted assets (BHCKA223).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Tier1Capital/Assets](https://frds.io/measures/bhc_tier1cap_to_assets)
    * Tier 1 capital allowable under the risk-based capital guidelines (BHCK8274) normalized by risk-weighted assets (BHCKA223).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Gross IR Hedging](https://frds.io/measures/bhc_gross_ir_hedging)
    * Total gross notional amount of interest rate derivatives held for purposes other than trading (BHCK8725) over total assets (BHCK2170); for the period 1995 to 2000, contracts not marked to market (BHCK8729) are added.
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Gross FX Hedging](https://frds.io/measures/bhc_gross_fx_hedging)
    * Total gross notional amount of foreign exchange rate derivatives held for purposes other than trading (BHCK8726) over total assets (BHCK2170); for the period 1995 to 2000, contracts not marked to market (BHCK8730) are added.
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Maturity Gap & Narrow Maturity Gap](https://frds.io/measures/bhc_maturity_gap)
    * Maturity gap is defined as the earning assets that are repriceable or mature within one year (BHCK3197) minus interest-bearing deposits that mature or reprice within one year (BHCK3296) minus long-term debt that reprices or matures within one year (BHCK3298 + BHCK3409) minus variable rate preferred stock (BHCK3408) minus other borrowed money with a maturity of one year or less (BHCK2332) minus commercial paper (BHCK2309) minus federal funds and repo liabilities (BHDMB993 + BHCKB995), normalized by total assets.
    * Narrow maturity gap does not subtract interest-bearing deposits that mature or reprice within one year (BHCK3296).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).

## Installation & Configuration

[`frds`](https://github.com/mgao6767/frds/) requires Python3.8 or higher. To install using `pip`:

```bash
$ pip install frds
```
After installation, a folder `frds` will be created under your user's home directory, which contains a `data` folder, a `result` folder and a default configuration file `config.ini`:

```ini
[Paths]
base_dir: ~/frds
data_dir: ${base_dir}/data
result_dir: ${base_dir}/result

[Login]
wrds_username: 
wrds_password: 
```

You need to enter your WRDS username and password under the login section.

## Usage

To start estimating various measures, run `frds` as a module:

```bash
$ python -m frds.gui.run
```

Alternatively, run without GUI to estimate all measures with default parameters:

```bash
$ python -m frds.run
```

You can also use the following example to use `frds` programmatically.

```python
from frds import Professor
from frds.measures import AccountingRestatement, ROA, FirmSize

measures = [
    AccountingRestatement(years=1),
    AccountingRestatement(years=2),
    ROA(),
    FirmSize(),
]

config = dict(
        wrds_username="your_wrds_username",
        wrds_password="you_wrds_password",
        result_dir="path/to/where/you/want/to/store/the/result/",
        data_dir="path/to/where/you/want/to/store/the/data/",
    )

if __name__ == "__main__":
    with Professor(config=config) as prof:
        prof.calculate(measures)
```

The output data will be saved as STATA `.dta` file in the `result` folder.

For example, below is a screenshot of the output for `frds.measures.AccountingRestatement`.

![result](https://github.com/mgao6767/frds/raw/master/images/result-restatements.png)


## Developing New Measures

New measures can be easily added by subclassing `frds.measures.Measure` and 
implement the `estimate` function, as shown in the template below. The best working example would be [`frds.measures.ROA`](https://github.com/mgao6767/frds/blob/master/frds/measures/roa.py).

```python
from typing import List
import numpy as np
import pandas as pd
from frds.measures import Measure
from frds.data import Dataset

DATASETS_REQUIRED = [
    Dataset(source='wrds',
            library="comp",
            table="funda",
            vars=[
                "datadate",
                "gvkey",
                "at",
                "ib",
            ],
            date_vars=["datadate"],
    )
]
VARIABLE_LABELS = {}

class NewMeasure(Measure):
    
    def __init__(self):
        # Note: `name` will be used to name the result dataset. 
        # So this will lead to a `New Measure.dta` in the `result` folder.
        # If the new measure contains custom parameters, please also implement
        # the `__str__()` function to differentiate
        super().__init__(name="New Measure", datasets_required=DATASETS_REQUIRED)

    def estimate(self, nparrays: List[np.recarray]):
        # do something with nparrays and produce a `result` pandas DataFrame
        # ...
        assert isinstance(result, pd.DataFrame)
        return result, VARIABLE_LABELS
```

Then to estimate `NewMeasure`:

```python
from frds import Professor

measures = [
    NewMeasure(),
]

config = dict(
        wrds_username="your_wrds_username",
        wrds_password="you_wrds_password",
        result_dir="path/to/where/you/want/to/store/the/result/",
        data_dir="path/to/where/you/want/to/store/the/data/",
    )

if __name__ == "__main__":
    with Professor(config=config) as prof:
        prof.calculate(measures)
```

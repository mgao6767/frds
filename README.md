![frds](https://github.com/mgao6767/frds/raw/master/images/frds_logo.png)

# FRDS - Financial Research Data Services
![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) ![PyPIDownloads](https://img.shields.io/pypi/dm/frds) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[`frds`](https://github.com/mgao6767/frds/) is a Python framework that aims to provide the simplest way to compute [a collection of major academic measures](#supported-measures) used in the finance literature, one-click with a Graphical User Interface (GUI).

![GUI](https://github.com/mgao6767/frds/raw/master/images/frds_demo.gif)

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

For example, below is a screenshot of the output for `frds.measures.asset_tangibility` (new versions also have variable labels).

![result-tangibility](https://github.com/mgao6767/frds/raw/master/images/result-tangibility.png)

## Supported Measures

The built-in measures currently supported by `frds` are as below. New measures will be added to `frds` gradually and can be easily developed following a template shown in [Developing New Measures](#developing-new-measures).
| Measure                                                                                                        | Description                                                                                                     | Datasets Used                            |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------|
| [Accounting Restatement](https://github.com/mgao6767/frds/blob/master/frds/measures/accounting_restatement.py) | Number of various accounting restatements during the fiscal year                                                | wrds.comp.funda, wrds.audit.auditnonreli |
| [Asset Tangibility](https://github.com/mgao6767/frds/blob/master/frds/measures/asset_tangibility.py)           | Property, Plant and Equipment (Net) scaled by Assets (Total)                                                    | wrds.comp.funda                          |
| [Book Leverage](https://github.com/mgao6767/frds/blob/master/frds/measures/book_leverage.py)                   | (Long-term Debt + Debt in Current Liabilities) / (Long-term Debt + Debt in Current Liabilities + Common Equity) | wrds.comp.funda                          |
| [Capital Expenditure](https://github.com/mgao6767/frds/blob/master/frds/measures/capital_expenditure.py)       | Capital Expenditures scaled by Assets (Total)                                                                   | wrds.comp.funda                          |
| [Executive Ownership](https://github.com/mgao6767/frds/blob/master/frds/measures/executive_ownership.py)       | Executive stock ownership                                                                                       | wrds.comp.funda, wrds.execcomp.anncomp   |
| [Firm Size](https://github.com/mgao6767/frds/blob/master/frds/measures/firm_size.py)                           | Natural logarithm of total assets                                                                               | wrds.comp.funda                          |
| [Market-to-Book ratio](https://github.com/mgao6767/frds/blob/master/frds/measures/market_to_book.py)           | Market Value of Common Equity to Book Common Equity                                                             | wrds.comp.funda                          |
| [ROA](https://github.com/mgao6767/frds/blob/master/frds/measures/roa.py)                                       | Income Before Extraordinary Items scaled by Assets (Total)                                                      | wrds.comp.funda                          |
| [ROE](https://github.com/mgao6767/frds/blob/master/frds/measures/roe.py)                                       | Income Before Extraordinary Items scaled by Common Equity (Total)                                               | wrds.comp.funda                          |


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
        return result, variable_labels
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

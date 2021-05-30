![frds](https://github.com/mgao6767/frds/raw/master/images/frds_logo.png)

# FRDS - Financial Research Data Services
![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=blue) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[`frds`](https://github.com/mgao6767/frds/) aims to provide the simplest way to compute [a collection of major academic measures](#supported-measures) used in the finance literature.

[Getting started](https://frds.io/getting-started/) by checking out this notebook.

## Example usage

### Import
We start by importing relevant modules.

Specifically, we import the `Funda` class from the `frds.data.wrds.comp` library since the demo uses only the Fundamentals Annual dataset from Compustat via WRDS. We next import the `setup` and `load` functions from `frds.io.wrds`, which are used to configure WRDS credentials and data management for WRDS datasets.

```Python
from frds.data.wrds.comp import Funda
from frds.io.wrds import setup, load
```

### (Optional) Setup

Then, set WRDS credentials in case later we need to download from WRDS.

```Python
setup(username='username', password='password', save_credentials=True)
```

### Load data
We now download the `Funda` (Fundamentals Annual) dataset and assign it to the variable `FUNDA`.

```Python
FUNDA = load(Funda, use_cache=True, obs=100)
```

### Compute

Let's now compute a few metrics to showcase how easy it is.

```Python
import numpy as np
import pandas as pd
from frds.measures.corporate import roa

pd.DataFrame(
    {
        # We can calculate metrics on the go
        "Fyear": FUNDA.FYEAR,
        "Tangibility": FUNDA.PPENT / FUNDA.AT,
        "Firm_Size": np.log(FUNDA.AT),
        "MTB": FUNDA.PRCC_F * FUNDA.CSHO / FUNDA.CEQ,
        # Or we can use the built-in measures available in FRDS:
        "ROA_v1": roa(FUNDA),
        "ROA_v2": roa(FUNDA, use_lagged_total_assets=True)
    }
).dropna().head(10)
```

The result would be a nice `pd.DataFrame`:

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Fyear</th>
      <th>Tangibility</th>
      <th>Firm_Size</th>
      <th>MTB</th>
      <th>ROA_v1</th>
      <th>ROA_v2</th>
    </tr>
    <tr>
      <th>GVKEY</th>
      <th>DATADATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">001000</th>
      <th>1970-12-31</th>
      <td>1970</td>
      <td>0.265351</td>
      <td>3.510052</td>
      <td>2.319803</td>
      <td>0.056143</td>
      <td>0.065408</td>
    </tr>
    <tr>
      <th>1971-12-31</th>
      <td>1971</td>
      <td>0.260450</td>
      <td>3.378611</td>
      <td>2.054797</td>
      <td>0.004705</td>
      <td>0.004126</td>
    </tr>
    <tr>
      <th>1976-12-31</th>
      <td>1976</td>
      <td>0.426061</td>
      <td>3.652890</td>
      <td>0.899635</td>
      <td>0.088996</td>
      <td>0.947310</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">001001</th>
      <th>1984-12-31</th>
      <td>1984</td>
      <td>0.781644</td>
      <td>2.789139</td>
      <td>1.492970</td>
      <td>0.069958</td>
      <td>0.080441</td>
    </tr>
    <tr>
      <th>1985-12-31</th>
      <td>1985</td>
      <td>0.567439</td>
      <td>3.676174</td>
      <td>3.102697</td>
      <td>0.065223</td>
      <td>0.158357</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">001002</th>
      <th>1970-12-31</th>
      <td>1970</td>
      <td>0.181825</td>
      <td>2.619000</td>
      <td>0.499715</td>
      <td>0.035490</td>
      <td>0.032331</td>
    </tr>
    <tr>
      <th>1971-12-31</th>
      <td>1971</td>
      <td>0.207127</td>
      <td>2.495104</td>
      <td>0.827517</td>
      <td>0.065660</td>
      <td>0.058009</td>
    </tr>
    <tr>
      <th>1972-12-31</th>
      <td>1972</td>
      <td>0.166369</td>
      <td>2.752131</td>
      <td>0.561460</td>
      <td>0.057285</td>
      <td>0.074074</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">001003</th>
      <th>1983-12-31</th>
      <td>1983</td>
      <td>0.030015</td>
      <td>2.143472</td>
      <td>2.311034</td>
      <td>0.123109</td>
      <td>0.186435</td>
    </tr>
    <tr>
      <th>1984-12-31</th>
      <td>1984</td>
      <td>0.051450</td>
      <td>2.109122</td>
      <td>1.138268</td>
      <td>0.046960</td>
      <td>0.138214</td>
    </tr>
  </tbody>
</table>

## Built-in Measures

Check the [built-in measures and documentation](https://frds.io/api/measures/).

## Note

This library is still under development and breaking changes may be expected.
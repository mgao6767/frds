# Working with WRDS Data

## Optional Setup for WRDS

`frds` primarily uses WRDS to obtain data, so it requires WRDS login credentials. The setup is done by `frds.io.wrds.setup`:

```python
from frds.io.wrds import setup

setup(username='username', password='password', save_credentials=False)
```

If `save_credentials=True`, the username and password will be saved locally in `credentials.json` in the `frds` folder. As such, this step is no longer required in later uses. Otherwise, the login credentials are only valid for the current Python session.

## Usage

### Download data from WRDS

As an example, let's say we want to download the Compustat Fundamentals Annual dataset.

``` python
>>> from frds.data.wrds.comp import Funda
>>> from frds.io.wrds import load # (1)
>>> FUNDA = load(Funda, use_cache=True, obs=100) # (2)
>>> FUNDA.data.head()
                                    FYEAR INDFMT CONSOL POPSRC DATAFMT   TIC      CUSIP                   CONM  ... PRCL_F   ADJEX_F RANK    AU  AUOP  AUOPIC CEOSO CFOSO
GVKEY  DATADATE                                                                                                 ...
001000 1961-12-31 00:00:00.000000  1961.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.341831  NaN  None  None    None  None  None
       1962-12-31 00:00:00.000000  1962.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.341831  NaN  None  None    None  None  None
       1963-12-31 00:00:00.000000  1963.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.244497  NaN  None  None    None  None  None
       1964-12-31 00:00:00.000000  1964.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.089999  NaN  None  None    None  None  None
       1965-12-31 00:00:00.000000  1965.0   INDL      C      D     STD  AE.2  000032102  A & E PLASTIK PAK INC  ...    NaN  3.089999  NaN  None  None    None  None  None

[5 rows x 946 columns]
```

1.  Here it skips the setup of WRDS login credentials. To do so, run the following script. 

    ``` python
    from frds.io.wrds import setup
    setup(username="username", password="password", save_credentials=True)
    ```

    If `save_credentials=True`, the username and password will be saved locally in `credentials.json` in the `frds` folder. Then in later uses, no more setup is required (no just current session).

    The `frds` folder is created under the user's home directory to store downloaded data upon installation.

2. `#!python use_cache=True` attempts to load the data from local cache instead of downloading it again.

### Compute metrics on the go

A simple example of using `frds` to compute assert tangibility for firms in the Compustat Annual dataset:

``` python
>>> tangibility = FUNDA.PPENT / FUNDA.AT # (1)
>>> type(tangibility)
<class 'pandas.core.series.Series'>
>>> tangibility.sample(10).sort_index()
GVKEY   DATADATE
001000  1965-12-31 00:00:00.000000    0.604762
        1967-12-31 00:00:00.000000    0.539495
        1968-12-31 00:00:00.000000    0.654171
        1977-12-31 00:00:00.000000    0.452402
001001  1985-12-31 00:00:00.000000    0.567439
001003  1980-12-31 00:00:00.000000         NaN
        1988-01-31 00:00:00.000000    0.073495
001004  1967-05-31 00:00:00.000000    0.175518
        1980-05-31 00:00:00.000000    0.183682
        1982-05-31 00:00:00.000000    0.286231
dtype: float64
```

1. The `frds.data.wrds.comp.Funda` class has all the variables in the Fundamental Annual dataset as attributes with proper docstrings. 

    Hence, we can write much simpler expressions whenever possible. 


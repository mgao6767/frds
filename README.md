# FRDS - Financial Research Data Services

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=green)

[frds](https://gihub.com/mgao6767/frds/) is a Python framework that aims to provide the simplest way to compute a collection of major academic measures used in the finance literature, one-click with a Graphical User Interface (GUI).

![GUI](/images/frds_demo.gif)

## Installation & Configuration

[frds](https://gihub.com/mgao6767/frds/) requires Python3.8 or higher. To install using `pip`:

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

Alternatively, run without GUI:

```bash
$ python -m frds.run
```

The output data will be saved as STATA `.dta` file in the `result` folder.

## Example Output

Below is an example output for [`tangibility`](https://github.com/mgao6767/frds/blob/master/frds/measures/tangibility.py), defined as the Property, Plant and Equipment (Net) scaled by Assets (Total), estimated for all firms in the Compustat Fundamental Annual. The result dataset is saved in `/result/Tangibility.dta`.

![result-tangibility](https://github.com/mgao6767/frds/raw/master/images/result-tangibility.png)

## Supported Measures

| Measure                                                                                                  | Description                                                                                                     | Datasets Used   |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------|
| [tangibility](https://github.com/mgao6767/frds/blob/master/frds/measures/tangibility.py)                 | Property, Plant and Equipment (Net) scaled by Assets (Total)                                                    | wrds.comp.funda |
| [roa](https://github.com/mgao6767/frds/blob/master/frds/measures/roa.py)                                 | Income Before Extraordinary Items scaled by Assets (Total)                                                      | wrds.comp.funda |
| [roe](https://github.com/mgao6767/frds/blob/master/frds/measures/roe.py)                                 | Income Before Extraordinary Items scaled by Common Equity (Total)                                               | wrds.comp.funda |
| [book leverage](https://github.com/mgao6767/frds/blob/master/frds/measures/book_leverage.py)             | (Long-term Debt + Debt in Current Liabilities) / (Long-term Debt + Debt in Current Liabilities + Common Equity) | wrds.comp.funda |
| [capital expenditure](https://github.com/mgao6767/frds/blob/master/frds/measures/capital_expenditure.py) | Capital Expenditures scaled by Assets (Total)                                                                   | wrds.comp.funda |
| [market to book](https://github.com/mgao6767/frds/blob/master/frds/measures/market_to_book.py)           | Market Value of Common Equity to Book Common Equity                                                             | wrds.comp.funda |
# FRDS

![LICENSE](https://img.shields.io/github/license/mgao6767/frds?color=green)

General purpose financial research data services.

## Install

FRDS requires Python3.8 or higher. To install using `pip`:

```bash
pip install frds
```
After installation, a folder `frds` will be created under the user's home
directory, which contains a `data` folder, a `result` folder and the
configuration file `config.ini`.

## Configuration

The default `config.ini` is:

```ini
[Author]
name: Mingze Gao
email: mingze.gao@sydney.edu.au
institution: University of Sydney

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
python -m frds.run
```
The output data will be saved as STATA `.dta` file in the `result` folder.

## Example

The default measure is
[`tangibility`](https://github.com/mgao6767/frds/blob/master/frds/measures/tangibility.py),
defined as the Property, Plant and Equipment (Net) scaled by Assets (Total),
estimated for all firms in the Compustat Fundamental Annual. The result dataset
is saved in `/result/Tangibility.dta`.

![result-tangibility](https://github.com/mgao6767/frds/blob/master/images/result-tangibility.png)

## Structure

| Module          | Description                                        |
|-----------------|----------------------------------------------------|
| `frds.data`     | provides data access to various data vendors.      |
| `frds.ra`       | the hard-working RA that does all the computation. |
| `frds.measures` | the collection of measures to estimate.            |

## Measures

| Measure                                                                                  | Description                                                  | Datasets Used |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------|---------------|
| [tangibility](https://github.com/mgao6767/frds/blob/master/frds/measures/tangibility.py) | Property, Plant and Equipment (Net) scaled by Assets (Total) | wrds.funda    |
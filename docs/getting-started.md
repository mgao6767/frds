# Getting started

## Installation 

[`frds`](/) requires Python3.8 or higher.

### Install via `PyPI`

Using `pip` is the simplest way. To install using `pip`:

```bash
$ pip install frds --upgrade
```

### Install from source

[`frds`](https://github.com/mgao6767/frds/) is available on GitHub. 
You can download the source code and install:

```bash
$ cd ~
$ git clone https://github.com/mgao6767/frds.git
$ cd frds
$ pip install .
```

## Setup

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

You need to enter your WRDS username and password under the login section if you wish to use [`frds`](/) via CLI.
Alternatively, you can leave them empty and enter manually when using the GUI.

## Usage

[`frds`](/) can be used via a Graphical User Interface (GUI) or a Command Line Interface (CLI).

### GUI

To start [`frds`](/) in GUI, open a terminal and enter:

```bash
$ python -m frds.gui.run
```

A window similar to the one below should show up.

The configuration is default to the settings in the [`config.ini`](/installation).

![frds-demo](/images/frds_demo.gif)

!!! note
    If you don't want to store your login data locally on your computer, you can 
    leave them empty in the [`config.ini`](/installation). Then each time using
    [`frds`](/) GUI you can enter your username and password manually.

### CLI

Alternatively, run without GUI:

```bash
$ python -m frds.run
```

This will estimate all measures and save the results as STATA `.dta` file in the `result` folder.

## Example Output

For example, below is a screenshot of the output for `frds.measures.AccountingRestatement`.

![result-restatements](/images/result-restatements.png)


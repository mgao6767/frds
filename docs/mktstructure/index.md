# frds-mktstructure

`frds-mktstructure` is a simple command-line tool to download data from Refinitiv Tick History and compute some market microstructure measures.

!!! info Note

    You will need a **Refinitiv DataScope Select** login to be able to use this tool.

## Features

You can use this tool to download tick-by-tick quote and transaction data for your selected securities, or for all S&P500 constituents, in one line of code!

!!! example

    Let's download the tick history for all S&P500 component stocks from Jan 1, 2022, to Jan 31, 2022:

    ``` bash
    frds-mktstructure download -u {username} -p {password} --sp500 --parse --data_dir "./data" -b 2022-01-01 -e 2022-01-31
    ```

    where `{username}` and `{password}` are the login credentials of Refinitiv DataScope Select.

    Note that we set the `--parse` flag to parse the downloaded data (gzip) into csv files by stock and date into the `./data` folder.
    
Similarly, you can use one line of code to compute a collection of selected measures:

!!! example

    Use the `compute` subcommand to compute specified market microstructure measures:

    ``` bash
    frds-mktstructure compute --all --data_dir "./data" --out out.csv --bid_ask_spread --price_impact
    ```

## Installation

There is no extra installation or setup required. It is available as long as you have `frds` installed.

!!! tip "`frds-mktstructure` is available once `frds` is installed"

    To check if it's correctly installed, type `frds-mktstructure -v` in your terminal, you should see something like:

    ```bash
    frds-mktstructure version 0.2.0
    ```

    Use `-h` or `--help` to see the usage instruction:

    ``` bash
    $ frds-mktstructure -h
    usage: frds-mktstructure [OPTION]...

    Download data from Refinitiv Tick History and compute some market microstructure measures.

    optional arguments:
    -h, --help            show this help message and exit
    -v, --version         show program's version number and exit

    Sub-commands:
    Choose one from the following. Use `frds-mktstructure subcommand -h` to see help for each sub-command.

    {download,clean,classify,compute}
        download            Download data from Refinitiv Tick History
        clean               Clean downloaded data
        classify            Classify ticks into buy and sell orders
        compute             Compute market microstructure measures
    ```

## Sub-commands

`frds-mktstructure` is consisted of a few standalone subcommands.

- [`download`](/mktstructure/download-data): to download raw data from Refinitiv.
- [`clean`](/mktstructure/clean-data): to sort and remove duplicates, etc.
- [`classify`](/mktstructure/classify-trade-direction): to classify trade direction.
- [`compute`](/mktstructure/compute-measures): to compute selected market microstructure measures.

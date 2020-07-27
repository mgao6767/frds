# Usage

[`frds`](/) can be used via a Graphical User Interface (GUI) or a Command Line Interface (CLI).

## GUI

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

## CLI

Alternatively, run without GUI:

```bash
$ python -m frds.run
```

This will estimate all measures and save the results as STATA `.dta` file in the `result` folder.

## Example Output

For example, below is a screenshot of the output for `frds.measures.AccountingRestatement`.

![result-restatements](/images/result-restatements.png)


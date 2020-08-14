---
path: tree/master/frds
source: measures/tobin_q.py
---

# Tobin's Q

## Definition

Tobin's Q is defined as the ratio between the market value of the firm over the replacement cost of its assets.

$$
\text{TobinQ} = \frac{\text{Marekt value of the firm}}{\text{Replacement cost of assets}}
$$

There're a number of ways to estimate Toin's Q empirically. [Gompers, Ishii and Metrick (2003 QJE)](https://doi.org/10.1162/00335530360535162), following [Kaplan and Zingales (1997 QJE)](https://doi.org/10.1162/003355397555163), define Tobin's Q as:

> The market value of assets divided by the book value of assets (Compustat item 6), where the market value of assets is computed as book value of assets plus the market value of common stock less the sum of the book value of common stock (Compustat item 60) and balance sheet deferred taxes (Compustat item 74). All book values for fiscal year t (from Compustat) are combined with the market value of common equity at the calendar end of year t.[^1]

[^1]: See Appendix 2 of [Gompers, Ishii and Metrick (2003 QJE)](https://doi.org/10.1162/00335530360535162).

which gives:

$$
\text{TobinQ}_{i,t} = \frac{\text{Total Assets}_{i,t} + \text{Market Equity}_{i,t} - \text{Book Equity}_{i,t}}{\text{Total Assets}_{i,t}}
$$

where:

* $\text{Total Assets}$ is the book value total assets as reported
* $\text{Market Equity}=PRCC\_C \times CSHO$
* $\text{Book Equity}=SEQ+TXDB+ITCB-PREF$ and 
* $PREF=\text{coalesce}(PSTKRV,PSTKL,PSTK)$

## Variables

| Variable  | Description                                                                |
| --------- | -------------------------------------------------------------------------- |
| `PRCC_C ` | Stock price at the calendar year end for a fair cross sectional comparison |
| `CSHO`    | Common shares outstanding                                                  |
| `SEQ`     | Shareholder equity                                                         |
| `TXDB`    | Deferred taxes                                                             |
| `ITCB`    | Investment Tax Credit                                                      |
| `PREF`    | Preferred Stock                                                            |
| `PSTKRV`  | Preferred stock - redemption value                                         |
| `PSTKL`   | Preferred stock - liquidating value                                        |
| `PSTK`    | Preferred stock - carrying value                                           |
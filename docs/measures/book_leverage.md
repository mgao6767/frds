---
path: tree/master/frds
source: measures/book_leverage.py
---

# Book Leverage

## Definition

The book leverage is defined as the amount of debts scaled by the firm's total debts plus common equity.

$$
\text{Book Leverage}_{i,t} = \frac{DLTT_{i,t}+DLC_{i,t}}{DLTT_{i,t}+DLC_{i,t}+CEQ_{i,t}}
$$

where $DLTT$ is the long-term debt, $DLC$ is the debt in current liabilities, and $CEQ$ is the common equity, all from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

If $CEQ$ is missing, the book leverage is treated as missing.


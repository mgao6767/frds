---
path: tree/master/frds
source: measures/market_to_book.py
---

# Market to Book Ratio

## Definition

Market value of common equity scaled by the book value common equity.

$$
MTB_{i,t} = \frac{PRCC\_F_{i,t}\times CSHO_{i,t}}{CEQ_{i,t}}
$$

where $PRCC\_F$ is the share price at fiscal year end, $CSHO$ is the common shares outstanding, and $CEQ$ is common equity, all from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.
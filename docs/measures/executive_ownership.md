---
path: tree/master/frds
source: measures/executive_ownership.py
---

# Executive Ownership

## Definition

Executive ownership measures the proportion of the firm's common equity owned by its executives.

### `ExecSharePct`

Executive-year level, executive share ownership:

$$
\text{ExecSharePct}_{e,t} = \frac{SHROWN\_TOT_{e,t}}{CSHO_{i,t}}
$$

where $SHROWN\_TOT$ is the shares owned by the executive (as reported) from Compustat Execucomp Annual Compensation `EXECCOMP.ANNCOMP`, and $CSHO$ is the common shares outstanding from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

### `ExecSharePctExclOpt`

Executive-year level, executive share ownership excluding options:

$$
\text{ExecSharePctExclOpt}_{e,t} = \frac{SHROWN\_EXCL\_OPTS_{e,t}}{CSHO_{i,t}}
$$

where $SHROWN\_EXCL\_OPTS$ is the shares owned by the executive, excluding options, from Compustat Execucomp Annual Compensation `EXECCOMP.ANNCOMP`, and $CSHO$ is the common shares outstanding from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

### `ExecOptPct`

Executive-year level, executive share ownership based on shares acquired on option exercise:

$$
\text{ExecOptPct}_{e,t} = \frac{OPT\_EXER\_NUM_{e,t}}{CSHO_{i,t}}
$$

where $OPT\_EXER\_NUM$ is the number of shares acquired on option exercise by the executive from Compustat Execucomp Annual Compensation `EXECCOMP.ANNCOMP`, and $CSHO$ is the common shares outstanding from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

### `ExecShareVestPct`

Executive-year level, executive share ownership based on shared acquired on vesting:

$$
\text{ExecShareVestPct}_{e,t} = \frac{SHRS\_VEST\_NUM_{e,t}}{CSHO_{i,t}}
$$

where $SHRS\_VEST\_NUM$ is the number of shares acquired on vesting by the executive from Compustat Execucomp Annual Compensation `EXECCOMP.ANNCOMP`, and $CSHO$ is the common shares outstanding from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.


### `ExecIncentivePct`

Executive-year level, value realized on option exercise and vesting scaled by total compensation:

$$
\text{ExecIncentivePct}_{e,t} = \frac{OPT\_EXER\_VAL_{e,t} + SHRS\_VEST\_NUM_{e,t}}{TDC1_{e,t}}
$$

where $OPT\_EXER\_VAL$ is the value realized on vesting and $TDC1$ is the executive's total compensation, both from Compustat Execucomp Annual Compensation `EXECCOMP.ANNCOMP`. 
from typing import Union
import numpy as np
import pandas as pd
from frds.data.wrds.comp import Funda, Fundq


def book_leverage(data: Union[Funda, Fundq]) -> Union[pd.Series, None]:
    r"""Book leverage

    The book leverage is defined as the amount of debts scaled by the firm's total debts plus common equity.

    $$
    \text{Book Leverage}_{i,t} = \frac{DLTT_{i,t}+DLC_{i,t}}{DLTT_{i,t}+DLC_{i,t}+CEQ_{i,t}}
    $$

    where $DLTT$ is the long-term debt, $DLC$ is the debt in current liabilities, and $CEQ$ is the common equity, all from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    Note:
        If $CEQ$ is missing, the book leverage is treated as missing.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    \text{Book Leverage}_{i,t} = \frac{DLTTQ_{i,t}+DLCQ_{i,t}}{DLTTQ_{i,t}+DLCQ_{i,t}+CEQQ_{i,t}}
    $$

    Args:
        data (Union[Funda, Fundq]): Input dataset

    Returns:
        Union[pd.Series, None]: Book leverage
    """
    leverage = None
    if isinstance(data, Funda):
        leverage = (data.DLTT + data.DLC) / (data.DLTT + data.DLC + data.CEQ)
    if isinstance(data, Fundq):
        leverage = (data.DLTTQ + data.DLCQ) / (data.DLTTQ + data.DLCQ + data.CEQQ)
    return leverage.rename("Leverage") if leverage is not None else None


def capital_expenditure(data: Funda) -> Union[pd.Series, None]:
    r"""Capital expenditure

    The capital expenditures scaled by total assets.

    $$
    \text{Capital Expenditure}_{i,t} = \frac{CAPX_{i,t}}{AT_{i,t}}
    $$

    where $CAPX$ and $AT$ are from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    Args:
        data (Funda): Input dataset

    Returns:
        Union[pd.Series, None]: Capital expenditure
    """
    capx = None
    if isinstance(data, Funda):
        capx = data.CAPX / data.AT
    return capx.rename("Capital_Expenditure") if capx is not None else None


def market_to_book(data: Union[Funda, Fundq]) -> Union[pd.Series, None]:
    r"""Market-to-Book ratio

    Market value of common equity scaled by the book value common equity.

    $$
    \text{MTB}_{i,t} = \frac{PRCC\_F_{i,t}\times CSHO_{i,t}}{CEQ_{i,t}}
    $$

    where $PRCC\_F$ is the share price at fiscal year end, $CSHO$ is the common shares outstanding, and $CEQ$ is common equity, all from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    \text{MTB}_{i,t} = \frac{PRCCQ_{i,t}\times CSHOQ_{i,t}}{CEQQ_{i,t}}
    $$

    Args:
        data (Union[Funda, Fundq]): Input dataset

    Returns:
        Union[pd.Series, None]: Market-to-Book ratio
    """
    mtb = None
    if isinstance(data, Funda):
        mtb = data.PRCC_F * data.CSHO / data.CEQ
    if isinstance(data, Fundq):
        mtb = data.PRCCQ * data.CSHOQ / data.CEQQ
    return mtb.rename("MTB") if mtb is not None else None


def roa(
    data: Union[Funda, Fundq], use_lagged_total_asset=False
) -> Union[pd.Series, None]:
    r"""ROA

    Income before extraordinary items scaled by total assets

    $$
    ROA_{i,t} = \frac{IB_{i,t}}{AT_{i,t}}
    $$

    where $IB$ and $AT$ are from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    ROA_{i,t} = \frac{IBQ_{i,t}}{ATQ_{i,t}}
    $$

    Args:
        data (Union[Funda, Fundq]): Input dataset
        use_lagged_total_asset (bool, optional): Use lagged total assets. Defaults to False.

    Returns:
        Union[pd.Series, None]: ROA

    Examples:
        >>> from frds.data.wrds.comp import Funda
        >>> from frds.io.wrds import load
        >>> from frds.measures.corporate import roa
        >>> FUNDA = load(Funda)
        >>> roa(FUNDA)
        GVKEY   DATADATE
        001000  1961-12-31         NaN
                1962-12-31         NaN
                1963-12-31         NaN
                1964-12-31    0.027542
                1965-12-31   -0.085281
                                ...
        001004  1982-05-31    0.010778
                1983-05-31    0.025115
                1984-05-31    0.032697
                1985-05-31    0.058370
                1986-05-31    0.058027
        Name: ROA, Length: 100, dtype: float64

    See Also:
        * [ROE](/measures/roe) - Return on equity
    """
    roa = None

    if isinstance(data, Funda):
        if use_lagged_total_asset:
            roa = data.IB / Funda.lag(data.AT, lags=1)
        else:
            roa = data.IB / data.AT

    if isinstance(data, Fundq):
        if use_lagged_total_asset:
            roa = data.IBQ / Fundq.lag(data.ATQ, lags=1)
        else:
            roa = data.IBQ / data.ATQ

    return roa.rename("ROA") if roa is not None else None


def roe(data: Union[Funda, Fundq], use_lagged_ceq=False) -> Union[pd.Series, None]:
    r"""ROE

    Income before extraordinary items scaled by common equity

    $$
    ROE_{i,t} = \frac{IB_{i,t}}{CEQ_{i,t}}
    $$

    where $IB$ and $CEQ$ are from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    ROE_{i,t} = \frac{IBQ_{i,t}}{CEQQ_{i,t}}
    $$

    Args:
        data (Union[Funda, Fundq]): Input dataset
        use_lagged_ceq (bool, optional): Use lagged common equity. Defaults to False.

    Returns:
        Union[pd.Series, None]: ROE

    Examples:
        >>> from frds.data.wrds.comp import Funda
        >>> from frds.io.wrds import load
        >>> from frds.measures.corporate import roe
        >>> FUNDA = load(Funda)
        >>> roe(FUNDA)
        GVKEY   DATADATE
        001000  1961-12-31         NaN
                1962-12-31    0.217391
                1963-12-31    0.005425
                1964-12-31    0.064250
                1965-12-31   -0.401222
                                ...
        001004  1982-05-31    0.028876
                1983-05-31    0.064662
                1984-05-31    0.055337
                1985-05-31    0.103805
                1986-05-31    0.120632
        Name: ROE, Length: 100, dtype: float64

    See Also:
        * [ROA](/measures/roa) - Return on assets
    """
    roe = None

    if isinstance(data, Funda):
        if use_lagged_ceq:
            roe = data.IB / Funda.lag(data.CEQ, lags=1)
        else:
            roe = data.IB / data.CEQ

    if isinstance(data, Fundq):
        if use_lagged_ceq:
            roe = data.IBQ / Fundq.lag(data.CEQQ, lags=1)
        else:
            roe = data.IBQ / data.CEQQ

    return roe.rename("ROE") if roe is not None else None


def firm_size(data: Union[Funda, Fundq]) -> Union[pd.Series, None]:
    r"""Firm size

    The natural logarithm of total assets.

    $$
    \text{Size}_{i,t} = \ln \left( AT_{i,t} \right)
    $$

    where $AT$ is from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    \text{Size}_{i,t} = \ln \left( ATQ_{i,t} \right)
    $$

    Note:
        If $AT$ or $ATQ$ is missing or negative, $\text{Size}$ is set to missing (`np.nan`).

    Args:
        data (Union[Funda, Fundq]): Input dataset

    Returns:
        Union[pd.Series, None]: Firm size
    """
    size = None

    if isinstance(data, Funda):
        size = np.log(data.AT, where=(data.AT > 0))
        size[np.isnan(data.AT) | (data.AT <= 0)] = np.nan

    if isinstance(data, Fundq):
        size = np.log(data.ATQ, where=(data.ATQ > 0))
        size[np.isnan(data.ATQ) | (data.ATQ <= 0)] = np.nan

    return size.rename("Firm_Size") if size is not None else None


def tangibility(data: Union[Funda, Fundq]) -> Union[pd.Series, None]:
    r"""Asset tangibility

    Property, plant and equipment (net) scaled by total assets

    $$
    \text{Tangibility}_{i,t} = \frac{PPENT_{i,t}}{AT_{i,t}}
    $$

    where $PPENT$ and $AT$ are from Compustat Fundamentals Annual `WRDS.COMP.FUNDA`.

    If `data` is a Fundamentals Quarterly dataset:

    $$
    \text{Tangibility}_{i,t} = \frac{PPENTQ_{i,t}}{ATQ_{i,t}}
    $$

    Args:
        data (Union[Funda, Fundq]): Input dataset

    Returns:
        Union[pd.Series, None]: Tangibility

    Examples:
        >>> from frds.data.wrds.comp import Funda
        >>> from frds.io.wrds import load
        >>> from frds.measures.corporate import tangibility
        >>> FUNDA = load(Funda)
        >>> tangibility(FUNDA)
        GVKEY   DATADATE
        001000  1961-12-31         NaN
                1962-12-31         NaN
                1963-12-31         NaN
                1964-12-31    0.397599
                1965-12-31    0.604762
                                ...
        001004  1982-05-31    0.286231
                1983-05-31    0.287057
                1984-05-31    0.245198
                1985-05-31    0.221782
                1986-05-31    0.194738
        Name: Tangibility, Length: 100, dtype: float64
    """
    if isinstance(data, Funda):
        return (data.PPENT / data.AT).rename("Tangibility")
    if isinstance(data, Fundq):
        return (data.PPENTQ / data.ATQ).rename("Tangibility")
    return None

from typing import Union
import pandas as pd
from frds.data.wrds.comp import Funda, Fundq


def roa(
    data: Union[Funda, Fundq], use_lagged_total_asset=False
) -> Union[pd.Series, None]:
    r"""Income before extraordinary items scaled by total assets

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
    r"""Income before extraordinary items scaled by common equity

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


def tangibility(data: Union[Funda, Fundq]) -> Union[pd.Series, None]:
    r"""Property, plant and equipment (net) scaled by total assets

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

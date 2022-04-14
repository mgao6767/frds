import os
import json
from typing import List, Dict
from numba import jit
import pandas as pd
import numpy as np
from .request_templates import INDEX_COMPONENTS, INTRADAY_TICKS


SP500_RIC = "0#.SPX"
NASDAQ_RIC = "0#.NDX"
NYSE_RIC = "0#.NYA"


def extract_index_components_ric(chain_result_json, mkt_index: List[str]):
    """
    Example json result:
    https://developers.refinitiv.com/thomson-reuters-tick-history-trth/thomson-reuters-tick-history-trth-rest-api/learning?content=13754&type=learning_material_item
    """
    if isinstance(chain_result_json, str):
        chain_result = json.loads(chain_result_json)
    elif isinstance(chain_result_json, dict):
        chain_result = chain_result_json
    vals: List[Dict] = chain_result.get("value")
    securities = set()
    for val in vals:
        identifier = val.get("Identifier")
        if identifier in mkt_index:
            constituents = val.get("Constituents")
            for security in constituents:
                ric: str = security.get("Identifier")
                if ric and not ric.startswith("."):
                    securities.add(ric)
    return list(securities)


def make_request_index_components(mkt_index: List[str], date_start, date_end):
    request = INDEX_COMPONENTS.copy()
    request["Request"]["ChainRics"] = mkt_index
    request["Request"]["Range"]["Start"] = date_start
    request["Request"]["Range"]["End"] = date_end
    return json.dumps(request)


def make_request_tick_history(rics: List[str], date_start, date_end):
    request = INTRADAY_TICKS.copy()
    request["ExtractionRequest"]["IdentifierList"]["InstrumentIdentifiers"] = [
        {"Identifier": ric, "IdentifierType": "Ric"} for ric in rics
    ]
    request["ExtractionRequest"]["Condition"]["QueryStartDate"] = date_start
    request["ExtractionRequest"]["Condition"]["QueryEndDate"] = date_end
    return request


def lee_and_ready(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to pd.DatetimeIndex to preserve nanoseconds.
    df["Date-Time"] = pd.DatetimeIndex(df["Date-Time"])
    # Get GMT Offset
    offset = np.timedelta64(df["GMT Offset"].iloc[0], "h")
    # Convert from GMTUTC to local time.
    df["Date-Time"] = df["Date-Time"] + offset
    # Set local time as index.
    df.set_index("Date-Time", inplace=True)
    # Keep only trades/quotes during normal trading hours.
    # TODO: Check RIC and get trading hours for non US exchanges.
    # without `.copy()` there is hidden chaining causing `SettingSettingWithCopyWarning` warning
    df = df.between_time(start_time="09:30", end_time="16:00").copy()
    # Prepare for Lee and Ready.
    prices = df["Price"].to_numpy()
    bids = df["Bid Price"].to_numpy()
    asks = df["Ask Price"].to_numpy()
    bidsize = df["Bid Size"].to_numpy()
    asksize = df["Ask Size"].to_numpy()
    directions, bbids, basks = _lee_and_ready_classify(
        prices, bids, asks, bidsize, asksize
    )
    df["Direction"] = pd.Series(directions, index=df.index)
    df["Bid Price"] = pd.Series(bbids, index=df.index)
    df["Ask Price"] = pd.Series(basks, index=df.index)
    df["Mid Point"] = (df["Bid Price"] + df["Ask Price"]) / 2
    # If the first observation is a Trade, there will not be a Mid Point.
    return df[df["Type"] == "Trade"].dropna(subset=["Mid Point"])


@jit(nopython=True, nogil=True, cache=True)
def _lee_and_ready_classify(prices, bids, asks, bidsize, asksize):
    n = len(prices)
    directions = np.zeros(n, dtype=np.int8)
    last_bid, last_ask = np.nan, np.nan
    last_trade_price, last2_trade_price = np.nan, np.nan
    last_quote_midpoint = np.nan
    for i in range(n):
        # If price[i] is np.nan then this is a quote.
        if np.isnan(prices[i]) and asks[i] and bids[i] and bidsize[i] and asksize[i]:
            last_quote_midpoint = (last_bid + last_ask) / 2
            last_bid, last_ask = bids[i], asks[i]
            continue
        # Up here we know this is a trade.
        p = prices[i]
        # Quote Test
        if np.isnan(last_quote_midpoint):
            pass
        elif p > last_quote_midpoint:
            directions[i] = 1
        elif p < last_quote_midpoint:
            directions[i] = -1
        # Ticke Test when price = last midpoint
        elif np.isnan(last_trade_price):
            pass
        elif p > last_trade_price:
            directions[i] = 1
        elif p < last_trade_price:
            directions[i] = -1
        elif np.isnan(last2_trade_price):
            pass
        elif p > last2_trade_price:
            directions[i] = 1
        elif p < last2_trade_price:
            directions[i] = -1
        # The immediate bid/ask before the trade.
        if not np.isnan(last_bid):
            bids[i] = last_bid
        if not np.isnan(last_ask):
            asks[i] = last_ask
        last2_trade_price = last_trade_price
        last_trade_price = p
    return directions, bids, asks


def _sort_and_rm_duplicates(data_path, replace=True):
    """
    Remove trades/quotes with same Bid/Ask/Volume/Price at the same nanosecond.
    It is highly unlikely that two quotes/trades of exactly the same parameters happen at the same nanosecond.
    """
    if "gz" in data_path:
        gz = True
        # Parse_dates here will result in loss of nanosecond precision!
        df = pd.read_csv(data_path, compression="gzip")
    else:
        gz = False
        df = pd.read_csv(data_path)

    # obs = len(df.index)
    # Drop duplicates first before converting Date-Time to DatetimeIndex, otherwise it'll be ignored.
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
    df.drop_duplicates(inplace=True)
    # This conversion preserves the nanoseconds.
    df["Date-Time"] = pd.DatetimeIndex(df["Date-Time"])
    df.set_index(["Date-Time"], inplace=True)
    df.sort_index(inplace=True)
    # new_len = len(df.index)
    df.to_csv(
        data_path if replace else f"{data_path.replace('.csv', '.sorted.csv')}",
        compression="gzip" if gz else "infer",
        mode="w",
    )

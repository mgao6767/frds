import json
from typing import List, Dict

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

INDEX_COMPONENTS = {
    "Request": {
        "ChainRics": ["0#.DJI", "0#.SPX", "0#.NDX"],
        "Range": {
            "Start": "2019-01-01T00:00:00.000Z",
            "End": "2020-01-01T00:00:00.000Z",
        },
    }
}

INTRADAY_TICKS = {
    "ExtractionRequest": {
        "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.TickHistoryTimeAndSalesExtractionRequest",
        "ContentFieldNames": [
            "Quote - Bid Price",
            "Quote - Bid Size",
            "Quote - Ask Price",
            "Quote - Ask Size",
            "Trade - Bid Price",
            "Trade - Ask Price",
            "Trade - Price",
            "Trade - Volume",
        ],
        "IdentifierList": {
            "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.InstrumentIdentifierList",
            "InstrumentIdentifiers": [
                {"Identifier": "AAPL.OQ", "IdentifierType": "Ric"}
            ],
            "UseUserPreferencesForValidationOptions": "False",
        },
        "Condition": {
            "MessageTimeStampIn": "GmtUtc",
            "ReportDateRangeType": "Range",
            "QueryStartDate": "2020-03-01T00:00:00.000Z",
            "QueryEndDate": "2020-03-03T00:00:00.000Z",
            "DisplaySourceRIC": "False",
        },
    }
}

RIC_IDENTIFIERS = {
    "ExtractionRequest": {
        "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.TermsAndConditionsExtractionRequest",
        "ContentFieldNames": [
            "RIC",
            "ISIN",
            "Currency Code",
            "Exchange Code",
            "Exchange Code List",
            "SEDOL",
            "CUSIP",
        ],
        "IdentifierList": {
            "@odata.type": "#ThomsonReuters.Dss.Api.Extractions.ExtractionRequests.InstrumentIdentifierList",
            "InstrumentIdentifiers": [
                {"Identifier": "AAPL.OQ", "IdentifierType": "Ric"}
            ],
            "ValidationOptions": {
                "AllowHistoricalInstruments": "True",
                "AllowInactiveInstruments": "True",
                "AllowOpenAccessInstruments": "True",
            },
            "UseUserPreferencesForValidationOptions": "False",
        },
    }
}

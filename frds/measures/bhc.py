import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.measures import Measure

DATASETS = [
    Dataset(
        source="frb_chicago",
        library="bhc",
        table="bhcf",
        vars=[
            "RSSD9001",  # RSSD ID
            "RSSD9999",  # Reporting date
            "BHCK2170",  # Total assets
            "BHCK4059",  # Fee and interest income from loans in foreign offices
            "BHCK4107",  # Total interest income
            "BHCK4340",  # Net income
            "BHCK4460",  # Cash dividends on common stock
            "BHCK3792",  # Total qualifying capital allowable under the risk-based capital guidelines
            "BHCKA223",  # Risk-weighted assets
            "BHCK8274",  # Tier 1 capital allowable under the risk-based capital guidelines
            "BHCK8725",  # Total gross notional amount of interest rate derivatives held for purposes other than trading (marked to market)
            "BHCK8729",  # Total gross notional amount of interest rate derivatives held for purposes other than trading (not marked to market)
            "BHCK8726",  # Total gross notional amount of foreign exchange rate derivatives held for purposes other than trading (marked to market)
            "BHCK8730",  # Total gross notional amount of foreign exchange rate derivatives held for purposes other than trading (not marked to market)
            "BHCK3197",  # Earning assets that are repriceable or mature within one year
            "BHCK3296",  # Interest-bearing deposits that mature or reprice within one year
            "BHCK3298",  # Long term debt that reprices within one year
            "BHCK3409",  # Long-term debt reported in schedule hc
            "BHCK3408",  # Variable rate preferred stock
            "BHCK2332",  # Other borrowed money with a remaining maturity of one year or less
            "BHCK2309",  # Commercial paper
            "BHDMB993",  # Federal funds purchased in domestic offices
            "BHCKB995",  # Securities sold under agreements to repurchase (repo liabilities)
            "BHCK2122",  # Total loans and leases, net of unearned income
        ],
        date_vars=["RSSD9999"],
    )
]
VARIABLE_LABELS = {
    "BHCSize": "Natural logarithm of total assets (BHCK2170)",
    "BHCFxExposure": "BHCK4059/BHCK4107",
    "BHCGrossIRHedging": "Gross interest rate hedging",
    "BHCGrossFXHedging": "Gross foeign exchange rate hedging",
    "BHCLoanGrowth": "ln(BHCK2122/last quarter's BHCK2122)",
    "BHCLoanGrowthPct": "Percentage change (%) in the total loans and leases (BHCK2122)",
    "RSSD9001": "RSSD ID",
    "RSSD9999": "Reporting date",
    "BHCK2170": "Total assets",
    "BHCK4340": "Net income",
    "BHCK4460": "Cash dividends on common stock",
    "BHCK3792": "Total qualifying capital allowable under the risk-based capital guidelines",
    "BHCK8274": "Tier 1 capital allowable under the risk-based capital guidelines",
    "BHCKA223": "Risk-weighted assets",
    "BHCK8725": "Total gross notional amount of IR derivatives held for non-trading purposes",
    "BHCK8729": "Total gross notional amount of IR derivatives held for non-trading purposes (not marked to market)",
    "BHCK8726": "Total gross notional amount of FX derivatives held for non-trading purposes",
    "BHCK8730": "Total gross notional amount of FX derivatives held for non-trading purposes (not marked to market)",
}
KEY_VARS = ["RSSD9001", "RSSD9999"]


class BHCSize(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany Size", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = np.log(bhcf["BHCK2170"])
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCFxExposure(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany FX Exposure", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4059 / bhcf.BHCK4107
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCNetIncomeToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany NetIncomeToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4340 / bhcf.BHCK2170
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCDividendToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany DividendToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK4460 / bhcf.BHCK2170
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCRegCapToAssets(Measure):
    def __init__(self):
        super().__init__(
            "BankHoldingCompany RegulatoryCapitalToAssets", DATASETS
        )

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK3792 / bhcf.BHCKA223
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCTier1CapToAssets(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany Tier1CapitalToAssets", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        bhcf[type(self).__name__] = bhcf.BHCK8274 / bhcf.BHCKA223
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCGrossIRHedging(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany GrossIRHedging", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        # Total gross notional amount of interest rate derivatives held for
        # purposes other than trading (bhck8725) over total assets;
        # for the period 1995 to 2000, contracts not marked to market (bhck8729)
        # are added;
        bhcf[type(self).__name__] = np.where(
            np.in1d(bhcf.RSSD9999.dt.year, range(1995, 2000 + 1)),
            (bhcf.BHCK8725 + bhcf.BHCK8729) / bhcf.BHCK2170,
            bhcf.BHCK8725 / bhcf.BHCK2170,
        )
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCGrossFXHedging(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany GrossFXHedging", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        # Total gross notional amount of foreign exchange rate derivatives held
        # for purposes other than trading (bhck8726) over total assets;
        # for the period 1995 to 2000, contracts not marked to market (bhck8730)
        # are added;
        bhcf[type(self).__name__] = np.where(
            np.in1d(bhcf.RSSD9999.dt.year, range(1995, 2000 + 1)),
            (bhcf.BHCK8726 + bhcf.BHCK8730) / bhcf.BHCK2170,
            bhcf.BHCK8726 / bhcf.BHCK2170,
        )
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, type(self).__name__]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCMaturityGap(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany MaturityGap", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])
        # Earning assets that are repriceable or mature within one year (bhck3197)
        # minus interest-bearing deposits that mature or reprice within one year (bhck3296)
        # minus long-term debt that reprices or matures within one year (bhck3298 + bhck3409)
        # minus variable rate preferred stock (bhck3408)
        # minus other borrowed money with a maturity of one year or less (bhck2332)
        # minus commercial paper (bhck2309)
        # minus federal funds and repo liabilities (bhdmb993 + bhckb995),
        # normalized by total assets.
        bhcf["BHCMaturityGap"] = (
            np.nansum(
                [
                    bhcf.BHCK3197,
                    -bhcf.BHCK3296,
                    -bhcf.BHCK3298,
                    -bhcf.BHCK3409,
                    -bhcf.BHCK3408,
                    -bhcf.BHCK2332,
                    -bhcf.BHCK2309,
                    -bhcf.BHDMB993,
                    -bhcf.BHCKB995,
                ],
                axis=0,
            )
        ) / bhcf.BHCK2170  # No propagation of NaNs
        # Narrow maturity gap does not subtract bhck3296
        bhcf["BHCNarrowMaturityGap"] = (
            np.nansum(
                [
                    bhcf.BHCK3197,
                    -bhcf.BHCK3298,
                    -bhcf.BHCK3409,
                    -bhcf.BHCK3408,
                    -bhcf.BHCK2332,
                    -bhcf.BHCK2309,
                    -bhcf.BHDMB993,
                    -bhcf.BHCKB995,
                ],
                axis=0,
            )
        ) / bhcf.BHCK2170  # No propagation of NaNs
        bhcf.replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, "BHCMaturityGap", "BHCNarrowMaturityGap"]
        return bhcf[keep_cols], VARIABLE_LABELS


class BHCLoanGrowth(Measure):
    def __init__(self):
        super().__init__("BankHoldingCompany LoanGrowth", DATASETS)

    def estimate(self, nparrays):
        bhcf = pd.DataFrame.from_records(nparrays[0])[
            ["RSSD9001", "RSSD9999", "BHCK2122"]
        ]  # FIXME: somehow `columns=` doesn't work here. Is it a bug?

        # No need to sort since we use "merge by" here. This is also the correct
        # way since some banks may not report consistently every quarter.
        bhcf["last_qtr"] = bhcf.RSSD9999 + pd.tseries.offsets.QuarterEnd(n=-1)
        tmp = bhcf.merge(
            bhcf,
            left_on=["RSSD9001", "last_qtr"],
            right_on=["RSSD9001", "RSSD9999"],
            suffixes=("", "_lagged"),
        )
        current_to_lagged_loan = np.true_divide(
            tmp.BHCK2122, tmp.BHCK2122_lagged, where=tmp.BHCK2122_lagged != 0,
        )
        current_to_lagged_loan[np.isnan(current_to_lagged_loan)] = np.nan
        # The simple measure: percentage change (%)
        # In Stata:
        # ```stata
        # use "~/frds/result/BankHoldingCompany LoanGrowth.dta", clear
        # gen qtr = qofd(RSSD9999)
        # format qtr %tq
        # xtset RSSD9001 qtr, quarterly
        # gen BHCLoanGrowthPct = (BHCK2122 / L.BHCK2122 - 1) * 100
        # ```
        tmp["BHCLoanGrowthPct"] = (current_to_lagged_loan - 1) * 100

        # ln(current loan / lagged loan) as in Zheng (2020 JBF)
        # If current loan is 0, ln(0) is undefined. The growth is -100% in fact.
        current_to_lagged_loan = np.where(
            current_to_lagged_loan == 0, 1 / np.e, current_to_lagged_loan
        )
        tmp["BHCLoanGrowth"] = np.log(current_to_lagged_loan)

        tmp["BHCLoanGrowthPct"].replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp["BHCLoanGrowth"].replace([np.inf, -np.inf], np.nan, inplace=True)
        keep_cols = [*KEY_VARS, "BHCK2122", "BHCLoanGrowthPct", "BHCLoanGrowth"]
        return tmp[keep_cols], VARIABLE_LABELS


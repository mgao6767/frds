# Supported Measures

## Firm Characteristics

* [Accounting Restatements](https://frds.io/measures/accounting_restatement)
    * Number of various accounting restatements during the past (*n*) fiscal year.
    * Source: `wrds.comp.funda`, `wrds.audit.auditnonreli`. 
* [Asset Tangibility](https://frds.io/measures/asset_tangibility) 
    * Property, Plant and Equipment (Net) scaled by total assets.
    * Source: `wrds.comp.funda`.
* [Board Independence](https://frds.io/measures/board_independence)
    * Board size and independence measured as the ratio of independent board members to board size.
    * Source: `wrds.funda`, `wrds.boardex.na_wrds_company_profile`, `wrds.boardex.na_wrds_org_composition`.
* [Book Leverage](https://frds.io/measures/book_leverage)
    * Amount of debts scaled by the firm's total debts plus common equity.
    * Source: `wrds.comp.funda`.
* [Capital Expenditure](https://frds.io/measures/capital_expenditure)
    * Capital expenditures scaled by total assets.
    * Source: `wrds.comp.funda`.
* [Executive Ownership](https://frds.io/measures/executive_ownership)
    * Various measures of executive stock ownership.
    * Source: `wrds.comp.funda`, `wrds.execcomp.anncomp`.
* [Firm Size](https://frds.io/measures/firm_size)
    * Natural logarithm of total assets.
    * Source: `wrds.comp.funda`.
* [Market-to-Book Ratio](https://frds.io/measures/market_to_book)
    * Market value of common equity to book value of common equity.
    * Source: `wrds.comp.funda`.
* [ROA](https://frds.io/measures/roa)
    * Income before extraordinary items scaled by total assets.
    * Source: `wrds.comp.funda`.
* [ROE](https://frds.io/measures/roe)
    * Income before extraordinary items scaled by common equity.
    * Source: `wrds.comp.funda`.
* [Stock Delisting](https://frds.io/measures/stock_delisting)
    * Stocks delisted due to financial troubles or as a result of being merged.
    * Source: `wrds.crsp.dse`.

## Bank Holding Company (BHC) Characteristics

* [BHC Size](https://frds.io/measures/bhc_size)
    * Natural logarithm of total assets.
    * Source: `frb_chicago.bhc.bhcf`.
* [BHC FX Exposure](https://frds.io/measures/bhc_fx_exposure)
    * Fee and interest income from loans in foreign offices (BHCK4059) scaled by total interest income (BHCK4107).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC NetIncome/Assets](https://frds.io/measures/bhc_netincome_to_assets)
    * Net income (BHCK4340) / total assets (BHCK2170).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Dividend/Assets](https://frds.io/measures/bhc_dividend_to_assets)
    * Cash dividends on common stock (BHCK4460) / total assets (BHCK2170).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC RegulatoryCapital/Assets](https://frds.io/measures/bhc_regcap_to_assets)
    * Total qualifying capital allowable under the risk-based capital guidelines (BHCK3792) normalized by risk-weighted assets (BHCKA223).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Tier1Capital/Assets](https://frds.io/measures/bhc_tier1cap_to_assets)
    * Tier 1 capital allowable under the risk-based capital guidelines (BHCK8274) normalized by risk-weighted assets (BHCKA223).
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
* [BHC Gross IR Hedging](https://frds.io/measures/bhc_gross_ir_hedging)
    * Total gross notional amount of interest rate derivatives held for purposes other than trading (BHCK8725) over total assets (BHCK2170); for the period 1995 to 2000, contracts not marked to market (BHCK8729) are added.
    * Source: `frb_chicago.bhc.bhcf`.
    * Reference: [Rampini, Viswanathan and Vuillemey (2020 JF)](https://doi.org/10.1111/jofi.12868).
from dataclasses import dataclass
import numpy as np
import pandas as pd
from frds.data.wrds import WRDSDataset


@dataclass
class Fundq(WRDSDataset):
    """Fundamentals Quarterly"""

    data: pd.DataFrame
    library = "comp"
    table = "fundq"
    index_col = ["gvkey", "datadate"]
    date_cols = ["datadate"]

    def __post_init__(self):
        idx = [c.upper() for c in self.index_col]
        if set(self.data.index.names) != set(idx):
            self.data.reset_index(inplace=True, drop=True)
        self.data.rename(columns=str.upper, inplace=True)
        self.data.set_index(idx, inplace=True)

        # Some variables are not available
        attrs = [
            varname
            for varname, prop in vars(Fundq).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            try:
                self.__getattribute__(attr)
            except KeyError:
                delattr(Fundq, attr)

        # Automatically apply the default filtering rules
        self.filter()

    def filter(self):
        """Default filter applied on the FUNDA dataset"""
        # TODO: This filter doesn't guarantee unique gvkey-datadate
        self.data = self.data[
            np.in1d(self.data.DATAFMT, ("STD"))
            & np.in1d(self.data.INDFMT, ("INDL"))
            & np.in1d(self.data.POPSRC, ("D"))
            & np.in1d(self.data.CONSOL, ("C"))
        ]

    @staticmethod
    def lag(series: pd.Series, lags: int = 1, *args, **kwargs):
        return series.shift(lags, *args, **kwargs)

    @staticmethod
    def lead(series: pd.Series, leads: int = 1, *args, **kwargs):
        return series.shift(-leads, *args, **kwargs)

    @property
    def CONM(self) -> pd.Series:
        """Company Name (CONM): string"""
        return self.data["CONM"]

    @property
    def TIC(self) -> pd.Series:
        """Ticker Symbol (TIC): string"""
        return self.data["TIC"]

    @property
    def CUSIP(self) -> pd.Series:
        """CUSIP (CUSIP): string"""
        return self.data["CUSIP"]

    @property
    def CIK(self) -> pd.Series:
        """CIK Number (CIK): string"""
        return self.data["CIK"]

    @property
    def EXCHG(self) -> pd.Series:
        """Stock Exchange Code (EXCHG): double"""
        return self.data["EXCHG"]

    @property
    def FYR(self) -> pd.Series:
        """Fiscal Year-End (FYR): double"""
        return self.data["FYR"]

    @property
    def FIC(self) -> pd.Series:
        """Foreign Incorporation Code (FIC): string"""
        return self.data["FIC"]

    @property
    def ADD1(self) -> pd.Series:
        """ADD1 -- Address Line 1 (ADD1): string"""
        return self.data["ADD1"]

    @property
    def ADD2(self) -> pd.Series:
        """ADD2 -- Address Line 2 (ADD2): string"""
        return self.data["ADD2"]

    @property
    def ADD3(self) -> pd.Series:
        """ADD3 -- Address Line 3 (ADD3): string"""
        return self.data["ADD3"]

    @property
    def ADD4(self) -> pd.Series:
        """ADD4 -- Address Line 4 (ADD4): string"""
        return self.data["ADD4"]

    @property
    def ADDZIP(self) -> pd.Series:
        """ADDZIP -- Postal Code (ADDZIP): string"""
        return self.data["ADDZIP"]

    @property
    def BUSDESC(self) -> pd.Series:
        """BUSDESC -- S&P Business Description (BUSDESC): string"""
        return self.data["BUSDESC"]

    @property
    def CITY(self) -> pd.Series:
        """CITY -- City (CITY): string"""
        return self.data["CITY"]

    @property
    def CONML(self) -> pd.Series:
        """CONML -- Company Legal Name (CONML): string"""
        return self.data["CONML"]

    @property
    def COUNTY(self) -> pd.Series:
        """COUNTY -- County Code (COUNTY): string"""
        return self.data["COUNTY"]

    @property
    def DLDTE(self) -> pd.Series:
        """DLDTE -- Research Company Deletion Date (DLDTE): date"""
        return self.data["DLDTE"]

    @property
    def DLRSN(self) -> pd.Series:
        """DLRSN -- Research Co Reason for Deletion (DLRSN): string"""
        return self.data["DLRSN"]

    @property
    def EIN(self) -> pd.Series:
        """EIN -- Employer Identification Number (EIN): string"""
        return self.data["EIN"]

    @property
    def FAX(self) -> pd.Series:
        """FAX -- Fax Number (FAX): string"""
        return self.data["FAX"]

    @property
    def FYRC(self) -> pd.Series:
        """FYRC -- Current Fiscal Year End Month (FYRC): double"""
        return self.data["FYRC"]

    @property
    def GGROUP(self) -> pd.Series:
        """GGROUP -- GIC Groups (GGROUP): string"""
        return self.data["GGROUP"]

    @property
    def GIND(self) -> pd.Series:
        """GIND -- GIC Industries (GIND): string"""
        return self.data["GIND"]

    @property
    def GSECTOR(self) -> pd.Series:
        """GSECTOR -- GIC Sectors (GSECTOR): string"""
        return self.data["GSECTOR"]

    @property
    def GSUBIND(self) -> pd.Series:
        """GSUBIND -- GIC Sub-Industries (GSUBIND): string"""
        return self.data["GSUBIND"]

    @property
    def IDBFLAG(self) -> pd.Series:
        """IDBFLAG -- International, Domestic, Both Indicator (IDBFLAG): string"""
        return self.data["IDBFLAG"]

    @property
    def INCORP(self) -> pd.Series:
        """INCORP -- Current State/Province of Incorporation Code (INCORP): string"""
        return self.data["INCORP"]

    @property
    def IPODATE(self) -> pd.Series:
        """IPODATE -- Company Initial Public Offering Date (IPODATE): date"""
        return self.data["IPODATE"]

    @property
    def LOC(self) -> pd.Series:
        """LOC -- Current ISO Country Code - Headquarters (LOC): string"""
        return self.data["LOC"]

    @property
    def NAICS(self) -> pd.Series:
        """NAICS -- North American Industry Classification Code (NAICS): string"""
        return self.data["NAICS"]

    @property
    def PHONE(self) -> pd.Series:
        """PHONE -- Phone Number (PHONE): string"""
        return self.data["PHONE"]

    @property
    def PRICAN(self) -> pd.Series:
        """PRICAN -- Current Primary Issue Tag - Canada (PRICAN): string"""
        return self.data["PRICAN"]

    @property
    def PRIROW(self) -> pd.Series:
        """PRIROW -- Primary Issue Tag - Rest of World (PRIROW): string"""
        return self.data["PRIROW"]

    @property
    def PRIUSA(self) -> pd.Series:
        """PRIUSA -- Current Primary Issue Tag - US (PRIUSA): string"""
        return self.data["PRIUSA"]

    @property
    def SIC(self) -> pd.Series:
        """SIC -- Standard Industry Classification Code (SIC): string"""
        return self.data["SIC"]

    @property
    def SPCINDCD(self) -> pd.Series:
        """SPCINDCD -- S&P Industry Sector Code (SPCINDCD): double"""
        return self.data["SPCINDCD"]

    @property
    def SPCSECCD(self) -> pd.Series:
        """SPCSECCD -- S&P Economic Sector Code (SPCSECCD): double"""
        return self.data["SPCSECCD"]

    @property
    def SPCSRC(self) -> pd.Series:
        """SPCSRC -- S&P Quality Ranking - Current (SPCSRC): string"""
        return self.data["SPCSRC"]

    @property
    def STATE(self) -> pd.Series:
        """STATE -- State/Province (STATE): string"""
        return self.data["STATE"]

    @property
    def STKO(self) -> pd.Series:
        """STKO -- Stock Ownership Code (STKO): double"""
        return self.data["STKO"]

    @property
    def WEBURL(self) -> pd.Series:
        """WEBURL -- Web URL (WEBURL): string"""
        return self.data["WEBURL"]

    @property
    def ACCTCHGQ(self) -> pd.Series:
        """ACCTCHGQ -- Adoption of Accounting Changes (ACCTCHGQ): string"""
        return self.data["ACCTCHGQ"]

    @property
    def ACCTSTDQ(self) -> pd.Series:
        """ACCTSTDQ -- Accounting Standard (ACCTSTDQ): string"""
        return self.data["ACCTSTDQ"]

    @property
    def ADRRQ(self) -> pd.Series:
        """ADRRQ -- ADR Ratio (ADRRQ): double"""
        return self.data["ADRRQ"]

    @property
    def AJEXQ(self) -> pd.Series:
        """AJEXQ -- Adjustment Factor (Company) - Cumulative by Ex-Date (AJEXQ): double"""
        return self.data["AJEXQ"]

    @property
    def AJPQ(self) -> pd.Series:
        """AJPQ -- Adjustment Factor (Company) - Cumulative byPay-Date (AJPQ): double"""
        return self.data["AJPQ"]

    @property
    def APDEDATEQ(self) -> pd.Series:
        """APDEDATEQ -- Actual Period End date (APDEDATEQ): date"""
        return self.data["APDEDATEQ"]

    @property
    def BSPRQ(self) -> pd.Series:
        """BSPRQ -- Balance Sheet Presentation (BSPRQ): string"""
        return self.data["BSPRQ"]

    @property
    def COMPSTQ(self) -> pd.Series:
        """COMPSTQ -- Comparability Status (COMPSTQ): string"""
        return self.data["COMPSTQ"]

    @property
    def CURNCDQ(self) -> pd.Series:
        """CURNCDQ -- Native Currency Code (CURNCDQ): string"""
        return self.data["CURNCDQ"]

    @property
    def CURRTRQ(self) -> pd.Series:
        """CURRTRQ -- Currency Translation Rate (CURRTRQ): double"""
        return self.data["CURRTRQ"]

    @property
    def CURUSCNQ(self) -> pd.Series:
        """CURUSCNQ -- US Canadian Translation Rate - Interim (CURUSCNQ): double"""
        return self.data["CURUSCNQ"]

    @property
    def DATACQTR(self) -> pd.Series:
        """DATACQTR -- Calendar Data Year and Quarter (DATACQTR): string"""
        return self.data["DATACQTR"]

    @property
    def DATAFQTR(self) -> pd.Series:
        """DATAFQTR -- Fiscal Data Year and Quarter (DATAFQTR): string"""
        return self.data["DATAFQTR"]

    @property
    def FDATEQ(self) -> pd.Series:
        """FDATEQ -- Final Date (FDATEQ): date"""
        return self.data["FDATEQ"]

    @property
    def FINALQ(self) -> pd.Series:
        """FINALQ -- Final Indicator Flag (FINALQ): string"""
        return self.data["FINALQ"]

    @property
    def FQTR(self) -> pd.Series:
        """FQTR -- Fiscal Quarter (FQTR): double"""
        return self.data["FQTR"]

    @property
    def FYEARQ(self) -> pd.Series:
        """FYEARQ -- Fiscal Year (FYEARQ): double"""
        return self.data["FYEARQ"]

    @property
    def OGMQ(self) -> pd.Series:
        """OGMQ -- OIL & GAS METHOD (OGMQ): string"""
        return self.data["OGMQ"]

    @property
    def PDATEQ(self) -> pd.Series:
        """PDATEQ -- Preliminary Date (PDATEQ): date"""
        return self.data["PDATEQ"]

    @property
    def RDQ(self) -> pd.Series:
        """RDQ -- Report Date of Quarterly Earnings (RDQ): date"""
        return self.data["RDQ"]

    @property
    def RP(self) -> pd.Series:
        """RP -- Reporting Periodicity (RP): string"""
        return self.data["RP"]

    @property
    def SCFQ(self) -> pd.Series:
        """SCFQ -- Cash Flow Model (SCFQ): double"""
        return self.data["SCFQ"]

    @property
    def SRCQ(self) -> pd.Series:
        """SRCQ -- Source Code (SRCQ): double"""
        return self.data["SRCQ"]

    @property
    def STALTQ(self) -> pd.Series:
        """STALTQ -- Status Alert (STALTQ): string"""
        return self.data["STALTQ"]

    @property
    def UPDQ(self) -> pd.Series:
        """UPDQ -- Update Code (UPDQ): double"""
        return self.data["UPDQ"]

    @property
    def ACCHGQ(self) -> pd.Series:
        """ACCHGQ -- Accounting Changes - Cumulative Effect (ACCHGQ): double"""
        return self.data["ACCHGQ"]

    @property
    def ACOMINCQ(self) -> pd.Series:
        """ACOMINCQ -- Accumulated Other Comprehensive Income (Loss) (ACOMINCQ): double"""
        return self.data["ACOMINCQ"]

    @property
    def ACOQ(self) -> pd.Series:
        """ACOQ -- Current Assets - Other - Total (ACOQ): double"""
        return self.data["ACOQ"]

    @property
    def ACTQ(self) -> pd.Series:
        """ACTQ -- Current Assets - Total (ACTQ): double"""
        return self.data["ACTQ"]

    @property
    def ALTOQ(self) -> pd.Series:
        """ALTOQ -- Other Long-term Assets (ALTOQ): double"""
        return self.data["ALTOQ"]

    @property
    def ANCQ(self) -> pd.Series:
        """ANCQ -- Non-Current Assets - Total (ANCQ): double"""
        return self.data["ANCQ"]

    @property
    def ANOQ(self) -> pd.Series:
        """ANOQ -- Assets Netting & Other Adjustments (ANOQ): double"""
        return self.data["ANOQ"]

    @property
    def AOCIDERGLQ(self) -> pd.Series:
        """AOCIDERGLQ -- Accum Other Comp Inc - Derivatives Unrealized Gain/Loss (AOCIDERGLQ): double"""
        return self.data["AOCIDERGLQ"]

    @property
    def AOCIOTHERQ(self) -> pd.Series:
        """AOCIOTHERQ -- Accum Other Comp Inc - Other Adjustments (AOCIOTHERQ): double"""
        return self.data["AOCIOTHERQ"]

    @property
    def AOCIPENQ(self) -> pd.Series:
        """AOCIPENQ -- Accum Other Comp Inc - Min Pension Liab Adj (AOCIPENQ): double"""
        return self.data["AOCIPENQ"]

    @property
    def AOCISECGLQ(self) -> pd.Series:
        """AOCISECGLQ -- Accum Other Comp Inc - Unreal G/L Ret Int in Sec Assets (AOCISECGLQ): double"""
        return self.data["AOCISECGLQ"]

    @property
    def AOL2Q(self) -> pd.Series:
        """AOL2Q -- Assets Level2 (Observable) (AOL2Q): double"""
        return self.data["AOL2Q"]

    @property
    def AOQ(self) -> pd.Series:
        """AOQ -- Assets - Other - Total (AOQ): double"""
        return self.data["AOQ"]

    @property
    def APQ(self) -> pd.Series:
        """APQ -- Account Payable/Creditors - Trade (APQ): double"""
        return self.data["APQ"]

    @property
    def AQAQ(self) -> pd.Series:
        """AQAQ -- Acquisition/Merger After-Tax (AQAQ): double"""
        return self.data["AQAQ"]

    @property
    def AQDQ(self) -> pd.Series:
        """AQDQ -- Acquisition/Merger Diluted EPS Effect (AQDQ): double"""
        return self.data["AQDQ"]

    @property
    def AQEPSQ(self) -> pd.Series:
        """AQEPSQ -- Acquisition/Merger Basic EPS Effect (AQEPSQ): double"""
        return self.data["AQEPSQ"]

    @property
    def AQPL1Q(self) -> pd.Series:
        """AQPL1Q -- Assets Level1 (Quoted Prices) (AQPL1Q): double"""
        return self.data["AQPL1Q"]

    @property
    def AQPQ(self) -> pd.Series:
        """AQPQ -- Acquisition/Merger Pretax (AQPQ): double"""
        return self.data["AQPQ"]

    @property
    def ARCEDQ(self) -> pd.Series:
        """ARCEDQ -- As Reported Core - Diluted EPS Effect (ARCEDQ): double"""
        return self.data["ARCEDQ"]

    @property
    def ARCEEPSQ(self) -> pd.Series:
        """ARCEEPSQ -- As Reported Core - Basic EPS Effect (ARCEEPSQ): double"""
        return self.data["ARCEEPSQ"]

    @property
    def ARCEQ(self) -> pd.Series:
        """ARCEQ -- As Reported Core - After-tax (ARCEQ): double"""
        return self.data["ARCEQ"]

    @property
    def ATQ(self) -> pd.Series:
        """ATQ -- Assets - Total (ATQ): double"""
        return self.data["ATQ"]

    @property
    def AUL3Q(self) -> pd.Series:
        """AUL3Q -- Assets Level3 (Unobservable) (AUL3Q): double"""
        return self.data["AUL3Q"]

    @property
    def BILLEXCEQ(self) -> pd.Series:
        """BILLEXCEQ -- Billings in Excess of Cost & Earnings (BILLEXCEQ): double"""
        return self.data["BILLEXCEQ"]

    @property
    def CAPR1Q(self) -> pd.Series:
        """CAPR1Q -- Risk-Adjusted Capital Ratio - Tier 1 (CAPR1Q): double"""
        return self.data["CAPR1Q"]

    @property
    def CAPR2Q(self) -> pd.Series:
        """CAPR2Q -- Risk-Adjusted Capital Ratio - Tier 2 (CAPR2Q): double"""
        return self.data["CAPR2Q"]

    @property
    def CAPR3Q(self) -> pd.Series:
        """CAPR3Q -- Risk-Adjusted Capital Ratio - Combined (CAPR3Q): double"""
        return self.data["CAPR3Q"]

    @property
    def CAPSFTQ(self) -> pd.Series:
        """CAPSFTQ -- Capitalized Software (CAPSFTQ): double"""
        return self.data["CAPSFTQ"]

    @property
    def CAPSQ(self) -> pd.Series:
        """CAPSQ -- Capital Surplus/Share Premium Reserve (CAPSQ): double"""
        return self.data["CAPSQ"]

    @property
    def CEIEXBILLQ(self) -> pd.Series:
        """CEIEXBILLQ -- Cost & Earnings in Excess of Billings (CEIEXBILLQ): double"""
        return self.data["CEIEXBILLQ"]

    @property
    def CEQQ(self) -> pd.Series:
        """CEQQ -- Common/Ordinary Equity - Total (CEQQ): double"""
        return self.data["CEQQ"]

    @property
    def CHEQ(self) -> pd.Series:
        """CHEQ -- Cash and Short-Term Investments (CHEQ): double"""
        return self.data["CHEQ"]

    @property
    def CHQ(self) -> pd.Series:
        """CHQ -- Cash (CHQ): double"""
        return self.data["CHQ"]

    @property
    def CIBEGNIQ(self) -> pd.Series:
        """CIBEGNIQ -- Comp Inc - Beginning Net Income (CIBEGNIQ): double"""
        return self.data["CIBEGNIQ"]

    @property
    def CICURRQ(self) -> pd.Series:
        """CICURRQ -- Comp Inc - Currency Trans Adj (CICURRQ): double"""
        return self.data["CICURRQ"]

    @property
    def CIDERGLQ(self) -> pd.Series:
        """CIDERGLQ -- Comp Inc - Derivative Gains/Losses (CIDERGLQ): double"""
        return self.data["CIDERGLQ"]

    @property
    def CIMIIQ(self) -> pd.Series:
        """CIMIIQ -- Comprehensive Income - Noncontrolling Interest (CIMIIQ): double"""
        return self.data["CIMIIQ"]

    @property
    def CIOTHERQ(self) -> pd.Series:
        """CIOTHERQ -- Comp Inc - Other Adj (CIOTHERQ): double"""
        return self.data["CIOTHERQ"]

    @property
    def CIPENQ(self) -> pd.Series:
        """CIPENQ -- Comp Inc - Minimum Pension Adj (CIPENQ): double"""
        return self.data["CIPENQ"]

    @property
    def CIQ(self) -> pd.Series:
        """CIQ -- Comprehensive Income - Total (CIQ): double"""
        return self.data["CIQ"]

    @property
    def CISECGLQ(self) -> pd.Series:
        """CISECGLQ -- Comp Inc - Securities Gains/Losses (CISECGLQ): double"""
        return self.data["CISECGLQ"]

    @property
    def CITOTALQ(self) -> pd.Series:
        """CITOTALQ -- Comprehensive Income - Parent (CITOTALQ): double"""
        return self.data["CITOTALQ"]

    @property
    def COGSQ(self) -> pd.Series:
        """COGSQ -- Cost of Goods Sold (COGSQ): double"""
        return self.data["COGSQ"]

    @property
    def CSH12Q(self) -> pd.Series:
        """CSH12Q -- Common Shares Used to Calculate Earnings Per Share - 12 Months Moving (CSH12Q): double"""
        return self.data["CSH12Q"]

    @property
    def CSHFD12(self) -> pd.Series:
        """CSHFD12 -- Common Shares Used to Calc Earnings Per Share - Fully Diluted - 12 Months M (CSHFD12): double"""
        return self.data["CSHFD12"]

    @property
    def CSHFDQ(self) -> pd.Series:
        """CSHFDQ -- Com Shares for Diluted EPS (CSHFDQ): double"""
        return self.data["CSHFDQ"]

    @property
    def CSHIQ(self) -> pd.Series:
        """CSHIQ -- Common Shares Issued (CSHIQ): double"""
        return self.data["CSHIQ"]

    @property
    def CSHOPQ(self) -> pd.Series:
        """CSHOPQ -- Total Shares Repurchased - Quarter (CSHOPQ): double"""
        return self.data["CSHOPQ"]

    @property
    def CSHOQ(self) -> pd.Series:
        """CSHOQ -- Common Shares Outstanding (CSHOQ): double"""
        return self.data["CSHOQ"]

    @property
    def CSHPRQ(self) -> pd.Series:
        """CSHPRQ -- Common Shares Used to Calculate Earnings Per Share - Basic (CSHPRQ): double"""
        return self.data["CSHPRQ"]

    @property
    def CSTKCVQ(self) -> pd.Series:
        """CSTKCVQ -- Carrying Value (CSTKCVQ): double"""
        return self.data["CSTKCVQ"]

    @property
    def CSTKEQ(self) -> pd.Series:
        """CSTKEQ -- Common Stock Equivalents - Dollar Savings (CSTKEQ): double"""
        return self.data["CSTKEQ"]

    @property
    def CSTKQ(self) -> pd.Series:
        """CSTKQ -- Common/Ordinary Stock (Capital) (CSTKQ): double"""
        return self.data["CSTKQ"]

    @property
    def DCOMQ(self) -> pd.Series:
        """DCOMQ -- Deferred Compensation (DCOMQ): double"""
        return self.data["DCOMQ"]

    @property
    def DD1Q(self) -> pd.Series:
        """DD1Q -- Long-Term Debt Due in One Year (DD1Q): double"""
        return self.data["DD1Q"]

    @property
    def DERACQ(self) -> pd.Series:
        """DERACQ -- Derivative Assets - Current (DERACQ): double"""
        return self.data["DERACQ"]

    @property
    def DERALTQ(self) -> pd.Series:
        """DERALTQ -- Derivative Assets Long-Term (DERALTQ): double"""
        return self.data["DERALTQ"]

    @property
    def DERHEDGLQ(self) -> pd.Series:
        """DERHEDGLQ -- Gains/Losses on Derivatives and Hedging (DERHEDGLQ): double"""
        return self.data["DERHEDGLQ"]

    @property
    def DERLCQ(self) -> pd.Series:
        """DERLCQ -- Derivative Liabilities- Current (DERLCQ): double"""
        return self.data["DERLCQ"]

    @property
    def DERLLTQ(self) -> pd.Series:
        """DERLLTQ -- Derivative Liabilities Long-Term (DERLLTQ): double"""
        return self.data["DERLLTQ"]

    @property
    def DILADQ(self) -> pd.Series:
        """DILADQ -- Dilution Adjustment (DILADQ): double"""
        return self.data["DILADQ"]

    @property
    def DILAVQ(self) -> pd.Series:
        """DILAVQ -- Dilution Available - Excluding Extraordinary Items (DILAVQ): double"""
        return self.data["DILAVQ"]

    @property
    def DLCQ(self) -> pd.Series:
        """DLCQ -- Debt in Current Liabilities (DLCQ): double"""
        return self.data["DLCQ"]

    @property
    def DLTTQ(self) -> pd.Series:
        """DLTTQ -- Long-Term Debt - Total (DLTTQ): double"""
        return self.data["DLTTQ"]

    @property
    def DOQ(self) -> pd.Series:
        """DOQ -- Discontinued Operations (DOQ): double"""
        return self.data["DOQ"]

    @property
    def DPACREQ(self) -> pd.Series:
        """DPACREQ -- Accumulated Depreciation of RE Property (DPACREQ): double"""
        return self.data["DPACREQ"]

    @property
    def DPACTQ(self) -> pd.Series:
        """DPACTQ -- Depreciation, Depletion and Amortization (Accumulated) (DPACTQ): double"""
        return self.data["DPACTQ"]

    @property
    def DPQ(self) -> pd.Series:
        """DPQ -- Depreciation and Amortization - Total (DPQ): double"""
        return self.data["DPQ"]

    @property
    def DPRETQ(self) -> pd.Series:
        """DPRETQ -- Depr/Amort of Property (DPRETQ): double"""
        return self.data["DPRETQ"]

    @property
    def DRCQ(self) -> pd.Series:
        """DRCQ -- Deferred Revenue - Current (DRCQ): double"""
        return self.data["DRCQ"]

    @property
    def DRLTQ(self) -> pd.Series:
        """DRLTQ -- Deferred Revenue - Long-term (DRLTQ): double"""
        return self.data["DRLTQ"]

    @property
    def DTEAQ(self) -> pd.Series:
        """DTEAQ -- Extinguishment of Debt After-tax (DTEAQ): double"""
        return self.data["DTEAQ"]

    @property
    def DTEDQ(self) -> pd.Series:
        """DTEDQ -- Extinguishment of Debt Diluted EPS Effect (DTEDQ): double"""
        return self.data["DTEDQ"]

    @property
    def DTEEPSQ(self) -> pd.Series:
        """DTEEPSQ -- Extinguishment of Debt Basic EPS Effect (DTEEPSQ): double"""
        return self.data["DTEEPSQ"]

    @property
    def DTEPQ(self) -> pd.Series:
        """DTEPQ -- Extinguishment of Debt Pretax (DTEPQ): double"""
        return self.data["DTEPQ"]

    @property
    def DVINTFQ(self) -> pd.Series:
        """DVINTFQ -- Dividends & Interest Receivable (Cash Flow) (DVINTFQ): double"""
        return self.data["DVINTFQ"]

    @property
    def DVPQ(self) -> pd.Series:
        """DVPQ -- Dividends - Preferred/Preference (DVPQ): double"""
        return self.data["DVPQ"]

    @property
    def EPSF12(self) -> pd.Series:
        """EPSF12 -- Earnings Per Share (Diluted) - Excluding Extraordinary Items - 12 Months Mo (EPSF12): double"""
        return self.data["EPSF12"]

    @property
    def EPSFI12(self) -> pd.Series:
        """EPSFI12 -- Earnings Per Share (Diluted) - Including Extraordinary Items (EPSFI12): double"""
        return self.data["EPSFI12"]

    @property
    def EPSFIQ(self) -> pd.Series:
        """EPSFIQ -- Earnings Per Share (Diluted) - Including Extraordinary Items (EPSFIQ): double"""
        return self.data["EPSFIQ"]

    @property
    def EPSFXQ(self) -> pd.Series:
        """EPSFXQ -- Earnings Per Share (Diluted) - Excluding Extraordinary items (EPSFXQ): double"""
        return self.data["EPSFXQ"]

    @property
    def EPSPI12(self) -> pd.Series:
        """EPSPI12 -- Earnings Per Share (Basic) - Including Extraordinary Items - 12 Months Movi (EPSPI12): double"""
        return self.data["EPSPI12"]

    @property
    def EPSPIQ(self) -> pd.Series:
        """EPSPIQ -- Earnings Per Share (Basic) - Including Extraordinary Items (EPSPIQ): double"""
        return self.data["EPSPIQ"]

    @property
    def EPSPXQ(self) -> pd.Series:
        """EPSPXQ -- Earnings Per Share (Basic) - Excluding Extraordinary Items (EPSPXQ): double"""
        return self.data["EPSPXQ"]

    @property
    def EPSX12(self) -> pd.Series:
        """EPSX12 -- Earnings Per Share (Basic) - Excluding Extraordinary Items - 12 Months Movi (EPSX12): double"""
        return self.data["EPSX12"]

    @property
    def ESOPCTQ(self) -> pd.Series:
        """ESOPCTQ -- Common ESOP Obligation - Total (ESOPCTQ): double"""
        return self.data["ESOPCTQ"]

    @property
    def ESOPNRQ(self) -> pd.Series:
        """ESOPNRQ -- Preferred ESOP Obligation - Non-Redeemable (ESOPNRQ): double"""
        return self.data["ESOPNRQ"]

    @property
    def ESOPRQ(self) -> pd.Series:
        """ESOPRQ -- Preferred ESOP Obligation - Redeemable (ESOPRQ): double"""
        return self.data["ESOPRQ"]

    @property
    def ESOPTQ(self) -> pd.Series:
        """ESOPTQ -- Preferred ESOP Obligation - Total (ESOPTQ): double"""
        return self.data["ESOPTQ"]

    @property
    def ESUBQ(self) -> pd.Series:
        """ESUBQ -- Equity in Earnings (I/S) - Unconsolidated Subsidiaries (ESUBQ): double"""
        return self.data["ESUBQ"]

    @property
    def FCAQ(self) -> pd.Series:
        """FCAQ -- Foreign Exchange Income (Loss) (FCAQ): double"""
        return self.data["FCAQ"]

    @property
    def FFOQ(self) -> pd.Series:
        """FFOQ -- Funds From Operations (REIT) (FFOQ): double"""
        return self.data["FFOQ"]

    @property
    def FINACOQ(self) -> pd.Series:
        """FINACOQ -- Finance Division Other Current Assets, Total (FINACOQ): double"""
        return self.data["FINACOQ"]

    @property
    def FINAOQ(self) -> pd.Series:
        """FINAOQ -- Finance Division Other Long-Term Assets, Total (FINAOQ): double"""
        return self.data["FINAOQ"]

    @property
    def FINCHQ(self) -> pd.Series:
        """FINCHQ -- Finance Division - Cash (FINCHQ): double"""
        return self.data["FINCHQ"]

    @property
    def FINDLCQ(self) -> pd.Series:
        """FINDLCQ -- Finance Division Long-Term Debt - Current (FINDLCQ): double"""
        return self.data["FINDLCQ"]

    @property
    def FINDLTQ(self) -> pd.Series:
        """FINDLTQ -- Finance Division Debt - Long-Term (FINDLTQ): double"""
        return self.data["FINDLTQ"]

    @property
    def FINIVSTQ(self) -> pd.Series:
        """FINIVSTQ -- Finance Division - Short-Term Investments (FINIVSTQ): double"""
        return self.data["FINIVSTQ"]

    @property
    def FINLCOQ(self) -> pd.Series:
        """FINLCOQ -- Finance Division Other Current Liabilities, Total (FINLCOQ): double"""
        return self.data["FINLCOQ"]

    @property
    def FINLTOQ(self) -> pd.Series:
        """FINLTOQ -- Finance Division Other Long Term Liabilities, Total (FINLTOQ): double"""
        return self.data["FINLTOQ"]

    @property
    def FINNPQ(self) -> pd.Series:
        """FINNPQ -- Finance Division Notes Payable (FINNPQ): double"""
        return self.data["FINNPQ"]

    @property
    def FINRECCQ(self) -> pd.Series:
        """FINRECCQ -- Finance Division - Current Receivables (FINRECCQ): double"""
        return self.data["FINRECCQ"]

    @property
    def FINRECLTQ(self) -> pd.Series:
        """FINRECLTQ -- Finance Division - Long-Term Receivables (FINRECLTQ): double"""
        return self.data["FINRECLTQ"]

    @property
    def FINREVQ(self) -> pd.Series:
        """FINREVQ -- Finance Division Revenue (FINREVQ): double"""
        return self.data["FINREVQ"]

    @property
    def FINXINTQ(self) -> pd.Series:
        """FINXINTQ -- Finance Division Interest Expense (FINXINTQ): double"""
        return self.data["FINXINTQ"]

    @property
    def FINXOPRQ(self) -> pd.Series:
        """FINXOPRQ -- Finance Division Operating Expense (FINXOPRQ): double"""
        return self.data["FINXOPRQ"]

    @property
    def FYR(self) -> pd.Series:
        """FYR -- Fiscal Year-end Month (FYR): double"""
        return self.data["FYR"]

    @property
    def GDWLAMQ(self) -> pd.Series:
        """GDWLAMQ -- Amortization of Goodwill (GDWLAMQ): double"""
        return self.data["GDWLAMQ"]

    @property
    def GDWLIA12(self) -> pd.Series:
        """GDWLIA12 -- Impairments of Goodwill AfterTax - 12mm (GDWLIA12): double"""
        return self.data["GDWLIA12"]

    @property
    def GDWLIAQ(self) -> pd.Series:
        """GDWLIAQ -- Impairment of Goodwill After-tax (GDWLIAQ): double"""
        return self.data["GDWLIAQ"]

    @property
    def GDWLID12(self) -> pd.Series:
        """GDWLID12 -- Impairments Diluted EPS - 12mm (GDWLID12): double"""
        return self.data["GDWLID12"]

    @property
    def GDWLIDQ(self) -> pd.Series:
        """GDWLIDQ -- Impairment of Goodwill Diluted EPS Effect (GDWLIDQ): double"""
        return self.data["GDWLIDQ"]

    @property
    def GDWLIEPS12(self) -> pd.Series:
        """GDWLIEPS12 -- Impairment of Goodwill Basic EPS Effect 12MM (GDWLIEPS12): double"""
        return self.data["GDWLIEPS12"]

    @property
    def GDWLIEPSQ(self) -> pd.Series:
        """GDWLIEPSQ -- Impairment of Goodwill Basic EPS Effect (GDWLIEPSQ): double"""
        return self.data["GDWLIEPSQ"]

    @property
    def GDWLIPQ(self) -> pd.Series:
        """GDWLIPQ -- Impairment of Goodwill Pretax (GDWLIPQ): double"""
        return self.data["GDWLIPQ"]

    @property
    def GDWLQ(self) -> pd.Series:
        """GDWLQ -- Goodwill (net) (GDWLQ): double"""
        return self.data["GDWLQ"]

    @property
    def GLAQ(self) -> pd.Series:
        """GLAQ -- Gain/Loss After-Tax (GLAQ): double"""
        return self.data["GLAQ"]

    @property
    def GLCEA12(self) -> pd.Series:
        """GLCEA12 -- Gain/Loss on Sale (Core Earnings Adjusted) After-tax 12MM (GLCEA12): double"""
        return self.data["GLCEA12"]

    @property
    def GLCEAQ(self) -> pd.Series:
        """GLCEAQ -- Gain/Loss on Sale (Core Earnings Adjusted) After-tax (GLCEAQ): double"""
        return self.data["GLCEAQ"]

    @property
    def GLCED12(self) -> pd.Series:
        """GLCED12 -- Gain/Loss on Sale (Core Earnings Adjusted) Diluted EPS Effect 12MM (GLCED12): double"""
        return self.data["GLCED12"]

    @property
    def GLCEDQ(self) -> pd.Series:
        """GLCEDQ -- Gain/Loss on Sale (Core Earnings Adjusted) Diluted EPS (GLCEDQ): double"""
        return self.data["GLCEDQ"]

    @property
    def GLCEEPS12(self) -> pd.Series:
        """GLCEEPS12 -- Gain/Loss on Sale (Core Earnings Adjusted) Basic EPS Effect 12MM (GLCEEPS12): double"""
        return self.data["GLCEEPS12"]

    @property
    def GLCEEPSQ(self) -> pd.Series:
        """GLCEEPSQ -- Gain/Loss on Sale (Core Earnings Adjusted) Basic EPS Effect (GLCEEPSQ): double"""
        return self.data["GLCEEPSQ"]

    @property
    def GLCEPQ(self) -> pd.Series:
        """GLCEPQ -- Gain/Loss on Sale (Core Earnings Adjusted) Pretax (GLCEPQ): double"""
        return self.data["GLCEPQ"]

    @property
    def GLDQ(self) -> pd.Series:
        """GLDQ -- Gain/Loss Diluted EPS Effect (GLDQ): double"""
        return self.data["GLDQ"]

    @property
    def GLEPSQ(self) -> pd.Series:
        """GLEPSQ -- Gain/Loss Basic EPS Effect (GLEPSQ): double"""
        return self.data["GLEPSQ"]

    @property
    def GLIVQ(self) -> pd.Series:
        """GLIVQ -- Gains/Losses on investments (GLIVQ): double"""
        return self.data["GLIVQ"]

    @property
    def GLPQ(self) -> pd.Series:
        """GLPQ -- Gain/Loss Pretax (GLPQ): double"""
        return self.data["GLPQ"]

    @property
    def HEDGEGLQ(self) -> pd.Series:
        """HEDGEGLQ -- Gain/Loss on Ineffective Hedges (HEDGEGLQ): double"""
        return self.data["HEDGEGLQ"]

    @property
    def IBADJ12(self) -> pd.Series:
        """IBADJ12 -- Income Before Extra Items - Adj for Common Stock Equivalents - 12MM (IBADJ12): double"""
        return self.data["IBADJ12"]

    @property
    def IBADJQ(self) -> pd.Series:
        """IBADJQ -- Income Before Extraordinary Items - Adjusted for Common Stock Equivalents (IBADJQ): double"""
        return self.data["IBADJQ"]

    @property
    def IBCOMQ(self) -> pd.Series:
        """IBCOMQ -- Income Before Extraordinary Items - Available for Common (IBCOMQ): double"""
        return self.data["IBCOMQ"]

    @property
    def IBMIIQ(self) -> pd.Series:
        """IBMIIQ -- Income before Extraordinary Items and Noncontrolling Interests (IBMIIQ): double"""
        return self.data["IBMIIQ"]

    @property
    def IBQ(self) -> pd.Series:
        """IBQ -- Income Before Extraordinary Items (IBQ): double"""
        return self.data["IBQ"]

    @property
    def ICAPTQ(self) -> pd.Series:
        """ICAPTQ -- Invested Capital - Total - Quarterly (ICAPTQ): double"""
        return self.data["ICAPTQ"]

    @property
    def INTACCQ(self) -> pd.Series:
        """INTACCQ -- Interest Accrued (INTACCQ): double"""
        return self.data["INTACCQ"]

    @property
    def INTANOQ(self) -> pd.Series:
        """INTANOQ -- Other Intangibles (INTANOQ): double"""
        return self.data["INTANOQ"]

    @property
    def INTANQ(self) -> pd.Series:
        """INTANQ -- Intangible Assets - Total (INTANQ): double"""
        return self.data["INTANQ"]

    @property
    def INVFGQ(self) -> pd.Series:
        """INVFGQ -- Inventory - Finished Goods (INVFGQ): double"""
        return self.data["INVFGQ"]

    @property
    def INVOQ(self) -> pd.Series:
        """INVOQ -- Inventory - Other (INVOQ): double"""
        return self.data["INVOQ"]

    @property
    def INVRMQ(self) -> pd.Series:
        """INVRMQ -- Inventory - Raw Materials (INVRMQ): double"""
        return self.data["INVRMQ"]

    @property
    def INVTQ(self) -> pd.Series:
        """INVTQ -- Inventories - Total (INVTQ): double"""
        return self.data["INVTQ"]

    @property
    def INVWIPQ(self) -> pd.Series:
        """INVWIPQ -- Inventory - Work in Process (INVWIPQ): double"""
        return self.data["INVWIPQ"]

    @property
    def IVAEQQ(self) -> pd.Series:
        """IVAEQQ -- Investment and Advances - Equity (IVAEQQ): double"""
        return self.data["IVAEQQ"]

    @property
    def IVAOQ(self) -> pd.Series:
        """IVAOQ -- Investment and Advances - Other (IVAOQ): double"""
        return self.data["IVAOQ"]

    @property
    def IVLTQ(self) -> pd.Series:
        """IVLTQ -- Total Long-term Investments (IVLTQ): double"""
        return self.data["IVLTQ"]

    @property
    def IVSTQ(self) -> pd.Series:
        """IVSTQ -- Short-Term Investments- Total (IVSTQ): double"""
        return self.data["IVSTQ"]

    @property
    def LCOQ(self) -> pd.Series:
        """LCOQ -- Current Liabilities - Other - Total (LCOQ): double"""
        return self.data["LCOQ"]

    @property
    def LCTQ(self) -> pd.Series:
        """LCTQ -- Current Liabilities - Total (LCTQ): double"""
        return self.data["LCTQ"]

    @property
    def LLTQ(self) -> pd.Series:
        """LLTQ -- Long-Term Liabilities (Total) (LLTQ): double"""
        return self.data["LLTQ"]

    @property
    def LNOQ(self) -> pd.Series:
        """LNOQ -- Liabilities Netting & Other Adjustments (LNOQ): double"""
        return self.data["LNOQ"]

    @property
    def LOL2Q(self) -> pd.Series:
        """LOL2Q -- Liabilities Level2 (Observable) (LOL2Q): double"""
        return self.data["LOL2Q"]

    @property
    def LOQ(self) -> pd.Series:
        """LOQ -- Liabilities - Other (LOQ): double"""
        return self.data["LOQ"]

    @property
    def LOXDRQ(self) -> pd.Series:
        """LOXDRQ -- Liabilities - Other - Excluding Deferred Revenue (LOXDRQ): double"""
        return self.data["LOXDRQ"]

    @property
    def LQPL1Q(self) -> pd.Series:
        """LQPL1Q -- Liabilities Level1 (Quoted Prices) (LQPL1Q): double"""
        return self.data["LQPL1Q"]

    @property
    def LSEQ(self) -> pd.Series:
        """LSEQ -- Liabilities and Stockholders Equity - Total (LSEQ): double"""
        return self.data["LSEQ"]

    @property
    def LTMIBQ(self) -> pd.Series:
        """LTMIBQ -- Liabilities - Total and Noncontrolling Interest (LTMIBQ): double"""
        return self.data["LTMIBQ"]

    @property
    def LTQ(self) -> pd.Series:
        """LTQ -- Liabilities - Total (LTQ): double"""
        return self.data["LTQ"]

    @property
    def LUL3Q(self) -> pd.Series:
        """LUL3Q -- Liabilities Level3 (Unobservable) (LUL3Q): double"""
        return self.data["LUL3Q"]

    @property
    def MIBNQ(self) -> pd.Series:
        """MIBNQ -- Noncontrolling Interests - Nonredeemable - Balance Sheet (MIBNQ): double"""
        return self.data["MIBNQ"]

    @property
    def MIBQ(self) -> pd.Series:
        """MIBQ -- Noncontrolling Interest - Redeemable - Balance Sheet (MIBQ): double"""
        return self.data["MIBQ"]

    @property
    def MIBTQ(self) -> pd.Series:
        """MIBTQ -- Noncontrolling Interests - Total - Balance Sheet (MIBTQ): double"""
        return self.data["MIBTQ"]

    @property
    def MIIQ(self) -> pd.Series:
        """MIIQ -- Noncontrolling Interest - Income Account (MIIQ): double"""
        return self.data["MIIQ"]

    @property
    def MSAQ(self) -> pd.Series:
        """MSAQ -- Accum Other Comp Inc - Marketable Security Adjustments (MSAQ): double"""
        return self.data["MSAQ"]

    @property
    def NCOQ(self) -> pd.Series:
        """NCOQ -- Net Charge-Offs (NCOQ): double"""
        return self.data["NCOQ"]

    @property
    def NIITQ(self) -> pd.Series:
        """NIITQ -- Net Interest Income (Tax Equivalent) (NIITQ): double"""
        return self.data["NIITQ"]

    @property
    def NIMQ(self) -> pd.Series:
        """NIMQ -- Net Interest Margin (NIMQ): double"""
        return self.data["NIMQ"]

    @property
    def NIQ(self) -> pd.Series:
        """NIQ -- Net Income (Loss) (NIQ): double"""
        return self.data["NIQ"]

    @property
    def NOPIQ(self) -> pd.Series:
        """NOPIQ -- Non-Operating Income (Expense) - Total (NOPIQ): double"""
        return self.data["NOPIQ"]

    @property
    def NPATQ(self) -> pd.Series:
        """NPATQ -- Nonperforming Assets - Total (NPATQ): double"""
        return self.data["NPATQ"]

    @property
    def NPQ(self) -> pd.Series:
        """NPQ -- Notes Payable (NPQ): double"""
        return self.data["NPQ"]

    @property
    def NRTXTDQ(self) -> pd.Series:
        """NRTXTDQ -- Nonrecurring Income Taxes Diluted EPS Effect (NRTXTDQ): double"""
        return self.data["NRTXTDQ"]

    @property
    def NRTXTEPSQ(self) -> pd.Series:
        """NRTXTEPSQ -- Nonrecurring Income Taxes Basic EPS Effect (NRTXTEPSQ): double"""
        return self.data["NRTXTEPSQ"]

    @property
    def NRTXTQ(self) -> pd.Series:
        """NRTXTQ -- Nonrecurring Income Taxes - After-tax (NRTXTQ): double"""
        return self.data["NRTXTQ"]

    @property
    def OBKQ(self) -> pd.Series:
        """OBKQ -- Order backlog (OBKQ): double"""
        return self.data["OBKQ"]

    @property
    def OEPF12(self) -> pd.Series:
        """OEPF12 -- Earnings Per Share - Diluted - from Operations - 12MM (OEPF12): double"""
        return self.data["OEPF12"]

    @property
    def OEPS12(self) -> pd.Series:
        """OEPS12 -- Earnings Per Share from Operations - 12 Months Moving (OEPS12): double"""
        return self.data["OEPS12"]

    @property
    def OEPSXQ(self) -> pd.Series:
        """OEPSXQ -- Earnings Per Share - Diluted - from Operations (OEPSXQ): double"""
        return self.data["OEPSXQ"]

    @property
    def OIADPQ(self) -> pd.Series:
        """OIADPQ -- Operating Income After Depreciation - Quarterly (OIADPQ): double"""
        return self.data["OIADPQ"]

    @property
    def OIBDPQ(self) -> pd.Series:
        """OIBDPQ -- Operating Income Before Depreciation - Quarterly (OIBDPQ): double"""
        return self.data["OIBDPQ"]

    @property
    def OPEPSQ(self) -> pd.Series:
        """OPEPSQ -- Earnings Per Share from Operations (OPEPSQ): double"""
        return self.data["OPEPSQ"]

    @property
    def OPTDRQ(self) -> pd.Series:
        """OPTDRQ -- Dividend Rate - Assumption (%) (OPTDRQ): double"""
        return self.data["OPTDRQ"]

    @property
    def OPTFVGRQ(self) -> pd.Series:
        """OPTFVGRQ -- Options - Fair Value of Options Granted (OPTFVGRQ): double"""
        return self.data["OPTFVGRQ"]

    @property
    def OPTLIFEQ(self) -> pd.Series:
        """OPTLIFEQ -- Life of Options - Assumption (# yrs) (OPTLIFEQ): double"""
        return self.data["OPTLIFEQ"]

    @property
    def OPTRFRQ(self) -> pd.Series:
        """OPTRFRQ -- Risk Free Rate - Assumption (%) (OPTRFRQ): double"""
        return self.data["OPTRFRQ"]

    @property
    def OPTVOLQ(self) -> pd.Series:
        """OPTVOLQ -- Volatility - Assumption (%) (OPTVOLQ): double"""
        return self.data["OPTVOLQ"]

    @property
    def PIQ(self) -> pd.Series:
        """PIQ -- Pretax Income (PIQ): double"""
        return self.data["PIQ"]

    @property
    def PLLQ(self) -> pd.Series:
        """PLLQ -- Provision for Loan/Asset Losses (PLLQ): double"""
        return self.data["PLLQ"]

    @property
    def PNC12(self) -> pd.Series:
        """PNC12 -- Pension Core Adjustment - 12mm (PNC12): double"""
        return self.data["PNC12"]

    @property
    def PNCD12(self) -> pd.Series:
        """PNCD12 -- Core Pension Adjustment Diluted EPS Effect 12MM (PNCD12): double"""
        return self.data["PNCD12"]

    @property
    def PNCDQ(self) -> pd.Series:
        """PNCDQ -- Core Pension Adjustment Diluted EPS Effect (PNCDQ): double"""
        return self.data["PNCDQ"]

    @property
    def PNCEPS12(self) -> pd.Series:
        """PNCEPS12 -- Core Pension Adjustment Basic EPS Effect 12MM (PNCEPS12): double"""
        return self.data["PNCEPS12"]

    @property
    def PNCEPSQ(self) -> pd.Series:
        """PNCEPSQ -- Core Pension Adjustment Basic EPS Effect (PNCEPSQ): double"""
        return self.data["PNCEPSQ"]

    @property
    def PNCIAPQ(self) -> pd.Series:
        """PNCIAPQ -- Core Pension Interest Adjustment After-tax Preliminary (PNCIAPQ): double"""
        return self.data["PNCIAPQ"]

    @property
    def PNCIAQ(self) -> pd.Series:
        """PNCIAQ -- Core Pension Interest Adjustment After-tax (PNCIAQ): double"""
        return self.data["PNCIAQ"]

    @property
    def PNCIDPQ(self) -> pd.Series:
        """PNCIDPQ -- Core Pension Interest Adjustment Diluted EPS Effect Preliminary (PNCIDPQ): double"""
        return self.data["PNCIDPQ"]

    @property
    def PNCIDQ(self) -> pd.Series:
        """PNCIDQ -- Core Pension Interest Adjustment Diluted EPS Effect (PNCIDQ): double"""
        return self.data["PNCIDQ"]

    @property
    def PNCIEPSPQ(self) -> pd.Series:
        """PNCIEPSPQ -- Core Pension Interest Adjustment Basic EPS Effect Preliminary (PNCIEPSPQ): double"""
        return self.data["PNCIEPSPQ"]

    @property
    def PNCIEPSQ(self) -> pd.Series:
        """PNCIEPSQ -- Core Pension Interest Adjustment Basic EPS Effect (PNCIEPSQ): double"""
        return self.data["PNCIEPSQ"]

    @property
    def PNCIPPQ(self) -> pd.Series:
        """PNCIPPQ -- Core Pension Interest Adjustment Pretax Preliminary (PNCIPPQ): double"""
        return self.data["PNCIPPQ"]

    @property
    def PNCIPQ(self) -> pd.Series:
        """PNCIPQ -- Core Pension Interest Adjustment Pretax (PNCIPQ): double"""
        return self.data["PNCIPQ"]

    @property
    def PNCPD12(self) -> pd.Series:
        """PNCPD12 -- Core Pension Adjustment 12MM Diluted EPS Effect Preliminary (PNCPD12): double"""
        return self.data["PNCPD12"]

    @property
    def PNCPDQ(self) -> pd.Series:
        """PNCPDQ -- Core Pension Adjustment Diluted EPS Effect Preliminary (PNCPDQ): double"""
        return self.data["PNCPDQ"]

    @property
    def PNCPEPS12(self) -> pd.Series:
        """PNCPEPS12 -- Core Pension Adjustment 12MM Basic EPS Effect Preliminary (PNCPEPS12): double"""
        return self.data["PNCPEPS12"]

    @property
    def PNCPEPSQ(self) -> pd.Series:
        """PNCPEPSQ -- Core Pension Adjustment Basic EPS Effect Preliminary (PNCPEPSQ): double"""
        return self.data["PNCPEPSQ"]

    @property
    def PNCPQ(self) -> pd.Series:
        """PNCPQ -- Core Pension Adjustment Preliminary (PNCPQ): double"""
        return self.data["PNCPQ"]

    @property
    def PNCQ(self) -> pd.Series:
        """PNCQ -- Core Pension Adjustment (PNCQ): double"""
        return self.data["PNCQ"]

    @property
    def PNCWIAPQ(self) -> pd.Series:
        """PNCWIAPQ -- Core Pension w/o Interest Adjustment After-tax Preliminary (PNCWIAPQ): double"""
        return self.data["PNCWIAPQ"]

    @property
    def PNCWIAQ(self) -> pd.Series:
        """PNCWIAQ -- Core Pension w/o Interest Adjustment After-tax (PNCWIAQ): double"""
        return self.data["PNCWIAQ"]

    @property
    def PNCWIDPQ(self) -> pd.Series:
        """PNCWIDPQ -- Core Pension w/o Interest Adjustment Diluted EPS Effect Preliminary (PNCWIDPQ): double"""
        return self.data["PNCWIDPQ"]

    @property
    def PNCWIDQ(self) -> pd.Series:
        """PNCWIDQ -- Core Pension w/o Interest Adjustment Diluted EPS Effect (PNCWIDQ): double"""
        return self.data["PNCWIDQ"]

    @property
    def PNCWIEPQ(self) -> pd.Series:
        """PNCWIEPQ -- Core Pension w/o Interest Adjustment Basic EPS Effect Preliminary (PNCWIEPQ): double"""
        return self.data["PNCWIEPQ"]

    @property
    def PNCWIEPSQ(self) -> pd.Series:
        """PNCWIEPSQ -- Core Pension w/o Interest Adjustment Basic EPS Effect (PNCWIEPSQ): double"""
        return self.data["PNCWIEPSQ"]

    @property
    def PNCWIPPQ(self) -> pd.Series:
        """PNCWIPPQ -- Core Pension w/o Interest Adjustment Pretax Preliminary (PNCWIPPQ): double"""
        return self.data["PNCWIPPQ"]

    @property
    def PNCWIPQ(self) -> pd.Series:
        """PNCWIPQ -- Core Pension w/o Interest Adjustment Pretax (PNCWIPQ): double"""
        return self.data["PNCWIPQ"]

    @property
    def PNRSHOQ(self) -> pd.Series:
        """PNRSHOQ -- Nonred Pfd Shares Outs (000) - Quarterly (PNRSHOQ): double"""
        return self.data["PNRSHOQ"]

    @property
    def PPEGTQ(self) -> pd.Series:
        """PPEGTQ -- Property, Plant and Equipment - Total (Gross) - Quarterly (PPEGTQ): double"""
        return self.data["PPEGTQ"]

    @property
    def PPENTQ(self) -> pd.Series:
        """PPENTQ -- Property Plant and Equipment - Total (Net) (PPENTQ): double"""
        return self.data["PPENTQ"]

    @property
    def PRCAQ(self) -> pd.Series:
        """PRCAQ -- Core Post Retirement Adjustment (PRCAQ): double"""
        return self.data["PRCAQ"]

    @property
    def PRCD12(self) -> pd.Series:
        """PRCD12 -- Core Post Retirement Adjustment Diluted EPS Effect 12MM (PRCD12): double"""
        return self.data["PRCD12"]

    @property
    def PRCDQ(self) -> pd.Series:
        """PRCDQ -- Core Post Retirement Adjustment Diluted EPS Effect (PRCDQ): double"""
        return self.data["PRCDQ"]

    @property
    def PRCE12(self) -> pd.Series:
        """PRCE12 -- Core Post Retirement Adjustment 12MM (PRCE12): double"""
        return self.data["PRCE12"]

    @property
    def PRCEPS12(self) -> pd.Series:
        """PRCEPS12 -- Core Post Retirement Adjustment Basic EPS Effect 12MM (PRCEPS12): double"""
        return self.data["PRCEPS12"]

    @property
    def PRCEPSQ(self) -> pd.Series:
        """PRCEPSQ -- Core Post Retirement Adjustment Basic EPS Effect (PRCEPSQ): double"""
        return self.data["PRCEPSQ"]

    @property
    def PRCPD12(self) -> pd.Series:
        """PRCPD12 -- Core Post Retirement Adjustment 12MM Diluted EPS Effect Preliminary (PRCPD12): double"""
        return self.data["PRCPD12"]

    @property
    def PRCPDQ(self) -> pd.Series:
        """PRCPDQ -- Core Post Retirement Adjustment Diluted EPS Effect Preliminary (PRCPDQ): double"""
        return self.data["PRCPDQ"]

    @property
    def PRCPEPS12(self) -> pd.Series:
        """PRCPEPS12 -- Core Post Retirement Adjustment 12MM Basic EPS Effect Preliminary (PRCPEPS12): double"""
        return self.data["PRCPEPS12"]

    @property
    def PRCPEPSQ(self) -> pd.Series:
        """PRCPEPSQ -- Core Post Retirement Adjustment Basic EPS Effect Preliminary (PRCPEPSQ): double"""
        return self.data["PRCPEPSQ"]

    @property
    def PRCPQ(self) -> pd.Series:
        """PRCPQ -- Core Post Retirement Adjustment Preliminary (PRCPQ): double"""
        return self.data["PRCPQ"]

    @property
    def PRCRAQ(self) -> pd.Series:
        """PRCRAQ -- Repurchase Price - Average per share Quarter (PRCRAQ): double"""
        return self.data["PRCRAQ"]

    @property
    def PRSHOQ(self) -> pd.Series:
        """PRSHOQ -- Redeem Pfd Shares Outs (000) (PRSHOQ): double"""
        return self.data["PRSHOQ"]

    @property
    def PSTKNQ(self) -> pd.Series:
        """PSTKNQ -- Preferred/Preference Stock - Nonredeemable (PSTKNQ): double"""
        return self.data["PSTKNQ"]

    @property
    def PSTKQ(self) -> pd.Series:
        """PSTKQ -- Preferred/Preference Stock (Capital) - Total (PSTKQ): double"""
        return self.data["PSTKQ"]

    @property
    def PSTKRQ(self) -> pd.Series:
        """PSTKRQ -- Preferred/Preference Stock - Redeemable (PSTKRQ): double"""
        return self.data["PSTKRQ"]

    @property
    def RCAQ(self) -> pd.Series:
        """RCAQ -- Restructuring Cost After-tax (RCAQ): double"""
        return self.data["RCAQ"]

    @property
    def RCDQ(self) -> pd.Series:
        """RCDQ -- Restructuring Cost Diluted EPS Effect (RCDQ): double"""
        return self.data["RCDQ"]

    @property
    def RCEPSQ(self) -> pd.Series:
        """RCEPSQ -- Restructuring Cost Basic EPS Effect (RCEPSQ): double"""
        return self.data["RCEPSQ"]

    @property
    def RCPQ(self) -> pd.Series:
        """RCPQ -- Restructuring Cost Pretax (RCPQ): double"""
        return self.data["RCPQ"]

    @property
    def RDIPAQ(self) -> pd.Series:
        """RDIPAQ -- In Process R&D Expense After-tax (RDIPAQ): double"""
        return self.data["RDIPAQ"]

    @property
    def RDIPDQ(self) -> pd.Series:
        """RDIPDQ -- In Process R&D Expense Diluted EPS Effect (RDIPDQ): double"""
        return self.data["RDIPDQ"]

    @property
    def RDIPEPSQ(self) -> pd.Series:
        """RDIPEPSQ -- In Process R&D Expense Basic EPS Effect (RDIPEPSQ): double"""
        return self.data["RDIPEPSQ"]

    @property
    def RDIPQ(self) -> pd.Series:
        """RDIPQ -- In Process R&D (RDIPQ): double"""
        return self.data["RDIPQ"]

    @property
    def RECDQ(self) -> pd.Series:
        """RECDQ -- Receivables - Estimated Doubtful (RECDQ): double"""
        return self.data["RECDQ"]

    @property
    def RECTAQ(self) -> pd.Series:
        """RECTAQ -- Accum Other Comp Inc - Cumulative Translation Adjustments (RECTAQ): double"""
        return self.data["RECTAQ"]

    @property
    def RECTOQ(self) -> pd.Series:
        """RECTOQ -- Receivables - Current Other incl Tax Refunds (RECTOQ): double"""
        return self.data["RECTOQ"]

    @property
    def RECTQ(self) -> pd.Series:
        """RECTQ -- Receivables - Total (RECTQ): double"""
        return self.data["RECTQ"]

    @property
    def RECTRQ(self) -> pd.Series:
        """RECTRQ -- Receivables - Trade (RECTRQ): double"""
        return self.data["RECTRQ"]

    @property
    def RECUBQ(self) -> pd.Series:
        """RECUBQ -- Unbilled Receivables - Quarterly (RECUBQ): double"""
        return self.data["RECUBQ"]

    @property
    def REQ(self) -> pd.Series:
        """REQ -- Retained Earnings (REQ): double"""
        return self.data["REQ"]

    @property
    def RETQ(self) -> pd.Series:
        """RETQ -- Total RE Property (RETQ): double"""
        return self.data["RETQ"]

    @property
    def REUNAQ(self) -> pd.Series:
        """REUNAQ -- Unadjusted Retained Earnings (REUNAQ): double"""
        return self.data["REUNAQ"]

    @property
    def REVTQ(self) -> pd.Series:
        """REVTQ -- Revenue - Total (REVTQ): double"""
        return self.data["REVTQ"]

    @property
    def RLLQ(self) -> pd.Series:
        """RLLQ -- Reserve for Loan/Asset Losses (RLLQ): double"""
        return self.data["RLLQ"]

    @property
    def RRA12(self) -> pd.Series:
        """RRA12 -- Reversal - Restructruring/Acquisition Aftertax 12MM (RRA12): double"""
        return self.data["RRA12"]

    @property
    def RRAQ(self) -> pd.Series:
        """RRAQ -- Reversal - Restructruring/Acquisition Aftertax (RRAQ): double"""
        return self.data["RRAQ"]

    @property
    def RRD12(self) -> pd.Series:
        """RRD12 -- Reversal - Restructuring/Acq Diluted EPS Effect 12MM (RRD12): double"""
        return self.data["RRD12"]

    @property
    def RRDQ(self) -> pd.Series:
        """RRDQ -- Reversal - Restructuring/Acq Diluted EPS Effect (RRDQ): double"""
        return self.data["RRDQ"]

    @property
    def RREPS12(self) -> pd.Series:
        """RREPS12 -- Reversal - Restructuring/Acq Basic EPS Effect 12MM (RREPS12): double"""
        return self.data["RREPS12"]

    @property
    def RREPSQ(self) -> pd.Series:
        """RREPSQ -- Reversal - Restructuring/Acq Basic EPS Effect (RREPSQ): double"""
        return self.data["RREPSQ"]

    @property
    def RRPQ(self) -> pd.Series:
        """RRPQ -- Reversal - Restructruring/Acquisition Pretax (RRPQ): double"""
        return self.data["RRPQ"]

    @property
    def RSTCHELTQ(self) -> pd.Series:
        """RSTCHELTQ -- Long-Term Restricted Cash & Investments (RSTCHELTQ): double"""
        return self.data["RSTCHELTQ"]

    @property
    def RSTCHEQ(self) -> pd.Series:
        """RSTCHEQ -- Restricted Cash & Investments - Current (RSTCHEQ): double"""
        return self.data["RSTCHEQ"]

    @property
    def SALEQ(self) -> pd.Series:
        """SALEQ -- Sales/Turnover (Net) (SALEQ): double"""
        return self.data["SALEQ"]

    @property
    def SEQOQ(self) -> pd.Series:
        """SEQOQ -- Other Stockholders- Equity Adjustments (SEQOQ): double"""
        return self.data["SEQOQ"]

    @property
    def SEQQ(self) -> pd.Series:
        """SEQQ -- Stockholders Equity > Parent > Index Fundamental > Quarterly (SEQQ): double"""
        return self.data["SEQQ"]

    @property
    def SETA12(self) -> pd.Series:
        """SETA12 -- Settlement (Litigation/Insurance) AfterTax - 12mm (SETA12): double"""
        return self.data["SETA12"]

    @property
    def SETAQ(self) -> pd.Series:
        """SETAQ -- Settlement (Litigation/Insurance) After-tax (SETAQ): double"""
        return self.data["SETAQ"]

    @property
    def SETD12(self) -> pd.Series:
        """SETD12 -- Settlement (Litigation/Insurance) Diluted EPS Effect 12MM (SETD12): double"""
        return self.data["SETD12"]

    @property
    def SETDQ(self) -> pd.Series:
        """SETDQ -- Settlement (Litigation/Insurance) Diluted EPS Effect (SETDQ): double"""
        return self.data["SETDQ"]

    @property
    def SETEPS12(self) -> pd.Series:
        """SETEPS12 -- Settlement (Litigation/Insurance) Basic EPS Effect 12MM (SETEPS12): double"""
        return self.data["SETEPS12"]

    @property
    def SETEPSQ(self) -> pd.Series:
        """SETEPSQ -- Settlement (Litigation/Insurance) Basic EPS Effect (SETEPSQ): double"""
        return self.data["SETEPSQ"]

    @property
    def SETPQ(self) -> pd.Series:
        """SETPQ -- Settlement (Litigation/Insurance) Pretax (SETPQ): double"""
        return self.data["SETPQ"]

    @property
    def SPCE12(self) -> pd.Series:
        """SPCE12 -- S&P Core Earnings 12MM (SPCE12): double"""
        return self.data["SPCE12"]

    @property
    def SPCED12(self) -> pd.Series:
        """SPCED12 -- S&P Core Earnings EPS Diluted 12MM (SPCED12): double"""
        return self.data["SPCED12"]

    @property
    def SPCEDPQ(self) -> pd.Series:
        """SPCEDPQ -- S&P Core Earnings EPS Diluted - Preliminary (SPCEDPQ): double"""
        return self.data["SPCEDPQ"]

    @property
    def SPCEDQ(self) -> pd.Series:
        """SPCEDQ -- S&P Core Earnings EPS Diluted (SPCEDQ): double"""
        return self.data["SPCEDQ"]

    @property
    def SPCEEPS12(self) -> pd.Series:
        """SPCEEPS12 -- S&P Core Earnings EPS Basic 12MM (SPCEEPS12): double"""
        return self.data["SPCEEPS12"]

    @property
    def SPCEEPSP12(self) -> pd.Series:
        """SPCEEPSP12 -- S&P Core 12MM EPS - Basic - Preliminary (SPCEEPSP12): double"""
        return self.data["SPCEEPSP12"]

    @property
    def SPCEEPSPQ(self) -> pd.Series:
        """SPCEEPSPQ -- S&P Core Earnings EPS Basic - Preliminary (SPCEEPSPQ): double"""
        return self.data["SPCEEPSPQ"]

    @property
    def SPCEEPSQ(self) -> pd.Series:
        """SPCEEPSQ -- S&P Core Earnings EPS Basic (SPCEEPSQ): double"""
        return self.data["SPCEEPSQ"]

    @property
    def SPCEP12(self) -> pd.Series:
        """SPCEP12 -- S&P Core Earnings 12MM - Preliminary (SPCEP12): double"""
        return self.data["SPCEP12"]

    @property
    def SPCEPD12(self) -> pd.Series:
        """SPCEPD12 -- S&P Core Earnings 12MM EPS Diluted - Preliminary (SPCEPD12): double"""
        return self.data["SPCEPD12"]

    @property
    def SPCEPQ(self) -> pd.Series:
        """SPCEPQ -- S&P Core Earnings - Preliminary (SPCEPQ): double"""
        return self.data["SPCEPQ"]

    @property
    def SPCEQ(self) -> pd.Series:
        """SPCEQ -- S&P Core Earnings (SPCEQ): double"""
        return self.data["SPCEQ"]

    @property
    def SPIDQ(self) -> pd.Series:
        """SPIDQ -- Other Special Items Diluted EPS Effect (SPIDQ): double"""
        return self.data["SPIDQ"]

    @property
    def SPIEPSQ(self) -> pd.Series:
        """SPIEPSQ -- Other Special Items Basic EPS Effect (SPIEPSQ): double"""
        return self.data["SPIEPSQ"]

    @property
    def SPIOAQ(self) -> pd.Series:
        """SPIOAQ -- Other Special Items After-tax (SPIOAQ): double"""
        return self.data["SPIOAQ"]

    @property
    def SPIOPQ(self) -> pd.Series:
        """SPIOPQ -- Other Special Items Pretax (SPIOPQ): double"""
        return self.data["SPIOPQ"]

    @property
    def SPIQ(self) -> pd.Series:
        """SPIQ -- Special Items (SPIQ): double"""
        return self.data["SPIQ"]

    @property
    def SRETQ(self) -> pd.Series:
        """SRETQ -- Gain/Loss on Sale of Property (SRETQ): double"""
        return self.data["SRETQ"]

    @property
    def STKCOQ(self) -> pd.Series:
        """STKCOQ -- Stock Compensation Expense (STKCOQ): double"""
        return self.data["STKCOQ"]

    @property
    def STKCPAQ(self) -> pd.Series:
        """STKCPAQ -- After-tax stock compensation (STKCPAQ): double"""
        return self.data["STKCPAQ"]

    @property
    def TEQQ(self) -> pd.Series:
        """TEQQ -- Stockholders Equity - Total (TEQQ): double"""
        return self.data["TEQQ"]

    @property
    def TFVAQ(self) -> pd.Series:
        """TFVAQ -- Total Fair Value Assets (TFVAQ): double"""
        return self.data["TFVAQ"]

    @property
    def TFVCEQ(self) -> pd.Series:
        """TFVCEQ -- Total Fair Value Changes including Earnings (TFVCEQ): double"""
        return self.data["TFVCEQ"]

    @property
    def TFVLQ(self) -> pd.Series:
        """TFVLQ -- Total Fair Value Liabilities (TFVLQ): double"""
        return self.data["TFVLQ"]

    @property
    def TIEQ(self) -> pd.Series:
        """TIEQ -- Interest Expense - Total (Financial Services) (TIEQ): double"""
        return self.data["TIEQ"]

    @property
    def TIIQ(self) -> pd.Series:
        """TIIQ -- Interest Income - Total (Financial Services) (TIIQ): double"""
        return self.data["TIIQ"]

    @property
    def TSTKNQ(self) -> pd.Series:
        """TSTKNQ -- Treasury Stock - Number of Common Shares (TSTKNQ): double"""
        return self.data["TSTKNQ"]

    @property
    def TSTKQ(self) -> pd.Series:
        """TSTKQ -- Treasury Stock - Total (All Capital) (TSTKQ): double"""
        return self.data["TSTKQ"]

    @property
    def TXDBAQ(self) -> pd.Series:
        """TXDBAQ -- Deferred Tax Asset - Long Term (TXDBAQ): double"""
        return self.data["TXDBAQ"]

    @property
    def TXDBCAQ(self) -> pd.Series:
        """TXDBCAQ -- Current Deferred Tax Asset (TXDBCAQ): double"""
        return self.data["TXDBCAQ"]

    @property
    def TXDBCLQ(self) -> pd.Series:
        """TXDBCLQ -- Current Deferred Tax Liability (TXDBCLQ): double"""
        return self.data["TXDBCLQ"]

    @property
    def TXDBQ(self) -> pd.Series:
        """TXDBQ -- Deferred Taxes - Balance Sheet (TXDBQ): double"""
        return self.data["TXDBQ"]

    @property
    def TXDIQ(self) -> pd.Series:
        """TXDIQ -- Income Taxes - Deferred (TXDIQ): double"""
        return self.data["TXDIQ"]

    @property
    def TXDITCQ(self) -> pd.Series:
        """TXDITCQ -- Deferred Taxes and Investment Tax Credit (TXDITCQ): double"""
        return self.data["TXDITCQ"]

    @property
    def TXPQ(self) -> pd.Series:
        """TXPQ -- Income Taxes Payable (TXPQ): double"""
        return self.data["TXPQ"]

    @property
    def TXTQ(self) -> pd.Series:
        """TXTQ -- Income Taxes - Total (TXTQ): double"""
        return self.data["TXTQ"]

    @property
    def TXWQ(self) -> pd.Series:
        """TXWQ -- Excise Taxes (TXWQ): double"""
        return self.data["TXWQ"]

    @property
    def UACOQ(self) -> pd.Series:
        """UACOQ -- Current Assets - Other - Utility (UACOQ): double"""
        return self.data["UACOQ"]

    @property
    def UAOQ(self) -> pd.Series:
        """UAOQ -- Other Assets - Utility (UAOQ): double"""
        return self.data["UAOQ"]

    @property
    def UAPTQ(self) -> pd.Series:
        """UAPTQ -- Accounts Payable - Utility (UAPTQ): double"""
        return self.data["UAPTQ"]

    @property
    def UCAPSQ(self) -> pd.Series:
        """UCAPSQ -- Paid In Capital - Other - Utility (UCAPSQ): double"""
        return self.data["UCAPSQ"]

    @property
    def UCCONSQ(self) -> pd.Series:
        """UCCONSQ -- Contributions In Aid Of Construction (UCCONSQ): double"""
        return self.data["UCCONSQ"]

    @property
    def UCEQQ(self) -> pd.Series:
        """UCEQQ -- Common Equity - Total - Utility (UCEQQ): double"""
        return self.data["UCEQQ"]

    @property
    def UDDQ(self) -> pd.Series:
        """UDDQ -- Debt (Debentures) - Utility (UDDQ): double"""
        return self.data["UDDQ"]

    @property
    def UDMBQ(self) -> pd.Series:
        """UDMBQ -- Debt (Mortgage Bonds) (UDMBQ): double"""
        return self.data["UDMBQ"]

    @property
    def UDOLTQ(self) -> pd.Series:
        """UDOLTQ -- Debt (Other Long-Term) (UDOLTQ): double"""
        return self.data["UDOLTQ"]

    @property
    def UDPCOQ(self) -> pd.Series:
        """UDPCOQ -- Debt (Pollution Control Obligations) (UDPCOQ): double"""
        return self.data["UDPCOQ"]

    @property
    def UDVPQ(self) -> pd.Series:
        """UDVPQ -- Preferred Dividend Requirements (UDVPQ): double"""
        return self.data["UDVPQ"]

    @property
    def UGIQ(self) -> pd.Series:
        """UGIQ -- Gross Income (Income Before Interest Charges) (UGIQ): double"""
        return self.data["UGIQ"]

    @property
    def UINVQ(self) -> pd.Series:
        """UINVQ -- Inventories (UINVQ): double"""
        return self.data["UINVQ"]

    @property
    def ULCOQ(self) -> pd.Series:
        """ULCOQ -- Current Liabilities - Other (ULCOQ): double"""
        return self.data["ULCOQ"]

    @property
    def UNIAMIQ(self) -> pd.Series:
        """UNIAMIQ -- Net Income before Extraordinary Items After Noncontrolling Interest (UNIAMIQ): double"""
        return self.data["UNIAMIQ"]

    @property
    def UNOPINCQ(self) -> pd.Series:
        """UNOPINCQ -- Nonoperating Income (Net) - Other (UNOPINCQ): double"""
        return self.data["UNOPINCQ"]

    @property
    def UOPIQ(self) -> pd.Series:
        """UOPIQ -- Operating Income - Total - Utility (UOPIQ): double"""
        return self.data["UOPIQ"]

    @property
    def UPDVPQ(self) -> pd.Series:
        """UPDVPQ -- Preference Dividend Requirements - Utility (UPDVPQ): double"""
        return self.data["UPDVPQ"]

    @property
    def UPMCSTKQ(self) -> pd.Series:
        """UPMCSTKQ -- Premium On Common Stock - Utility (UPMCSTKQ): double"""
        return self.data["UPMCSTKQ"]

    @property
    def UPMPFQ(self) -> pd.Series:
        """UPMPFQ -- Premium On Preferred Stock - Utility (UPMPFQ): double"""
        return self.data["UPMPFQ"]

    @property
    def UPMPFSQ(self) -> pd.Series:
        """UPMPFSQ -- Premium On Preference Stock - Utility (UPMPFSQ): double"""
        return self.data["UPMPFSQ"]

    @property
    def UPMSUBPQ(self) -> pd.Series:
        """UPMSUBPQ -- Premium On Subsidiary Preferred Stock - Utility (UPMSUBPQ): double"""
        return self.data["UPMSUBPQ"]

    @property
    def UPSTKCQ(self) -> pd.Series:
        """UPSTKCQ -- Preference Stock At Carrying Value - Utility (UPSTKCQ): double"""
        return self.data["UPSTKCQ"]

    @property
    def UPSTKQ(self) -> pd.Series:
        """UPSTKQ -- Preferred Stock At Carrying Value - Utility (UPSTKQ): double"""
        return self.data["UPSTKQ"]

    @property
    def URECTQ(self) -> pd.Series:
        """URECTQ -- Receivables (Net) - Utility (URECTQ): double"""
        return self.data["URECTQ"]

    @property
    def USPIQ(self) -> pd.Series:
        """USPIQ -- Special Items - Utility (USPIQ): double"""
        return self.data["USPIQ"]

    @property
    def USUBDVPQ(self) -> pd.Series:
        """USUBDVPQ -- Subsidiary Preferred Dividends - Utility (USUBDVPQ): double"""
        return self.data["USUBDVPQ"]

    @property
    def USUBPCVQ(self) -> pd.Series:
        """USUBPCVQ -- Subsidiary Preferred Stock At Carrying Value - Utility (USUBPCVQ): double"""
        return self.data["USUBPCVQ"]

    @property
    def UTEMQ(self) -> pd.Series:
        """UTEMQ -- Maintenance Expense - Total (UTEMQ): double"""
        return self.data["UTEMQ"]

    @property
    def WCAPQ(self) -> pd.Series:
        """WCAPQ -- Working Capital (Balance Sheet) (WCAPQ): double"""
        return self.data["WCAPQ"]

    @property
    def WDAQ(self) -> pd.Series:
        """WDAQ -- Writedowns After-tax (WDAQ): double"""
        return self.data["WDAQ"]

    @property
    def WDDQ(self) -> pd.Series:
        """WDDQ -- Writedowns Diluted EPS Effect (WDDQ): double"""
        return self.data["WDDQ"]

    @property
    def WDEPSQ(self) -> pd.Series:
        """WDEPSQ -- Writedowns Basic EPS Effect (WDEPSQ): double"""
        return self.data["WDEPSQ"]

    @property
    def WDPQ(self) -> pd.Series:
        """WDPQ -- Writedowns Pretax (WDPQ): double"""
        return self.data["WDPQ"]

    @property
    def XACCQ(self) -> pd.Series:
        """XACCQ -- Accrued Expenses (XACCQ): double"""
        return self.data["XACCQ"]

    @property
    def XIDOQ(self) -> pd.Series:
        """XIDOQ -- Extraordinary Items and Discontinued Operations (XIDOQ): double"""
        return self.data["XIDOQ"]

    @property
    def XINTQ(self) -> pd.Series:
        """XINTQ -- Interest and Related Expense- Total (XINTQ): double"""
        return self.data["XINTQ"]

    @property
    def XIQ(self) -> pd.Series:
        """XIQ -- Extraordinary Items (XIQ): double"""
        return self.data["XIQ"]

    @property
    def XOPRQ(self) -> pd.Series:
        """XOPRQ -- Operating Expense- Total (XOPRQ): double"""
        return self.data["XOPRQ"]

    @property
    def XOPT12(self) -> pd.Series:
        """XOPT12 -- Implied Option Expense - 12mm (XOPT12): double"""
        return self.data["XOPT12"]

    @property
    def XOPTD12(self) -> pd.Series:
        """XOPTD12 -- Implied Option EPS Diluted 12MM (XOPTD12): double"""
        return self.data["XOPTD12"]

    @property
    def XOPTD12P(self) -> pd.Series:
        """XOPTD12P -- Implied Option 12MM EPS Diluted Preliminary (XOPTD12P): double"""
        return self.data["XOPTD12P"]

    @property
    def XOPTDQ(self) -> pd.Series:
        """XOPTDQ -- Implied Option EPS Diluted (XOPTDQ): double"""
        return self.data["XOPTDQ"]

    @property
    def XOPTDQP(self) -> pd.Series:
        """XOPTDQP -- Implied Option EPS Diluted Preliminary (XOPTDQP): double"""
        return self.data["XOPTDQP"]

    @property
    def XOPTEPS12(self) -> pd.Series:
        """XOPTEPS12 -- Implied Option EPS Basic 12MM (XOPTEPS12): double"""
        return self.data["XOPTEPS12"]

    @property
    def XOPTEPSP12(self) -> pd.Series:
        """XOPTEPSP12 -- Implied Option 12MM EPS Basic Preliminary (XOPTEPSP12): double"""
        return self.data["XOPTEPSP12"]

    @property
    def XOPTEPSQ(self) -> pd.Series:
        """XOPTEPSQ -- Implied Option EPS Basic (XOPTEPSQ): double"""
        return self.data["XOPTEPSQ"]

    @property
    def XOPTEPSQP(self) -> pd.Series:
        """XOPTEPSQP -- Implied Option EPS Basic Preliminary (XOPTEPSQP): double"""
        return self.data["XOPTEPSQP"]

    @property
    def XOPTQ(self) -> pd.Series:
        """XOPTQ -- Implied Option Expense (XOPTQ): double"""
        return self.data["XOPTQ"]

    @property
    def XOPTQP(self) -> pd.Series:
        """XOPTQP -- Implied Option Expense Preliminary (XOPTQP): double"""
        return self.data["XOPTQP"]

    @property
    def XRDQ(self) -> pd.Series:
        """XRDQ -- Research and Development Expense (XRDQ): double"""
        return self.data["XRDQ"]

    @property
    def XSGAQ(self) -> pd.Series:
        """XSGAQ -- Selling, General and Administrative Expenses (XSGAQ): double"""
        return self.data["XSGAQ"]

    @property
    def ACCHGY(self) -> pd.Series:
        """ACCHGY -- Accounting Changes - Cumulative Effect (ACCHGY): double"""
        return self.data["ACCHGY"]

    @property
    def AFUDCCY(self) -> pd.Series:
        """AFUDCCY -- Allowance for Funds Used During Construction (Cash Flow) (AFUDCCY): double"""
        return self.data["AFUDCCY"]

    @property
    def AFUDCIY(self) -> pd.Series:
        """AFUDCIY -- Allowance for Funds Used During Construction (Investing) (Cash Flow) (AFUDCIY): double"""
        return self.data["AFUDCIY"]

    @property
    def AMCY(self) -> pd.Series:
        """AMCY -- Amortization (Cash Flow) (AMCY): double"""
        return self.data["AMCY"]

    @property
    def AOLOCHY(self) -> pd.Series:
        """AOLOCHY -- Assets and Liabilities - Other (Net Change) (AOLOCHY): double"""
        return self.data["AOLOCHY"]

    @property
    def APALCHY(self) -> pd.Series:
        """APALCHY -- Accounts Payable and Accrued Liabilities - Increase (Decrease) (APALCHY): double"""
        return self.data["APALCHY"]

    @property
    def AQAY(self) -> pd.Series:
        """AQAY -- Acquisition/Merger After-Tax (AQAY): double"""
        return self.data["AQAY"]

    @property
    def AQCY(self) -> pd.Series:
        """AQCY -- Acquisitions (AQCY): double"""
        return self.data["AQCY"]

    @property
    def AQDY(self) -> pd.Series:
        """AQDY -- Acquisition/Merger Diluted EPS Effect (AQDY): double"""
        return self.data["AQDY"]

    @property
    def AQEPSY(self) -> pd.Series:
        """AQEPSY -- Acquisition/Merger Basic EPS Effect (AQEPSY): double"""
        return self.data["AQEPSY"]

    @property
    def AQPY(self) -> pd.Series:
        """AQPY -- Acquisition/Merger Pretax (AQPY): double"""
        return self.data["AQPY"]

    @property
    def ARCEDY(self) -> pd.Series:
        """ARCEDY -- As Reported Core - Diluted EPS Effect (ARCEDY): double"""
        return self.data["ARCEDY"]

    @property
    def ARCEEPSY(self) -> pd.Series:
        """ARCEEPSY -- As Reported Core - Basic EPS Effect (ARCEEPSY): double"""
        return self.data["ARCEEPSY"]

    @property
    def ARCEY(self) -> pd.Series:
        """ARCEY -- As Reported Core - After-tax (ARCEY): double"""
        return self.data["ARCEY"]

    @property
    def CAPXY(self) -> pd.Series:
        """CAPXY -- Capital Expenditures (CAPXY): double"""
        return self.data["CAPXY"]

    @property
    def CDVCY(self) -> pd.Series:
        """CDVCY -- Cash Dividends on Common Stock (Cash Flow) (CDVCY): double"""
        return self.data["CDVCY"]

    @property
    def CHECHY(self) -> pd.Series:
        """CHECHY -- Cash and Cash Equivalents - Increase (Decrease) (CHECHY): double"""
        return self.data["CHECHY"]

    @property
    def CIBEGNIY(self) -> pd.Series:
        """CIBEGNIY -- Comp Inc - Beginning Net Income (CIBEGNIY): double"""
        return self.data["CIBEGNIY"]

    @property
    def CICURRY(self) -> pd.Series:
        """CICURRY -- Comp Inc - Currency Trans Adj (CICURRY): double"""
        return self.data["CICURRY"]

    @property
    def CIDERGLY(self) -> pd.Series:
        """CIDERGLY -- Comp Inc - Derivative Gains/Losses (CIDERGLY): double"""
        return self.data["CIDERGLY"]

    @property
    def CIMIIY(self) -> pd.Series:
        """CIMIIY -- Comprehensive Income - Noncontrolling Interest (CIMIIY): double"""
        return self.data["CIMIIY"]

    @property
    def CIOTHERY(self) -> pd.Series:
        """CIOTHERY -- Comp Inc - Other Adj (CIOTHERY): double"""
        return self.data["CIOTHERY"]

    @property
    def CIPENY(self) -> pd.Series:
        """CIPENY -- Comp Inc - Minimum Pension Adj (CIPENY): double"""
        return self.data["CIPENY"]

    @property
    def CISECGLY(self) -> pd.Series:
        """CISECGLY -- Comp Inc - Securities Gains/Losses (CISECGLY): double"""
        return self.data["CISECGLY"]

    @property
    def CITOTALY(self) -> pd.Series:
        """CITOTALY -- Comprehensive Income - Parent (CITOTALY): double"""
        return self.data["CITOTALY"]

    @property
    def CIY(self) -> pd.Series:
        """CIY -- Comprehensive Income - Total (CIY): double"""
        return self.data["CIY"]

    @property
    def COGSY(self) -> pd.Series:
        """COGSY -- Cost of Goods Sold (COGSY): double"""
        return self.data["COGSY"]

    @property
    def CSHFDY(self) -> pd.Series:
        """CSHFDY -- Com Shares for Diluted EPS (CSHFDY): double"""
        return self.data["CSHFDY"]

    @property
    def CSHPRY(self) -> pd.Series:
        """CSHPRY -- Common Shares Used to Calculate Earnings Per Share - Basic (CSHPRY): double"""
        return self.data["CSHPRY"]

    @property
    def CSTKEY(self) -> pd.Series:
        """CSTKEY -- Common Stock Equivalents - Dollar Savings (CSTKEY): double"""
        return self.data["CSTKEY"]

    @property
    def DEPCY(self) -> pd.Series:
        """DEPCY -- Depreciation and Depletion (Cash Flow) (DEPCY): double"""
        return self.data["DEPCY"]

    @property
    def DERHEDGLY(self) -> pd.Series:
        """DERHEDGLY -- Gains/Losses on Derivatives and Hedging (DERHEDGLY): double"""
        return self.data["DERHEDGLY"]

    @property
    def DILADY(self) -> pd.Series:
        """DILADY -- Dilution Adjustment (DILADY): double"""
        return self.data["DILADY"]

    @property
    def DILAVY(self) -> pd.Series:
        """DILAVY -- Dilution Available - Excluding Extraordinary Items (DILAVY): double"""
        return self.data["DILAVY"]

    @property
    def DLCCHY(self) -> pd.Series:
        """DLCCHY -- Changes in Current Debt (DLCCHY): double"""
        return self.data["DLCCHY"]

    @property
    def DLTISY(self) -> pd.Series:
        """DLTISY -- Long-Term Debt - Issuance (DLTISY): double"""
        return self.data["DLTISY"]

    @property
    def DLTRY(self) -> pd.Series:
        """DLTRY -- Long-Term Debt - Reduction (DLTRY): double"""
        return self.data["DLTRY"]

    @property
    def DOY(self) -> pd.Series:
        """DOY -- Discontinued Operations (DOY): double"""
        return self.data["DOY"]

    @property
    def DPCY(self) -> pd.Series:
        """DPCY -- Depreciation and Amortization - Statement of Cash Flows (DPCY): double"""
        return self.data["DPCY"]

    @property
    def DPRETY(self) -> pd.Series:
        """DPRETY -- Depr/Amort of Property (DPRETY): double"""
        return self.data["DPRETY"]

    @property
    def DPY(self) -> pd.Series:
        """DPY -- Depreciation and Amortization - Total (DPY): double"""
        return self.data["DPY"]

    @property
    def DTEAY(self) -> pd.Series:
        """DTEAY -- Extinguishment of Debt After-tax (DTEAY): double"""
        return self.data["DTEAY"]

    @property
    def DTEDY(self) -> pd.Series:
        """DTEDY -- Extinguishment of Debt Diluted EPS Effect (DTEDY): double"""
        return self.data["DTEDY"]

    @property
    def DTEEPSY(self) -> pd.Series:
        """DTEEPSY -- Extinguishment of Debt Basic EPS Effect (DTEEPSY): double"""
        return self.data["DTEEPSY"]

    @property
    def DTEPY(self) -> pd.Series:
        """DTEPY -- Extinguishment of Debt Pretax (DTEPY): double"""
        return self.data["DTEPY"]

    @property
    def DVPY(self) -> pd.Series:
        """DVPY -- Dividends - Preferred/Preference (DVPY): double"""
        return self.data["DVPY"]

    @property
    def DVY(self) -> pd.Series:
        """DVY -- Cash Dividends (DVY): double"""
        return self.data["DVY"]

    @property
    def EPSFIY(self) -> pd.Series:
        """EPSFIY -- Earnings Per Share (Diluted) - Including Extraordinary Items (EPSFIY): double"""
        return self.data["EPSFIY"]

    @property
    def EPSFXY(self) -> pd.Series:
        """EPSFXY -- Earnings Per Share (Diluted) - Excluding Extraordinary items (EPSFXY): double"""
        return self.data["EPSFXY"]

    @property
    def EPSPIY(self) -> pd.Series:
        """EPSPIY -- Earnings Per Share (Basic) - Including Extraordinary Items (EPSPIY): double"""
        return self.data["EPSPIY"]

    @property
    def EPSPXY(self) -> pd.Series:
        """EPSPXY -- Earnings Per Share (Basic) - Excluding Extraordinary Items (EPSPXY): double"""
        return self.data["EPSPXY"]

    @property
    def ESUBCY(self) -> pd.Series:
        """ESUBCY -- Equity in Net Loss/Earnings (C/F) (ESUBCY): double"""
        return self.data["ESUBCY"]

    @property
    def ESUBY(self) -> pd.Series:
        """ESUBY -- Equity in Earnings (I/S)- Unconsolidated Subsidiaries (ESUBY): double"""
        return self.data["ESUBY"]

    @property
    def EXREY(self) -> pd.Series:
        """EXREY -- Exchange Rate Effect (EXREY): double"""
        return self.data["EXREY"]

    @property
    def FCAY(self) -> pd.Series:
        """FCAY -- Foreign Exchange Income (Loss) (FCAY): double"""
        return self.data["FCAY"]

    @property
    def FFOY(self) -> pd.Series:
        """FFOY -- Funds From Operations (REIT) (FFOY): double"""
        return self.data["FFOY"]

    @property
    def FIAOY(self) -> pd.Series:
        """FIAOY -- Financing Activities - Other (FIAOY): double"""
        return self.data["FIAOY"]

    @property
    def FINCFY(self) -> pd.Series:
        """FINCFY -- Financing Activities - Net Cash Flow (FINCFY): double"""
        return self.data["FINCFY"]

    @property
    def FINREVY(self) -> pd.Series:
        """FINREVY -- Finance Division Revenue (FINREVY): double"""
        return self.data["FINREVY"]

    @property
    def FINXINTY(self) -> pd.Series:
        """FINXINTY -- Finance Division Interest Expense (FINXINTY): double"""
        return self.data["FINXINTY"]

    @property
    def FINXOPRY(self) -> pd.Series:
        """FINXOPRY -- Finance Division Operating Expense (FINXOPRY): double"""
        return self.data["FINXOPRY"]

    @property
    def FOPOXY(self) -> pd.Series:
        """FOPOXY -- Funds from Operations - Other excluding Option Tax Benefit (FOPOXY): double"""
        return self.data["FOPOXY"]

    @property
    def FOPOY(self) -> pd.Series:
        """FOPOY -- Funds from Operations - Other (FOPOY): double"""
        return self.data["FOPOY"]

    @property
    def FOPTY(self) -> pd.Series:
        """FOPTY -- Funds From Operations - Total (FOPTY): double"""
        return self.data["FOPTY"]

    @property
    def FSRCOY(self) -> pd.Series:
        """FSRCOY -- Sources of Funds - Other (FSRCOY): double"""
        return self.data["FSRCOY"]

    @property
    def FSRCTY(self) -> pd.Series:
        """FSRCTY -- Sources of Funds - Total (FSRCTY): double"""
        return self.data["FSRCTY"]

    @property
    def FUSEOY(self) -> pd.Series:
        """FUSEOY -- Uses of Funds - Other (FUSEOY): double"""
        return self.data["FUSEOY"]

    @property
    def FUSETY(self) -> pd.Series:
        """FUSETY -- Uses of Funds - Total (FUSETY): double"""
        return self.data["FUSETY"]

    @property
    def FYR(self) -> pd.Series:
        """FYR -- Fiscal Year-end Month (FYR): double"""
        return self.data["FYR"]

    @property
    def GDWLAMY(self) -> pd.Series:
        """GDWLAMY -- Amortization of Goodwill (GDWLAMY): double"""
        return self.data["GDWLAMY"]

    @property
    def GDWLIAY(self) -> pd.Series:
        """GDWLIAY -- Impairment of Goodwill After-tax (GDWLIAY): double"""
        return self.data["GDWLIAY"]

    @property
    def GDWLIDY(self) -> pd.Series:
        """GDWLIDY -- Impairment of Goodwill Diluted EPS Effect (GDWLIDY): double"""
        return self.data["GDWLIDY"]

    @property
    def GDWLIEPSY(self) -> pd.Series:
        """GDWLIEPSY -- Impairment of Goodwill Basic EPS Effect (GDWLIEPSY): double"""
        return self.data["GDWLIEPSY"]

    @property
    def GDWLIPY(self) -> pd.Series:
        """GDWLIPY -- Impairment of Goodwill Pretax (GDWLIPY): double"""
        return self.data["GDWLIPY"]

    @property
    def GLAY(self) -> pd.Series:
        """GLAY -- Gain/Loss After-Tax (GLAY): double"""
        return self.data["GLAY"]

    @property
    def GLCEAY(self) -> pd.Series:
        """GLCEAY -- Gain/Loss on Sale (Core Earnings Adjusted) After-tax (GLCEAY): double"""
        return self.data["GLCEAY"]

    @property
    def GLCEDY(self) -> pd.Series:
        """GLCEDY -- Gain/Loss on Sale (Core Earnings Adjusted) Diluted EPS (GLCEDY): double"""
        return self.data["GLCEDY"]

    @property
    def GLCEEPSY(self) -> pd.Series:
        """GLCEEPSY -- Gain/Loss on Sale (Core Earnings Adjusted) Basic EPS Effect (GLCEEPSY): double"""
        return self.data["GLCEEPSY"]

    @property
    def GLCEPY(self) -> pd.Series:
        """GLCEPY -- Gain/Loss on Sale (Core Earnings Adjusted) Pretax (GLCEPY): double"""
        return self.data["GLCEPY"]

    @property
    def GLDY(self) -> pd.Series:
        """GLDY -- Gain/Loss Diluted EPS Effect (GLDY): double"""
        return self.data["GLDY"]

    @property
    def GLEPSY(self) -> pd.Series:
        """GLEPSY -- Gain/Loss Basic EPS Effect (GLEPSY): double"""
        return self.data["GLEPSY"]

    @property
    def GLIVY(self) -> pd.Series:
        """GLIVY -- Gains/Losses on investments (GLIVY): double"""
        return self.data["GLIVY"]

    @property
    def GLPY(self) -> pd.Series:
        """GLPY -- Gain/Loss Pretax (GLPY): double"""
        return self.data["GLPY"]

    @property
    def HEDGEGLY(self) -> pd.Series:
        """HEDGEGLY -- Gain/Loss on Ineffective Hedges (HEDGEGLY): double"""
        return self.data["HEDGEGLY"]

    @property
    def IBADJY(self) -> pd.Series:
        """IBADJY -- Income Before Extraordinary Items - Adjusted for Common Stock Equivalents (IBADJY): double"""
        return self.data["IBADJY"]

    @property
    def IBCOMY(self) -> pd.Series:
        """IBCOMY -- Income Before Extraordinary Items - Available for Common (IBCOMY): double"""
        return self.data["IBCOMY"]

    @property
    def IBCY(self) -> pd.Series:
        """IBCY -- Income Before Extraordinary Items - Statement of Cash Flows (IBCY): double"""
        return self.data["IBCY"]

    @property
    def IBMIIY(self) -> pd.Series:
        """IBMIIY -- Income before Extraordinary Items and Noncontrolling Interests (IBMIIY): double"""
        return self.data["IBMIIY"]

    @property
    def IBY(self) -> pd.Series:
        """IBY -- Income Before Extraordinary Items (IBY): double"""
        return self.data["IBY"]

    @property
    def INTPNY(self) -> pd.Series:
        """INTPNY -- Interest Paid - Net (INTPNY): double"""
        return self.data["INTPNY"]

    @property
    def INVCHY(self) -> pd.Series:
        """INVCHY -- Inventory - Decrease (Increase) (INVCHY): double"""
        return self.data["INVCHY"]

    @property
    def ITCCY(self) -> pd.Series:
        """ITCCY -- Investment Tax Credit - Net (Cash Flow) (ITCCY): double"""
        return self.data["ITCCY"]

    @property
    def IVACOY(self) -> pd.Series:
        """IVACOY -- Investing Activities - Other (IVACOY): double"""
        return self.data["IVACOY"]

    @property
    def IVCHY(self) -> pd.Series:
        """IVCHY -- Increase in Investments (IVCHY): double"""
        return self.data["IVCHY"]

    @property
    def IVNCFY(self) -> pd.Series:
        """IVNCFY -- Investing Activities - Net Cash Flow (IVNCFY): double"""
        return self.data["IVNCFY"]

    @property
    def IVSTCHY(self) -> pd.Series:
        """IVSTCHY -- Short-Term Investments - Change (IVSTCHY): double"""
        return self.data["IVSTCHY"]

    @property
    def MIIY(self) -> pd.Series:
        """MIIY -- Noncontrolling Interest - Income Account (MIIY): double"""
        return self.data["MIIY"]

    @property
    def NCOY(self) -> pd.Series:
        """NCOY -- Net Charge-Offs (NCOY): double"""
        return self.data["NCOY"]

    @property
    def NIITY(self) -> pd.Series:
        """NIITY -- Net Interest Income (Tax Equivalent) (NIITY): double"""
        return self.data["NIITY"]

    @property
    def NIMY(self) -> pd.Series:
        """NIMY -- Net Interest Margin (NIMY): double"""
        return self.data["NIMY"]

    @property
    def NIY(self) -> pd.Series:
        """NIY -- Net Income (Loss) (NIY): double"""
        return self.data["NIY"]

    @property
    def NOPIY(self) -> pd.Series:
        """NOPIY -- Non-Operating Income (Expense) - Total (NOPIY): double"""
        return self.data["NOPIY"]

    @property
    def NRTXTDY(self) -> pd.Series:
        """NRTXTDY -- Nonrecurring Income Taxes Diluted EPS Effect (NRTXTDY): double"""
        return self.data["NRTXTDY"]

    @property
    def NRTXTEPSY(self) -> pd.Series:
        """NRTXTEPSY -- Nonrecurring Income Taxes Basic EPS Effect (NRTXTEPSY): double"""
        return self.data["NRTXTEPSY"]

    @property
    def NRTXTY(self) -> pd.Series:
        """NRTXTY -- Nonrecurring Income Taxes - After-tax (NRTXTY): double"""
        return self.data["NRTXTY"]

    @property
    def OANCFY(self) -> pd.Series:
        """OANCFY -- Operating Activities - Net Cash Flow (OANCFY): double"""
        return self.data["OANCFY"]

    @property
    def OEPSXY(self) -> pd.Series:
        """OEPSXY -- Earnings Per Share - Diluted - from Operations (OEPSXY): double"""
        return self.data["OEPSXY"]

    @property
    def OIADPY(self) -> pd.Series:
        """OIADPY -- Operating Income After Depreciation - Year-to-Date (OIADPY): double"""
        return self.data["OIADPY"]

    @property
    def OIBDPY(self) -> pd.Series:
        """OIBDPY -- Operating Income Before Depreciation (OIBDPY): double"""
        return self.data["OIBDPY"]

    @property
    def OPEPSY(self) -> pd.Series:
        """OPEPSY -- Earnings Per Share from Operations (OPEPSY): double"""
        return self.data["OPEPSY"]

    @property
    def OPTDRY(self) -> pd.Series:
        """OPTDRY -- Dividend Rate - Assumption (%) (OPTDRY): double"""
        return self.data["OPTDRY"]

    @property
    def OPTFVGRY(self) -> pd.Series:
        """OPTFVGRY -- Options - Fair Value of Options Granted (OPTFVGRY): double"""
        return self.data["OPTFVGRY"]

    @property
    def OPTLIFEY(self) -> pd.Series:
        """OPTLIFEY -- Life of Options - Assumption (# yrs) (OPTLIFEY): double"""
        return self.data["OPTLIFEY"]

    @property
    def OPTRFRY(self) -> pd.Series:
        """OPTRFRY -- Risk Free Rate - Assumption (%) (OPTRFRY): double"""
        return self.data["OPTRFRY"]

    @property
    def OPTVOLY(self) -> pd.Series:
        """OPTVOLY -- Volatility - Assumption (%) (OPTVOLY): double"""
        return self.data["OPTVOLY"]

    @property
    def PDVCY(self) -> pd.Series:
        """PDVCY -- Cash Dividends on Preferred/Preference Stock (Cash Flow) (PDVCY): double"""
        return self.data["PDVCY"]

    @property
    def PIY(self) -> pd.Series:
        """PIY -- Pretax Income (PIY): double"""
        return self.data["PIY"]

    @property
    def PLLY(self) -> pd.Series:
        """PLLY -- Provision for Loan/Asset Losses (PLLY): double"""
        return self.data["PLLY"]

    @property
    def PNCDY(self) -> pd.Series:
        """PNCDY -- Core Pension Adjustment Diluted EPS Effect (PNCDY): double"""
        return self.data["PNCDY"]

    @property
    def PNCEPSY(self) -> pd.Series:
        """PNCEPSY -- Core Pension Adjustment Basic EPS Effect (PNCEPSY): double"""
        return self.data["PNCEPSY"]

    @property
    def PNCIAPY(self) -> pd.Series:
        """PNCIAPY -- Core Pension Interest Adjustment After-tax Preliminary (PNCIAPY): double"""
        return self.data["PNCIAPY"]

    @property
    def PNCIAY(self) -> pd.Series:
        """PNCIAY -- Core Pension Interest Adjustment After-tax (PNCIAY): double"""
        return self.data["PNCIAY"]

    @property
    def PNCIDPY(self) -> pd.Series:
        """PNCIDPY -- Core Pension Interest Adjustment Diluted EPS Effect Preliminary (PNCIDPY): double"""
        return self.data["PNCIDPY"]

    @property
    def PNCIDY(self) -> pd.Series:
        """PNCIDY -- Core Pension Interest Adjustment Diluted EPS Effect (PNCIDY): double"""
        return self.data["PNCIDY"]

    @property
    def PNCIEPSPY(self) -> pd.Series:
        """PNCIEPSPY -- Core Pension Interest Adjustment Basic EPS Effect Preliminary (PNCIEPSPY): double"""
        return self.data["PNCIEPSPY"]

    @property
    def PNCIEPSY(self) -> pd.Series:
        """PNCIEPSY -- Core Pension Interest Adjustment Basic EPS Effect (PNCIEPSY): double"""
        return self.data["PNCIEPSY"]

    @property
    def PNCIPPY(self) -> pd.Series:
        """PNCIPPY -- Core Pension Interest Adjustment Pretax Preliminary (PNCIPPY): double"""
        return self.data["PNCIPPY"]

    @property
    def PNCIPY(self) -> pd.Series:
        """PNCIPY -- Core Pension Interest Adjustment Pretax (PNCIPY): double"""
        return self.data["PNCIPY"]

    @property
    def PNCPDY(self) -> pd.Series:
        """PNCPDY -- Core Pension Adjustment Diluted EPS Effect Preliminary (PNCPDY): double"""
        return self.data["PNCPDY"]

    @property
    def PNCPEPSY(self) -> pd.Series:
        """PNCPEPSY -- Core Pension Adjustment Basic EPS Effect Preliminary (PNCPEPSY): double"""
        return self.data["PNCPEPSY"]

    @property
    def PNCPY(self) -> pd.Series:
        """PNCPY -- Core Pension Adjustment Preliminary (PNCPY): double"""
        return self.data["PNCPY"]

    @property
    def PNCWIAPY(self) -> pd.Series:
        """PNCWIAPY -- Core Pension w/o Interest Adjustment After-tax Preliminary (PNCWIAPY): double"""
        return self.data["PNCWIAPY"]

    @property
    def PNCWIAY(self) -> pd.Series:
        """PNCWIAY -- Core Pension w/o Interest Adjustment After-tax (PNCWIAY): double"""
        return self.data["PNCWIAY"]

    @property
    def PNCWIDPY(self) -> pd.Series:
        """PNCWIDPY -- Core Pension w/o Interest Adjustment Diluted EPS Effect Preliminary (PNCWIDPY): double"""
        return self.data["PNCWIDPY"]

    @property
    def PNCWIDY(self) -> pd.Series:
        """PNCWIDY -- Core Pension w/o Interest Adjustment Diluted EPS Effect (PNCWIDY): double"""
        return self.data["PNCWIDY"]

    @property
    def PNCWIEPSY(self) -> pd.Series:
        """PNCWIEPSY -- Core Pension w/o Interest Adjustment Basic EPS Effect (PNCWIEPSY): double"""
        return self.data["PNCWIEPSY"]

    @property
    def PNCWIEPY(self) -> pd.Series:
        """PNCWIEPY -- Core Pension w/o Interest Adjustment Basic EPS Effect Preliminary (PNCWIEPY): double"""
        return self.data["PNCWIEPY"]

    @property
    def PNCWIPPY(self) -> pd.Series:
        """PNCWIPPY -- Core Pension w/o Interest Adjustment Pretax Preliminary (PNCWIPPY): double"""
        return self.data["PNCWIPPY"]

    @property
    def PNCWIPY(self) -> pd.Series:
        """PNCWIPY -- Core Pension w/o Interest Adjustment Pretax (PNCWIPY): double"""
        return self.data["PNCWIPY"]

    @property
    def PNCY(self) -> pd.Series:
        """PNCY -- Core Pension Adjustment (PNCY): double"""
        return self.data["PNCY"]

    @property
    def PRCAY(self) -> pd.Series:
        """PRCAY -- Core Post Retirement Adjustment (PRCAY): double"""
        return self.data["PRCAY"]

    @property
    def PRCDY(self) -> pd.Series:
        """PRCDY -- Core Post Retirement Adjustment Diluted EPS Effect (PRCDY): double"""
        return self.data["PRCDY"]

    @property
    def PRCEPSY(self) -> pd.Series:
        """PRCEPSY -- Core Post Retirement Adjustment Basic EPS Effect (PRCEPSY): double"""
        return self.data["PRCEPSY"]

    @property
    def PRCPDY(self) -> pd.Series:
        """PRCPDY -- Core Post Retirement Adjustment Diluted EPS Effect Preliminary (PRCPDY): double"""
        return self.data["PRCPDY"]

    @property
    def PRCPEPSY(self) -> pd.Series:
        """PRCPEPSY -- Core Post Retirement Adjustment Basic EPS Effect Preliminary (PRCPEPSY): double"""
        return self.data["PRCPEPSY"]

    @property
    def PRCPY(self) -> pd.Series:
        """PRCPY -- Core Post Retirement Adjustment Preliminary (PRCPY): double"""
        return self.data["PRCPY"]

    @property
    def PRSTKCCY(self) -> pd.Series:
        """PRSTKCCY -- Purchase of Common Stock (Cash Flow) (PRSTKCCY): double"""
        return self.data["PRSTKCCY"]

    @property
    def PRSTKCY(self) -> pd.Series:
        """PRSTKCY -- Purchase of Common and Preferred Stock (PRSTKCY): double"""
        return self.data["PRSTKCY"]

    @property
    def PRSTKPCY(self) -> pd.Series:
        """PRSTKPCY -- Purchase of Preferred/Preference Stock (Cash Flow) (PRSTKPCY): double"""
        return self.data["PRSTKPCY"]

    @property
    def RCAY(self) -> pd.Series:
        """RCAY -- Restructuring Cost After-tax (RCAY): double"""
        return self.data["RCAY"]

    @property
    def RCDY(self) -> pd.Series:
        """RCDY -- Restructuring Cost Diluted EPS Effect (RCDY): double"""
        return self.data["RCDY"]

    @property
    def RCEPSY(self) -> pd.Series:
        """RCEPSY -- Restructuring Cost Basic EPS Effect (RCEPSY): double"""
        return self.data["RCEPSY"]

    @property
    def RCPY(self) -> pd.Series:
        """RCPY -- Restructuring Cost Pretax (RCPY): double"""
        return self.data["RCPY"]

    @property
    def RDIPAY(self) -> pd.Series:
        """RDIPAY -- In Process R&D Expense After-tax (RDIPAY): double"""
        return self.data["RDIPAY"]

    @property
    def RDIPDY(self) -> pd.Series:
        """RDIPDY -- In Process R&D Expense Diluted EPS Effect (RDIPDY): double"""
        return self.data["RDIPDY"]

    @property
    def RDIPEPSY(self) -> pd.Series:
        """RDIPEPSY -- In Process R&D Expense Basic EPS Effect (RDIPEPSY): double"""
        return self.data["RDIPEPSY"]

    @property
    def RDIPY(self) -> pd.Series:
        """RDIPY -- In Process R&D (RDIPY): double"""
        return self.data["RDIPY"]

    @property
    def RECCHY(self) -> pd.Series:
        """RECCHY -- Accounts Receivable - Decrease (Increase) (RECCHY): double"""
        return self.data["RECCHY"]

    @property
    def REVTY(self) -> pd.Series:
        """REVTY -- Revenue - Total (REVTY): double"""
        return self.data["REVTY"]

    @property
    def RRAY(self) -> pd.Series:
        """RRAY -- Reversal - Restructruring/Acquisition Aftertax (RRAY): double"""
        return self.data["RRAY"]

    @property
    def RRDY(self) -> pd.Series:
        """RRDY -- Reversal - Restructuring/Acq Diluted EPS Effect (RRDY): double"""
        return self.data["RRDY"]

    @property
    def RREPSY(self) -> pd.Series:
        """RREPSY -- Reversal - Restructuring/Acq Basic EPS Effect (RREPSY): double"""
        return self.data["RREPSY"]

    @property
    def RRPY(self) -> pd.Series:
        """RRPY -- Reversal - Restructruring/Acquisition Pretax (RRPY): double"""
        return self.data["RRPY"]

    @property
    def SALEY(self) -> pd.Series:
        """SALEY -- Sales/Turnover (Net) (SALEY): double"""
        return self.data["SALEY"]

    @property
    def SCSTKCY(self) -> pd.Series:
        """SCSTKCY -- Sale of Common Stock (Cash Flow) (SCSTKCY): double"""
        return self.data["SCSTKCY"]

    @property
    def SETAY(self) -> pd.Series:
        """SETAY -- Settlement (Litigation/Insurance) After-tax (SETAY): double"""
        return self.data["SETAY"]

    @property
    def SETDY(self) -> pd.Series:
        """SETDY -- Settlement (Litigation/Insurance) Diluted EPS Effect (SETDY): double"""
        return self.data["SETDY"]

    @property
    def SETEPSY(self) -> pd.Series:
        """SETEPSY -- Settlement (Litigation/Insurance) Basic EPS Effect (SETEPSY): double"""
        return self.data["SETEPSY"]

    @property
    def SETPY(self) -> pd.Series:
        """SETPY -- Settlement (Litigation/Insurance) Pretax (SETPY): double"""
        return self.data["SETPY"]

    @property
    def SIVY(self) -> pd.Series:
        """SIVY -- Sale of Investments (SIVY): double"""
        return self.data["SIVY"]

    @property
    def SPCEDPY(self) -> pd.Series:
        """SPCEDPY -- S&P Core Earnings EPS Diluted - Preliminary (SPCEDPY): double"""
        return self.data["SPCEDPY"]

    @property
    def SPCEDY(self) -> pd.Series:
        """SPCEDY -- S&P Core Earnings EPS Diluted (SPCEDY): double"""
        return self.data["SPCEDY"]

    @property
    def SPCEEPSPY(self) -> pd.Series:
        """SPCEEPSPY -- S&P Core Earnings EPS Basic - Preliminary (SPCEEPSPY): double"""
        return self.data["SPCEEPSPY"]

    @property
    def SPCEEPSY(self) -> pd.Series:
        """SPCEEPSY -- S&P Core Earnings EPS Basic (SPCEEPSY): double"""
        return self.data["SPCEEPSY"]

    @property
    def SPCEPY(self) -> pd.Series:
        """SPCEPY -- S&P Core Earnings - Preliminary (SPCEPY): double"""
        return self.data["SPCEPY"]

    @property
    def SPCEY(self) -> pd.Series:
        """SPCEY -- S&P Core Earnings (SPCEY): double"""
        return self.data["SPCEY"]

    @property
    def SPIDY(self) -> pd.Series:
        """SPIDY -- Other Special Items Diluted EPS Effect (SPIDY): double"""
        return self.data["SPIDY"]

    @property
    def SPIEPSY(self) -> pd.Series:
        """SPIEPSY -- Other Special Items Basic EPS Effect (SPIEPSY): double"""
        return self.data["SPIEPSY"]

    @property
    def SPIOAY(self) -> pd.Series:
        """SPIOAY -- Other Special Items After-tax (SPIOAY): double"""
        return self.data["SPIOAY"]

    @property
    def SPIOPY(self) -> pd.Series:
        """SPIOPY -- Other Special Items Pretax (SPIOPY): double"""
        return self.data["SPIOPY"]

    @property
    def SPIY(self) -> pd.Series:
        """SPIY -- Special Items (SPIY): double"""
        return self.data["SPIY"]

    @property
    def SPPEY(self) -> pd.Series:
        """SPPEY -- Sale of Property (SPPEY): double"""
        return self.data["SPPEY"]

    @property
    def SPPIVY(self) -> pd.Series:
        """SPPIVY -- Sale of PP&E and Investments - (Gain) Loss (SPPIVY): double"""
        return self.data["SPPIVY"]

    @property
    def SPSTKCY(self) -> pd.Series:
        """SPSTKCY -- Sale of Preferred/Preference Stock (Cash Flow) (SPSTKCY): double"""
        return self.data["SPSTKCY"]

    @property
    def SRETY(self) -> pd.Series:
        """SRETY -- Gain/Loss on Sale of Property (SRETY): double"""
        return self.data["SRETY"]

    @property
    def SSTKY(self) -> pd.Series:
        """SSTKY -- Sale of Common and Preferred Stock (SSTKY): double"""
        return self.data["SSTKY"]

    @property
    def STKCOY(self) -> pd.Series:
        """STKCOY -- Stock Compensation Expense (STKCOY): double"""
        return self.data["STKCOY"]

    @property
    def STKCPAY(self) -> pd.Series:
        """STKCPAY -- After-tax stock compensation (STKCPAY): double"""
        return self.data["STKCPAY"]

    @property
    def TDCY(self) -> pd.Series:
        """TDCY -- Deferred Income Taxes - Net (Cash Flow) (TDCY): double"""
        return self.data["TDCY"]

    @property
    def TFVCEY(self) -> pd.Series:
        """TFVCEY -- Total Fair Value Changes including Earnings (TFVCEY): double"""
        return self.data["TFVCEY"]

    @property
    def TIEY(self) -> pd.Series:
        """TIEY -- Interest Expense - Total (Financial Services) (TIEY): double"""
        return self.data["TIEY"]

    @property
    def TIIY(self) -> pd.Series:
        """TIIY -- Interest Income - Total (Financial Services) (TIIY): double"""
        return self.data["TIIY"]

    @property
    def TSAFCY(self) -> pd.Series:
        """TSAFCY -- Total Srcs of Funds (FOF) (TSAFCY): double"""
        return self.data["TSAFCY"]

    @property
    def TXACHY(self) -> pd.Series:
        """TXACHY -- Income Taxes - Accrued - Increase (Decrease) (TXACHY): double"""
        return self.data["TXACHY"]

    @property
    def TXBCOFY(self) -> pd.Series:
        """TXBCOFY -- Excess Tax Benefit of Stock Options - Cash Flow Financing (TXBCOFY): double"""
        return self.data["TXBCOFY"]

    @property
    def TXBCOY(self) -> pd.Series:
        """TXBCOY -- Excess Tax Benefit of Stock Options - Cash Flow Operating (TXBCOY): double"""
        return self.data["TXBCOY"]

    @property
    def TXDCY(self) -> pd.Series:
        """TXDCY -- Deferred Taxes (Statement of Cash Flows) (TXDCY): double"""
        return self.data["TXDCY"]

    @property
    def TXDIY(self) -> pd.Series:
        """TXDIY -- Income Taxes - Deferred (TXDIY): double"""
        return self.data["TXDIY"]

    @property
    def TXPDY(self) -> pd.Series:
        """TXPDY -- Income Taxes Paid (TXPDY): double"""
        return self.data["TXPDY"]

    @property
    def TXTY(self) -> pd.Series:
        """TXTY -- Income Taxes - Total (TXTY): double"""
        return self.data["TXTY"]

    @property
    def TXWY(self) -> pd.Series:
        """TXWY -- Excise Taxes (TXWY): double"""
        return self.data["TXWY"]

    @property
    def UAOLOCHY(self) -> pd.Series:
        """UAOLOCHY -- Other Assets and Liabilities - Net Change (Statement of Cash Flows) (UAOLOCHY): double"""
        return self.data["UAOLOCHY"]

    @property
    def UDFCCY(self) -> pd.Series:
        """UDFCCY -- Deferred Fuel - Increase (Decrease) (Statement of Cash Flows) (UDFCCY): double"""
        return self.data["UDFCCY"]

    @property
    def UDVPY(self) -> pd.Series:
        """UDVPY -- Preferred Dividend Requirements - Utility (UDVPY): double"""
        return self.data["UDVPY"]

    @property
    def UFRETSDY(self) -> pd.Series:
        """UFRETSDY -- Tot Funds Ret ofSec&STD (FOF) (UFRETSDY): double"""
        return self.data["UFRETSDY"]

    @property
    def UGIY(self) -> pd.Series:
        """UGIY -- Gross Income (Income Before Interest Charges) - Utility (UGIY): double"""
        return self.data["UGIY"]

    @property
    def UNIAMIY(self) -> pd.Series:
        """UNIAMIY -- Net Income before Extraordinary Items After Noncontrolling Interest - Utili (UNIAMIY): double"""
        return self.data["UNIAMIY"]

    @property
    def UNOPINCY(self) -> pd.Series:
        """UNOPINCY -- Nonoperating Income (Net) - Other - Utility (UNOPINCY): double"""
        return self.data["UNOPINCY"]

    @property
    def UNWCCY(self) -> pd.Series:
        """UNWCCY -- Inc(Dec)Working Cap (FOF) (UNWCCY): double"""
        return self.data["UNWCCY"]

    @property
    def UOISY(self) -> pd.Series:
        """UOISY -- Other Internal Sources - Net (Cash Flow) (UOISY): double"""
        return self.data["UOISY"]

    @property
    def UPDVPY(self) -> pd.Series:
        """UPDVPY -- Preference Dividend Requirements - Utility (UPDVPY): double"""
        return self.data["UPDVPY"]

    @property
    def UPTACY(self) -> pd.Series:
        """UPTACY -- Utility Plant - Gross Additions (Cash Flow) (UPTACY): double"""
        return self.data["UPTACY"]

    @property
    def USPIY(self) -> pd.Series:
        """USPIY -- Special Items - Utility (USPIY): double"""
        return self.data["USPIY"]

    @property
    def USTDNCY(self) -> pd.Series:
        """USTDNCY -- Net Decr in ST Debt (FOF) (USTDNCY): double"""
        return self.data["USTDNCY"]

    @property
    def USUBDVPY(self) -> pd.Series:
        """USUBDVPY -- Subsidiary Preferred Dividends - Utility (USUBDVPY): double"""
        return self.data["USUBDVPY"]

    @property
    def UTFDOCY(self) -> pd.Series:
        """UTFDOCY -- Total Funds From Ops (FOF) (UTFDOCY): double"""
        return self.data["UTFDOCY"]

    @property
    def UTFOSCY(self) -> pd.Series:
        """UTFOSCY -- Tot Funds Frm Outside Sources (FOF) (UTFOSCY): double"""
        return self.data["UTFOSCY"]

    @property
    def UTMEY(self) -> pd.Series:
        """UTMEY -- Maintenance Expense - Total (UTMEY): double"""
        return self.data["UTMEY"]

    @property
    def UWKCAPCY(self) -> pd.Series:
        """UWKCAPCY -- Dec(Inc) in Working Capital (FOF) (UWKCAPCY): double"""
        return self.data["UWKCAPCY"]

    @property
    def WCAPCHY(self) -> pd.Series:
        """WCAPCHY -- Working Capital Changes - Total (WCAPCHY): double"""
        return self.data["WCAPCHY"]

    @property
    def WCAPCY(self) -> pd.Series:
        """WCAPCY -- Working Capital Change - Other - Increase/(Decrease) (WCAPCY): double"""
        return self.data["WCAPCY"]

    @property
    def WDAY(self) -> pd.Series:
        """WDAY -- Writedowns After-tax (WDAY): double"""
        return self.data["WDAY"]

    @property
    def WDDY(self) -> pd.Series:
        """WDDY -- Writedowns Diluted EPS Effect (WDDY): double"""
        return self.data["WDDY"]

    @property
    def WDEPSY(self) -> pd.Series:
        """WDEPSY -- Writedowns Basic EPS Effect (WDEPSY): double"""
        return self.data["WDEPSY"]

    @property
    def WDPY(self) -> pd.Series:
        """WDPY -- Writedowns Pretax (WDPY): double"""
        return self.data["WDPY"]

    @property
    def XIDOCY(self) -> pd.Series:
        """XIDOCY -- Extraordinary Items and Discontinued Operations (Statement of Cash Flows) (XIDOCY): double"""
        return self.data["XIDOCY"]

    @property
    def XIDOY(self) -> pd.Series:
        """XIDOY -- Extraordinary Items and Discontinued Operations (XIDOY): double"""
        return self.data["XIDOY"]

    @property
    def XINTY(self) -> pd.Series:
        """XINTY -- Interest and Related Expense- Total (XINTY): double"""
        return self.data["XINTY"]

    @property
    def XIY(self) -> pd.Series:
        """XIY -- Extraordinary Items (XIY): double"""
        return self.data["XIY"]

    @property
    def XOPRY(self) -> pd.Series:
        """XOPRY -- Operating Expense- Total (XOPRY): double"""
        return self.data["XOPRY"]

    @property
    def XOPTDQPY(self) -> pd.Series:
        """XOPTDQPY -- Implied Option EPS Diluted Preliminary (XOPTDQPY): double"""
        return self.data["XOPTDQPY"]

    @property
    def XOPTDY(self) -> pd.Series:
        """XOPTDY -- Implied Option EPS Diluted (XOPTDY): double"""
        return self.data["XOPTDY"]

    @property
    def XOPTEPSQPY(self) -> pd.Series:
        """XOPTEPSQPY -- Implied Option EPS Basic Preliminary (XOPTEPSQPY): double"""
        return self.data["XOPTEPSQPY"]

    @property
    def XOPTEPSY(self) -> pd.Series:
        """XOPTEPSY -- Implied Option EPS Basic (XOPTEPSY): double"""
        return self.data["XOPTEPSY"]

    @property
    def XOPTQPY(self) -> pd.Series:
        """XOPTQPY -- Implied Option Expense Preliminary (XOPTQPY): double"""
        return self.data["XOPTQPY"]

    @property
    def XOPTY(self) -> pd.Series:
        """XOPTY -- Implied Option Expense (XOPTY): double"""
        return self.data["XOPTY"]

    @property
    def XRDY(self) -> pd.Series:
        """XRDY -- Research and Development Expense (XRDY): double"""
        return self.data["XRDY"]

    @property
    def XSGAY(self) -> pd.Series:
        """XSGAY -- Selling, General and Administrative Expenses (XSGAY): double"""
        return self.data["XSGAY"]

    @property
    def ADJEX(self) -> pd.Series:
        """ADJEX -- Cumulative Adjustment Factor by Ex-Date (ADJEX): double"""
        return self.data["ADJEX"]

    @property
    def CSHTRQ(self) -> pd.Series:
        """CSHTRQ -- Common Shares Traded - Quarter (CSHTRQ): double"""
        return self.data["CSHTRQ"]

    @property
    def DVPSPQ(self) -> pd.Series:
        """DVPSPQ -- Dividends per Share - Pay Date - Quarter (DVPSPQ): double"""
        return self.data["DVPSPQ"]

    @property
    def DVPSXQ(self) -> pd.Series:
        """DVPSXQ -- Div per Share - Exdate - Quarter (DVPSXQ): double"""
        return self.data["DVPSXQ"]

    @property
    def MKVALTQ(self) -> pd.Series:
        """MKVALTQ -- Market Value - Total (MKVALTQ): double"""
        return self.data["MKVALTQ"]

    @property
    def PRCCQ(self) -> pd.Series:
        """PRCCQ -- Price Close - Quarter (PRCCQ): double"""
        return self.data["PRCCQ"]

    @property
    def PRCHQ(self) -> pd.Series:
        """PRCHQ -- Price High - Quarter (PRCHQ): double"""
        return self.data["PRCHQ"]

    @property
    def PRCLQ(self) -> pd.Series:
        """PRCLQ -- Price Low - Quarter (PRCLQ): double"""
        return self.data["PRCLQ"]

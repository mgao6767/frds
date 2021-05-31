from dataclasses import dataclass
import numpy as np
import pandas as pd
from frds.data.wrds import WRDSDataset


@dataclass
class Funda(WRDSDataset):
    """Fundamentals Annual"""

    data: pd.DataFrame
    library = "comp"
    table = "funda"
    index_col = ["gvkey", "datadate"]
    date_cols = ["datadate"]

    def __post_init__(self):
        idx = [c.upper() for c in self.index_col]
        if set(self.data.index.names) != set(idx):
            self.data.reset_index(inplace=True, drop=True)
        self.data.rename(columns=str.upper, inplace=True)
        self.data.set_index(idx, inplace=True)

        # Some variables are not available
        # e.g., ADD1 (address line 1) is not itself stored in FUNDA
        attrs = [
            varname
            for varname, prop in vars(Funda).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            try:
                self.__getattribute__(attr)
            except KeyError:
                delattr(Funda, attr)

        # Automatically apply the default filtering rules
        self.filter()

    def filter(self):
        """Default filter applied on the FUNDA dataset"""
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
    def GVKEY(self) -> pd.Series:
        """GVKEY -- Global Company Key (GVKEY): string"""
        return self.data["GVKEY"]

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
    def ACCTCHG(self) -> pd.Series:
        """ACCTCHG -- Adoption of Accounting Changes (ACCTCHG): string"""
        return self.data["ACCTCHG"]

    @property
    def ACCTSTD(self) -> pd.Series:
        """ACCTSTD -- Accounting Standard (ACCTSTD): string"""
        return self.data["ACCTSTD"]

    @property
    def ACQMETH(self) -> pd.Series:
        """ACQMETH -- Acquisition Method (ACQMETH): string"""
        return self.data["ACQMETH"]

    @property
    def ADRR(self) -> pd.Series:
        """ADRR -- ADR Ratio (ADRR): double"""
        return self.data["ADRR"]

    @property
    def AJEX(self) -> pd.Series:
        """AJEX -- Adjustment Factor (Company) - Cumulative by Ex-Date (AJEX): double"""
        return self.data["AJEX"]

    @property
    def AJP(self) -> pd.Series:
        """AJP -- Adjustment Factor (Company) - Cumulative byPay-Date (AJP): double"""
        return self.data["AJP"]

    @property
    def APDEDATE(self) -> pd.Series:
        """APDEDATE -- Actual Period End date (APDEDATE): date"""
        return self.data["APDEDATE"]

    @property
    def BSPR(self) -> pd.Series:
        """BSPR -- Balance Sheet Presentation (BSPR): string"""
        return self.data["BSPR"]

    @property
    def COMPST(self) -> pd.Series:
        """COMPST -- Comparability Status (COMPST): string"""
        return self.data["COMPST"]

    @property
    def CURNCD(self) -> pd.Series:
        """CURNCD -- Native Currency Code (CURNCD): string"""
        return self.data["CURNCD"]

    @property
    def CURRTR(self) -> pd.Series:
        """CURRTR -- Currency Translation Rate (CURRTR): double"""
        return self.data["CURRTR"]

    @property
    def CURUSCN(self) -> pd.Series:
        """CURUSCN -- US Canadian Translation Rate (CURUSCN): double"""
        return self.data["CURUSCN"]

    @property
    def FDATE(self) -> pd.Series:
        """FDATE -- Final Date (FDATE): date"""
        return self.data["FDATE"]

    @property
    def FINAL(self) -> pd.Series:
        """FINAL -- Final Indicator Flag (FINAL): string"""
        return self.data["FINAL"]

    @property
    def FYEAR(self) -> pd.Series:
        """FYEAR -- Data Year - Fiscal (FYEAR): int"""
        return self.data["FYEAR"].astype("Int32")

    @property
    def ISMOD(self) -> pd.Series:
        """ISMOD -- Income Statement Model Number (ISMOD): double"""
        return self.data["ISMOD"]

    @property
    def LTCM(self) -> pd.Series:
        """LTCM -- Long Term Contract Method (LTCM): string"""
        return self.data["LTCM"]

    @property
    def OGM(self) -> pd.Series:
        """OGM -- OIL & GAS METHOD (OGM): string"""
        return self.data["OGM"]

    @property
    def PDATE(self) -> pd.Series:
        """PDATE -- Preliminary Date (PDATE): date"""
        return self.data["PDATE"]

    @property
    def PDDUR(self) -> pd.Series:
        """PDDUR -- Period Duration (PDDUR): double"""
        return self.data["PDDUR"]

    @property
    def SCF(self) -> pd.Series:
        """SCF -- Cash Flow Format (SCF): double"""
        return self.data["SCF"]

    @property
    def SRC(self) -> pd.Series:
        """SRC -- Source Document (SRC): double"""
        return self.data["SRC"]

    @property
    def STALT(self) -> pd.Series:
        """STALT -- Status Alert (STALT): string"""
        return self.data["STALT"]

    @property
    def UDPL(self) -> pd.Series:
        """UDPL -- Utility - Liberalized Depreciation Code (UDPL): string"""
        return self.data["UDPL"]

    @property
    def UPD(self) -> pd.Series:
        """UPD -- Update Code (UPD): double"""
        return self.data["UPD"]

    @property
    def ACCO(self) -> pd.Series:
        """ACCO -- Acceptances Outstanding (ACCO): double"""
        return self.data["ACCO"]

    @property
    def ACDO(self) -> pd.Series:
        """ACDO -- Current Assets of Discontinued Operations (ACDO): double"""
        return self.data["ACDO"]

    @property
    def ACO(self) -> pd.Series:
        """ACO -- Current Assets Other Total (ACO): double"""
        return self.data["ACO"]

    @property
    def ACODO(self) -> pd.Series:
        """ACODO -- Other Current Assets Excl Discontinued Operations (ACODO): double"""
        return self.data["ACODO"]

    @property
    def ACOMINC(self) -> pd.Series:
        """ACOMINC -- Accumulated Other Comprehensive Income (Loss) (ACOMINC): double"""
        return self.data["ACOMINC"]

    @property
    def ACOX(self) -> pd.Series:
        """ACOX -- Current Assets Other Sundry (ACOX): double"""
        return self.data["ACOX"]

    @property
    def ACOXAR(self) -> pd.Series:
        """ACOXAR -- Current Assets - Other - Total As Reported (ACOXAR): double"""
        return self.data["ACOXAR"]

    @property
    def ACT(self) -> pd.Series:
        """ACT -- Current Assets - Total (ACT): double"""
        return self.data["ACT"]

    @property
    def AEDI(self) -> pd.Series:
        """AEDI -- Accrued Expenses and Deferred Income (AEDI): double"""
        return self.data["AEDI"]

    @property
    def ALDO(self) -> pd.Series:
        """ALDO -- Long-term Assets of Discontinued Operations (ALDO): double"""
        return self.data["ALDO"]

    @property
    def AO(self) -> pd.Series:
        """AO -- Assets - Other (AO): double"""
        return self.data["AO"]

    @property
    def AOCIDERGL(self) -> pd.Series:
        """AOCIDERGL -- Accum Other Comp Inc - Derivatives Unrealized Gain/Loss (AOCIDERGL): double"""
        return self.data["AOCIDERGL"]

    @property
    def AOCIOTHER(self) -> pd.Series:
        """AOCIOTHER -- Accum Other Comp Inc - Other Adjustments (AOCIOTHER): double"""
        return self.data["AOCIOTHER"]

    @property
    def AOCIPEN(self) -> pd.Series:
        """AOCIPEN -- Accum Other Comp Inc - Min Pension Liab Adj (AOCIPEN): double"""
        return self.data["AOCIPEN"]

    @property
    def AOCISECGL(self) -> pd.Series:
        """AOCISECGL -- Accum Other Comp Inc - Unreal G/L Ret Int in Sec Assets (AOCISECGL): double"""
        return self.data["AOCISECGL"]

    @property
    def AODO(self) -> pd.Series:
        """AODO -- Other Assets excluding Discontinued Operations (AODO): double"""
        return self.data["AODO"]

    @property
    def AOX(self) -> pd.Series:
        """AOX -- Assets - Other - Sundry (AOX): double"""
        return self.data["AOX"]

    @property
    def AP(self) -> pd.Series:
        """AP -- Accounts Payable - Trade (AP): double"""
        return self.data["AP"]

    @property
    def APB(self) -> pd.Series:
        """APB -- Accounts Payable/Creditors - Brokers, Dealers, and Clearing Organizations (APB): double"""
        return self.data["APB"]

    @property
    def APC(self) -> pd.Series:
        """APC -- Accounts Payable/Creditors - Customer (APC): double"""
        return self.data["APC"]

    @property
    def APOFS(self) -> pd.Series:
        """APOFS -- Accounts Payable/Creditors - Other - FS (APOFS): double"""
        return self.data["APOFS"]

    @property
    def ARB(self) -> pd.Series:
        """ARB -- Accounts Receivable/Debtors - Brokers, Dealers, and Clearing Organizations (ARB): double"""
        return self.data["ARB"]

    @property
    def ARC(self) -> pd.Series:
        """ARC -- Accounts Receivable/Debtors - Customer (ARC): double"""
        return self.data["ARC"]

    @property
    def ARTFS(self) -> pd.Series:
        """ARTFS -- Accounts Receivable/Debtors - Total (ARTFS): double"""
        return self.data["ARTFS"]

    @property
    def AT(self) -> pd.Series:
        """AT -- Assets - Total (AT): double"""
        return self.data["AT"]

    @property
    def BAST(self) -> pd.Series:
        """BAST -- Average Short-Term Borrowings (BAST): double"""
        return self.data["BAST"]

    @property
    def BKVLPS(self) -> pd.Series:
        """BKVLPS -- Book Value Per Share (BKVLPS): double"""
        return self.data["BKVLPS"]

    @property
    def CA(self) -> pd.Series:
        """CA -- Customers' Acceptance (CA): double"""
        return self.data["CA"]

    @property
    def CAPS(self) -> pd.Series:
        """CAPS -- Capital Surplus/Share Premium Reserve (CAPS): double"""
        return self.data["CAPS"]

    @property
    def CB(self) -> pd.Series:
        """CB -- Compensating Balance (CB): double"""
        return self.data["CB"]

    @property
    def CEQ(self) -> pd.Series:
        """CEQ -- Common/Ordinary Equity - Total (CEQ): double"""
        return self.data["CEQ"]

    @property
    def CEQL(self) -> pd.Series:
        """CEQL -- Common Equity Liquidation Value (CEQL): double"""
        return self.data["CEQL"]

    @property
    def CEQT(self) -> pd.Series:
        """CEQT -- Common Equity Tangible (CEQT): double"""
        return self.data["CEQT"]

    @property
    def CH(self) -> pd.Series:
        """CH -- Cash (CH): double"""
        return self.data["CH"]

    @property
    def CHE(self) -> pd.Series:
        """CHE -- Cash and Short-Term Investments (CHE): double"""
        return self.data["CHE"]

    @property
    def CHS(self) -> pd.Series:
        """CHS -- Cash and Deposits - Segregated (CHS): double"""
        return self.data["CHS"]

    @property
    def CLD2(self) -> pd.Series:
        """CLD2 -- Capitalized Leases - Due in 2nd Year (CLD2): double"""
        return self.data["CLD2"]

    @property
    def CLD3(self) -> pd.Series:
        """CLD3 -- Capitalized Leases - Due in 3rd Year (CLD3): double"""
        return self.data["CLD3"]

    @property
    def CLD4(self) -> pd.Series:
        """CLD4 -- Capitalized Leases - Due in 4th Year (CLD4): double"""
        return self.data["CLD4"]

    @property
    def CLD5(self) -> pd.Series:
        """CLD5 -- Capitalized Leases - Due in 5th Year (CLD5): double"""
        return self.data["CLD5"]

    @property
    def CLFC(self) -> pd.Series:
        """CLFC -- Contingent Liabilities - Forward and Future Contracts (CLFC): double"""
        return self.data["CLFC"]

    @property
    def CLFX(self) -> pd.Series:
        """CLFX -- Contingent Liabilities - Foreign Exchange Commitments (CLFX): double"""
        return self.data["CLFX"]

    @property
    def CLG(self) -> pd.Series:
        """CLG -- Contingent Liabilities - Guarantees (CLG): double"""
        return self.data["CLG"]

    @property
    def CLIS(self) -> pd.Series:
        """CLIS -- Contingent Liabilities - Interest Rate Swaps (CLIS): double"""
        return self.data["CLIS"]

    @property
    def CLL(self) -> pd.Series:
        """CLL -- Contingent Liabilities - Letters of Credit (CLL): double"""
        return self.data["CLL"]

    @property
    def CLLC(self) -> pd.Series:
        """CLLC -- Contingent Liabilities - Loan Commitments (CLLC): double"""
        return self.data["CLLC"]

    @property
    def CLO(self) -> pd.Series:
        """CLO -- Contingent Liabilities - Other (CLO): double"""
        return self.data["CLO"]

    @property
    def CLRLL(self) -> pd.Series:
        """CLRLL -- Credit Loss Reserve Allocated for LDC Loans (CLRLL): double"""
        return self.data["CLRLL"]

    @property
    def CLT(self) -> pd.Series:
        """CLT -- Contingent Liabilities - Total (CLT): double"""
        return self.data["CLT"]

    @property
    def CMP(self) -> pd.Series:
        """CMP -- Commercial Paper (CMP): double"""
        return self.data["CMP"]

    @property
    def CRV(self) -> pd.Series:
        """CRV -- Consolidation Reserves (CRV): double"""
        return self.data["CRV"]

    @property
    def CRVNLI(self) -> pd.Series:
        """CRVNLI -- Reserves for Claims (Losses) - Nonlife (Insurance) (CRVNLI): double"""
        return self.data["CRVNLI"]

    @property
    def CSTK(self) -> pd.Series:
        """CSTK -- Common/Ordinary Stock (Capital) (CSTK): double"""
        return self.data["CSTK"]

    @property
    def CSTKCV(self) -> pd.Series:
        """CSTKCV -- Common Stock-Carrying Value (CSTKCV): double"""
        return self.data["CSTKCV"]

    @property
    def DC(self) -> pd.Series:
        """DC -- Deferred Charges (DC): double"""
        return self.data["DC"]

    @property
    def DCLO(self) -> pd.Series:
        """DCLO -- Debt Capitalized Lease Obligations (DCLO): double"""
        return self.data["DCLO"]

    @property
    def DCOM(self) -> pd.Series:
        """DCOM -- Deferred Compensation (DCOM): double"""
        return self.data["DCOM"]

    @property
    def DCPSTK(self) -> pd.Series:
        """DCPSTK -- Convertible Debt and Preferred Stock (DCPSTK): double"""
        return self.data["DCPSTK"]

    @property
    def DCS(self) -> pd.Series:
        """DCS -- Debt Consolidated Subsidiary (DCS): double"""
        return self.data["DCS"]

    @property
    def DCVSR(self) -> pd.Series:
        """DCVSR -- Debt Senior Convertible (DCVSR): double"""
        return self.data["DCVSR"]

    @property
    def DCVSUB(self) -> pd.Series:
        """DCVSUB -- Debt Subordinated Convertible (DCVSUB): double"""
        return self.data["DCVSUB"]

    @property
    def DCVT(self) -> pd.Series:
        """DCVT -- Debt - Convertible (DCVT): double"""
        return self.data["DCVT"]

    @property
    def DD(self) -> pd.Series:
        """DD -- Debt Debentures (DD): double"""
        return self.data["DD"]

    @property
    def DD1(self) -> pd.Series:
        """DD1 -- Long-Term Debt Due in One Year (DD1): double"""
        return self.data["DD1"]

    @property
    def DD2(self) -> pd.Series:
        """DD2 -- Debt Due in 2nd Year (DD2): double"""
        return self.data["DD2"]

    @property
    def DD3(self) -> pd.Series:
        """DD3 -- Debt Due in 3rd Year (DD3): double"""
        return self.data["DD3"]

    @property
    def DD4(self) -> pd.Series:
        """DD4 -- Debt Due in 4th Year (DD4): double"""
        return self.data["DD4"]

    @property
    def DD5(self) -> pd.Series:
        """DD5 -- Debt Due in 5th Year (DD5): double"""
        return self.data["DD5"]

    @property
    def DFPAC(self) -> pd.Series:
        """DFPAC -- Deferred Policy Acquisition Costs (DFPAC): double"""
        return self.data["DFPAC"]

    @property
    def DFS(self) -> pd.Series:
        """DFS -- Debt Finance Subsidiary (DFS): double"""
        return self.data["DFS"]

    @property
    def DLC(self) -> pd.Series:
        """DLC -- Debt in Current Liabilities - Total (DLC): double"""
        return self.data["DLC"]

    @property
    def DLTO(self) -> pd.Series:
        """DLTO -- Other Long-term Debt (DLTO): double"""
        return self.data["DLTO"]

    @property
    def DLTP(self) -> pd.Series:
        """DLTP -- Long-Term Debt Tied to Prime (DLTP): double"""
        return self.data["DLTP"]

    @property
    def DLTSUB(self) -> pd.Series:
        """DLTSUB -- Long-Term Debt - Subordinated (DLTSUB): double"""
        return self.data["DLTSUB"]

    @property
    def DLTT(self) -> pd.Series:
        """DLTT -- Long-Term Debt - Total (DLTT): double"""
        return self.data["DLTT"]

    @property
    def DM(self) -> pd.Series:
        """DM -- Debt Mortgages & Other Secured (DM): double"""
        return self.data["DM"]

    @property
    def DN(self) -> pd.Series:
        """DN -- Debt Notes (DN): double"""
        return self.data["DN"]

    @property
    def DPACB(self) -> pd.Series:
        """DPACB -- Depreciation (Accumulated) Buildings (DPACB): double"""
        return self.data["DPACB"]

    @property
    def DPACC(self) -> pd.Series:
        """DPACC -- Depreciation (Accumulated) Construction in Progress (DPACC): double"""
        return self.data["DPACC"]

    @property
    def DPACLI(self) -> pd.Series:
        """DPACLI -- Depreciation (Accumulated) Land and Improvements (DPACLI): double"""
        return self.data["DPACLI"]

    @property
    def DPACLS(self) -> pd.Series:
        """DPACLS -- Depreciation (Accumulated) Leases (DPACLS): double"""
        return self.data["DPACLS"]

    @property
    def DPACME(self) -> pd.Series:
        """DPACME -- Depreciation (Accumulated) Machinery and Equipment (DPACME): double"""
        return self.data["DPACME"]

    @property
    def DPACNR(self) -> pd.Series:
        """DPACNR -- Depreciation (Accumulated) Natural Resources (DPACNR): double"""
        return self.data["DPACNR"]

    @property
    def DPACO(self) -> pd.Series:
        """DPACO -- Depreciation (Accumulated) Other (DPACO): double"""
        return self.data["DPACO"]

    @property
    def DPACRE(self) -> pd.Series:
        """DPACRE -- Accumulated Depreciation of RE Property (DPACRE): double"""
        return self.data["DPACRE"]

    @property
    def DPACT(self) -> pd.Series:
        """DPACT -- Depreciation, Depletion and Amortization (Accumulated) (DPACT): double"""
        return self.data["DPACT"]

    @property
    def DPDC(self) -> pd.Series:
        """DPDC -- Deposits - Demand - Customer (DPDC): double"""
        return self.data["DPDC"]

    @property
    def DPLTB(self) -> pd.Series:
        """DPLTB -- Deposits - Long-Term Time - Bank (DPLTB): double"""
        return self.data["DPLTB"]

    @property
    def DPSC(self) -> pd.Series:
        """DPSC -- Deposits vings - Customer (DPSC): double"""
        return self.data["DPSC"]

    @property
    def DPSTB(self) -> pd.Series:
        """DPSTB -- Deposits - Short-Term Demand - Bank (DPSTB): double"""
        return self.data["DPSTB"]

    @property
    def DPTB(self) -> pd.Series:
        """DPTB -- Deposits - Total - Banks (DPTB): double"""
        return self.data["DPTB"]

    @property
    def DPTC(self) -> pd.Series:
        """DPTC -- Deposits - Total - Customer (DPTC): double"""
        return self.data["DPTC"]

    @property
    def DPTIC(self) -> pd.Series:
        """DPTIC -- Deposits - Time - Customer (DPTIC): double"""
        return self.data["DPTIC"]

    @property
    def DPVIEB(self) -> pd.Series:
        """DPVIEB -- Depreciation (Accumulated) Ending Balance (Schedule VI) (DPVIEB): double"""
        return self.data["DPVIEB"]

    @property
    def DPVIO(self) -> pd.Series:
        """DPVIO -- Depreciation (Accumulated) Other Changes (Schedule VI) (DPVIO): double"""
        return self.data["DPVIO"]

    @property
    def DPVIR(self) -> pd.Series:
        """DPVIR -- Depreciation (Accumulated) Retirements (Schedule VI) (DPVIR): double"""
        return self.data["DPVIR"]

    @property
    def DRC(self) -> pd.Series:
        """DRC -- Deferred Revenue Current (DRC): double"""
        return self.data["DRC"]

    @property
    def DRCI(self) -> pd.Series:
        """DRCI -- Deduction From Policy and Claims Reserves for Reinsurance Ceded (DRCI): double"""
        return self.data["DRCI"]

    @property
    def DRLT(self) -> pd.Series:
        """DRLT -- Deferred Revenue Long-term (DRLT): double"""
        return self.data["DRLT"]

    @property
    def DS(self) -> pd.Series:
        """DS -- Debt-Subordinated (DS): double"""
        return self.data["DS"]

    @property
    def DUDD(self) -> pd.Series:
        """DUDD -- Debt Unamortized Debt Discount and Other (DUDD): double"""
        return self.data["DUDD"]

    @property
    def DVPA(self) -> pd.Series:
        """DVPA -- Preferred Dividends in Arrears (DVPA): double"""
        return self.data["DVPA"]

    @property
    def DVPIBB(self) -> pd.Series:
        """DVPIBB -- Depreciation (Accumulated) Beginning Balance (Schedule VI) (DVPIBB): double"""
        return self.data["DVPIBB"]

    @property
    def DXD2(self) -> pd.Series:
        """DXD2 -- Debt (excl Capitalized Leases) - Due in 2nd Year (DXD2): double"""
        return self.data["DXD2"]

    @property
    def DXD3(self) -> pd.Series:
        """DXD3 -- Debt (excl Capitalized Leases) - Due in 3rd Year (DXD3): double"""
        return self.data["DXD3"]

    @property
    def DXD4(self) -> pd.Series:
        """DXD4 -- Debt (excl Capitalized Leases) - Due in 4th Year (DXD4): double"""
        return self.data["DXD4"]

    @property
    def DXD5(self) -> pd.Series:
        """DXD5 -- Debt (excl Capitalized Leases) - Due in 5th Year (DXD5): double"""
        return self.data["DXD5"]

    @property
    def EA(self) -> pd.Series:
        """EA -- Exchange Adjustments (Assets) (EA): double"""
        return self.data["EA"]

    @property
    def ESOPCT(self) -> pd.Series:
        """ESOPCT -- ESOP Obligation (Common) - Total (ESOPCT): double"""
        return self.data["ESOPCT"]

    @property
    def ESOPDLT(self) -> pd.Series:
        """ESOPDLT -- ESOP Debt - Long Term (ESOPDLT): double"""
        return self.data["ESOPDLT"]

    @property
    def ESOPNR(self) -> pd.Series:
        """ESOPNR -- Preferred ESOP Obligation - Non-Redeemable (ESOPNR): double"""
        return self.data["ESOPNR"]

    @property
    def ESOPR(self) -> pd.Series:
        """ESOPR -- Preferred ESOP Obligation - Redeemable (ESOPR): double"""
        return self.data["ESOPR"]

    @property
    def ESOPT(self) -> pd.Series:
        """ESOPT -- Preferred ESOP Obligation - Total (ESOPT): double"""
        return self.data["ESOPT"]

    @property
    def EXCADJ(self) -> pd.Series:
        """EXCADJ -- Exchange Adjustments (Liabilities) (EXCADJ): double"""
        return self.data["EXCADJ"]

    @property
    def FATB(self) -> pd.Series:
        """FATB -- Property, Plant, and Equipment Buildings at Cost (FATB): double"""
        return self.data["FATB"]

    @property
    def FATC(self) -> pd.Series:
        """FATC -- Property, Plant, and Equipment Construction in Progress at Cost (FATC): double"""
        return self.data["FATC"]

    @property
    def FATE(self) -> pd.Series:
        """FATE -- Property, Plant, and Equipment Machinery and Equipment at Cost (FATE): double"""
        return self.data["FATE"]

    @property
    def FATL(self) -> pd.Series:
        """FATL -- Property, Plant, and Equipment Leases at Cost (FATL): double"""
        return self.data["FATL"]

    @property
    def FATN(self) -> pd.Series:
        """FATN -- Property, Plant, and Equipment Natural Resources at Cost (FATN): double"""
        return self.data["FATN"]

    @property
    def FATO(self) -> pd.Series:
        """FATO -- Property, Plant, and Equipment Other at Cost (FATO): double"""
        return self.data["FATO"]

    @property
    def FATP(self) -> pd.Series:
        """FATP -- Property, Plant, and Equipment Land and Improvements at Cost (FATP): double"""
        return self.data["FATP"]

    @property
    def FDFR(self) -> pd.Series:
        """FDFR -- Federal Funds Purchased (FDFR): double"""
        return self.data["FDFR"]

    @property
    def FEA(self) -> pd.Series:
        """FEA -- Foreign Exchange Assets (FEA): double"""
        return self.data["FEA"]

    @property
    def FEL(self) -> pd.Series:
        """FEL -- Foreign Exchange Liabilities (FEL): double"""
        return self.data["FEL"]

    @property
    def FFS(self) -> pd.Series:
        """FFS -- Federal Funds Sold (FFS): double"""
        return self.data["FFS"]

    @property
    def GDWL(self) -> pd.Series:
        """GDWL -- Goodwill (GDWL): double"""
        return self.data["GDWL"]

    @property
    def GEQRV(self) -> pd.Series:
        """GEQRV -- Grants - Equity Reserves (GEQRV): double"""
        return self.data["GEQRV"]

    @property
    def GOVGR(self) -> pd.Series:
        """GOVGR -- Government Grants (GOVGR): double"""
        return self.data["GOVGR"]

    @property
    def IAEQ(self) -> pd.Series:
        """IAEQ -- Investment Assets - Equity Securities (Insurance) (IAEQ): double"""
        return self.data["IAEQ"]

    @property
    def IAEQCI(self) -> pd.Series:
        """IAEQCI -- Investment Assets (Insurance) - Equity Securities (Cost) (IAEQCI): double"""
        return self.data["IAEQCI"]

    @property
    def IAEQMI(self) -> pd.Series:
        """IAEQMI -- Investment Assets (Insurance) - Equity Securities (Market) (IAEQMI): double"""
        return self.data["IAEQMI"]

    @property
    def IAFICI(self) -> pd.Series:
        """IAFICI -- Investment Assets (Insurance) - Fixed Income Securities (Cost) (IAFICI): double"""
        return self.data["IAFICI"]

    @property
    def IAFXI(self) -> pd.Series:
        """IAFXI -- Investment Assets - Fixed Income Securities (Insurance) (IAFXI): double"""
        return self.data["IAFXI"]

    @property
    def IAFXMI(self) -> pd.Series:
        """IAFXMI -- Investment Assets (Insurance) - Fixed Income Securities (Market) (IAFXMI): double"""
        return self.data["IAFXMI"]

    @property
    def IALI(self) -> pd.Series:
        """IALI -- Investment Assets (Insurance) - Listed Securities-Total (IALI): double"""
        return self.data["IALI"]

    @property
    def IALOI(self) -> pd.Series:
        """IALOI -- Investment Assets - Loans - Other (Insurance) (IALOI): double"""
        return self.data["IALOI"]

    @property
    def IALTI(self) -> pd.Series:
        """IALTI -- Investment Assets - Loans - Total (Insurance) (IALTI): double"""
        return self.data["IALTI"]

    @property
    def IAMLI(self) -> pd.Series:
        """IAMLI -- Investment Assets - Mortgage Loans (Insurance) (IAMLI): double"""
        return self.data["IAMLI"]

    @property
    def IAOI(self) -> pd.Series:
        """IAOI -- Investment Assets - Other (Insurance) (IAOI): double"""
        return self.data["IAOI"]

    @property
    def IAPLI(self) -> pd.Series:
        """IAPLI -- Investment Assets - Policy Loans (Insurance) (IAPLI): double"""
        return self.data["IAPLI"]

    @property
    def IAREI(self) -> pd.Series:
        """IAREI -- Investment Assets - Real Estate (Insurance) (IAREI): double"""
        return self.data["IAREI"]

    @property
    def IASCI(self) -> pd.Series:
        """IASCI -- Investment Assets (Insurance) - Securities - Sundry (Cost) (IASCI): double"""
        return self.data["IASCI"]

    @property
    def IASMI(self) -> pd.Series:
        """IASMI -- Investment Assets (Insurance) - Securities - Sundry (Market) (IASMI): double"""
        return self.data["IASMI"]

    @property
    def IASSI(self) -> pd.Series:
        """IASSI -- Investment Assets - Securities - Sundry (Insurance) (IASSI): double"""
        return self.data["IASSI"]

    @property
    def IASTI(self) -> pd.Series:
        """IASTI -- Investment Assets - Securities - Total (Insurance) (IASTI): double"""
        return self.data["IASTI"]

    @property
    def IATCI(self) -> pd.Series:
        """IATCI -- Investment Assets (Insurance) - Securities - Total (Cost) (IATCI): double"""
        return self.data["IATCI"]

    @property
    def IATI(self) -> pd.Series:
        """IATI -- Investment Assets - Total (Insurance) (IATI): double"""
        return self.data["IATI"]

    @property
    def IATMI(self) -> pd.Series:
        """IATMI -- Investment Assets (Insurance) - Securities - Total (Market) (IATMI): double"""
        return self.data["IATMI"]

    @property
    def IAUI(self) -> pd.Series:
        """IAUI -- Investment Assets (Insurance) - Unlisted Securities - Total (IAUI): double"""
        return self.data["IAUI"]

    @property
    def ICAPT(self) -> pd.Series:
        """ICAPT -- Invested Capital - Total (ICAPT): double"""
        return self.data["ICAPT"]

    @property
    def INTAN(self) -> pd.Series:
        """INTAN -- Intangible Assets - Total (INTAN): double"""
        return self.data["INTAN"]

    @property
    def INTANO(self) -> pd.Series:
        """INTANO -- Other Intangibles (INTANO): double"""
        return self.data["INTANO"]

    @property
    def INVFG(self) -> pd.Series:
        """INVFG -- Inventories Finished Goods (INVFG): double"""
        return self.data["INVFG"]

    @property
    def INVO(self) -> pd.Series:
        """INVO -- Inventories Other (INVO): double"""
        return self.data["INVO"]

    @property
    def INVOFS(self) -> pd.Series:
        """INVOFS -- Inventory/Stock - Other (INVOFS): double"""
        return self.data["INVOFS"]

    @property
    def INVREH(self) -> pd.Series:
        """INVREH -- Inventory/Stock - Real Estate Held for Development (INVREH): double"""
        return self.data["INVREH"]

    @property
    def INVREI(self) -> pd.Series:
        """INVREI -- Inventory/Stock - Real Estate Under Development (INVREI): double"""
        return self.data["INVREI"]

    @property
    def INVRES(self) -> pd.Series:
        """INVRES -- Inventory/Stock - Real Estate Held for Sale (INVRES): double"""
        return self.data["INVRES"]

    @property
    def INVRM(self) -> pd.Series:
        """INVRM -- Inventories Raw Materials (INVRM): double"""
        return self.data["INVRM"]

    @property
    def INVT(self) -> pd.Series:
        """INVT -- Inventories - Total (INVT): double"""
        return self.data["INVT"]

    @property
    def INVWIP(self) -> pd.Series:
        """INVWIP -- Inventories Work In Process (INVWIP): double"""
        return self.data["INVWIP"]

    @property
    def IP(self) -> pd.Series:
        """IP -- Investment Property (IP): double"""
        return self.data["IP"]

    @property
    def IPC(self) -> pd.Series:
        """IPC -- Investment Property (Cost) (IPC): double"""
        return self.data["IPC"]

    @property
    def IPV(self) -> pd.Series:
        """IPV -- Investment Property (Valuation) (IPV): double"""
        return self.data["IPV"]

    @property
    def ISEQ(self) -> pd.Series:
        """ISEQ -- Investment Securities - Equity (ISEQ): double"""
        return self.data["ISEQ"]

    @property
    def ISEQC(self) -> pd.Series:
        """ISEQC -- Investment Securities - Equity (Cost) (ISEQC): double"""
        return self.data["ISEQC"]

    @property
    def ISEQM(self) -> pd.Series:
        """ISEQM -- Investment Securities - Equity (Market) (ISEQM): double"""
        return self.data["ISEQM"]

    @property
    def ISFI(self) -> pd.Series:
        """ISFI -- Investment Securities - Fixed Income (ISFI): double"""
        return self.data["ISFI"]

    @property
    def ISFXC(self) -> pd.Series:
        """ISFXC -- Investment Securities - Fixed Income (Cost) (ISFXC): double"""
        return self.data["ISFXC"]

    @property
    def ISFXM(self) -> pd.Series:
        """ISFXM -- Investment Securities - Fixed Income (Market) (ISFXM): double"""
        return self.data["ISFXM"]

    @property
    def ISLG(self) -> pd.Series:
        """ISLG -- Investment Securities - Local Governments (ISLG): double"""
        return self.data["ISLG"]

    @property
    def ISLGC(self) -> pd.Series:
        """ISLGC -- Investment Securities - Local Governments (Cost) (ISLGC): double"""
        return self.data["ISLGC"]

    @property
    def ISLGM(self) -> pd.Series:
        """ISLGM -- Investment Securities - Local Governments (Market) (ISLGM): double"""
        return self.data["ISLGM"]

    @property
    def ISLT(self) -> pd.Series:
        """ISLT -- Investment Securities - Listed - Total (ISLT): double"""
        return self.data["ISLT"]

    @property
    def ISNG(self) -> pd.Series:
        """ISNG -- Investment Securities - National Governments (ISNG): double"""
        return self.data["ISNG"]

    @property
    def ISNGC(self) -> pd.Series:
        """ISNGC -- Investment Securities - National Governments (Cost) (ISNGC): double"""
        return self.data["ISNGC"]

    @property
    def ISNGM(self) -> pd.Series:
        """ISNGM -- Investment Securities - National Governments (Market) (ISNGM): double"""
        return self.data["ISNGM"]

    @property
    def ISOTC(self) -> pd.Series:
        """ISOTC -- Invetsment Securities - Other (Cost) (ISOTC): double"""
        return self.data["ISOTC"]

    @property
    def ISOTH(self) -> pd.Series:
        """ISOTH -- Investment Securities - Other (ISOTH): double"""
        return self.data["ISOTH"]

    @property
    def ISOTM(self) -> pd.Series:
        """ISOTM -- Invetsment Securities - Other (Market) (ISOTM): double"""
        return self.data["ISOTM"]

    @property
    def ISSC(self) -> pd.Series:
        """ISSC -- Investment Securities - Sundry (Cost) (ISSC): double"""
        return self.data["ISSC"]

    @property
    def ISSM(self) -> pd.Series:
        """ISSM -- Investment Securities - Sundry (Market) (ISSM): double"""
        return self.data["ISSM"]

    @property
    def ISSU(self) -> pd.Series:
        """ISSU -- Investment Securities - Sundry (ISSU): double"""
        return self.data["ISSU"]

    @property
    def IST(self) -> pd.Series:
        """IST -- Investment Securities -Total (IST): double"""
        return self.data["IST"]

    @property
    def ISTC(self) -> pd.Series:
        """ISTC -- Investment Securities - Total (Cost) (ISTC): double"""
        return self.data["ISTC"]

    @property
    def ISTM(self) -> pd.Series:
        """ISTM -- Investment Securities - Total (Market) (ISTM): double"""
        return self.data["ISTM"]

    @property
    def ISUT(self) -> pd.Series:
        """ISUT -- Investment Securities - Unlisted - Total (ISUT): double"""
        return self.data["ISUT"]

    @property
    def ITCB(self) -> pd.Series:
        """ITCB -- Investment Tax Credit (Balance Sheet) (ITCB): double"""
        return self.data["ITCB"]

    @property
    def IVAEQ(self) -> pd.Series:
        """IVAEQ -- Investment and Advances - Equity (IVAEQ): double"""
        return self.data["IVAEQ"]

    @property
    def IVAO(self) -> pd.Series:
        """IVAO -- Investment and Advances Other (IVAO): double"""
        return self.data["IVAO"]

    @property
    def IVGOD(self) -> pd.Series:
        """IVGOD -- Investments Grants and Other Deductions (IVGOD): double"""
        return self.data["IVGOD"]

    @property
    def IVPT(self) -> pd.Series:
        """IVPT -- Investments - Permanent - Total (IVPT): double"""
        return self.data["IVPT"]

    @property
    def IVST(self) -> pd.Series:
        """IVST -- Short-Term Investments - Total (IVST): double"""
        return self.data["IVST"]

    @property
    def LCABG(self) -> pd.Series:
        """LCABG -- Loans/Claims/Advances - Banks and Government - Total (LCABG): double"""
        return self.data["LCABG"]

    @property
    def LCACL(self) -> pd.Series:
        """LCACL -- Loans/Claims/Advances - Commercial (LCACL): double"""
        return self.data["LCACL"]

    @property
    def LCACR(self) -> pd.Series:
        """LCACR -- Loans/Claims/Advances - Consumer (LCACR): double"""
        return self.data["LCACR"]

    @property
    def LCAG(self) -> pd.Series:
        """LCAG -- Loans/Claims/Advances - Government (LCAG): double"""
        return self.data["LCAG"]

    @property
    def LCAL(self) -> pd.Series:
        """LCAL -- Loans/Claims/Advances - Lease (LCAL): double"""
        return self.data["LCAL"]

    @property
    def LCALT(self) -> pd.Series:
        """LCALT -- Loans/Claims/Advances - Long-Term (Banks) (LCALT): double"""
        return self.data["LCALT"]

    @property
    def LCAM(self) -> pd.Series:
        """LCAM -- Loans/Claims/Advances - Mortgage (LCAM): double"""
        return self.data["LCAM"]

    @property
    def LCAO(self) -> pd.Series:
        """LCAO -- Loans/Claims/Advances - Other (LCAO): double"""
        return self.data["LCAO"]

    @property
    def LCAST(self) -> pd.Series:
        """LCAST -- Loans/Claims/Advances - Short-Term - Banks (LCAST): double"""
        return self.data["LCAST"]

    @property
    def LCAT(self) -> pd.Series:
        """LCAT -- Loans/Claims/Advances - Total (LCAT): double"""
        return self.data["LCAT"]

    @property
    def LCO(self) -> pd.Series:
        """LCO -- Current Liabilities Other Total (LCO): double"""
        return self.data["LCO"]

    @property
    def LCOX(self) -> pd.Series:
        """LCOX -- Current Liabilities Other Sundry (LCOX): double"""
        return self.data["LCOX"]

    @property
    def LCOXAR(self) -> pd.Series:
        """LCOXAR -- Current Liabilities - Other - Total As Reported (LCOXAR): double"""
        return self.data["LCOXAR"]

    @property
    def LCOXDR(self) -> pd.Series:
        """LCOXDR -- Current Liabilities - Other - Excluding Deferred Revenue (LCOXDR): double"""
        return self.data["LCOXDR"]

    @property
    def LCT(self) -> pd.Series:
        """LCT -- Current Liabilities - Total (LCT): double"""
        return self.data["LCT"]

    @property
    def LCUACU(self) -> pd.Series:
        """LCUACU -- Loans/Claims/Advances - Customer - Total (LCUACU): double"""
        return self.data["LCUACU"]

    @property
    def LIF(self) -> pd.Series:
        """LIF -- Life Insurance in Force (LIF): double"""
        return self.data["LIF"]

    @property
    def LIFR(self) -> pd.Series:
        """LIFR -- LIFO Reserve (LIFR): double"""
        return self.data["LIFR"]

    @property
    def LLOML(self) -> pd.Series:
        """LLOML -- LDC Loans Outstanding - Medium and Long-Term (LLOML): double"""
        return self.data["LLOML"]

    @property
    def LLOO(self) -> pd.Series:
        """LLOO -- LDC Loans Outstanding - Other (LLOO): double"""
        return self.data["LLOO"]

    @property
    def LLOT(self) -> pd.Series:
        """LLOT -- LDC Loans Outstanding - Total (LLOT): double"""
        return self.data["LLOT"]

    @property
    def LO(self) -> pd.Series:
        """LO -- Liabilities - Other - Total (LO): double"""
        return self.data["LO"]

    @property
    def LOXDR(self) -> pd.Series:
        """LOXDR -- Liabilities - Other - Excluding Deferred Revenue (LOXDR): double"""
        return self.data["LOXDR"]

    @property
    def LRV(self) -> pd.Series:
        """LRV -- Legal Reserves (LRV): double"""
        return self.data["LRV"]

    @property
    def LS(self) -> pd.Series:
        """LS -- Liabilities - Other - Sundry (LS): double"""
        return self.data["LS"]

    @property
    def LSE(self) -> pd.Series:
        """LSE -- Liabilities and Stockholders Equity - Total (LSE): double"""
        return self.data["LSE"]

    @property
    def LT(self) -> pd.Series:
        """LT -- Liabilities - Total (LT): double"""
        return self.data["LT"]

    @property
    def MIB(self) -> pd.Series:
        """MIB -- Minority Interest (Balance Sheet) (MIB): double"""
        return self.data["MIB"]

    @property
    def MRC1(self) -> pd.Series:
        """MRC1 -- Rental Commitments Minimum 1st Year (MRC1): double"""
        return self.data["MRC1"]

    @property
    def MRC2(self) -> pd.Series:
        """MRC2 -- Rental Commitments Minimum 2nd Year (MRC2): double"""
        return self.data["MRC2"]

    @property
    def MRC3(self) -> pd.Series:
        """MRC3 -- Rental Commitments Minimum 3rd Year (MRC3): double"""
        return self.data["MRC3"]

    @property
    def MRC4(self) -> pd.Series:
        """MRC4 -- Rental Commitments Minimum 4th Year (MRC4): double"""
        return self.data["MRC4"]

    @property
    def MRC5(self) -> pd.Series:
        """MRC5 -- Rental Commitments Minimum 5th Year (MRC5): double"""
        return self.data["MRC5"]

    @property
    def MRCT(self) -> pd.Series:
        """MRCT -- Rental Commitments Minimum 5 Year Total (MRCT): double"""
        return self.data["MRCT"]

    @property
    def MRCTA(self) -> pd.Series:
        """MRCTA -- Thereafter Portion of Leases (MRCTA): double"""
        return self.data["MRCTA"]

    @property
    def MSA(self) -> pd.Series:
        """MSA -- Marketable Securities Adjustment (MSA): double"""
        return self.data["MSA"]

    @property
    def MSVRV(self) -> pd.Series:
        """MSVRV -- Mandatory Securities Valuation Reserve (Statutory) (MSVRV): double"""
        return self.data["MSVRV"]

    @property
    def MTL(self) -> pd.Series:
        """MTL -- Loans From Securities Finance Companies for Margin Transactions (MTL): double"""
        return self.data["MTL"]

    @property
    def NAT(self) -> pd.Series:
        """NAT -- Nonadmitted Assets - Total (Statutory) (NAT): double"""
        return self.data["NAT"]

    @property
    def NP(self) -> pd.Series:
        """NP -- Notes Payable Short-Term Borrowings (NP): double"""
        return self.data["NP"]

    @property
    def NPANL(self) -> pd.Series:
        """NPANL -- Nonperforming Assets - Nonaccrual Loans (NPANL): double"""
        return self.data["NPANL"]

    @property
    def NPAORE(self) -> pd.Series:
        """NPAORE -- Nonperforming Assets - Other Real Estate Owned (NPAORE): double"""
        return self.data["NPAORE"]

    @property
    def NPARL(self) -> pd.Series:
        """NPARL -- Nonperforming Assets - Restructured Loans (NPARL): double"""
        return self.data["NPARL"]

    @property
    def NPAT(self) -> pd.Series:
        """NPAT -- Nonperforming Assets - Total (NPAT): double"""
        return self.data["NPAT"]

    @property
    def OB(self) -> pd.Series:
        """OB -- Order Backlog (OB): double"""
        return self.data["OB"]

    @property
    def OPTPRCCA(self) -> pd.Series:
        """OPTPRCCA -- Options Cancelled - Price (OPTPRCCA): double"""
        return self.data["OPTPRCCA"]

    @property
    def OPTPRCEX(self) -> pd.Series:
        """OPTPRCEX -- Options Exercised - Price (OPTPRCEX): double"""
        return self.data["OPTPRCEX"]

    @property
    def OPTPRCEY(self) -> pd.Series:
        """OPTPRCEY -- Options Outstanding End of Year - Price (OPTPRCEY): double"""
        return self.data["OPTPRCEY"]

    @property
    def OPTPRCGR(self) -> pd.Series:
        """OPTPRCGR -- Options Granted - Price (OPTPRCGR): double"""
        return self.data["OPTPRCGR"]

    @property
    def OPTPRCWA(self) -> pd.Series:
        """OPTPRCWA -- Options Exercisable - Weighted Avg Price (OPTPRCWA): double"""
        return self.data["OPTPRCWA"]

    @property
    def PPEGT(self) -> pd.Series:
        """PPEGT -- Property, Plant and Equipment - Total (Gross) (PPEGT): double"""
        return self.data["PPEGT"]

    @property
    def PPENB(self) -> pd.Series:
        """PPENB -- Property, Plant, and Equipment Buildings (Net) (PPENB): double"""
        return self.data["PPENB"]

    @property
    def PPENC(self) -> pd.Series:
        """PPENC -- Property, Plant, and Equipment Construction in Progress (Net) (PPENC): double"""
        return self.data["PPENC"]

    @property
    def PPENLI(self) -> pd.Series:
        """PPENLI -- Property, Plant, and Equipment Land and Improvements (Net) (PPENLI): double"""
        return self.data["PPENLI"]

    @property
    def PPENLS(self) -> pd.Series:
        """PPENLS -- Property, Plant, and Equipment Leases (Net) (PPENLS): double"""
        return self.data["PPENLS"]

    @property
    def PPENME(self) -> pd.Series:
        """PPENME -- Property, Plant, and Equipment Machinery and Equipment (Net) (PPENME): double"""
        return self.data["PPENME"]

    @property
    def PPENNR(self) -> pd.Series:
        """PPENNR -- Property, Plant, and Equipment Natural Resources (Net) (PPENNR): double"""
        return self.data["PPENNR"]

    @property
    def PPENO(self) -> pd.Series:
        """PPENO -- Property, Plant, and Equipment Other (Net) (PPENO): double"""
        return self.data["PPENO"]

    @property
    def PPENT(self) -> pd.Series:
        """PPENT -- Property, Plant and Equipment - Total (Net) (PPENT): double"""
        return self.data["PPENT"]

    @property
    def PPEVBB(self) -> pd.Series:
        """PPEVBB -- Property, Plant and Equipment Beginning Balance (Schedule V) (PPEVBB): double"""
        return self.data["PPEVBB"]

    @property
    def PPEVEB(self) -> pd.Series:
        """PPEVEB -- Property, Plant, and Equipment Ending Balance (Schedule V) (PPEVEB): double"""
        return self.data["PPEVEB"]

    @property
    def PPEVO(self) -> pd.Series:
        """PPEVO -- Property, Plant, and Equipment Other Changes (Schedule V) (PPEVO): double"""
        return self.data["PPEVO"]

    @property
    def PPEVR(self) -> pd.Series:
        """PPEVR -- Property, Plant and Equipment Retirements (Schedule V) (PPEVR): double"""
        return self.data["PPEVR"]

    @property
    def PRC(self) -> pd.Series:
        """PRC -- Participation Rights Certificates (PRC): double"""
        return self.data["PRC"]

    @property
    def PRODV(self) -> pd.Series:
        """PRODV -- Proposed Dividends (PRODV): double"""
        return self.data["PRODV"]

    @property
    def PRVT(self) -> pd.Series:
        """PRVT -- Policy Reserves - Total (Statutory) (PRVT): double"""
        return self.data["PRVT"]

    @property
    def PSTK(self) -> pd.Series:
        """PSTK -- Preferred/Preference Stock (Capital) - Total (PSTK): double"""
        return self.data["PSTK"]

    @property
    def PSTKC(self) -> pd.Series:
        """PSTKC -- Preferred Stock Convertible (PSTKC): double"""
        return self.data["PSTKC"]

    @property
    def PSTKL(self) -> pd.Series:
        """PSTKL -- Preferred Stock Liquidating Value (PSTKL): double"""
        return self.data["PSTKL"]

    @property
    def PSTKN(self) -> pd.Series:
        """PSTKN -- Preferred/Preference Stock - Nonredeemable (PSTKN): double"""
        return self.data["PSTKN"]

    @property
    def PSTKR(self) -> pd.Series:
        """PSTKR -- Preferred/Preference Stock - Redeemable (PSTKR): double"""
        return self.data["PSTKR"]

    @property
    def PSTKRV(self) -> pd.Series:
        """PSTKRV -- Preferred Stock Redemption Value (PSTKRV): double"""
        return self.data["PSTKRV"]

    @property
    def PVCL(self) -> pd.Series:
        """PVCL -- Provision - Credit Losses (Balance Sheet) (PVCL): double"""
        return self.data["PVCL"]

    @property
    def PVPL(self) -> pd.Series:
        """PVPL -- Provision - Pension Liabilities (PVPL): double"""
        return self.data["PVPL"]

    @property
    def PVT(self) -> pd.Series:
        """PVT -- Provisions - Total (PVT): double"""
        return self.data["PVT"]

    @property
    def RADP(self) -> pd.Series:
        """RADP -- Reinsurance Assets - Deposits and Other (Insurance) (RADP): double"""
        return self.data["RADP"]

    @property
    def RAGR(self) -> pd.Series:
        """RAGR -- Resale Agreements (RAGR): double"""
        return self.data["RAGR"]

    @property
    def RARI(self) -> pd.Series:
        """RARI -- Reinsurance Assets - Receivable/Debtors (Insurance) (RARI): double"""
        return self.data["RARI"]

    @property
    def RATI(self) -> pd.Series:
        """RATI -- Reinsurance Assets - Total (Insurance) (RATI): double"""
        return self.data["RATI"]

    @property
    def RCL(self) -> pd.Series:
        """RCL -- Reserves for Credit Losses (Assets) (RCL): double"""
        return self.data["RCL"]

    @property
    def RDP(self) -> pd.Series:
        """RDP -- Regulatory Deposits (RDP): double"""
        return self.data["RDP"]

    @property
    def RE(self) -> pd.Series:
        """RE -- Retained Earnings (RE): double"""
        return self.data["RE"]

    @property
    def REA(self) -> pd.Series:
        """REA -- Retained Earnings Restatement (REA): double"""
        return self.data["REA"]

    @property
    def REAJO(self) -> pd.Series:
        """REAJO -- Retained Earnings Other Adjustments (REAJO): double"""
        return self.data["REAJO"]

    @property
    def RECCO(self) -> pd.Series:
        """RECCO -- Receivables - Current - Other (RECCO): double"""
        return self.data["RECCO"]

    @property
    def RECD(self) -> pd.Series:
        """RECD -- Receivables - Estimated Doubtful (RECD): double"""
        return self.data["RECD"]

    @property
    def RECT(self) -> pd.Series:
        """RECT -- Receivables Total (RECT): double"""
        return self.data["RECT"]

    @property
    def RECTA(self) -> pd.Series:
        """RECTA -- Retained Earnings Cumulative Translation Adjustment (RECTA): double"""
        return self.data["RECTA"]

    @property
    def RECTR(self) -> pd.Series:
        """RECTR -- Receivables - Trade (RECTR): double"""
        return self.data["RECTR"]

    @property
    def RECUB(self) -> pd.Series:
        """RECUB -- Unbilled Receivables (RECUB): double"""
        return self.data["RECUB"]

    @property
    def RET(self) -> pd.Series:
        """RET -- Total RE Property (RET): double"""
        return self.data["RET"]

    @property
    def REUNA(self) -> pd.Series:
        """REUNA -- Retained Earnings Unadjusted (REUNA): double"""
        return self.data["REUNA"]

    @property
    def REUNR(self) -> pd.Series:
        """REUNR -- Retained Earnings Unrestricted (REUNR): double"""
        return self.data["REUNR"]

    @property
    def RLL(self) -> pd.Series:
        """RLL -- Reserve for Loan/Asset Losses (RLL): double"""
        return self.data["RLL"]

    @property
    def RLO(self) -> pd.Series:
        """RLO -- Reinsurance Liabilities - Other (RLO): double"""
        return self.data["RLO"]

    @property
    def RLP(self) -> pd.Series:
        """RLP -- Reinsurance Liabilities - Payables/Creditors (RLP): double"""
        return self.data["RLP"]

    @property
    def RLRI(self) -> pd.Series:
        """RLRI -- Reinsurers' Liability for Reserves (Insurance) (RLRI): double"""
        return self.data["RLRI"]

    @property
    def RLT(self) -> pd.Series:
        """RLT -- Reinsurance Liabilities - Total (RLT): double"""
        return self.data["RLT"]

    @property
    def RPAG(self) -> pd.Series:
        """RPAG -- Repurchase Agreements (RPAG): double"""
        return self.data["RPAG"]

    @property
    def RREPS(self) -> pd.Series:
        """RREPS -- Reversal Restructuring/Acq Basic EPS Effect (RREPS): double"""
        return self.data["RREPS"]

    @property
    def RVBCI(self) -> pd.Series:
        """RVBCI -- Reserves for Benefits - Life - Claims (Insurance) (RVBCI): double"""
        return self.data["RVBCI"]

    @property
    def RVBPI(self) -> pd.Series:
        """RVBPI -- Reserves for Benefits - Life - Policy (Insurance) (RVBPI): double"""
        return self.data["RVBPI"]

    @property
    def RVBTI(self) -> pd.Series:
        """RVBTI -- Reserves for Benefits - Life - Total (Insurance) (RVBTI): double"""
        return self.data["RVBTI"]

    @property
    def RVDO(self) -> pd.Series:
        """RVDO -- Reserves - Distributable - Other (RVDO): double"""
        return self.data["RVDO"]

    @property
    def RVDT(self) -> pd.Series:
        """RVDT -- Reserves - Distributable - Total (RVDT): double"""
        return self.data["RVDT"]

    @property
    def RVEQT(self) -> pd.Series:
        """RVEQT -- Equity Reserves - Total (RVEQT): double"""
        return self.data["RVEQT"]

    @property
    def RVLRV(self) -> pd.Series:
        """RVLRV -- Revaluation Reserve (RVLRV): double"""
        return self.data["RVLRV"]

    @property
    def RVNO(self) -> pd.Series:
        """RVNO -- Reserves - Nondistributable - Other (RVNO): double"""
        return self.data["RVNO"]

    @property
    def RVNT(self) -> pd.Series:
        """RVNT -- Reserves - Nondistributable - Total (RVNT): double"""
        return self.data["RVNT"]

    @property
    def RVRI(self) -> pd.Series:
        """RVRI -- Reserves - Reinsurance (Insurance) (RVRI): double"""
        return self.data["RVRI"]

    @property
    def RVSI(self) -> pd.Series:
        """RVSI -- Reserves - Sundry (Insurance) (RVSI): double"""
        return self.data["RVSI"]

    @property
    def RVTI(self) -> pd.Series:
        """RVTI -- Reserves - Total (RVTI): double"""
        return self.data["RVTI"]

    @property
    def RVTXR(self) -> pd.Series:
        """RVTXR -- Reserves - Tax-Regulated (RVTXR): double"""
        return self.data["RVTXR"]

    @property
    def RVUPI(self) -> pd.Series:
        """RVUPI -- Reserves for Unearned Premiums (Insurance) (RVUPI): double"""
        return self.data["RVUPI"]

    @property
    def RVUTX(self) -> pd.Series:
        """RVUTX -- Reserves - Untaxed (RVUTX): double"""
        return self.data["RVUTX"]

    @property
    def SAA(self) -> pd.Series:
        """SAA -- Separate Account Assets (SAA): double"""
        return self.data["SAA"]

    @property
    def SAL(self) -> pd.Series:
        """SAL -- Separate Account Liabilities (SAL): double"""
        return self.data["SAL"]

    @property
    def SBDC(self) -> pd.Series:
        """SBDC -- Securities Borrowed and Deposited by Customers (SBDC): double"""
        return self.data["SBDC"]

    @property
    def SC(self) -> pd.Series:
        """SC -- Securities In Custody (SC): double"""
        return self.data["SC"]

    @property
    def SCO(self) -> pd.Series:
        """SCO -- Share Capital - Other (SCO): double"""
        return self.data["SCO"]

    @property
    def SECU(self) -> pd.Series:
        """SECU -- Securities Gains (Losses) - Unrealized (SECU): double"""
        return self.data["SECU"]

    @property
    def SEQ(self) -> pd.Series:
        """SEQ -- Stockholders' Equity - Total (SEQ): double"""
        return self.data["SEQ"]

    @property
    def SEQO(self) -> pd.Series:
        """SEQO -- Other Stockholders Equity Adjustments (SEQO): double"""
        return self.data["SEQO"]

    @property
    def SRT(self) -> pd.Series:
        """SRT -- Surplus - Total (Statutory) (SRT): double"""
        return self.data["SRT"]

    @property
    def SSNP(self) -> pd.Series:
        """SSNP -- Securities Sold Not Yet Purchased (SSNP): double"""
        return self.data["SSNP"]

    @property
    def STBO(self) -> pd.Series:
        """STBO -- Short-Term Borrowings - Other (STBO): double"""
        return self.data["STBO"]

    @property
    def STIO(self) -> pd.Series:
        """STIO -- Short-Term Investments - Other (STIO): double"""
        return self.data["STIO"]

    @property
    def TDSCD(self) -> pd.Series:
        """TDSCD -- Trading/Dealing Account Securities - Corporate Debt (TDSCD): double"""
        return self.data["TDSCD"]

    @property
    def TDSCE(self) -> pd.Series:
        """TDSCE -- Trading/Dealing Account Securities - Corporate Equity (TDSCE): double"""
        return self.data["TDSCE"]

    @property
    def TDSLG(self) -> pd.Series:
        """TDSLG -- Trading/Dealing Account Securities - Local Governments (TDSLG): double"""
        return self.data["TDSLG"]

    @property
    def TDSMM(self) -> pd.Series:
        """TDSMM -- Trading/Dealing Account Securities - Money Market (TDSMM): double"""
        return self.data["TDSMM"]

    @property
    def TDSNG(self) -> pd.Series:
        """TDSNG -- Trading/Dealing Account Securities - National Governments (TDSNG): double"""
        return self.data["TDSNG"]

    @property
    def TDSO(self) -> pd.Series:
        """TDSO -- Trading/Dealing Account Securities - Other (TDSO): double"""
        return self.data["TDSO"]

    @property
    def TDSS(self) -> pd.Series:
        """TDSS -- Trading/Dealing Account Securities - Sundry (TDSS): double"""
        return self.data["TDSS"]

    @property
    def TDST(self) -> pd.Series:
        """TDST -- Trading/Dealing Account Securities - Total (TDST): double"""
        return self.data["TDST"]

    @property
    def TLCF(self) -> pd.Series:
        """TLCF -- Tax Loss Carry Forward (TLCF): double"""
        return self.data["TLCF"]

    @property
    def TRANSA(self) -> pd.Series:
        """TRANSA -- Cumulative Translation Adjustment (TRANSA): double"""
        return self.data["TRANSA"]

    @property
    def TSA(self) -> pd.Series:
        """TSA -- Treasury Stock (Assets) (TSA): double"""
        return self.data["TSA"]

    @property
    def TSO(self) -> pd.Series:
        """TSO -- Treasury Stock - Other Share Capital (TSO): double"""
        return self.data["TSO"]

    @property
    def TSTK(self) -> pd.Series:
        """TSTK -- Treasury Stock - Total (All Capital) (TSTK): double"""
        return self.data["TSTK"]

    @property
    def TSTKC(self) -> pd.Series:
        """TSTKC -- Treasury Stock - Common (TSTKC): double"""
        return self.data["TSTKC"]

    @property
    def TSTKME(self) -> pd.Series:
        """TSTKME -- Treasury Stock Book Value Memo Entry (TSTKME): double"""
        return self.data["TSTKME"]

    @property
    def TSTKP(self) -> pd.Series:
        """TSTKP -- Treasury Stock - Preferrred (TSTKP): double"""
        return self.data["TSTKP"]

    @property
    def TXDB(self) -> pd.Series:
        """TXDB -- Deferred Taxes (Balance Sheet) (TXDB): double"""
        return self.data["TXDB"]

    @property
    def TXDBA(self) -> pd.Series:
        """TXDBA -- Deferred Tax Asset - Long Term (TXDBA): double"""
        return self.data["TXDBA"]

    @property
    def TXDBCA(self) -> pd.Series:
        """TXDBCA -- Deferred Tax Asset - Current (TXDBCA): double"""
        return self.data["TXDBCA"]

    @property
    def TXDBCL(self) -> pd.Series:
        """TXDBCL -- Deferred Tax Liability - Current (TXDBCL): double"""
        return self.data["TXDBCL"]

    @property
    def TXDITC(self) -> pd.Series:
        """TXDITC -- Deferred Taxes and Investment Tax Credit (TXDITC): double"""
        return self.data["TXDITC"]

    @property
    def TXNDB(self) -> pd.Series:
        """TXNDB -- Net Deferred Tax Asset (Liab) - Total (TXNDB): double"""
        return self.data["TXNDB"]

    @property
    def TXNDBA(self) -> pd.Series:
        """TXNDBA -- Net Deferred Tax Asset (TXNDBA): double"""
        return self.data["TXNDBA"]

    @property
    def TXNDBL(self) -> pd.Series:
        """TXNDBL -- Net Deferred Tax Liability (TXNDBL): double"""
        return self.data["TXNDBL"]

    @property
    def TXNDBR(self) -> pd.Series:
        """TXNDBR -- Deferred Tax Residual (TXNDBR): double"""
        return self.data["TXNDBR"]

    @property
    def TXP(self) -> pd.Series:
        """TXP -- Income Taxes Payable (TXP): double"""
        return self.data["TXP"]

    @property
    def TXR(self) -> pd.Series:
        """TXR -- Income Tax Refund (TXR): double"""
        return self.data["TXR"]

    @property
    def UAOX(self) -> pd.Series:
        """UAOX -- Other Assets - Utility (UAOX): double"""
        return self.data["UAOX"]

    @property
    def UAPT(self) -> pd.Series:
        """UAPT -- Accounts Payable - Utility (UAPT): double"""
        return self.data["UAPT"]

    @property
    def UCAPS(self) -> pd.Series:
        """UCAPS -- Paid in Capital - Other (UCAPS): double"""
        return self.data["UCAPS"]

    @property
    def UCCONS(self) -> pd.Series:
        """UCCONS -- Contributions in Aid of Construction (UCCONS): double"""
        return self.data["UCCONS"]

    @property
    def UCEQ(self) -> pd.Series:
        """UCEQ -- Common Equity Total - Utility (UCEQ): double"""
        return self.data["UCEQ"]

    @property
    def UCUSTAD(self) -> pd.Series:
        """UCUSTAD -- Customer Advances for Construction (UCUSTAD): double"""
        return self.data["UCUSTAD"]

    @property
    def UDCOPRES(self) -> pd.Series:
        """UDCOPRES -- Deferred Credits and Operating Reserves - Other (UDCOPRES): double"""
        return self.data["UDCOPRES"]

    @property
    def UDD(self) -> pd.Series:
        """UDD -- Debt (Debentures) (UDD): double"""
        return self.data["UDD"]

    @property
    def UDMB(self) -> pd.Series:
        """UDMB -- Debt (Mortgage Bonds) - Utility (UDMB): double"""
        return self.data["UDMB"]

    @property
    def UDOLT(self) -> pd.Series:
        """UDOLT -- Debt (Other Long-Term) - Utility (UDOLT): double"""
        return self.data["UDOLT"]

    @property
    def UDPCO(self) -> pd.Series:
        """UDPCO -- Debt (Pollution Control Obligations) - Utility (UDPCO): double"""
        return self.data["UDPCO"]

    @property
    def UI(self) -> pd.Series:
        """UI -- Unearned Income (UI): double"""
        return self.data["UI"]

    @property
    def UINVT(self) -> pd.Series:
        """UINVT -- Inventories - Utility (UINVT): double"""
        return self.data["UINVT"]

    @property
    def ULCM(self) -> pd.Series:
        """ULCM -- Current Liabilities - Miscellaneous (ULCM): double"""
        return self.data["ULCM"]

    @property
    def ULCO(self) -> pd.Series:
        """ULCO -- Current Liabilities - Other - Utility (ULCO): double"""
        return self.data["ULCO"]

    @property
    def UNL(self) -> pd.Series:
        """UNL -- Unappropriated Net Loss (UNL): double"""
        return self.data["UNL"]

    @property
    def UNNP(self) -> pd.Series:
        """UNNP -- Unappropriated Net Profit (Stockholders' Equity) (UNNP): double"""
        return self.data["UNNP"]

    @property
    def UNNPL(self) -> pd.Series:
        """UNNPL -- Unappropriated Net Profit (UNNPL): double"""
        return self.data["UNNPL"]

    @property
    def UOPRES(self) -> pd.Series:
        """UOPRES -- Operating Reserves (UOPRES): double"""
        return self.data["UOPRES"]

    @property
    def UPMCSTK(self) -> pd.Series:
        """UPMCSTK -- Premium on Common Stock* (UPMCSTK): double"""
        return self.data["UPMCSTK"]

    @property
    def UPMPF(self) -> pd.Series:
        """UPMPF -- Premium on Preferred Stock* (UPMPF): double"""
        return self.data["UPMPF"]

    @property
    def UPMPFS(self) -> pd.Series:
        """UPMPFS -- Premium on Preference Stock* (UPMPFS): double"""
        return self.data["UPMPFS"]

    @property
    def UPMSUBP(self) -> pd.Series:
        """UPMSUBP -- Premium on Subsidiary Preferred Stock* (UPMSUBP): double"""
        return self.data["UPMSUBP"]

    @property
    def UPSTK(self) -> pd.Series:
        """UPSTK -- Preferred Stock at Carrying Value (UPSTK): double"""
        return self.data["UPSTK"]

    @property
    def UPSTKC(self) -> pd.Series:
        """UPSTKC -- Preference Stock at Carrying Value* (UPSTKC): double"""
        return self.data["UPSTKC"]

    @property
    def UPSTKSF(self) -> pd.Series:
        """UPSTKSF -- Preferred/Preference Stock Sinking Fund Requirement (UPSTKSF): double"""
        return self.data["UPSTKSF"]

    @property
    def URECT(self) -> pd.Series:
        """URECT -- Receivables (Net) (URECT): double"""
        return self.data["URECT"]

    @property
    def URECTR(self) -> pd.Series:
        """URECTR -- Accounts Receivable - Trade - Utility (URECTR): double"""
        return self.data["URECTR"]

    @property
    def UREVUB(self) -> pd.Series:
        """UREVUB -- Accrued Unbilled Revenues (Balance Sheet) (UREVUB): double"""
        return self.data["UREVUB"]

    @property
    def USUBPSTK(self) -> pd.Series:
        """USUBPSTK -- Subsidiary Preferred Stock at Carrying Value (USUBPSTK): double"""
        return self.data["USUBPSTK"]

    @property
    def VPAC(self) -> pd.Series:
        """VPAC -- Investments - Permanent - Associated Companies (VPAC): double"""
        return self.data["VPAC"]

    @property
    def VPO(self) -> pd.Series:
        """VPO -- Investments - Permanent - Other (VPO): double"""
        return self.data["VPO"]

    @property
    def WCAP(self) -> pd.Series:
        """WCAP -- Working Capital (Balance Sheet) (WCAP): double"""
        return self.data["WCAP"]

    @property
    def XACC(self) -> pd.Series:
        """XACC -- Accrued Expenses (XACC): double"""
        return self.data["XACC"]

    @property
    def XPP(self) -> pd.Series:
        """XPP -- Prepaid Expenses (XPP): double"""
        return self.data["XPP"]

    @property
    def ACCHG(self) -> pd.Series:
        """ACCHG -- Accounting Changes Cumulative Effect (ACCHG): double"""
        return self.data["ACCHG"]

    @property
    def ADPAC(self) -> pd.Series:
        """ADPAC -- Amortization of Deferred Policy Acquisition Costs (ADPAC): double"""
        return self.data["ADPAC"]

    @property
    def AM(self) -> pd.Series:
        """AM -- Amortization of Intangibles (AM): double"""
        return self.data["AM"]

    @property
    def AMDC(self) -> pd.Series:
        """AMDC -- Amortization of Deferred Charges (AMDC): double"""
        return self.data["AMDC"]

    @property
    def AMGW(self) -> pd.Series:
        """AMGW -- Amortization of Goodwill (AMGW): double"""
        return self.data["AMGW"]

    @property
    def AQA(self) -> pd.Series:
        """AQA -- Acquisition/Merger After-tax (AQA): double"""
        return self.data["AQA"]

    @property
    def AQD(self) -> pd.Series:
        """AQD -- Acquisition/Merger Diluted EPS Effect (AQD): double"""
        return self.data["AQD"]

    @property
    def AQEPS(self) -> pd.Series:
        """AQEPS -- Acquisition/Merger Basic EPS Effect (AQEPS): double"""
        return self.data["AQEPS"]

    @property
    def AQI(self) -> pd.Series:
        """AQI -- Acquisitions Income Contribution (AQI): double"""
        return self.data["AQI"]

    @property
    def AQP(self) -> pd.Series:
        """AQP -- Acquisition/Merger Pretax (AQP): double"""
        return self.data["AQP"]

    @property
    def AQS(self) -> pd.Series:
        """AQS -- Acquisitions Sales Contribution (AQS): double"""
        return self.data["AQS"]

    @property
    def ARCE(self) -> pd.Series:
        """ARCE -- As Reported Core After-tax (ARCE): double"""
        return self.data["ARCE"]

    @property
    def ARCED(self) -> pd.Series:
        """ARCED -- As Reported Core Diluted EPS Effect (ARCED): double"""
        return self.data["ARCED"]

    @property
    def ARCEEPS(self) -> pd.Series:
        """ARCEEPS -- As Reported Core Basic EPS Effect (ARCEEPS): double"""
        return self.data["ARCEEPS"]

    @property
    def AUTXR(self) -> pd.Series:
        """AUTXR -- Appropriations to Untaxed Reserves (AUTXR): double"""
        return self.data["AUTXR"]

    @property
    def BALR(self) -> pd.Series:
        """BALR -- Benefits Assumed - Life (BALR): double"""
        return self.data["BALR"]

    @property
    def BANLR(self) -> pd.Series:
        """BANLR -- Benefits Assumed - Nonlife (BANLR): double"""
        return self.data["BANLR"]

    @property
    def BATR(self) -> pd.Series:
        """BATR -- Benefits Assumed - Total (BATR): double"""
        return self.data["BATR"]

    @property
    def BCEF(self) -> pd.Series:
        """BCEF -- Brokerage, Clearing and Exchange Fees (BCEF): double"""
        return self.data["BCEF"]

    @property
    def BCLR(self) -> pd.Series:
        """BCLR -- Benefits Ceded - Life (BCLR): double"""
        return self.data["BCLR"]

    @property
    def BCLTBL(self) -> pd.Series:
        """BCLTBL -- Benefits and Claims - Total (Business Line) (BCLTBL): double"""
        return self.data["BCLTBL"]

    @property
    def BCNLR(self) -> pd.Series:
        """BCNLR -- Benefits Ceded - Nonlife (BCNLR): double"""
        return self.data["BCNLR"]

    @property
    def BCRBL(self) -> pd.Series:
        """BCRBL -- Benefits and Claims - Reinsurance (Business Line) (BCRBL): double"""
        return self.data["BCRBL"]

    @property
    def BCT(self) -> pd.Series:
        """BCT -- Benefits and Claims - Total (Insurance) (BCT): double"""
        return self.data["BCT"]

    @property
    def BCTBL(self) -> pd.Series:
        """BCTBL -- Benefits and Claims - Other (Business Line) (BCTBL): double"""
        return self.data["BCTBL"]

    @property
    def BCTR(self) -> pd.Series:
        """BCTR -- Benefits Ceded - Total (BCTR): double"""
        return self.data["BCTR"]

    @property
    def BLTBL(self) -> pd.Series:
        """BLTBL -- Benefits - Life - Total (Business Line) (BLTBL): double"""
        return self.data["BLTBL"]

    @property
    def CBI(self) -> pd.Series:
        """CBI -- Claims Incurred - Insurance (CBI): double"""
        return self.data["CBI"]

    @property
    def CDPAC(self) -> pd.Series:
        """CDPAC -- Capitalized Deferred Polcy Acquisition Costs (CDPAC): double"""
        return self.data["CDPAC"]

    @property
    def CFBD(self) -> pd.Series:
        """CFBD -- Commissions and Fees - (Broker/Dealer) (CFBD): double"""
        return self.data["CFBD"]

    @property
    def CFERE(self) -> pd.Series:
        """CFERE -- Commissions and Fees - (Real Estate) (CFERE): double"""
        return self.data["CFERE"]

    @property
    def CFO(self) -> pd.Series:
        """CFO -- Commissions and Fees - Other (CFO): double"""
        return self.data["CFO"]

    @property
    def CFPDO(self) -> pd.Series:
        """CFPDO -- Commissions and Fees Paid - Other (CFPDO): double"""
        return self.data["CFPDO"]

    @property
    def CGA(self) -> pd.Series:
        """CGA -- Capital Gains - After-Tax (CGA): double"""
        return self.data["CGA"]

    @property
    def CGRI(self) -> pd.Series:
        """CGRI -- Capital Gains - Realized (Insurance) (CGRI): double"""
        return self.data["CGRI"]

    @property
    def CGTI(self) -> pd.Series:
        """CGTI -- Capital Gains - Total (Insurance) (CGTI): double"""
        return self.data["CGTI"]

    @property
    def CGUI(self) -> pd.Series:
        """CGUI -- Capital Gains - Unrealized (Insurance) (CGUI): double"""
        return self.data["CGUI"]

    @property
    def CIBEGNI(self) -> pd.Series:
        """CIBEGNI -- Comp Inc - Beginning Net Income (CIBEGNI): double"""
        return self.data["CIBEGNI"]

    @property
    def CICURR(self) -> pd.Series:
        """CICURR -- Comp Inc - Currency Trans Adj (CICURR): double"""
        return self.data["CICURR"]

    @property
    def CIDERGL(self) -> pd.Series:
        """CIDERGL -- Comp Inc - Derivative Gains/Losses (CIDERGL): double"""
        return self.data["CIDERGL"]

    @property
    def CIOTHER(self) -> pd.Series:
        """CIOTHER -- Comp Inc - Other Adj (CIOTHER): double"""
        return self.data["CIOTHER"]

    @property
    def CIPEN(self) -> pd.Series:
        """CIPEN -- Comp Inc - Minimum Pension Adj (CIPEN): double"""
        return self.data["CIPEN"]

    @property
    def CISECGL(self) -> pd.Series:
        """CISECGL -- Comp Inc - Securities Gains/Losses (CISECGL): double"""
        return self.data["CISECGL"]

    @property
    def CITOTAL(self) -> pd.Series:
        """CITOTAL -- Comprehensive Income - Total (CITOTAL): double"""
        return self.data["CITOTAL"]

    @property
    def CNLTBL(self) -> pd.Series:
        """CNLTBL -- Claims - Non-Life - Total (Business Line) (CNLTBL): double"""
        return self.data["CNLTBL"]

    @property
    def COGS(self) -> pd.Series:
        """COGS -- Cost of Goods Sold (COGS): double"""
        return self.data["COGS"]

    @property
    def CPCBL(self) -> pd.Series:
        """CPCBL -- Commercial Property and Casualty Claims (Business Line) (CPCBL): double"""
        return self.data["CPCBL"]

    @property
    def CPDOI(self) -> pd.Series:
        """CPDOI -- Claims Paid - Other (CPDOI): double"""
        return self.data["CPDOI"]

    @property
    def CPNLI(self) -> pd.Series:
        """CPNLI -- Claims Paid - Non-Life (CPNLI): double"""
        return self.data["CPNLI"]

    @property
    def CPPBL(self) -> pd.Series:
        """CPPBL -- Commercial Property and Casualty Premiums (Business Line) (CPPBL): double"""
        return self.data["CPPBL"]

    @property
    def CPREI(self) -> pd.Series:
        """CPREI -- Claims Paid - Reinsurance (CPREI): double"""
        return self.data["CPREI"]

    @property
    def CSTKE(self) -> pd.Series:
        """CSTKE -- Common Stock Equivalents - Dollar Savings (CSTKE): double"""
        return self.data["CSTKE"]

    @property
    def DBI(self) -> pd.Series:
        """DBI -- Death Benefits - Insurance (DBI): double"""
        return self.data["DBI"]

    @property
    def DFXA(self) -> pd.Series:
        """DFXA -- Depreciation of Tangible Fixed Assets (DFXA): double"""
        return self.data["DFXA"]

    @property
    def DILADJ(self) -> pd.Series:
        """DILADJ -- Dilution Adjustment (DILADJ): double"""
        return self.data["DILADJ"]

    @property
    def DILAVX(self) -> pd.Series:
        """DILAVX -- Dilution Available Excluding Extraordinary Items (DILAVX): double"""
        return self.data["DILAVX"]

    @property
    def DO(self) -> pd.Series:
        """DO -- Discontinued Operations (DO): double"""
        return self.data["DO"]

    @property
    def DONR(self) -> pd.Series:
        """DONR -- Nonrecurring Disc Operations (DONR): double"""
        return self.data["DONR"]

    @property
    def DP(self) -> pd.Series:
        """DP -- Depreciation and Amortization (DP): double"""
        return self.data["DP"]

    @property
    def DPRET(self) -> pd.Series:
        """DPRET -- Depr/Amort of Property (DPRET): double"""
        return self.data["DPRET"]

    @property
    def DTEA(self) -> pd.Series:
        """DTEA -- Extinguishment of Debt After-tax (DTEA): double"""
        return self.data["DTEA"]

    @property
    def DTED(self) -> pd.Series:
        """DTED -- Extinguishment of Debt Diluted EPS Effect (DTED): double"""
        return self.data["DTED"]

    @property
    def DTEEPS(self) -> pd.Series:
        """DTEEPS -- Extinguishment of Debt Basic EPS Effect (DTEEPS): double"""
        return self.data["DTEEPS"]

    @property
    def DTEP(self) -> pd.Series:
        """DTEP -- Extinguishment of Debt Pretax (DTEP): double"""
        return self.data["DTEP"]

    @property
    def DVC(self) -> pd.Series:
        """DVC -- Dividends Common/Ordinary (DVC): double"""
        return self.data["DVC"]

    @property
    def DVDNP(self) -> pd.Series:
        """DVDNP -- Dividends Declared and Not Provided (DVDNP): double"""
        return self.data["DVDNP"]

    @property
    def DVP(self) -> pd.Series:
        """DVP -- Dividends - Preferred/Preference (DVP): double"""
        return self.data["DVP"]

    @property
    def DVPD(self) -> pd.Series:
        """DVPD -- Cash Dividends Paid (DVPD): double"""
        return self.data["DVPD"]

    @property
    def DVPDP(self) -> pd.Series:
        """DVPDP -- Dividends and Bonuses Paid Policyholders (DVPDP): double"""
        return self.data["DVPDP"]

    @property
    def DVRPIV(self) -> pd.Series:
        """DVRPIV -- Dividends Received from Permanent Investments (DVRPIV): double"""
        return self.data["DVRPIV"]

    @property
    def DVRRE(self) -> pd.Series:
        """DVRRE -- Development Revenue (Real Estate) (DVRRE): double"""
        return self.data["DVRRE"]

    @property
    def DVSCO(self) -> pd.Series:
        """DVSCO -- Dividends - Share Capital - Other (DVSCO): double"""
        return self.data["DVSCO"]

    @property
    def DVT(self) -> pd.Series:
        """DVT -- Dividends - Total (DVT): double"""
        return self.data["DVT"]

    @property
    def EBIT(self) -> pd.Series:
        """EBIT -- Earnings Before Interest and Taxes (EBIT): double"""
        return self.data["EBIT"]

    @property
    def EBITDA(self) -> pd.Series:
        """EBITDA -- Earnings Before Interest (EBITDA): double"""
        return self.data["EBITDA"]

    @property
    def EIEA(self) -> pd.Series:
        """EIEA -- Equity in Earnings - After-Tax (EIEA): double"""
        return self.data["EIEA"]

    @property
    def EMOL(self) -> pd.Series:
        """EMOL -- Directors' Emoluments (EMOL): double"""
        return self.data["EMOL"]

    @property
    def EPSFI(self) -> pd.Series:
        """EPSFI -- Earnings Per Share (Diluted) Including Extraordinary Items (EPSFI): double"""
        return self.data["EPSFI"]

    @property
    def EPSFX(self) -> pd.Series:
        """EPSFX -- Earnings Per Share (Diluted) Excluding Extraordinary Items (EPSFX): double"""
        return self.data["EPSFX"]

    @property
    def EPSPI(self) -> pd.Series:
        """EPSPI -- Earnings Per Share (Basic) Including Extraordinary Items (EPSPI): double"""
        return self.data["EPSPI"]

    @property
    def EPSPX(self) -> pd.Series:
        """EPSPX -- Earnings Per Share (Basic) Excluding Extraordinary Items (EPSPX): double"""
        return self.data["EPSPX"]

    @property
    def ESUB(self) -> pd.Series:
        """ESUB -- Equity in Earnings - Unconsolidated Subsidiaries (ESUB): double"""
        return self.data["ESUB"]

    @property
    def FATD(self) -> pd.Series:
        """FATD -- Fixed Assets and Investments - Disposals - Gain (Loss) (FATD): double"""
        return self.data["FATD"]

    @property
    def FCA(self) -> pd.Series:
        """FCA -- Foreign Exchange Income (Loss) (FCA): double"""
        return self.data["FCA"]

    @property
    def FFO(self) -> pd.Series:
        """FFO -- Funds From Operations (REIT) (FFO): double"""
        return self.data["FFO"]

    @property
    def GBBL(self) -> pd.Series:
        """GBBL -- Group Benefits (Business Line) (GBBL): double"""
        return self.data["GBBL"]

    @property
    def GDWLAM(self) -> pd.Series:
        """GDWLAM -- Goodwill Amortization (GDWLAM): double"""
        return self.data["GDWLAM"]

    @property
    def GDWLIA(self) -> pd.Series:
        """GDWLIA -- Impairments of Goodwill After-tax (GDWLIA): double"""
        return self.data["GDWLIA"]

    @property
    def GDWLID(self) -> pd.Series:
        """GDWLID -- Impairments of Goodwill Diluted EPS Effect (GDWLID): double"""
        return self.data["GDWLID"]

    @property
    def GDWLIEPS(self) -> pd.Series:
        """GDWLIEPS -- Impairments of Goodwill Basic EPS Effect (GDWLIEPS): double"""
        return self.data["GDWLIEPS"]

    @property
    def GDWLIP(self) -> pd.Series:
        """GDWLIP -- Impairments of Goodwill Pretax (GDWLIP): double"""
        return self.data["GDWLIP"]

    @property
    def GLA(self) -> pd.Series:
        """GLA -- Gain/Loss After-tax (GLA): double"""
        return self.data["GLA"]

    @property
    def GLCEA(self) -> pd.Series:
        """GLCEA -- Gain/Loss on Sale (Core Earnings Adjusted) After-tax (GLCEA): double"""
        return self.data["GLCEA"]

    @property
    def GLCED(self) -> pd.Series:
        """GLCED -- Gain/Loss on Sale (Core Earnings Adjusted) Diluted EPS (GLCED): double"""
        return self.data["GLCED"]

    @property
    def GLCEEPS(self) -> pd.Series:
        """GLCEEPS -- Gain/Loss on Sale (Core Earnings Adjusted) Basic EPS Effect (GLCEEPS): double"""
        return self.data["GLCEEPS"]

    @property
    def GLCEP(self) -> pd.Series:
        """GLCEP -- Gain/Loss on Sale (Core Earnings Adjusted) Pretax (GLCEP): double"""
        return self.data["GLCEP"]

    @property
    def GLD(self) -> pd.Series:
        """GLD -- Gain/Loss Diluted EPS Effect (GLD): double"""
        return self.data["GLD"]

    @property
    def GLEPS(self) -> pd.Series:
        """GLEPS -- Gain/Loss Basic EPS Effect (GLEPS): double"""
        return self.data["GLEPS"]

    @property
    def GLP(self) -> pd.Series:
        """GLP -- Gain/Loss Pretax (GLP): double"""
        return self.data["GLP"]

    @property
    def GP(self) -> pd.Series:
        """GP -- Gross Profit (Loss) (GP): double"""
        return self.data["GP"]

    @property
    def GPHBL(self) -> pd.Series:
        """GPHBL -- Group Premiums - Health (Business Line) (GPHBL): double"""
        return self.data["GPHBL"]

    @property
    def GPLBL(self) -> pd.Series:
        """GPLBL -- Group Premiums - Life (Business Line) (GPLBL): double"""
        return self.data["GPLBL"]

    @property
    def GPOBL(self) -> pd.Series:
        """GPOBL -- Group Premiums - Other (Business Line) (GPOBL): double"""
        return self.data["GPOBL"]

    @property
    def GPRBL(self) -> pd.Series:
        """GPRBL -- Group Premiums - Retirement Benefits (Business Line) (GPRBL): double"""
        return self.data["GPRBL"]

    @property
    def GPTBL(self) -> pd.Series:
        """GPTBL -- Group Premiums - Total (Business Line) (GPTBL): double"""
        return self.data["GPTBL"]

    @property
    def GWO(self) -> pd.Series:
        """GWO -- Goodwill Written Off (GWO): double"""
        return self.data["GWO"]

    @property
    def HEDGEGL(self) -> pd.Series:
        """HEDGEGL -- Gain/Loss on Ineffective Hedges (HEDGEGL): double"""
        return self.data["HEDGEGL"]

    @property
    def IB(self) -> pd.Series:
        """IB -- Income Before Extraordinary Items (IB): double"""
        return self.data["IB"]

    @property
    def IBADJ(self) -> pd.Series:
        """IBADJ -- Income Before Extraordinary Items Adjusted for Common Stock Equivalents (IBADJ): double"""
        return self.data["IBADJ"]

    @property
    def IBBL(self) -> pd.Series:
        """IBBL -- Individual Benefits (Business Line) (IBBL): double"""
        return self.data["IBBL"]

    @property
    def IBCOM(self) -> pd.Series:
        """IBCOM -- Income Before Extraordinary Items Available for Common (IBCOM): double"""
        return self.data["IBCOM"]

    @property
    def IBKI(self) -> pd.Series:
        """IBKI -- Investment Banking Income (IBKI): double"""
        return self.data["IBKI"]

    @property
    def IDIIS(self) -> pd.Series:
        """IDIIS -- Interest and Dividend Income - Investment Securities (IDIIS): double"""
        return self.data["IDIIS"]

    @property
    def IDILB(self) -> pd.Series:
        """IDILB -- Interest and Dividend Income - Loans/Claims/Advances - Banks (IDILB): double"""
        return self.data["IDILB"]

    @property
    def IDILC(self) -> pd.Series:
        """IDILC -- Interest and Dividend Income - Loans/Claims/Advances - Customers (IDILC): double"""
        return self.data["IDILC"]

    @property
    def IDIS(self) -> pd.Series:
        """IDIS -- Interest and Dividend Income - Sundry (IDIS): double"""
        return self.data["IDIS"]

    @property
    def IDIST(self) -> pd.Series:
        """IDIST -- Interest and Dividend Income - Short-Term Investments (IDIST): double"""
        return self.data["IDIST"]

    @property
    def IDIT(self) -> pd.Series:
        """IDIT -- Interest and Related Income - Total (IDIT): double"""
        return self.data["IDIT"]

    @property
    def IDITS(self) -> pd.Series:
        """IDITS -- Interest and Dividend Income - Trading Securities (IDITS): double"""
        return self.data["IDITS"]

    @property
    def IIRE(self) -> pd.Series:
        """IIRE -- Investment Income (Real Estate) (IIRE): double"""
        return self.data["IIRE"]

    @property
    def INITB(self) -> pd.Series:
        """INITB -- Income - Non-interest - Total (Bank) (INITB): double"""
        return self.data["INITB"]

    @property
    def INTC(self) -> pd.Series:
        """INTC -- Interest Capitalized (INTC): double"""
        return self.data["INTC"]

    @property
    def IOBD(self) -> pd.Series:
        """IOBD -- Income - Other (Broker Dealer) (IOBD): double"""
        return self.data["IOBD"]

    @property
    def IOI(self) -> pd.Series:
        """IOI -- Income - Other (Insurance) (IOI): double"""
        return self.data["IOI"]

    @property
    def IORE(self) -> pd.Series:
        """IORE -- Income - Other (Real Estate) (IORE): double"""
        return self.data["IORE"]

    @property
    def IPABL(self) -> pd.Series:
        """IPABL -- Individual Premiums - Annuity (Business Line) (IPABL): double"""
        return self.data["IPABL"]

    @property
    def IPHBL(self) -> pd.Series:
        """IPHBL -- Individual Premiums - Health (Business Line) (IPHBL): double"""
        return self.data["IPHBL"]

    @property
    def IPLBL(self) -> pd.Series:
        """IPLBL -- Individual Premiums - Life (Business Line) (IPLBL): double"""
        return self.data["IPLBL"]

    @property
    def IPOBL(self) -> pd.Series:
        """IPOBL -- Individual Premiums - Other (Business Line) (IPOBL): double"""
        return self.data["IPOBL"]

    @property
    def IPTBL(self) -> pd.Series:
        """IPTBL -- Individual Premiums - Total (Business Line) (IPTBL): double"""
        return self.data["IPTBL"]

    @property
    def IPTI(self) -> pd.Series:
        """IPTI -- Insurance Premiums - Total (Insurance) (IPTI): double"""
        return self.data["IPTI"]

    @property
    def IREI(self) -> pd.Series:
        """IREI -- Interest and Related Income - Reinsurance (Insurance) (IREI): double"""
        return self.data["IREI"]

    @property
    def IRENT(self) -> pd.Series:
        """IRENT -- Rental Income (IRENT): double"""
        return self.data["IRENT"]

    @property
    def IRII(self) -> pd.Series:
        """IRII -- Interest and Related Income (Insurance) (IRII): double"""
        return self.data["IRII"]

    @property
    def IRLI(self) -> pd.Series:
        """IRLI -- Interest and Related Income - Life (Insurance) (IRLI): double"""
        return self.data["IRLI"]

    @property
    def IRNLI(self) -> pd.Series:
        """IRNLI -- Interest and Related Income - Non-Life (Insurance) (IRNLI): double"""
        return self.data["IRNLI"]

    @property
    def IRSI(self) -> pd.Series:
        """IRSI -- Interest and Related Income - Sundry (Insurance) (IRSI): double"""
        return self.data["IRSI"]

    @property
    def ISGR(self) -> pd.Series:
        """ISGR -- Investment Securities - Gain (Loss) - Realized (ISGR): double"""
        return self.data["ISGR"]

    @property
    def ISGT(self) -> pd.Series:
        """ISGT -- Investment Securities - Gain (Loss) - Total (ISGT): double"""
        return self.data["ISGT"]

    @property
    def ISGU(self) -> pd.Series:
        """ISGU -- Investment Securities - Gain (Loss) - Unrealized (ISGU): double"""
        return self.data["ISGU"]

    @property
    def ITCI(self) -> pd.Series:
        """ITCI -- Investment Tax Credit (Income Account) (ITCI): double"""
        return self.data["ITCI"]

    @property
    def IVI(self) -> pd.Series:
        """IVI -- Investment Income - Total (Insurance) (IVI): double"""
        return self.data["IVI"]

    @property
    def LI(self) -> pd.Series:
        """LI -- Leasing Income (LI): double"""
        return self.data["LI"]

    @property
    def LLRCI(self) -> pd.Series:
        """LLRCI -- Loan Loss Recoveries - Credited to Income (LLRCI): double"""
        return self.data["LLRCI"]

    @property
    def LLRCR(self) -> pd.Series:
        """LLRCR -- Loan Loss Recoveries - Credited to Reserves (LLRCR): double"""
        return self.data["LLRCR"]

    @property
    def LLWOCI(self) -> pd.Series:
        """LLWOCI -- Loan Loss Written Off - Charged to Income (LLWOCI): double"""
        return self.data["LLWOCI"]

    @property
    def LLWOCR(self) -> pd.Series:
        """LLWOCR -- Loan Loss Written Off - Charged to Reserves (LLWOCR): double"""
        return self.data["LLWOCR"]

    @property
    def LST(self) -> pd.Series:
        """LST -- Life Insurance Surrenders and Terminations (LST): double"""
        return self.data["LST"]

    @property
    def MII(self) -> pd.Series:
        """MII -- Minority Interest (Income Account) (MII): double"""
        return self.data["MII"]

    @property
    def NCO(self) -> pd.Series:
        """NCO -- Net Charge-Offs (NCO): double"""
        return self.data["NCO"]

    @property
    def NFSR(self) -> pd.Series:
        """NFSR -- Non-Financial Services Revenue (NFSR): double"""
        return self.data["NFSR"]

    @property
    def NI(self) -> pd.Series:
        """NI -- Net Income (Loss) (NI): double"""
        return self.data["NI"]

    @property
    def NIADJ(self) -> pd.Series:
        """NIADJ -- Net Income Adjusted for Common/Ordinary Stock (Capital) Equivalents (NIADJ): double"""
        return self.data["NIADJ"]

    @property
    def NIECI(self) -> pd.Series:
        """NIECI -- Net Income Effect Capitalized Interest (NIECI): double"""
        return self.data["NIECI"]

    @property
    def NIINT(self) -> pd.Series:
        """NIINT -- Net Interest Income (NIINT): double"""
        return self.data["NIINT"]

    @property
    def NIIT(self) -> pd.Series:
        """NIIT -- Net Interest Income (Tax Equivalent) (NIIT): double"""
        return self.data["NIIT"]

    @property
    def NIM(self) -> pd.Series:
        """NIM -- Net Interest Margin (NIM): double"""
        return self.data["NIM"]

    @property
    def NIO(self) -> pd.Series:
        """NIO -- Net Items - Other (NIO): double"""
        return self.data["NIO"]

    @property
    def NIT(self) -> pd.Series:
        """NIT -- Net Item - Total (NIT): double"""
        return self.data["NIT"]

    @property
    def NITS(self) -> pd.Series:
        """NITS -- Net Income - Total (Statutory) (NITS): double"""
        return self.data["NITS"]

    @property
    def NOPI(self) -> pd.Series:
        """NOPI -- Nonoperating Income (Expense) (NOPI): double"""
        return self.data["NOPI"]

    @property
    def NOPIO(self) -> pd.Series:
        """NOPIO -- Nonoperating Income (Expense) Other (NOPIO): double"""
        return self.data["NOPIO"]

    @property
    def NRTXT(self) -> pd.Series:
        """NRTXT -- Nonrecurring Income Taxes After-tax (NRTXT): double"""
        return self.data["NRTXT"]

    @property
    def NRTXTD(self) -> pd.Series:
        """NRTXTD -- Nonrecurring Income Tax Diluted EPS Effect (NRTXTD): double"""
        return self.data["NRTXTD"]

    @property
    def NRTXTEPS(self) -> pd.Series:
        """NRTXTEPS -- Nonrecurring Income Tax Basic EPS Effect (NRTXTEPS): double"""
        return self.data["NRTXTEPS"]

    @property
    def OIADP(self) -> pd.Series:
        """OIADP -- Operating Income After Depreciation (OIADP): double"""
        return self.data["OIADP"]

    @property
    def OIBDP(self) -> pd.Series:
        """OIBDP -- Operating Income Before Depreciation (OIBDP): double"""
        return self.data["OIBDP"]

    @property
    def OPEPS(self) -> pd.Series:
        """OPEPS -- Earnings Per Share from Operations (OPEPS): double"""
        return self.data["OPEPS"]

    @property
    def OPILI(self) -> pd.Series:
        """OPILI -- Operating Income - Life (OPILI): double"""
        return self.data["OPILI"]

    @property
    def OPINCAR(self) -> pd.Series:
        """OPINCAR -- Operating Income - As Reported (OPINCAR): double"""
        return self.data["OPINCAR"]

    @property
    def OPINI(self) -> pd.Series:
        """OPINI -- Operating Income - Non-Life (OPINI): double"""
        return self.data["OPINI"]

    @property
    def OPIOI(self) -> pd.Series:
        """OPIOI -- Operating Income - Other (OPIOI): double"""
        return self.data["OPIOI"]

    @property
    def OPIRI(self) -> pd.Series:
        """OPIRI -- Operating Income - Reinsurance (OPIRI): double"""
        return self.data["OPIRI"]

    @property
    def OPITI(self) -> pd.Series:
        """OPITI -- Operating Income - Total (OPITI): double"""
        return self.data["OPITI"]

    @property
    def OPREPSX(self) -> pd.Series:
        """OPREPSX -- Earnings Per Share Diluted from Operations (OPREPSX): double"""
        return self.data["OPREPSX"]

    @property
    def PALR(self) -> pd.Series:
        """PALR -- Premiums Assumed - Life (PALR): double"""
        return self.data["PALR"]

    @property
    def PANLR(self) -> pd.Series:
        """PANLR -- Premiums Assumed - Nonlife (PANLR): double"""
        return self.data["PANLR"]

    @property
    def PATR(self) -> pd.Series:
        """PATR -- Premiums Assumed - Total (PATR): double"""
        return self.data["PATR"]

    @property
    def PCL(self) -> pd.Series:
        """PCL -- Provision - Credit Losses (Income Account) (PCL): double"""
        return self.data["PCL"]

    @property
    def PCLR(self) -> pd.Series:
        """PCLR -- Premiums Ceded - Life (PCLR): double"""
        return self.data["PCLR"]

    @property
    def PCNLR(self) -> pd.Series:
        """PCNLR -- Premiums Ceded - Nonlife (PCNLR): double"""
        return self.data["PCNLR"]

    @property
    def PCTR(self) -> pd.Series:
        """PCTR -- Premiums Ceded - Total (PCTR): double"""
        return self.data["PCTR"]

    @property
    def PI(self) -> pd.Series:
        """PI -- Pretax Income (PI): double"""
        return self.data["PI"]

    @property
    def PIDOM(self) -> pd.Series:
        """PIDOM -- Pretax Income Domestic (PIDOM): double"""
        return self.data["PIDOM"]

    @property
    def PIFO(self) -> pd.Series:
        """PIFO -- Pretax Income Foreign (PIFO): double"""
        return self.data["PIFO"]

    @property
    def PLL(self) -> pd.Series:
        """PLL -- Provision for Loan/Asset Losses (PLL): double"""
        return self.data["PLL"]

    @property
    def PLTBL(self) -> pd.Series:
        """PLTBL -- Premiums - Life - Total (Business Line) (PLTBL): double"""
        return self.data["PLTBL"]

    @property
    def PNCA(self) -> pd.Series:
        """PNCA -- Core Pension Adjustment (PNCA): double"""
        return self.data["PNCA"]

    @property
    def PNCAD(self) -> pd.Series:
        """PNCAD -- Core Pension Adjustment Diluted EPS Effect (PNCAD): double"""
        return self.data["PNCAD"]

    @property
    def PNCAEPS(self) -> pd.Series:
        """PNCAEPS -- Core Pension Adjustment Basic EPS Effect (PNCAEPS): double"""
        return self.data["PNCAEPS"]

    @property
    def PNCIA(self) -> pd.Series:
        """PNCIA -- Core Pension Interest Adjustment After-tax (PNCIA): double"""
        return self.data["PNCIA"]

    @property
    def PNCID(self) -> pd.Series:
        """PNCID -- Core Pension Interest Adjustment Diluted EPS Effect (PNCID): double"""
        return self.data["PNCID"]

    @property
    def PNCIEPS(self) -> pd.Series:
        """PNCIEPS -- Core Pension Interest Adjustment Basic EPS Effect (PNCIEPS): double"""
        return self.data["PNCIEPS"]

    @property
    def PNCIP(self) -> pd.Series:
        """PNCIP -- Core Pension Interest Adjustment Pretax (PNCIP): double"""
        return self.data["PNCIP"]

    @property
    def PNCWIA(self) -> pd.Series:
        """PNCWIA -- Core Pension w/o Interest Adjustment After-tax (PNCWIA): double"""
        return self.data["PNCWIA"]

    @property
    def PNCWID(self) -> pd.Series:
        """PNCWID -- Core Pension w/o Interest Adjustment Diluted EPS Effect (PNCWID): double"""
        return self.data["PNCWID"]

    @property
    def PNCWIEPS(self) -> pd.Series:
        """PNCWIEPS -- Core Pension w/o Interest Adjustment Basic EPS Effect (PNCWIEPS): double"""
        return self.data["PNCWIEPS"]

    @property
    def PNCWIP(self) -> pd.Series:
        """PNCWIP -- Core Pension w/o Interest Adjustment Pretax (PNCWIP): double"""
        return self.data["PNCWIP"]

    @property
    def PNLBL(self) -> pd.Series:
        """PNLBL -- Premiums - Nonlife - Total (Business Line) (PNLBL): double"""
        return self.data["PNLBL"]

    @property
    def PNLI(self) -> pd.Series:
        """PNLI -- Premiums Written - Non-Life (PNLI): double"""
        return self.data["PNLI"]

    @property
    def POBL(self) -> pd.Series:
        """POBL -- Premiums - Other (Business Line) (POBL): double"""
        return self.data["POBL"]

    @property
    def PPCBL(self) -> pd.Series:
        """PPCBL -- Personal Property and Casualty Claims (Business Line) (PPCBL): double"""
        return self.data["PPCBL"]

    @property
    def PPPABL(self) -> pd.Series:
        """PPPABL -- Personal Property and Casualty Premiums - Automobile (Business Line) (PPPABL): double"""
        return self.data["PPPABL"]

    @property
    def PPPHBL(self) -> pd.Series:
        """PPPHBL -- Personal Property and Casualty Premiums - Homeowners (Business Line) (PPPHBL): double"""
        return self.data["PPPHBL"]

    @property
    def PPPOBL(self) -> pd.Series:
        """PPPOBL -- Personal Property and Casualty Premiums - Other (Business Line) (PPPOBL): double"""
        return self.data["PPPOBL"]

    @property
    def PPPTBL(self) -> pd.Series:
        """PPPTBL -- Personal Property & Casualty Premiums - Total (Business Line) (PPPTBL): double"""
        return self.data["PPPTBL"]

    @property
    def PRCA(self) -> pd.Series:
        """PRCA -- Core Post Retirement Adjustment (PRCA): double"""
        return self.data["PRCA"]

    @property
    def PRCAD(self) -> pd.Series:
        """PRCAD -- Core Post Retirement Adjustment Diluted EPS Effect (PRCAD): double"""
        return self.data["PRCAD"]

    @property
    def PRCAEPS(self) -> pd.Series:
        """PRCAEPS -- Core Post Retirement Adjustment Basic EPS Effect (PRCAEPS): double"""
        return self.data["PRCAEPS"]

    @property
    def PREBL(self) -> pd.Series:
        """PREBL -- Premiums - Reinsurance (Business Line) (PREBL): double"""
        return self.data["PREBL"]

    @property
    def PRI(self) -> pd.Series:
        """PRI -- Premiums Written - Reinsurance (PRI): double"""
        return self.data["PRI"]

    @property
    def PTBL(self) -> pd.Series:
        """PTBL -- Premiums - Total (Business Line) (PTBL): double"""
        return self.data["PTBL"]

    @property
    def PTRAN(self) -> pd.Series:
        """PTRAN -- Principal Transactions (PTRAN): double"""
        return self.data["PTRAN"]

    @property
    def PVO(self) -> pd.Series:
        """PVO -- Provision - Other (PVO): double"""
        return self.data["PVO"]

    @property
    def PVON(self) -> pd.Series:
        """PVON -- Provisions - Other (Net) (PVON): double"""
        return self.data["PVON"]

    @property
    def PWOI(self) -> pd.Series:
        """PWOI -- Premiums Written - Other (PWOI): double"""
        return self.data["PWOI"]

    @property
    def RCA(self) -> pd.Series:
        """RCA -- Restructuring Costs After-tax (RCA): double"""
        return self.data["RCA"]

    @property
    def RCD(self) -> pd.Series:
        """RCD -- Restructuring Costs Diluted EPS Effect (RCD): double"""
        return self.data["RCD"]

    @property
    def RCEPS(self) -> pd.Series:
        """RCEPS -- Restructuring Costs Basic EPS Effect (RCEPS): double"""
        return self.data["RCEPS"]

    @property
    def RCP(self) -> pd.Series:
        """RCP -- Restructuring Costs Pretax (RCP): double"""
        return self.data["RCP"]

    @property
    def RDIP(self) -> pd.Series:
        """RDIP -- In Process R&D Expense (RDIP): double"""
        return self.data["RDIP"]

    @property
    def RDIPA(self) -> pd.Series:
        """RDIPA -- In Process R&D Expense After-tax (RDIPA): double"""
        return self.data["RDIPA"]

    @property
    def RDIPD(self) -> pd.Series:
        """RDIPD -- In Process R&D Expense Diluted EPS Effect (RDIPD): double"""
        return self.data["RDIPD"]

    @property
    def RDIPEPS(self) -> pd.Series:
        """RDIPEPS -- In Process R&D Expense Basic EPS Effect (RDIPEPS): double"""
        return self.data["RDIPEPS"]

    @property
    def REVT(self) -> pd.Series:
        """REVT -- Revenue - Total (REVT): double"""
        return self.data["REVT"]

    @property
    def RIS(self) -> pd.Series:
        """RIS -- Revenue/Income - Sundry (RIS): double"""
        return self.data["RIS"]

    @property
    def RMUM(self) -> pd.Series:
        """RMUM -- Auditors' Remuneraton (RMUM): double"""
        return self.data["RMUM"]

    @property
    def RRA(self) -> pd.Series:
        """RRA -- Reversal Restructruring/Acquisition Aftertax (RRA): double"""
        return self.data["RRA"]

    @property
    def RRD(self) -> pd.Series:
        """RRD -- Reversal Restructuring/Acq Diluted EPS Effect (RRD): double"""
        return self.data["RRD"]

    @property
    def RRP(self) -> pd.Series:
        """RRP -- Reversal Restructruring/Acquisition Pretax (RRP): double"""
        return self.data["RRP"]

    @property
    def SALE(self) -> pd.Series:
        """SALE -- Sales/Turnover (Net) (SALE): double"""
        return self.data["SALE"]

    @property
    def SETA(self) -> pd.Series:
        """SETA -- Settlement (Litigation/Insurance) After-tax (SETA): double"""
        return self.data["SETA"]

    @property
    def SETD(self) -> pd.Series:
        """SETD -- Settlement (Litigation/Insurance) Diluted EPS Effect (SETD): double"""
        return self.data["SETD"]

    @property
    def SETEPS(self) -> pd.Series:
        """SETEPS -- Settlement (Litigation/Insurance) Basic EPS Effect (SETEPS): double"""
        return self.data["SETEPS"]

    @property
    def SETP(self) -> pd.Series:
        """SETP -- Settlement (Litigation/Insurance) Pretax (SETP): double"""
        return self.data["SETP"]

    @property
    def SPCE(self) -> pd.Series:
        """SPCE -- S&P Core Earnings (SPCE): double"""
        return self.data["SPCE"]

    @property
    def SPCED(self) -> pd.Series:
        """SPCED -- S&P Core Earnings EPS Diluted (SPCED): double"""
        return self.data["SPCED"]

    @property
    def SPCEEPS(self) -> pd.Series:
        """SPCEEPS -- S&P Core Earnings EPS Basic (SPCEEPS): double"""
        return self.data["SPCEEPS"]

    @property
    def SPI(self) -> pd.Series:
        """SPI -- Special Items (SPI): double"""
        return self.data["SPI"]

    @property
    def SPID(self) -> pd.Series:
        """SPID -- Other Special Items Diluted EPS Effect (SPID): double"""
        return self.data["SPID"]

    @property
    def SPIEPS(self) -> pd.Series:
        """SPIEPS -- Other Special Items Basic EPS Effect (SPIEPS): double"""
        return self.data["SPIEPS"]

    @property
    def SPIOA(self) -> pd.Series:
        """SPIOA -- Other Special Items After-tax (SPIOA): double"""
        return self.data["SPIOA"]

    @property
    def SPIOP(self) -> pd.Series:
        """SPIOP -- Other Special Items Pretax (SPIOP): double"""
        return self.data["SPIOP"]

    @property
    def SRET(self) -> pd.Series:
        """SRET -- Gain/Loss on Sale of Property (SRET): double"""
        return self.data["SRET"]

    @property
    def STKCO(self) -> pd.Series:
        """STKCO -- Stock Compensation Expense (STKCO): double"""
        return self.data["STKCO"]

    @property
    def STKCPA(self) -> pd.Series:
        """STKCPA -- After-tax stock compensation (STKCPA): double"""
        return self.data["STKCPA"]

    @property
    def TDSG(self) -> pd.Series:
        """TDSG -- Trading/Dealing Securities - Gain (Loss) (TDSG): double"""
        return self.data["TDSG"]

    @property
    def TF(self) -> pd.Series:
        """TF -- Trust Fees (TF): double"""
        return self.data["TF"]

    @property
    def TIE(self) -> pd.Series:
        """TIE -- Interest Expense Total (Financial Services) (TIE): double"""
        return self.data["TIE"]

    @property
    def TII(self) -> pd.Series:
        """TII -- Interest Income Total (Financial Services) (TII): double"""
        return self.data["TII"]

    @property
    def TXC(self) -> pd.Series:
        """TXC -- Income Taxes - Current (TXC): double"""
        return self.data["TXC"]

    @property
    def TXDFED(self) -> pd.Series:
        """TXDFED -- Deferred Taxes-Federal (TXDFED): double"""
        return self.data["TXDFED"]

    @property
    def TXDFO(self) -> pd.Series:
        """TXDFO -- Deferred Taxes-Foreign (TXDFO): double"""
        return self.data["TXDFO"]

    @property
    def TXDI(self) -> pd.Series:
        """TXDI -- Income Taxes - Deferred (TXDI): double"""
        return self.data["TXDI"]

    @property
    def TXDS(self) -> pd.Series:
        """TXDS -- Deferred Taxes-State (TXDS): double"""
        return self.data["TXDS"]

    @property
    def TXEQA(self) -> pd.Series:
        """TXEQA -- Tax - Equivalent Adjustment (TXEQA): double"""
        return self.data["TXEQA"]

    @property
    def TXEQII(self) -> pd.Series:
        """TXEQII -- Tax - Equivalent Interest Income (Gross) (TXEQII): double"""
        return self.data["TXEQII"]

    @property
    def TXFED(self) -> pd.Series:
        """TXFED -- Income Taxes Federal (TXFED): double"""
        return self.data["TXFED"]

    @property
    def TXFO(self) -> pd.Series:
        """TXFO -- Income Taxes - Foreign (TXFO): double"""
        return self.data["TXFO"]

    @property
    def TXO(self) -> pd.Series:
        """TXO -- Income Taxes - Other (TXO): double"""
        return self.data["TXO"]

    @property
    def TXS(self) -> pd.Series:
        """TXS -- Income Taxes State (TXS): double"""
        return self.data["TXS"]

    @property
    def TXT(self) -> pd.Series:
        """TXT -- Income Taxes - Total (TXT): double"""
        return self.data["TXT"]

    @property
    def TXVA(self) -> pd.Series:
        """TXVA -- Value Added Taxes (TXVA): double"""
        return self.data["TXVA"]

    @property
    def TXW(self) -> pd.Series:
        """TXW -- Excise Taxes (TXW): double"""
        return self.data["TXW"]

    @property
    def UDPFA(self) -> pd.Series:
        """UDPFA -- Depreciation of Fixed Assets (UDPFA): double"""
        return self.data["UDPFA"]

    @property
    def UDVP(self) -> pd.Series:
        """UDVP -- Preferred Dividend Requirements (UDVP): double"""
        return self.data["UDVP"]

    @property
    def UGI(self) -> pd.Series:
        """UGI -- Gross Income (Income Before Interest Charges) (UGI): double"""
        return self.data["UGI"]

    @property
    def UNIAMI(self) -> pd.Series:
        """UNIAMI -- Net Income before Extraordinary Items and after Minority Interest (UNIAMI): double"""
        return self.data["UNIAMI"]

    @property
    def UNOPINC(self) -> pd.Series:
        """UNOPINC -- Nonoperating Income (Net) - Other (UNOPINC): double"""
        return self.data["UNOPINC"]

    @property
    def UOPI(self) -> pd.Series:
        """UOPI -- Operating Income - Total - Utility (UOPI): double"""
        return self.data["UOPI"]

    @property
    def UPDVP(self) -> pd.Series:
        """UPDVP -- Preference Dividend Requirements* (UPDVP): double"""
        return self.data["UPDVP"]

    @property
    def USPI(self) -> pd.Series:
        """USPI -- Special Items (USPI): double"""
        return self.data["USPI"]

    @property
    def USUBDVP(self) -> pd.Series:
        """USUBDVP -- Subsidiary Preferred Dividends (USUBDVP): double"""
        return self.data["USUBDVP"]

    @property
    def UTME(self) -> pd.Series:
        """UTME -- Maintenance Expense - Total (UTME): double"""
        return self.data["UTME"]

    @property
    def UTXFED(self) -> pd.Series:
        """UTXFED -- Current Taxes - Federal (Operating) (UTXFED): double"""
        return self.data["UTXFED"]

    @property
    def UXINST(self) -> pd.Series:
        """UXINST -- Interest On Short-Term Debt - Utility (UXINST): double"""
        return self.data["UXINST"]

    @property
    def UXINTD(self) -> pd.Series:
        """UXINTD -- Interest on Long-Term Debt* (UXINTD): double"""
        return self.data["UXINTD"]

    @property
    def WDA(self) -> pd.Series:
        """WDA -- Writedowns After-tax (WDA): double"""
        return self.data["WDA"]

    @property
    def WDD(self) -> pd.Series:
        """WDD -- Writedowns Diluted EPS Effect (WDD): double"""
        return self.data["WDD"]

    @property
    def WDEPS(self) -> pd.Series:
        """WDEPS -- Writedowns Basic EPS Effect (WDEPS): double"""
        return self.data["WDEPS"]

    @property
    def WDP(self) -> pd.Series:
        """WDP -- Writedowns Pretax (WDP): double"""
        return self.data["WDP"]

    @property
    def XAD(self) -> pd.Series:
        """XAD -- Advertising Expense (XAD): double"""
        return self.data["XAD"]

    @property
    def XAGO(self) -> pd.Series:
        """XAGO -- Administrative and General Expense - Other (XAGO): double"""
        return self.data["XAGO"]

    @property
    def XAGT(self) -> pd.Series:
        """XAGT -- Administrative and General Expense - Total (XAGT): double"""
        return self.data["XAGT"]

    @property
    def XCOM(self) -> pd.Series:
        """XCOM -- Communications Expense (XCOM): double"""
        return self.data["XCOM"]

    @property
    def XCOMI(self) -> pd.Series:
        """XCOMI -- Commissions Expense (Insurance) (XCOMI): double"""
        return self.data["XCOMI"]

    @property
    def XDEPL(self) -> pd.Series:
        """XDEPL -- Depletion Expense (Schedule VI) (XDEPL): double"""
        return self.data["XDEPL"]

    @property
    def XDP(self) -> pd.Series:
        """XDP -- Depreciation Expense (Schedule VI) (XDP): double"""
        return self.data["XDP"]

    @property
    def XDVRE(self) -> pd.Series:
        """XDVRE -- Expense - Development (Real Estate) (XDVRE): double"""
        return self.data["XDVRE"]

    @property
    def XEQO(self) -> pd.Series:
        """XEQO -- Equipment and Occupancy Expense (XEQO): double"""
        return self.data["XEQO"]

    @property
    def XI(self) -> pd.Series:
        """XI -- Extraordinary Items (XI): double"""
        return self.data["XI"]

    @property
    def XIDO(self) -> pd.Series:
        """XIDO -- Extraordinary Items and Discontinued Operations (XIDO): double"""
        return self.data["XIDO"]

    @property
    def XINDB(self) -> pd.Series:
        """XINDB -- Interest Expense - Deposits - Banks (XINDB): double"""
        return self.data["XINDB"]

    @property
    def XINDC(self) -> pd.Series:
        """XINDC -- Interest Expense - Deposits - Customer (XINDC): double"""
        return self.data["XINDC"]

    @property
    def XINS(self) -> pd.Series:
        """XINS -- Interest Expense - Sundry (XINS): double"""
        return self.data["XINS"]

    @property
    def XINST(self) -> pd.Series:
        """XINST -- Interest Expense - Short-Term Borrowings (XINST): double"""
        return self.data["XINST"]

    @property
    def XINT(self) -> pd.Series:
        """XINT -- Interest and Related Expense - Total (XINT): double"""
        return self.data["XINT"]

    @property
    def XINTD(self) -> pd.Series:
        """XINTD -- Interest Expense - Long-Term Debt (XINTD): double"""
        return self.data["XINTD"]

    @property
    def XINTOPT(self) -> pd.Series:
        """XINTOPT -- Implied Option Expense (XINTOPT): double"""
        return self.data["XINTOPT"]

    @property
    def XIVI(self) -> pd.Series:
        """XIVI -- Investment Expense (Insurance) (XIVI): double"""
        return self.data["XIVI"]

    @property
    def XIVRE(self) -> pd.Series:
        """XIVRE -- Expense - Investment (Real Estate) (XIVRE): double"""
        return self.data["XIVRE"]

    @property
    def XLR(self) -> pd.Series:
        """XLR -- Staff Expense - Total (XLR): double"""
        return self.data["XLR"]

    @property
    def XNBI(self) -> pd.Series:
        """XNBI -- Other Insurance Expense (XNBI): double"""
        return self.data["XNBI"]

    @property
    def XNF(self) -> pd.Series:
        """XNF -- Non-Financial Services Expense (XNF): double"""
        return self.data["XNF"]

    @property
    def XNINS(self) -> pd.Series:
        """XNINS -- Other Expense - Noninsurance (XNINS): double"""
        return self.data["XNINS"]

    @property
    def XNITB(self) -> pd.Series:
        """XNITB -- Expense - Noninterest - Total (Bank) (XNITB): double"""
        return self.data["XNITB"]

    @property
    def XOBD(self) -> pd.Series:
        """XOBD -- Expense - Other (Broker/Dealer) (XOBD): double"""
        return self.data["XOBD"]

    @property
    def XOI(self) -> pd.Series:
        """XOI -- Expenses - Other (Insurance) (XOI): double"""
        return self.data["XOI"]

    @property
    def XOPR(self) -> pd.Series:
        """XOPR -- Operating Expenses Total (XOPR): double"""
        return self.data["XOPR"]

    @property
    def XOPRAR(self) -> pd.Series:
        """XOPRAR -- Operatings Expenses - As Reported (XOPRAR): double"""
        return self.data["XOPRAR"]

    @property
    def XOPTD(self) -> pd.Series:
        """XOPTD -- Implied Option EPS Diluted (XOPTD): double"""
        return self.data["XOPTD"]

    @property
    def XOPTEPS(self) -> pd.Series:
        """XOPTEPS -- Implied Option EPS Basic (XOPTEPS): double"""
        return self.data["XOPTEPS"]

    @property
    def XORE(self) -> pd.Series:
        """XORE -- Expense - Other (Real Estate) (XORE): double"""
        return self.data["XORE"]

    @property
    def XPR(self) -> pd.Series:
        """XPR -- Pension and Retirement Expense (XPR): double"""
        return self.data["XPR"]

    @property
    def XRD(self) -> pd.Series:
        """XRD -- Research and Development Expense (XRD): double"""
        return self.data["XRD"]

    @property
    def XRENT(self) -> pd.Series:
        """XRENT -- Rental Expense (XRENT): double"""
        return self.data["XRENT"]

    @property
    def XS(self) -> pd.Series:
        """XS -- Expense - Sundry (XS): double"""
        return self.data["XS"]

    @property
    def XSGA(self) -> pd.Series:
        """XSGA -- Selling, General and Administrative Expense (XSGA): double"""
        return self.data["XSGA"]

    @property
    def XSTF(self) -> pd.Series:
        """XSTF -- Staff Expense (Income Account) (XSTF): double"""
        return self.data["XSTF"]

    @property
    def XSTFO(self) -> pd.Series:
        """XSTFO -- Staff Expense - Other (XSTFO): double"""
        return self.data["XSTFO"]

    @property
    def XSTFWS(self) -> pd.Series:
        """XSTFWS -- Staff Expense - Wages and Salaries (XSTFWS): double"""
        return self.data["XSTFWS"]

    @property
    def XT(self) -> pd.Series:
        """XT -- Expense - Total (XT): double"""
        return self.data["XT"]

    @property
    def XUW(self) -> pd.Series:
        """XUW -- Other Underwriting Expenses - Insurance (XUW): double"""
        return self.data["XUW"]

    @property
    def XUWLI(self) -> pd.Series:
        """XUWLI -- Underwriting Expense - Life (XUWLI): double"""
        return self.data["XUWLI"]

    @property
    def XUWNLI(self) -> pd.Series:
        """XUWNLI -- Underwriting Expense - Non-Life (XUWNLI): double"""
        return self.data["XUWNLI"]

    @property
    def XUWOI(self) -> pd.Series:
        """XUWOI -- Underwriting Expense - Other (XUWOI): double"""
        return self.data["XUWOI"]

    @property
    def XUWREI(self) -> pd.Series:
        """XUWREI -- Underwriting Expense - Reinsurance (XUWREI): double"""
        return self.data["XUWREI"]

    @property
    def XUWTI(self) -> pd.Series:
        """XUWTI -- Underwriting Expense - Total (XUWTI): double"""
        return self.data["XUWTI"]

    @property
    def AFUDCC(self) -> pd.Series:
        """AFUDCC -- Allowance for Funds Used During Construction (Cash Flow) (AFUDCC): double"""
        return self.data["AFUDCC"]

    @property
    def AFUDCI(self) -> pd.Series:
        """AFUDCI -- Allowance for Funds Used During Construction (Investing) (Cash Flow) (AFUDCI): double"""
        return self.data["AFUDCI"]

    @property
    def AMC(self) -> pd.Series:
        """AMC -- Amortization (Cash Flow) - Utility (AMC): double"""
        return self.data["AMC"]

    @property
    def AOLOCH(self) -> pd.Series:
        """AOLOCH -- Assets and Liabilities Other Net Change (AOLOCH): double"""
        return self.data["AOLOCH"]

    @property
    def APALCH(self) -> pd.Series:
        """APALCH -- Accounts Payable and Accrued Liabilities Increase/(Decrease) (APALCH): double"""
        return self.data["APALCH"]

    @property
    def AQC(self) -> pd.Series:
        """AQC -- Acquisitions (AQC): double"""
        return self.data["AQC"]

    @property
    def CAPX(self) -> pd.Series:
        """CAPX -- Capital Expenditures (CAPX): double"""
        return self.data["CAPX"]

    @property
    def CAPXV(self) -> pd.Series:
        """CAPXV -- Capital Expend Property, Plant and Equipment Schd V (CAPXV): double"""
        return self.data["CAPXV"]

    @property
    def CDVC(self) -> pd.Series:
        """CDVC -- Cash Dividends on Common Stock (Cash Flow) (CDVC): double"""
        return self.data["CDVC"]

    @property
    def CHECH(self) -> pd.Series:
        """CHECH -- Cash and Cash Equivalents Increase/(Decrease) (CHECH): double"""
        return self.data["CHECH"]

    @property
    def DEPC(self) -> pd.Series:
        """DEPC -- Depreciation and Depletion (Cash Flow) (DEPC): double"""
        return self.data["DEPC"]

    @property
    def DLCCH(self) -> pd.Series:
        """DLCCH -- Current Debt Changes (DLCCH): double"""
        return self.data["DLCCH"]

    @property
    def DLTIS(self) -> pd.Series:
        """DLTIS -- Long-Term Debt Issuance (DLTIS): double"""
        return self.data["DLTIS"]

    @property
    def DLTR(self) -> pd.Series:
        """DLTR -- Long-Term Debt Reduction (DLTR): double"""
        return self.data["DLTR"]

    @property
    def DPC(self) -> pd.Series:
        """DPC -- Depreciation and Amortization (Cash Flow) (DPC): double"""
        return self.data["DPC"]

    @property
    def DV(self) -> pd.Series:
        """DV -- Cash Dividends (Cash Flow) (DV): double"""
        return self.data["DV"]

    @property
    def ESUBC(self) -> pd.Series:
        """ESUBC -- Equity in Net Loss Earnings (ESUBC): double"""
        return self.data["ESUBC"]

    @property
    def EXRE(self) -> pd.Series:
        """EXRE -- Exchange Rate Effect (EXRE): double"""
        return self.data["EXRE"]

    @property
    def FIAO(self) -> pd.Series:
        """FIAO -- Financing Activities Other (FIAO): double"""
        return self.data["FIAO"]

    @property
    def FINCF(self) -> pd.Series:
        """FINCF -- Financing Activities Net Cash Flow (FINCF): double"""
        return self.data["FINCF"]

    @property
    def FOPO(self) -> pd.Series:
        """FOPO -- Funds from Operations Other (FOPO): double"""
        return self.data["FOPO"]

    @property
    def FOPOX(self) -> pd.Series:
        """FOPOX -- Funds from Operations - Other excluding Option Tax Benefit (FOPOX): double"""
        return self.data["FOPOX"]

    @property
    def FOPT(self) -> pd.Series:
        """FOPT -- Funds From Operations Total (FOPT): double"""
        return self.data["FOPT"]

    @property
    def FSRCO(self) -> pd.Series:
        """FSRCO -- Sources of Funds Other (FSRCO): double"""
        return self.data["FSRCO"]

    @property
    def FSRCT(self) -> pd.Series:
        """FSRCT -- Sources of Funds Total (FSRCT): double"""
        return self.data["FSRCT"]

    @property
    def FUSEO(self) -> pd.Series:
        """FUSEO -- Uses of Funds Other (FUSEO): double"""
        return self.data["FUSEO"]

    @property
    def FUSET(self) -> pd.Series:
        """FUSET -- Uses of Funds Total (FUSET): double"""
        return self.data["FUSET"]

    @property
    def IBC(self) -> pd.Series:
        """IBC -- Income Before Extraordinary Items (Cash Flow) (IBC): double"""
        return self.data["IBC"]

    @property
    def INTPN(self) -> pd.Series:
        """INTPN -- Interest Paid Net (INTPN): double"""
        return self.data["INTPN"]

    @property
    def INVCH(self) -> pd.Series:
        """INVCH -- Inventory Decrease (Increase) (INVCH): double"""
        return self.data["INVCH"]

    @property
    def ITCC(self) -> pd.Series:
        """ITCC -- Investment Tax Credit - Net (Cash Flow) - Utility (ITCC): double"""
        return self.data["ITCC"]

    @property
    def IVACO(self) -> pd.Series:
        """IVACO -- Investing Activities Other (IVACO): double"""
        return self.data["IVACO"]

    @property
    def IVCH(self) -> pd.Series:
        """IVCH -- Increase in Investments (IVCH): double"""
        return self.data["IVCH"]

    @property
    def IVNCF(self) -> pd.Series:
        """IVNCF -- Investing Activities Net Cash Flow (IVNCF): double"""
        return self.data["IVNCF"]

    @property
    def IVSTCH(self) -> pd.Series:
        """IVSTCH -- Short-Term Investments Change (IVSTCH): double"""
        return self.data["IVSTCH"]

    @property
    def OANCF(self) -> pd.Series:
        """OANCF -- Operating Activities Net Cash Flow (OANCF): double"""
        return self.data["OANCF"]

    @property
    def PDVC(self) -> pd.Series:
        """PDVC -- Cash Dividends on Preferred/Preference Stock (Cash Flow) (PDVC): double"""
        return self.data["PDVC"]

    @property
    def PRSTKC(self) -> pd.Series:
        """PRSTKC -- Purchase of Common and Preferred Stock (PRSTKC): double"""
        return self.data["PRSTKC"]

    @property
    def PRSTKCC(self) -> pd.Series:
        """PRSTKCC -- Purchase of Common Stock (Cash Flow) (PRSTKCC): double"""
        return self.data["PRSTKCC"]

    @property
    def PRSTKPC(self) -> pd.Series:
        """PRSTKPC -- Purchase of Preferred/Preference Stock (Cash Flow) (PRSTKPC): double"""
        return self.data["PRSTKPC"]

    @property
    def RECCH(self) -> pd.Series:
        """RECCH -- Accounts Receivable Decrease (Increase) (RECCH): double"""
        return self.data["RECCH"]

    @property
    def SCSTKC(self) -> pd.Series:
        """SCSTKC -- Sale of Common Stock (Cash Flow) (SCSTKC): double"""
        return self.data["SCSTKC"]

    @property
    def SIV(self) -> pd.Series:
        """SIV -- Sale of Investments (SIV): double"""
        return self.data["SIV"]

    @property
    def SPPE(self) -> pd.Series:
        """SPPE -- Sale of Property (SPPE): double"""
        return self.data["SPPE"]

    @property
    def SPPIV(self) -> pd.Series:
        """SPPIV -- Sale of Property, Plant and Equipment and Investments Gain (Loss) (SPPIV): double"""
        return self.data["SPPIV"]

    @property
    def SPSTKC(self) -> pd.Series:
        """SPSTKC -- Sale of Preferred/Preference Stock (Cash Flow) (SPSTKC): double"""
        return self.data["SPSTKC"]

    @property
    def SSTK(self) -> pd.Series:
        """SSTK -- Sale of Common and Preferred Stock (SSTK): double"""
        return self.data["SSTK"]

    @property
    def TDC(self) -> pd.Series:
        """TDC -- Deferred Income Taxes - Net (Cash Flow) (TDC): double"""
        return self.data["TDC"]

    @property
    def TSAFC(self) -> pd.Series:
        """TSAFC -- Total Sources/Applications of Funds (Cash Flow) (TSAFC): double"""
        return self.data["TSAFC"]

    @property
    def TXACH(self) -> pd.Series:
        """TXACH -- Income Taxes Accrued Increase/(Decrease) (TXACH): double"""
        return self.data["TXACH"]

    @property
    def TXBCO(self) -> pd.Series:
        """TXBCO -- Excess Tax Benefit Stock Options - Cash Flow Operating (TXBCO): double"""
        return self.data["TXBCO"]

    @property
    def TXBCOF(self) -> pd.Series:
        """TXBCOF -- Excess Tax Benefit of Stock Options - Cash Flow Financing (TXBCOF): double"""
        return self.data["TXBCOF"]

    @property
    def TXDC(self) -> pd.Series:
        """TXDC -- Deferred Taxes (Cash Flow) (TXDC): double"""
        return self.data["TXDC"]

    @property
    def TXPD(self) -> pd.Series:
        """TXPD -- Income Taxes Paid (TXPD): double"""
        return self.data["TXPD"]

    @property
    def UAOLOCH(self) -> pd.Series:
        """UAOLOCH -- Other Assets and Liabilities - Net Change (Statement of Cash Flows) (UAOLOCH): double"""
        return self.data["UAOLOCH"]

    @property
    def UDFCC(self) -> pd.Series:
        """UDFCC -- Deferred Fuel - Increase (Decrease) (Statement of Cash Flows) (UDFCC): double"""
        return self.data["UDFCC"]

    @property
    def UFRETSD(self) -> pd.Series:
        """UFRETSD -- Funds for Retirement of Securities and Short-Term Debt (Cash Flow) (UFRETSD): double"""
        return self.data["UFRETSD"]

    @property
    def UNWCC(self) -> pd.Series:
        """UNWCC -- Working Capital (Use) - Increase (Decrease) (Cash Flow) (UNWCC): double"""
        return self.data["UNWCC"]

    @property
    def UOIS(self) -> pd.Series:
        """UOIS -- Other Internal Sources - Net (Cash Flow) (UOIS): double"""
        return self.data["UOIS"]

    @property
    def USTDNC(self) -> pd.Series:
        """USTDNC -- Short-Term Debt - Decrease (Increase) (Cash Flow) (USTDNC): double"""
        return self.data["USTDNC"]

    @property
    def UTFDOC(self) -> pd.Series:
        """UTFDOC -- Total Funds From Operations (Cash Flow) (UTFDOC): double"""
        return self.data["UTFDOC"]

    @property
    def UTFOSC(self) -> pd.Series:
        """UTFOSC -- Total Funds from Outside Sources (Cash Flow) (UTFOSC): double"""
        return self.data["UTFOSC"]

    @property
    def UWKCAPC(self) -> pd.Series:
        """UWKCAPC -- Working Capital (Source) - Decrease (Increase) (Cash Flow) (UWKCAPC): double"""
        return self.data["UWKCAPC"]

    @property
    def WCAPC(self) -> pd.Series:
        """WCAPC -- Working Capital Change Other Increase/(Decrease) (WCAPC): double"""
        return self.data["WCAPC"]

    @property
    def WCAPCH(self) -> pd.Series:
        """WCAPCH -- Working Capital Change Total (WCAPCH): double"""
        return self.data["WCAPCH"]

    @property
    def XIDOC(self) -> pd.Series:
        """XIDOC -- Extraordinary Items and Discontinued Operations (Cash Flow) (XIDOC): double"""
        return self.data["XIDOC"]

    @property
    def ACCRT(self) -> pd.Series:
        """ACCRT -- ARO Accretion Expense (ACCRT): double"""
        return self.data["ACCRT"]

    @property
    def ACQAO(self) -> pd.Series:
        """ACQAO -- Acquired Assets > Other Long-Term Assets (ACQAO): double"""
        return self.data["ACQAO"]

    @property
    def ACQCSHI(self) -> pd.Series:
        """ACQCSHI -- Shares Issued for Acquisition (ACQCSHI): double"""
        return self.data["ACQCSHI"]

    @property
    def ACQGDWL(self) -> pd.Series:
        """ACQGDWL -- Acquired Assets - Goodwill (ACQGDWL): double"""
        return self.data["ACQGDWL"]

    @property
    def ACQIC(self) -> pd.Series:
        """ACQIC -- Acquisitions - Current Income Contribution (ACQIC): double"""
        return self.data["ACQIC"]

    @property
    def ACQINTAN(self) -> pd.Series:
        """ACQINTAN -- Acquired Assets - Intangibles (ACQINTAN): double"""
        return self.data["ACQINTAN"]

    @property
    def ACQINVT(self) -> pd.Series:
        """ACQINVT -- Acquired Assets - Inventory (ACQINVT): double"""
        return self.data["ACQINVT"]

    @property
    def ACQLNTAL(self) -> pd.Series:
        """ACQLNTAL -- Acquired Loans (ACQLNTAL): double"""
        return self.data["ACQLNTAL"]

    @property
    def ACQNIINTC(self) -> pd.Series:
        """ACQNIINTC -- Net Interest Income Contribution (ACQNIINTC): double"""
        return self.data["ACQNIINTC"]

    @property
    def ACQPPE(self) -> pd.Series:
        """ACQPPE -- Acquired Assets > Property, Plant & Equipment (ACQPPE): double"""
        return self.data["ACQPPE"]

    @property
    def ACQSC(self) -> pd.Series:
        """ACQSC -- Acquisitions - Current Sales Contribution (ACQSC): double"""
        return self.data["ACQSC"]

    @property
    def ANO(self) -> pd.Series:
        """ANO -- Assets Netting & Other Adjustments (ANO): double"""
        return self.data["ANO"]

    @property
    def AOL2(self) -> pd.Series:
        """AOL2 -- Assets Level2 (Observable) (AOL2): double"""
        return self.data["AOL2"]

    @property
    def AQPL1(self) -> pd.Series:
        """AQPL1 -- Assets Level1 (Quoted Prices) (AQPL1): double"""
        return self.data["AQPL1"]

    @property
    def AU(self) -> pd.Series:
        """AU -- Auditor (AU): string"""
        return self.data["AU"]

    @property
    def AUL3(self) -> pd.Series:
        """AUL3 -- Assets Level3 (Unobservable) (AUL3): double"""
        return self.data["AUL3"]

    @property
    def AUOP(self) -> pd.Series:
        """AUOP -- Auditor Opinion (AUOP): string"""
        return self.data["AUOP"]

    @property
    def AUOPIC(self) -> pd.Series:
        """AUOPIC -- Auditor Opinion - Internal Control (AUOPIC): string"""
        return self.data["AUOPIC"]

    @property
    def BASTR(self) -> pd.Series:
        """BASTR -- Average Short-Term Borrowings Rate (BASTR): double"""
        return self.data["BASTR"]

    @property
    def BILLEXCE(self) -> pd.Series:
        """BILLEXCE -- Billings in Excess of Cost & Earnings (BILLEXCE): double"""
        return self.data["BILLEXCE"]

    @property
    def CAPR1(self) -> pd.Series:
        """CAPR1 -- Risk-Adjusted Capital Ratio - Tier 1 (CAPR1): double"""
        return self.data["CAPR1"]

    @property
    def CAPR2(self) -> pd.Series:
        """CAPR2 -- Risk-Adjusted Capital Ratio - Tier 2 (CAPR2): double"""
        return self.data["CAPR2"]

    @property
    def CAPR3(self) -> pd.Series:
        """CAPR3 -- Risk-Adjusted Capital Ratio - Combined (CAPR3): double"""
        return self.data["CAPR3"]

    @property
    def CAPSFT(self) -> pd.Series:
        """CAPSFT -- Capitalized Software (CAPSFT): double"""
        return self.data["CAPSFT"]

    @property
    def CEIEXBILL(self) -> pd.Series:
        """CEIEXBILL -- Cost & Earnings in Excess of Billings (CEIEXBILL): double"""
        return self.data["CEIEXBILL"]

    @property
    def CEOSO(self) -> pd.Series:
        """CEOSO -- Chief Executive Officer SOX Certification (CEOSO): string"""
        return self.data["CEOSO"]

    @property
    def CFOSO(self) -> pd.Series:
        """CFOSO -- Chief Financial Officer SOX Certification (CFOSO): string"""
        return self.data["CFOSO"]

    @property
    def CI(self) -> pd.Series:
        """CI -- Comprehensive Income - Total (CI): double"""
        return self.data["CI"]

    @property
    def CIMII(self) -> pd.Series:
        """CIMII -- Comprehensive Income - Noncontrolling Interest (CIMII): double"""
        return self.data["CIMII"]

    @property
    def CSHFD(self) -> pd.Series:
        """CSHFD -- Common Shares Used to Calc Earnings Per Share Fully Diluted (CSHFD): double"""
        return self.data["CSHFD"]

    @property
    def CSHI(self) -> pd.Series:
        """CSHI -- Common Shares Issued (CSHI): double"""
        return self.data["CSHI"]

    @property
    def CSHO(self) -> pd.Series:
        """CSHO -- Common Shares Outstanding (CSHO): double"""
        return self.data["CSHO"]

    @property
    def CSHPRI(self) -> pd.Series:
        """CSHPRI -- Common Shares Used to Calculate Earnings Per Share Basic (CSHPRI): double"""
        return self.data["CSHPRI"]

    @property
    def CSHR(self) -> pd.Series:
        """CSHR -- Common/Ordinary Shareholders (CSHR): double"""
        return self.data["CSHR"]

    @property
    def CSHRC(self) -> pd.Series:
        """CSHRC -- Common Shares Reserved for Conversion Convertible Debt (CSHRC): double"""
        return self.data["CSHRC"]

    @property
    def CSHRP(self) -> pd.Series:
        """CSHRP -- Common Shares Reserved for Conversion Preferred Stock (CSHRP): double"""
        return self.data["CSHRP"]

    @property
    def CSHRSO(self) -> pd.Series:
        """CSHRSO -- Common Shares Reserved for Conversion Stock Options (CSHRSO): double"""
        return self.data["CSHRSO"]

    @property
    def CSHRT(self) -> pd.Series:
        """CSHRT -- Common Shares Reserved for Conversion Total (CSHRT): double"""
        return self.data["CSHRT"]

    @property
    def CSHRW(self) -> pd.Series:
        """CSHRW -- Common Shares Reserved for Conversion Warrants and Other (CSHRW): double"""
        return self.data["CSHRW"]

    @property
    def DERAC(self) -> pd.Series:
        """DERAC -- Derivative Assets - Current (DERAC): double"""
        return self.data["DERAC"]

    @property
    def DERALT(self) -> pd.Series:
        """DERALT -- Derivative Assets Long-Term (DERALT): double"""
        return self.data["DERALT"]

    @property
    def DERHEDGL(self) -> pd.Series:
        """DERHEDGL -- Gains/Losses on Derivatives and Hedging (DERHEDGL): double"""
        return self.data["DERHEDGL"]

    @property
    def DERLC(self) -> pd.Series:
        """DERLC -- Derivative Liabilities- Current (DERLC): double"""
        return self.data["DERLC"]

    @property
    def DERLLT(self) -> pd.Series:
        """DERLLT -- Derivative Liabilities Long-Term (DERLLT): double"""
        return self.data["DERLLT"]

    @property
    def DT(self) -> pd.Series:
        """DT -- Total Debt Including Current (DT): double"""
        return self.data["DT"]

    @property
    def DVINTF(self) -> pd.Series:
        """DVINTF -- Dividends & Interest Receivable (Cash Flow) (DVINTF): double"""
        return self.data["DVINTF"]

    @property
    def EMP(self) -> pd.Series:
        """EMP -- Employees (EMP): double"""
        return self.data["EMP"]

    @property
    def FINACO(self) -> pd.Series:
        """FINACO -- Finance Division Other Current Assets, Total (FINACO): double"""
        return self.data["FINACO"]

    @property
    def FINAO(self) -> pd.Series:
        """FINAO -- Finance Division Other Long-Term Assets, Total (FINAO): double"""
        return self.data["FINAO"]

    @property
    def FINCH(self) -> pd.Series:
        """FINCH -- Finance Division - Cash (FINCH): double"""
        return self.data["FINCH"]

    @property
    def FINDLC(self) -> pd.Series:
        """FINDLC -- Finance Division Long-Term Debt - Current (FINDLC): double"""
        return self.data["FINDLC"]

    @property
    def FINDLT(self) -> pd.Series:
        """FINDLT -- Finance Division Debt - Long-Term (FINDLT): double"""
        return self.data["FINDLT"]

    @property
    def FINIVST(self) -> pd.Series:
        """FINIVST -- Finance Division - Short-Term Investments (FINIVST): double"""
        return self.data["FINIVST"]

    @property
    def FINLCO(self) -> pd.Series:
        """FINLCO -- Finance Division Other Current Liabilities, Total (FINLCO): double"""
        return self.data["FINLCO"]

    @property
    def FINLTO(self) -> pd.Series:
        """FINLTO -- Finance Division Other Long Term Liabilities, Total (FINLTO): double"""
        return self.data["FINLTO"]

    @property
    def FINNP(self) -> pd.Series:
        """FINNP -- Finance Division Notes Payable (FINNP): double"""
        return self.data["FINNP"]

    @property
    def FINRECC(self) -> pd.Series:
        """FINRECC -- Finance Division - Current Receivables (FINRECC): double"""
        return self.data["FINRECC"]

    @property
    def FINRECLT(self) -> pd.Series:
        """FINRECLT -- Finance Division - Long-Term Receivables (FINRECLT): double"""
        return self.data["FINRECLT"]

    @property
    def FINREV(self) -> pd.Series:
        """FINREV -- Finance Division Revenue (FINREV): double"""
        return self.data["FINREV"]

    @property
    def FINXINT(self) -> pd.Series:
        """FINXINT -- Finance Division Interest Expense (FINXINT): double"""
        return self.data["FINXINT"]

    @property
    def FINXOPR(self) -> pd.Series:
        """FINXOPR -- Finance Division Operating Expense (FINXOPR): double"""
        return self.data["FINXOPR"]

    @property
    def GLIV(self) -> pd.Series:
        """GLIV -- Gains/Losses on investments (GLIV): double"""
        return self.data["GLIV"]

    @property
    def GOVTOWN(self) -> pd.Series:
        """GOVTOWN -- Percent of Gov't Owned (GOVTOWN): double"""
        return self.data["GOVTOWN"]

    @property
    def IBMII(self) -> pd.Series:
        """IBMII -- Income before Extraordinary Items and Noncontrolling Interests (IBMII): double"""
        return self.data["IBMII"]

    @property
    def LIFRP(self) -> pd.Series:
        """LIFRP -- LIFO Reserve - Prior (LIFRP): double"""
        return self.data["LIFRP"]

    @property
    def LNO(self) -> pd.Series:
        """LNO -- Liabilities Netting & Other Adjustments (LNO): double"""
        return self.data["LNO"]

    @property
    def LOL2(self) -> pd.Series:
        """LOL2 -- Liabilities Level2 (Observable) (LOL2): double"""
        return self.data["LOL2"]

    @property
    def LQPL1(self) -> pd.Series:
        """LQPL1 -- Liabilities Level1 (Quoted Prices) (LQPL1): double"""
        return self.data["LQPL1"]

    @property
    def LUL3(self) -> pd.Series:
        """LUL3 -- Liabilities Level3 (Unobservable) (LUL3): double"""
        return self.data["LUL3"]

    @property
    def MIBN(self) -> pd.Series:
        """MIBN -- Noncontrolling Interests - Nonredeemable - Balance Sheet (MIBN): double"""
        return self.data["MIBN"]

    @property
    def MIBT(self) -> pd.Series:
        """MIBT -- Noncontrolling Interests - Total - Balance Sheet (MIBT): double"""
        return self.data["MIBT"]

    @property
    def NIINTPFC(self) -> pd.Series:
        """NIINTPFC -- Pro Forma Net Interest Income - Current (NIINTPFC): double"""
        return self.data["NIINTPFC"]

    @property
    def NIINTPFP(self) -> pd.Series:
        """NIINTPFP -- Pro Forma Net Interest Income - Prior (NIINTPFP): double"""
        return self.data["NIINTPFP"]

    @property
    def NIPFC(self) -> pd.Series:
        """NIPFC -- Pro Forma Net Income - Current (NIPFC): double"""
        return self.data["NIPFC"]

    @property
    def NIPFP(self) -> pd.Series:
        """NIPFP -- Pro Forma Net Income - Prior (NIPFP): double"""
        return self.data["NIPFP"]

    @property
    def OPTCA(self) -> pd.Series:
        """OPTCA -- Options - Cancelled (-) (OPTCA): double"""
        return self.data["OPTCA"]

    @property
    def OPTDR(self) -> pd.Series:
        """OPTDR -- Dividend Rate - Assumption (%) (OPTDR): double"""
        return self.data["OPTDR"]

    @property
    def OPTEX(self) -> pd.Series:
        """OPTEX -- Options Exercisable (000) (OPTEX): double"""
        return self.data["OPTEX"]

    @property
    def OPTEXD(self) -> pd.Series:
        """OPTEXD -- Options - Exercised (-) (OPTEXD): double"""
        return self.data["OPTEXD"]

    @property
    def OPTFVGR(self) -> pd.Series:
        """OPTFVGR -- Options - Fair Value of Options Granted (OPTFVGR): double"""
        return self.data["OPTFVGR"]

    @property
    def OPTGR(self) -> pd.Series:
        """OPTGR -- Options - Granted (OPTGR): double"""
        return self.data["OPTGR"]

    @property
    def OPTLIFE(self) -> pd.Series:
        """OPTLIFE -- Life of Options - Assumption (# yrs) (OPTLIFE): double"""
        return self.data["OPTLIFE"]

    @property
    def OPTOSBY(self) -> pd.Series:
        """OPTOSBY -- Options Outstanding - Beg of Year (OPTOSBY): double"""
        return self.data["OPTOSBY"]

    @property
    def OPTOSEY(self) -> pd.Series:
        """OPTOSEY -- Options Outstanding - End of Year (OPTOSEY): double"""
        return self.data["OPTOSEY"]

    @property
    def OPTPRCBY(self) -> pd.Series:
        """OPTPRCBY -- Options Outstanding Beg of Year - Price (OPTPRCBY): double"""
        return self.data["OPTPRCBY"]

    @property
    def OPTRFR(self) -> pd.Series:
        """OPTRFR -- Risk Free Rate - Assumption (%) (OPTRFR): double"""
        return self.data["OPTRFR"]

    @property
    def OPTVOL(self) -> pd.Series:
        """OPTVOL -- Volatility - Assumption (%) (OPTVOL): double"""
        return self.data["OPTVOL"]

    @property
    def PNRSHO(self) -> pd.Series:
        """PNRSHO -- Nonred Pfd Shares Outs (000) (PNRSHO): double"""
        return self.data["PNRSHO"]

    @property
    def PRSHO(self) -> pd.Series:
        """PRSHO -- Redeem Pfd Shares Outs (000) (PRSHO): double"""
        return self.data["PRSHO"]

    @property
    def RANK(self) -> pd.Series:
        """RANK -- Rank - Auditor (RANK): double"""
        return self.data["RANK"]

    @property
    def RSTCHE(self) -> pd.Series:
        """RSTCHE -- Restricted Cash & Investments - Current (RSTCHE): double"""
        return self.data["RSTCHE"]

    @property
    def RSTCHELT(self) -> pd.Series:
        """RSTCHELT -- Long-Term Restricted Cash & Investments (RSTCHELT): double"""
        return self.data["RSTCHELT"]

    @property
    def SALEPFC(self) -> pd.Series:
        """SALEPFC -- Pro Forma Net Sales - Current Year (SALEPFC): double"""
        return self.data["SALEPFC"]

    @property
    def SALEPFP(self) -> pd.Series:
        """SALEPFP -- Pro Forma Net Sales - Prior Year (SALEPFP): double"""
        return self.data["SALEPFP"]

    @property
    def TEQ(self) -> pd.Series:
        """TEQ -- Stockholders Equity - Total (TEQ): double"""
        return self.data["TEQ"]

    @property
    def TFVA(self) -> pd.Series:
        """TFVA -- Total Fair Value Assets (TFVA): double"""
        return self.data["TFVA"]

    @property
    def TFVCE(self) -> pd.Series:
        """TFVCE -- Total Fair Value Changes including Earnings (TFVCE): double"""
        return self.data["TFVCE"]

    @property
    def TFVL(self) -> pd.Series:
        """TFVL -- Total Fair Value Liabilities (TFVL): double"""
        return self.data["TFVL"]

    @property
    def TSTKN(self) -> pd.Series:
        """TSTKN -- Treasury Stock Number of Common Shares (TSTKN): double"""
        return self.data["TSTKN"]

    @property
    def TXTUBADJUST(self) -> pd.Series:
        """TXTUBADJUST -- Other Unrecog Tax Benefit Adj. (TXTUBADJUST): double"""
        return self.data["TXTUBADJUST"]

    @property
    def TXTUBBEGIN(self) -> pd.Series:
        """TXTUBBEGIN -- Unrecog. Tax Benefits - Beg of Year (TXTUBBEGIN): double"""
        return self.data["TXTUBBEGIN"]

    @property
    def TXTUBEND(self) -> pd.Series:
        """TXTUBEND -- Unrecog. Tax Benefits - End of Year (TXTUBEND): double"""
        return self.data["TXTUBEND"]

    @property
    def TXTUBMAX(self) -> pd.Series:
        """TXTUBMAX -- Chg. In Unrecog. Tax Benefits - Max (TXTUBMAX): double"""
        return self.data["TXTUBMAX"]

    @property
    def TXTUBMIN(self) -> pd.Series:
        """TXTUBMIN -- Chg. In Unrecog. Tax Benefits - Min (TXTUBMIN): double"""
        return self.data["TXTUBMIN"]

    @property
    def TXTUBPOSDEC(self) -> pd.Series:
        """TXTUBPOSDEC -- Decrease- Current Tax Positions (TXTUBPOSDEC): double"""
        return self.data["TXTUBPOSDEC"]

    @property
    def TXTUBPOSINC(self) -> pd.Series:
        """TXTUBPOSINC -- Increase- Current Tax Positions (TXTUBPOSINC): double"""
        return self.data["TXTUBPOSINC"]

    @property
    def TXTUBPOSPDEC(self) -> pd.Series:
        """TXTUBPOSPDEC -- Decrease- Prior Tax Positions (TXTUBPOSPDEC): double"""
        return self.data["TXTUBPOSPDEC"]

    @property
    def TXTUBPOSPINC(self) -> pd.Series:
        """TXTUBPOSPINC -- Increase- Prior Tax Positions (TXTUBPOSPINC): double"""
        return self.data["TXTUBPOSPINC"]

    @property
    def TXTUBSETTLE(self) -> pd.Series:
        """TXTUBSETTLE -- Settlements with Tax Authorities (TXTUBSETTLE): double"""
        return self.data["TXTUBSETTLE"]

    @property
    def TXTUBSOFLIMIT(self) -> pd.Series:
        """TXTUBSOFLIMIT -- Lapse of Statute of Limitations (TXTUBSOFLIMIT): double"""
        return self.data["TXTUBSOFLIMIT"]

    @property
    def TXTUBTXTR(self) -> pd.Series:
        """TXTUBTXTR -- Impact on Effective Tax Rate (TXTUBTXTR): double"""
        return self.data["TXTUBTXTR"]

    @property
    def TXTUBXINTBS(self) -> pd.Series:
        """TXTUBXINTBS -- Interest & Penalties Accrued - B/S (TXTUBXINTBS): double"""
        return self.data["TXTUBXINTBS"]

    @property
    def TXTUBXINTIS(self) -> pd.Series:
        """TXTUBXINTIS -- Interest & Penalties Reconized - I/S (TXTUBXINTIS): double"""
        return self.data["TXTUBXINTIS"]

    @property
    def XRDP(self) -> pd.Series:
        """XRDP -- Research & Development - Prior (XRDP): double"""
        return self.data["XRDP"]

    @property
    def ADJEX_C(self) -> pd.Series:
        """ADJEX_C -- Cumulative Adjustment Factor by Ex-Date - Calendar (ADJEX_C): double"""
        return self.data["ADJEX_C"]

    @property
    def ADJEX_F(self) -> pd.Series:
        """ADJEX_F -- Cumulative Adjustment Factor by Ex-Date - Fiscal (ADJEX_F): double"""
        return self.data["ADJEX_F"]

    @property
    def CSHTR_C(self) -> pd.Series:
        """CSHTR_C -- Common Shares Traded - Annual - Calendar (CSHTR_C): double"""
        return self.data["CSHTR_C"]

    @property
    def CSHTR_F(self) -> pd.Series:
        """CSHTR_F -- Common Shares Traded - Annual - Fiscal (CSHTR_F): double"""
        return self.data["CSHTR_F"]

    @property
    def DVPSP_C(self) -> pd.Series:
        """DVPSP_C -- Dividends per Share - Pay Date - Calendar (DVPSP_C): double"""
        return self.data["DVPSP_C"]

    @property
    def DVPSP_F(self) -> pd.Series:
        """DVPSP_F -- Dividends per Share - Pay Date - Fiscal (DVPSP_F): double"""
        return self.data["DVPSP_F"]

    @property
    def DVPSX_C(self) -> pd.Series:
        """DVPSX_C -- Dividends per Share - Ex-Date - Calendar (DVPSX_C): double"""
        return self.data["DVPSX_C"]

    @property
    def DVPSX_F(self) -> pd.Series:
        """DVPSX_F -- Dividends per Share - Ex-Date - Fiscal (DVPSX_F): double"""
        return self.data["DVPSX_F"]

    @property
    def MKVALT(self) -> pd.Series:
        """MKVALT -- Market Value - Total - Fiscal (MKVALT): double"""
        return self.data["MKVALT"]

    @property
    def NAICSH(self) -> pd.Series:
        """NAICSH -- North America Industrial Classification System - Historical (NAICSH): string"""
        return self.data["NAICSH"]

    @property
    def PRCC_C(self) -> pd.Series:
        """PRCC_C -- Price Close - Annual - Calendar (PRCC_C): double"""
        return self.data["PRCC_C"]

    @property
    def PRCC_F(self) -> pd.Series:
        """PRCC_F -- Price Close - Annual - Fiscal (PRCC_F): double"""
        return self.data["PRCC_F"]

    @property
    def PRCH_C(self) -> pd.Series:
        """PRCH_C -- Price High - Annual - Calendar (PRCH_C): double"""
        return self.data["PRCH_C"]

    @property
    def PRCH_F(self) -> pd.Series:
        """PRCH_F -- Price High - Annual - Fiscal (PRCH_F): double"""
        return self.data["PRCH_F"]

    @property
    def PRCL_C(self) -> pd.Series:
        """PRCL_C -- Price Low - Annual - Calendar (PRCL_C): double"""
        return self.data["PRCL_C"]

    @property
    def PRCL_F(self) -> pd.Series:
        """PRCL_F -- Price Low - Annual - Fiscal (PRCL_F): double"""
        return self.data["PRCL_F"]

    @property
    def SICH(self) -> pd.Series:
        """SICH -- Standard Industrial Classification - Historical (SICH): double"""
        return self.data["SICH"]

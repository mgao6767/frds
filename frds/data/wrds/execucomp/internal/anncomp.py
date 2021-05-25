from dataclasses import dataclass
import pandas as pd
from frds.data.wrds import WRDSDataset


@dataclass
class Anncomp(WRDSDataset):
    """Annual Compensation"""

    data: pd.DataFrame
    library = "execcomp"
    table = "anncomp"
    index_col = ["co_per_rol", "year"]
    date_cols = ["becameceo", "joined_co", "leftco", "leftofc", "rejoin", "releft"]

    def __post_init__(self):
        idx = [c.upper() for c in self.index_col]
        if set(self.data.index.names) != set(idx):
            self.data.reset_index(inplace=True, drop=True)
        self.data.rename(columns=str.upper, inplace=True)
        self.data.set_index(idx, inplace=True)

        # Some variables are not available
        attrs = [
            varname
            for varname, prop in vars(Anncomp).items()
            if isinstance(prop, property) and varname.isupper()
        ]
        for attr in attrs:
            try:
                self.__getattribute__(attr)
            except KeyError:
                delattr(Anncomp, attr)

    @staticmethod
    def lag(series: pd.Series, lags: int = 1, *args, **kwargs):
        return series.shift(lags, *args, **kwargs)

    @staticmethod
    def lead(series: pd.Series, leads: int = 1, *args, **kwargs):
        return series.shift(-leads, *args, **kwargs)

    @property
    def ADDRESS(self) -> pd.Series:
        """Street Address (ADDRESS): string"""
        return self.data["ADDRESS"]

    @property
    def CITY(self) -> pd.Series:
        """CITY - City (CITY): string"""
        return self.data["CITY"]

    @property
    def CONAME(self) -> pd.Series:
        """Company Name (CONAME): string"""
        return self.data["CONAME"]

    @property
    def CUSIP(self) -> pd.Series:
        """CUSIP (CUSIP): string"""
        return self.data["CUSIP"]

    @property
    def EXCHANGE(self) -> pd.Series:
        """Exchange (EXCHANGE): string"""
        return self.data["EXCHANGE"]

    @property
    def INDDESC(self) -> pd.Series:
        """Index Description (INDDESC): string"""
        return self.data["INDDESC"]

    @property
    def NAICS(self) -> pd.Series:
        """North American Industry Class System (NAICS): string"""
        return self.data["NAICS"]

    @property
    def NAICSDESC(self) -> pd.Series:
        """Description of NAICS Code (NAICSDESC): string"""
        return self.data["NAICSDESC"]

    @property
    def SIC(self) -> pd.Series:
        """Standard Industrial Classification (SIC) (SIC): double"""
        return self.data["SIC"]

    @property
    def SICDESC(self) -> pd.Series:
        """SIC Code Description (SICDESC): string"""
        return self.data["SICDESC"]

    @property
    def SPCODE(self) -> pd.Series:
        """S&P Index (SPCODE): string"""
        return self.data["SPCODE"]

    @property
    def SPINDEX(self) -> pd.Series:
        """S&Pindex (SPINDEX): double"""
        return self.data["SPINDEX"]

    @property
    def STATE(self) -> pd.Series:
        """State (STATE): string"""
        return self.data["STATE"]

    @property
    def SUB_TELE(self) -> pd.Series:
        """Area Code (SUB_TELE): double"""
        return self.data["SUB_TELE"]

    @property
    def TELE(self) -> pd.Series:
        """Bought From Telemarketing Promotion (TELE): string"""
        return self.data["TELE"]

    @property
    def TICKER(self) -> pd.Series:
        """(Current) Ticker Symbol (TICKER): string"""
        return self.data["TICKER"]

    @property
    def ZIP(self) -> pd.Series:
        """Zip (ZIP): string"""
        return self.data["ZIP"]

    @property
    def EXECID(self) -> pd.Series:
        """Executive ID number (EXECID): string"""
        return self.data["EXECID"]

    @property
    def EXEC_FNAME(self) -> pd.Series:
        """First Name (EXEC_FNAME): string"""
        return self.data["EXEC_FNAME"]

    @property
    def EXEC_FULLNAME(self) -> pd.Series:
        """EXEC_FULLNAME (EXEC_FULLNAME): string"""
        return self.data["EXEC_FULLNAME"]

    @property
    def EXEC_LNAME(self) -> pd.Series:
        """Last Name (EXEC_LNAME): string"""
        return self.data["EXEC_LNAME"]

    @property
    def EXEC_MNAME(self) -> pd.Series:
        """Middle Name (EXEC_MNAME): string"""
        return self.data["EXEC_MNAME"]

    @property
    def GENDER(self) -> pd.Series:
        """Gender (GENDER): string"""
        return self.data["GENDER"]

    @property
    def NAMEPREFIX(self) -> pd.Series:
        """Name Prefix (NAMEPREFIX): string"""
        return self.data["NAMEPREFIX"]

    @property
    def PAGE(self) -> pd.Series:
        """IRRC Page Number (PAGE): double"""
        return self.data["PAGE"]

    @property
    def BECAMECEO(self) -> pd.Series:
        """Date Became CEO (BECAMECEO): double"""
        return self.data["BECAMECEO"]

    @property
    def CO_PER_ROL(self) -> pd.Series:
        """ID number for each executive/company combination (CO_PER_ROL): double"""
        return self.data["CO_PER_ROL"]

    @property
    def EXECRANK(self) -> pd.Series:
        """Current Rank by Salary + Bonus (EXECRANK): double"""
        return self.data["EXECRANK"]

    @property
    def GVKEY(self) -> pd.Series:
        """Compustat's Global Company Key (GVKEY): string"""
        return self.data["GVKEY"]

    @property
    def JOINED_CO(self) -> pd.Series:
        """Date Joined Company (JOINED_CO): double"""
        return self.data["JOINED_CO"]

    @property
    def LEFTCO(self) -> pd.Series:
        """Date Left Company (LEFTCO): double"""
        return self.data["LEFTCO"]

    @property
    def LEFTOFC(self) -> pd.Series:
        """Date Left as CEO (LEFTOFC): double"""
        return self.data["LEFTOFC"]

    @property
    def PCEO(self) -> pd.Series:
        """Current CEO (PCEO): string"""
        return self.data["PCEO"]

    @property
    def PCFO(self) -> pd.Series:
        """Current CFO (PCFO): string"""
        return self.data["PCFO"]

    @property
    def REASON(self) -> pd.Series:
        """Reason (REASON): string"""
        return self.data["REASON"]

    @property
    def REJOIN(self) -> pd.Series:
        """Date Rejoined Company (REJOIN): double"""
        return self.data["REJOIN"]

    @property
    def RELEFT(self) -> pd.Series:
        """Date Releft Company (RELEFT): double"""
        return self.data["RELEFT"]

    @property
    def TITLE(self) -> pd.Series:
        """Title (TITLE): string"""
        return self.data["TITLE"]

    @property
    def AGE(self) -> pd.Series:
        """Director - Age (AGE): double"""
        return self.data["AGE"]

    @property
    def ALLOTHPD(self) -> pd.Series:
        """ALLOTHPD -- All Other Paid (ALLOTHPD): double"""
        return self.data["ALLOTHPD"]

    @property
    def ALLOTHTOT(self) -> pd.Series:
        """ALLOTHTOT -- All Other Total (ALLOTHTOT): double"""
        return self.data["ALLOTHTOT"]

    @property
    def BONUS(self) -> pd.Series:
        """BONUS -- Bonus ($) (BONUS): double"""
        return self.data["BONUS"]

    @property
    def CEOANN(self) -> pd.Series:
        """CEOANN -- Annual CEO Flag (CEOANN): string"""
        return self.data["CEOANN"]

    @property
    def CFOANN(self) -> pd.Series:
        """CFOANN -- Annual CFO Flag (CFOANN): string"""
        return self.data["CFOANN"]

    @property
    def CHG_CTRL_PYMT(self) -> pd.Series:
        """CHG_CTRL_PYMT -- Estimated Payments in event of change in cont (CHG_CTRL_PYMT): double"""
        return self.data["CHG_CTRL_PYMT"]

    @property
    def COMMENT(self) -> pd.Series:
        """Comment (COMMENT): string"""
        return self.data["COMMENT"]

    @property
    def DEFER_BALANCE_TOT(self) -> pd.Series:
        """DEFER_BALANCE_TOT -- Total Aggregate Balance in Deferred Compensat (DEFER_BALANCE_TOT): double"""
        return self.data["DEFER_BALANCE_TOT"]

    @property
    def DEFER_CONTRIB_CO_TOT(self) -> pd.Series:
        """DEFER_CONTRIB_CO_TOT -- Total Registrant Contributions to Deferred Co (DEFER_CONTRIB_CO_TOT): double"""
        return self.data["DEFER_CONTRIB_CO_TOT"]

    @property
    def DEFER_CONTRIB_EXEC_TOT(self) -> pd.Series:
        """DEFER_CONTRIB_EXEC_TOT -- Total Executive Contributions to Deferred Com (DEFER_CONTRIB_EXEC_TOT): double"""
        return self.data["DEFER_CONTRIB_EXEC_TOT"]

    @property
    def DEFER_EARNINGS_TOT(self) -> pd.Series:
        """DEFER_EARNINGS_TOT -- Total Aggregate Earnings in Deferred Compensa (DEFER_EARNINGS_TOT): double"""
        return self.data["DEFER_EARNINGS_TOT"]

    @property
    def DEFER_RPT_AS_COMP_TOT(self) -> pd.Series:
        """DEFER_RPT_AS_COMP_TOT -- Total Portion of Deferred Earnings Reported A (DEFER_RPT_AS_COMP_TOT): double"""
        return self.data["DEFER_RPT_AS_COMP_TOT"]

    @property
    def DEFER_WITHDR_TOT(self) -> pd.Series:
        """DEFER_WITHDR_TOT -- Total Aggregate Withdrawals/Distributions Fro (DEFER_WITHDR_TOT): double"""
        return self.data["DEFER_WITHDR_TOT"]

    @property
    def EIP_UNEARN_NUM(self) -> pd.Series:
        """EIP_UNEARN_NUM -- Equity Incentive Plan--Number of Unearned Sha (EIP_UNEARN_NUM): double"""
        return self.data["EIP_UNEARN_NUM"]

    @property
    def EIP_UNEARN_VAL(self) -> pd.Series:
        """EIP_UNEARN_VAL -- Equity Incentive Plan - Value of Unearned/Unv (EIP_UNEARN_VAL): double"""
        return self.data["EIP_UNEARN_VAL"]

    @property
    def EXECDIR(self) -> pd.Series:
        """EXECDIR -- Executive served as a director during the fis (EXECDIR): double"""
        return self.data["EXECDIR"]

    @property
    def EXECRANKANN(self) -> pd.Series:
        """EXECRANKANN -- Executive Rank by Salary + Bonus (EXECRANKANN): double"""
        return self.data["EXECRANKANN"]

    @property
    def INTERLOCK(self) -> pd.Series:
        """INTERLOCK -- Executive Is Listed in the Compensation Commi (INTERLOCK): double"""
        return self.data["INTERLOCK"]

    @property
    def LTIP(self) -> pd.Series:
        """LTIP -- LTIP Payouts (LTIP): double"""
        return self.data["LTIP"]

    @property
    def NONEQ_INCENT(self) -> pd.Series:
        """NONEQ_INCENT -- Non-Equity Incentive Plan Compensation ($) (NONEQ_INCENT): double"""
        return self.data["NONEQ_INCENT"]

    @property
    def OLD_DATAFMT_FLAG(self) -> pd.Series:
        """OLD_DATAFMT_FLAG -- True Indicates data in this table is based on (OLD_DATAFMT_FLAG): double"""
        return self.data["OLD_DATAFMT_FLAG"]

    @property
    def OPTION_AWARDS(self) -> pd.Series:
        """OPTION_AWARDS -- Value of Option Awards - FAS 123R ($) (OPTION_AWARDS): double"""
        return self.data["OPTION_AWARDS"]

    @property
    def OPTION_AWARDS_BLK_VALUE(self) -> pd.Series:
        """OPTION_AWARDS_BLK_VALUE -- Options Granted ($ - Compustat Black Scholes (OPTION_AWARDS_BLK_VALUE): double"""
        return self.data["OPTION_AWARDS_BLK_VALUE"]

    @property
    def OPTION_AWARDS_FV(self) -> pd.Series:
        """OPTION_AWARDS_FV -- Grant Date Fair Value of Options Granted ($ (OPTION_AWARDS_FV): double"""
        return self.data["OPTION_AWARDS_FV"]

    @property
    def OPTION_AWARDS_NUM(self) -> pd.Series:
        """OPTION_AWARDS_NUM -- Options Granted (OPTION_AWARDS_NUM): double"""
        return self.data["OPTION_AWARDS_NUM"]

    @property
    def OPTION_AWARDS_RPT_VALUE(self) -> pd.Series:
        """OPTION_AWARDS_RPT_VALUE -- Options Granted ($ - As Reported by Company) (OPTION_AWARDS_RPT_VALUE): double"""
        return self.data["OPTION_AWARDS_RPT_VALUE"]

    @property
    def OPT_EXER_NUM(self) -> pd.Series:
        """OPT_EXER_NUM -- Number of Shares Acquired on Option Exercise (OPT_EXER_NUM): double"""
        return self.data["OPT_EXER_NUM"]

    @property
    def OPT_EXER_VAL(self) -> pd.Series:
        """OPT_EXER_VAL -- Value Realized on Option Exercise ($) (OPT_EXER_VAL): double"""
        return self.data["OPT_EXER_VAL"]

    @property
    def OPT_UNEX_EXER_EST_VAL(self) -> pd.Series:
        """OPT_UNEX_EXER_EST_VAL -- Estimated Value of In-the-Money Unexercised E (OPT_UNEX_EXER_EST_VAL): double"""
        return self.data["OPT_UNEX_EXER_EST_VAL"]

    @property
    def OPT_UNEX_EXER_NUM(self) -> pd.Series:
        """OPT_UNEX_EXER_NUM -- Unexercised Exercisable Options (OPT_UNEX_EXER_NUM): double"""
        return self.data["OPT_UNEX_EXER_NUM"]

    @property
    def OPT_UNEX_UNEXER_EST_VAL(self) -> pd.Series:
        """OPT_UNEX_UNEXER_EST_VAL -- Estimated Value Of In-the-Money Unexercised U (OPT_UNEX_UNEXER_EST_VAL): double"""
        return self.data["OPT_UNEX_UNEXER_EST_VAL"]

    @property
    def OPT_UNEX_UNEXER_NUM(self) -> pd.Series:
        """OPT_UNEX_UNEXER_NUM -- Unexercised Unexercisable Options (OPT_UNEX_UNEXER_NUM): double"""
        return self.data["OPT_UNEX_UNEXER_NUM"]

    @property
    def OTHANN(self) -> pd.Series:
        """OTHANN -- Other Annual (OTHANN): double"""
        return self.data["OTHANN"]

    @property
    def OTHCOMP(self) -> pd.Series:
        """OTHCOMP -- All Ohter Compensation ($) (OTHCOMP): double"""
        return self.data["OTHCOMP"]

    @property
    def PENSION_CHG(self) -> pd.Series:
        """PENSION_CHG -- Change in Pension Value and NonQualified Defe (PENSION_CHG): double"""
        return self.data["PENSION_CHG"]

    @property
    def PENSION_PYMTS_TOT(self) -> pd.Series:
        """PENSION_PYMTS_TOT -- Total Payments Made From All Pension Plans Du (PENSION_PYMTS_TOT): double"""
        return self.data["PENSION_PYMTS_TOT"]

    @property
    def PENSION_VALUE_TOT(self) -> pd.Series:
        """PENSION_VALUE_TOT -- Present Value of Accumulated Pension Benefits (PENSION_VALUE_TOT): double"""
        return self.data["PENSION_VALUE_TOT"]

    @property
    def REPRICE(self) -> pd.Series:
        """REPRICE -- Executive Is Listed in a Stock Option Reprice (REPRICE): double"""
        return self.data["REPRICE"]

    @property
    def RET_YRS(self) -> pd.Series:
        """RET_YRS -- Number of Years of Credited Service (RET_YRS): double"""
        return self.data["RET_YRS"]

    @property
    def RSTKGRNT(self) -> pd.Series:
        """RSTKGRNT -- Restricted Stock Grant ($) (RSTKGRNT): double"""
        return self.data["RSTKGRNT"]

    @property
    def RSTKVYRS(self) -> pd.Series:
        """RSTKVYRS -- Years Until Restricted Stock Grant Begins to (RSTKVYRS): double"""
        return self.data["RSTKVYRS"]

    @property
    def SALARY(self) -> pd.Series:
        """SALARY -- Salary ($) (SALARY): double"""
        return self.data["SALARY"]

    @property
    def SAL_PCT(self) -> pd.Series:
        """SAL_PCT -- Salary Percent Change Year-to-Year (%) (SAL_PCT): double"""
        return self.data["SAL_PCT"]

    @property
    def SHROWN_EXCL_OPTS(self) -> pd.Series:
        """SHROWN_EXCL_OPTS -- Shares Owned - Options Excluded (SHROWN_EXCL_OPTS): double"""
        return self.data["SHROWN_EXCL_OPTS"]

    @property
    def SHROWN_EXCL_OPTS_PCT(self) -> pd.Series:
        """SHROWN_EXCL_OPTS_PCT -- Percentage of Total Shares Owned - Options Ex (SHROWN_EXCL_OPTS_PCT): double"""
        return self.data["SHROWN_EXCL_OPTS_PCT"]

    @property
    def SHROWN_TOT(self) -> pd.Series:
        """SHROWN_TOT -- Shares Owned - As Reported (SHROWN_TOT): double"""
        return self.data["SHROWN_TOT"]

    @property
    def SHROWN_TOT_PCT(self) -> pd.Series:
        """SHROWN_TOT_PCT -- Percentage of Total Shares Owned - As Reporte (SHROWN_TOT_PCT): double"""
        return self.data["SHROWN_TOT_PCT"]

    @property
    def SHRS_VEST_NUM(self) -> pd.Series:
        """SHRS_VEST_NUM -- Number of Shares Acquired on Vesting (SHRS_VEST_NUM): double"""
        return self.data["SHRS_VEST_NUM"]

    @property
    def SHRS_VEST_VAL(self) -> pd.Series:
        """SHRS_VEST_VAL -- Value Realized on Vesting ($) (SHRS_VEST_VAL): double"""
        return self.data["SHRS_VEST_VAL"]

    @property
    def STOCK_AWARDS(self) -> pd.Series:
        """STOCK_AWARDS -- Value of Stock Awards - FAS 123R ($) (STOCK_AWARDS): double"""
        return self.data["STOCK_AWARDS"]

    @property
    def STOCK_AWARDS_FV(self) -> pd.Series:
        """STOCK_AWARDS_FV -- Grant Date Fair Value of Stock Awarded Under (STOCK_AWARDS_FV): double"""
        return self.data["STOCK_AWARDS_FV"]

    @property
    def STOCK_UNVEST_NUM(self) -> pd.Series:
        """STOCK_UNVEST_NUM -- Restricted Stock Holdings (STOCK_UNVEST_NUM): double"""
        return self.data["STOCK_UNVEST_NUM"]

    @property
    def STOCK_UNVEST_VAL(self) -> pd.Series:
        """STOCK_UNVEST_VAL -- Restricted Stock Holdings ($) (STOCK_UNVEST_VAL): double"""
        return self.data["STOCK_UNVEST_VAL"]

    @property
    def TDC1(self) -> pd.Series:
        """TDC1 -- Total Compensation (Salary + Bonus + Other An (TDC1): double"""
        return self.data["TDC1"]

    @property
    def TDC1_PCT(self) -> pd.Series:
        """TDC1_PCT -- TDC1 Percent Change Year-to-Year (%) (TDC1_PCT): double"""
        return self.data["TDC1_PCT"]

    @property
    def TDC2(self) -> pd.Series:
        """TDC2 -- Total Compensation (Salary + Bonus + Other An (TDC2): double"""
        return self.data["TDC2"]

    @property
    def TDC2_PCT(self) -> pd.Series:
        """TDC2_PCT -- TDC2 Percent Change Year-to-Year (%) (TDC2_PCT): double"""
        return self.data["TDC2_PCT"]

    @property
    def TERM_PYMT(self) -> pd.Series:
        """TERM_PYMT -- Estimated Payments in event of involuntary te (TERM_PYMT): double"""
        return self.data["TERM_PYMT"]

    @property
    def TITLEANN(self) -> pd.Series:
        """TITLEANN -- Annual Title (TITLEANN): string"""
        return self.data["TITLEANN"]

    @property
    def TOTAL_ALT1(self) -> pd.Series:
        """TOTAL_ALT1 -- Total Compensation - Stock/Options Valued Usi (TOTAL_ALT1): double"""
        return self.data["TOTAL_ALT1"]

    @property
    def TOTAL_ALT1_PCT(self) -> pd.Series:
        """TOTAL_ALT1_PCT -- Total Compensation (fair value of stock/optio (TOTAL_ALT1_PCT): double"""
        return self.data["TOTAL_ALT1_PCT"]

    @property
    def TOTAL_ALT2(self) -> pd.Series:
        """TOTAL_ALT2 -- Total Compensation - Stock Valued at time of (TOTAL_ALT2): double"""
        return self.data["TOTAL_ALT2"]

    @property
    def TOTAL_ALT2_PCT(self) -> pd.Series:
        """TOTAL_ALT2_PCT -- Total Compensation (options exercised/stock v (TOTAL_ALT2_PCT): double"""
        return self.data["TOTAL_ALT2_PCT"]

    @property
    def TOTAL_CURR(self) -> pd.Series:
        """TOTAL_CURR -- Total Current Compensation (Salary + Bonus) (TOTAL_CURR): double"""
        return self.data["TOTAL_CURR"]

    @property
    def TOTAL_CURR_PCT(self) -> pd.Series:
        """TOTAL_CURR_PCT -- Total Current Compensation Percent Change Yea (TOTAL_CURR_PCT): double"""
        return self.data["TOTAL_CURR_PCT"]

    @property
    def TOTAL_SEC(self) -> pd.Series:
        """TOTAL_SEC -- Total Compensation - As Reported in SEC Filin (TOTAL_SEC): double"""
        return self.data["TOTAL_SEC"]

    @property
    def TOTAL_SEC_PCT(self) -> pd.Series:
        """TOTAL_SEC_PCT -- Total Compensation (SEC total) Percent Change (TOTAL_SEC_PCT): double"""
        return self.data["TOTAL_SEC_PCT"]

    @property
    def YEAR(self) -> pd.Series:
        """Year (YEAR): double"""
        return self.data["YEAR"].astype(int)

from datetime import datetime
import numpy as np

MOCK_WRDS_COMP_FUNDA = np.array(
    [
        # datadate, gvkey, fyear, cik, indfmt, datafmt, popsrc, consol
        (datetime(2019, 6, 30), "000100", 2019, 1, "INDL", "STD", "D", "C"),
        (datetime(2020, 6, 30), "000100", 2020, 1, "INDL", "STD", "D", "C"),
        (datetime(2019, 6, 30), "000101", 2019, 2, "INDL", "STD", "D", "C"),
        (datetime(2020, 6, 30), "000101", 2020, 2, "INDL", "STD", "D", "C"),
    ],
    dtype=[
        ("datadate", "<M8[ns]"),
        ("gvkey", "U6"),
        ("fyear", "int"),
        ("cik", "int"),
        ("indfmt", "U6"),
        ("datafmt", "U6"),
        ("popsrc", "U6"),
        ("consol", "U6"),
    ],
).view(np.recarray)

MOCK_WRDS_BOARDEX_NA_WRDS_COMPANY_PROFILE = np.array(
    [
        # cikcode, boardid
        (1, 11),
        (2, 22),
        (3, 33),
        (4, 44),
    ],
    dtype=[("cikcode", "int"), ("boardid", "int")],
).view(np.recarray)

MOCK_WRDS_BOARDEX_NA_WRDS_ORG_COMPOSITION = np.array(
    [
        # companyid, datestartrole, dateendrole, directorid, seniority, rolename
        (
            11,
            datetime(2000, 1, 1),
            datetime(2021, 12, 31),
            10000,
            "Executive Director",
            "CEO",
        ),
        (
            11,
            None,
            datetime(2019, 12, 31),
            10001,
            "Executive Director",
            "CTO/COO",
        ),
        (
            11,
            datetime(2000, 1, 1),
            datetime(2021, 12, 31),
            20000,
            "Supervisory Director",
            "Independent NED",
        ),
        (
            11,
            datetime(2000, 1, 1),
            datetime(2021, 12, 31),
            30000,
            "Supervisory Director",
            "Independent Director",
        ),
        (
            11,
            datetime(2000, 1, 1),
            datetime(2021, 12, 31),
            40000,
            "Executive Director",
            "CFO",
        ),
        (
            22,
            datetime(2000, 1, 1),
            datetime(2020, 12, 31),
            50000,
            "Supervisory Director",
            "Independent Director",
        ),
        (
            22,
            datetime(2000, 1, 1),
            datetime(2020, 12, 31),
            60000,
            "Executive Director",
            "CEO",
        ),
    ],
    dtype=[
        ("companyid", "int"),
        ("datestartrole", "<M8[ns]"),
        ("dateendrole", "<M8[ns]"),
        ("directorid", "int"),
        ("seniority", "U32"),
        ("rolename", "U32"),
    ],
).view(np.recarray)


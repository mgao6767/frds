---
path: tree/master/frds
source: measures/credit_rating.py
---

# Credit Rating

## Definition

S&P Credit Ratings retrieved from the WRDS Capital IQ database. Specifically, it merges `wrds.ciq.erating` and `wrds.ciq.gvkey` by Company ID and if there are multiple ratings issued in a day for the same entity, the last rating data is used. However, there are a few cases where the same entity receive different ratings the same day, and the reporting time data is insufficient to discern the most recent one.

For ease of empirical analysis, the S&P ratings are transformed into conventional numerical scores from 1 to 22, where 1 represents a AAA rating and 22 reflects a D rating. Specifically, the mapping is:

* "AAA": 1,
* "AA+": 2,
* "AA": 3,
* "AA-": 4,
* "A+": 5,
* "A": 6,
* "A-": 7,
* "BBB+": 8,
* "BBB": 9,
* "BBB-": 10,
* "BB+": 11,
* "BB": 12,
* "BB-": 13,
* "B+": 14,
* "B": 15,
* "B-": 16,
* "CCC+": 17,
* "CCC": 18,
* "CCC-": 19,
* "CC": 20,
* "C": 21,
* "D": 22.

## Equivalent SAS Code

```sas
proc sql;
create table credt_rating as 
select distinct company_id, rdate, rating, rtype, b.gvkey 
from ciq.wrds_erating as a inner join ciq.wrds_gvkey as b
on a.company_id=b.companyid and 
	(a.rdate <= b.enddate or b.enddate=.E) and (a.rdate >= b.startdate or b.startdate=.B)
	and not missing(b.gvkey) and rtype="Local Currency LT"
group by gvkey, rdate having rtime=max(rtime)
order by company_id, rdate; 
quit;
```
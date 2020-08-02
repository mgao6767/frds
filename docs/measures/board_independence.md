---
path: tree/master/frds
source: measures/board_independence.py
---

# Board Independence

## Definition

### `BoardSize`

Number of directors, where directors are identified as those whose `seniority` in the BoardEx `na_wrds_org_composition` table is either "Executiver Director" or "Supervisory Director".

### `IndependentMembers`

Number of independent members on board, identified as those directors whose `rolename` in BoardEx `na_wrds_org_composition` table contains "independent".
 
### `BoardIndependence`

Ratio of independent board members to board size.

$$
\text{BoardIndependence}_{i,t}=\frac{\text{IndependentMembers}_{i,t}}{\text{BoardSize}_{i,t}}
$$

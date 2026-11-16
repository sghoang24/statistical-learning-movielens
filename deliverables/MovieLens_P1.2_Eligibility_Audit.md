# MovieLens 25M P1.2 Eligibility Audit

**Status** — Evidence package for review. Counts are from the local PowerShell run.

## Eligibility specification

Use ratings strictly before 1 January 2017 UTC. A user qualifies with at least one historical rating across all movies. A movie qualifies with at least ten historical ratings. Freeze these eligibility sets for validation and test. The movie rule retains 17,965 of 37,363 historically rated movies and 20,740,259 of 20,798,765 historical ratings (99.7187%).

Use one rating as the primary user minimum to keep sparse-history users in scope. Retain 5, 10, and 20 as prespecified evaluation sensitivity cutoffs. Apply them to the same fitted models and saved recommendations to isolate changes in the evaluated cohort.

## User threshold audit

Count each user's pre-2017 ratings. Join those counts to the 2017 validation positives generated with the one-rating minimum. A positive pair is a distinct user–movie pair with a 2017 rating greater than 3, an eligible movie, and no pre-2017 rating by that user for the movie. At each threshold, count historically eligible users, users with at least one eligible 2017 positive, and eligible positive pairs.

| Minimum pre-2017 ratings | Historically eligible users | Users with eligible 2017 positives | Eligible 2017 positive pairs |
|---:|---:|---:|---:|
| 1 | 142,184 | 4,778 | 169,368 |
| 5 | 142,136 | 4,748 | 165,945 |
| 10 | 142,085 | 4,714 | 163,014 |
| 20 | 141,902 | 4,630 | 158,256 |

Relative to one rating, a minimum of ten removes 0.07% of historically eligible users, 1.34% of users with eligible validation positives, and 3.75% of positive pairs. A minimum of twenty removes 0.20%, 3.10%, and 6.56%, respectively.

Historical eligibility uses pre-2017 data alone. The positive counts describe the available validation feedback. This audit measures cohort retention. Ranking performance at the four cutoffs is a later sensitivity analysis on overlapping cohorts.

## Movie threshold audit

Hold the user minimum at one pre-2017 rating. Rebuild the validation positives from raw 2017 ratings using all historically rated movies, then filter those pairs by each movie's pre-2017 count. The same rating > 3 rule and previously rated exclusions apply throughout.

| Minimum pre-2017 ratings per movie | Eligible movies | Historical ratings retained | Historical share retained | Users with eligible 2017 positives | Eligible 2017 positive pairs |
|---:|---:|---:|---:|---:|---:|
| 1 | 37,363 | 20,798,765 | 100.0000% | 4,794 | 176,894 |
| 5 | 22,471 | 20,770,015 | 99.8618% | 4,788 | 172,117 |
| **10** | **17,965** | **20,740,259** | **99.7187%** | **4,778** | **169,368** |
| 20 | 14,558 | 20,693,243 | 99.4927% | 4,771 | 165,452 |

Keep 10 as the primary cutoff. Moving from 5 to 10 removes 4,506 movies (20.05%) and 2,749 positive pairs (1.60%). Moving from 10 to 20 removes another 3,407 movies (18.96%) and 3,916 positive pairs (2.31%). The cutoff of 10 balances catalogue size and observed feedback while retaining movies with 10–19 historical ratings. These counts describe the catalogue and available feedback. Ranking performance across movie cutoffs remains unmeasured.

The 10-rating row reproduced the existing P1.2 and P1.4 results.

## Reviewer commands

From the project root in PowerShell, use the project environment with the dependencies in `requirements.txt`. The audit needs the extracted MovieLens 25M CSV and the previously generated validation positives file generated with a one-rating user minimum and a ten-rating movie minimum.

```powershell
Test-Path .\data\raw\ml-25m\ratings.csv
Test-Path .\data\processed\validation_positives.csv
.\.venv\Scripts\python.exe -m src.audit_eligibility
```

Both path checks should return `True`. If the positives file needs regeneration, use the data-preparation record's reproduction command with `min_user_ratings=1` and `positive_threshold=3.0`. Compare the four rows in each printed table with the corresponding results above. The combined script checks that the 10-rating movie row reproduces the existing catalogue and validation counts. The reported tables come from the two local inline runs.

## Review criterion

A teammate can reproduce the counts and verify that eligibility thresholds use only historical ratings. Record approval and integrate the script and this audit record with the shared project before marking P1.2 Integrated.

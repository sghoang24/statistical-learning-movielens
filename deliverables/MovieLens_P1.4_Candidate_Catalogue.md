# MovieLens 25M P1.4 Candidate Catalogue

**Status** — Locally verified and ready for teammate review.

## Approved rule

Start every model from the same fixed catalogue of 17,965 movies. A movie enters this catalogue if it has at least 10 ratings before 1 January 2017 UTC. For each user, remove all movies that user rated strictly before the recommendation date, regardless of score. Preserve catalogue order in the candidate helper.

| Phase | Recommendation date (UTC) | History used for exclusions |
|---|---|---|
| Validation | 1 January 2017 | Pre-2017 ratings |
| Final test | 1 January 2018 | Pre-2018 ratings, including 2017 |

The shared catalogue keeps the model comparison consistent. Per-user exclusions make the ranking task about movies the user has yet to rate. Held-out positives serve as scoring labels. Candidate generation uses only the fixed catalogue and history available at the recommendation date.

## Observed results

| Check | Local result |
|---|---|
| Synthetic validation candidates | [20, 30, 40] |
| Synthetic final-test candidates | [30, 40] |
| Fixed catalogue | 17,965 unique movies |
| Saved validation positives checked | 169,368 unique pairs across 4,778 users |
| Positives outside the fixed catalogue | 0 |
| Positives previously rated by the same user | 0 |

Movie 20 is rated at the start of validation and is excluded by the final-test date. Movie 30 is rated at the start of test and remains a final-test candidate. Another user's rating of movie 40 has no effect on user 1. Movie 99 is outside the catalogue.

## Reproduce the checks

Run from the project root in PowerShell with the dependencies installed in the project virtual environment. The checks use `src/catalogue.py`, `src/splits.py`, and `src/ratings.py`. For the full-data check, extract MovieLens 25M to `data/raw/ml-25m/` and regenerate `eligible_movies.csv` and `validation_positives.csv` under `data/processed/` using the Phase 1 data-preparation record.

Run both checks with the standalone module in `src/check_candidates.py`.

```powershell
.\.venv\Scripts\python.exe -m src.check_candidates
```

Expected output from the verified local run:

```text
validation: [20, 30, 40] - passed
test: [30, 40] - passed
Validation positive pairs checked: 169,368
Outside the fixed catalogue: 0
Previously rated by the same user: 0
P1.4 validation candidate consistency - passed
```

## Review criterion

Reproduce both synthetic lists and both zero counts. The full-data check verifies membership of saved validation positives under the approved rule. The final-test boundary was verified with synthetic data. Ranker integration should use the shared candidate helper with each phase's history.

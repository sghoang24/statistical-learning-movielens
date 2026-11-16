# MovieLens 25M P1.3 Temporal Protocol

**Status** — Locally verified and ready for teammate review.

## Agreed protocol

Generate one set of recommendations at the start of each phase. The recommendation date separates the ratings available to the model from the later outcomes used for scoring. All boundaries are UTC. Each outcome window includes its start and excludes its end.

| Phase | Recommendation date | Training and user history | Scoring window |
|---|---|---|---|
| Validation | 1 January 2017 | Ratings before 1 January 2017 | 1 January 2017 to 1 January 2018 |
| Final test | 1 January 2018 | Ratings before 1 January 2018, including 2017 | 1 January 2018 to 1 January 2020 |

Use 2017 outcomes to select model settings. Freeze those settings before refitting on all available pre-2018 history. Score the final recommendations against the held-out test window. The last observed rating in that window is dated 21 November 2019 UTC.

This design uses 2017 ratings first as validation outcomes and later as history available at the final recommendation date. User and movie eligibility stays fixed from pre-2017 data under P1.2. For final-test candidates, a user's 2017 ratings also count toward the previously rated movie exclusion.

## Implementation and observed results

`src/splits.py` defines the UTC cutoffs and exposes `phase_ratings(ratings, phase)`. It returns a history frame with timestamps strictly before the recommendation date and an outcome frame within the phase's scoring window. The counts below describe those raw temporal partitions before eligibility and candidate filters.

| Phase | History rows | Outcome rows | Timestamp checks |
|---|---:|---:|---|
| Validation | 20,798,765 | 1,689,935 | Passed |
| Final test | 22,488,700 | 2,511,395 | Passed |

Both phases passed the row-count and timestamp assertions in the local PowerShell run. The final-test history count equals the sum of the validation history and validation outcomes.

## Reproduce the checks

From the `StatisticalLearningProj` root in PowerShell, use the project virtual environment with dependencies installed from `requirements.txt`. The command requires `src/ratings.py`, `src/splits.py` with `phase_ratings`, and the extracted MovieLens 25M file at `data/raw/ml-25m/ratings.csv`.

```powershell
Test-Path .\data\raw\ml-25m\ratings.csv

@'
from src.ratings import load_ratings
from src.splits import VALIDATION_START, TEST_START, TEST_END, phase_ratings

ratings = load_ratings()
checks = (
    ("validation", VALIDATION_START, TEST_START, 20_798_765, 1_689_935),
    ("test", TEST_START, TEST_END, 22_488_700, 2_511_395),
)

for phase, start, end, expected_history, expected_outcomes in checks:
    history, outcomes = phase_ratings(ratings, phase)

    assert len(history) == expected_history, f"{phase}: history count"
    assert len(outcomes) == expected_outcomes, f"{phase}: outcome count"
    assert (history["timestamp"] < start).all(), f"{phase}: history cutoff"
    assert (
        (outcomes["timestamp"] >= start) & (outcomes["timestamp"] < end)
    ).all(), f"{phase}: outcome window"

    print(f"{phase}: {len(history):,} history, {len(outcomes):,} outcomes - passed")
    del history, outcomes
'@ | .\.venv\Scripts\python.exe -
```

Expected output, with an ASCII hyphen for consistent terminal display.

```text
True
validation: 20,798,765 history, 1,689,935 outcomes - passed
test: 22,488,700 history, 2,511,395 outcomes - passed
```

## Review criterion

Confirm the recommendation dates, scoring windows, fixed eligibility cutoff, and refit policy. Rerun the checks and compare both rows with the observed results. These checks verify the split helper. Model integration must enforce the history and outcome separation when fitting, ranking, and scoring.

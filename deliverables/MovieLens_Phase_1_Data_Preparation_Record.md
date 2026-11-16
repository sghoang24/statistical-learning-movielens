# MovieLens 25M Phase 1 Data Preparation Record

**8 November 2026 · Draft for team review**  
**Tracker scope** — P1.2 data eligibility, P1.3 temporal protocol, P1.4 candidate catalogue, and P1.7 reproducible datasets.

This record documents local runs in `StatisticalLearningProj`. It supports the planned comparison of Item-CF, matrix factorization, and their hybrid on observed-positive NDCG@10. The results below describe data preparation and validation labels.

## 1. Decisions and rationale

| Decision | Rationale |
|---|---|
| Train before 2017, select models on 2017, and reserve 2018–2019 for final testing | The chronological order mirrors a recommendation setting where models learn from past ratings and are evaluated on later feedback. Reserving a separate test period protects the final comparison from model-selection feedback. |
| Use UTC calendar boundaries | Explicit UTC cutoffs give teammates identical partitions across machine time zones. |
| Freeze the movie catalogue using at least 10 ratings per movie before 2017 | A historical activity minimum keeps the candidate pool grounded in information available at the cutoff and gives every strategy the same catalogue. The observed trade-off was 51.92% fewer historically rated titles while retaining 99.7187% of historical ratings. Rating retention measures historical interaction volume. |
| Define user eligibility from pre-2017 history | Personalized strategies need user history available at the cutoff. The first run used a configurable minimum of one rating across all pre-2017 movies, including titles outside the catalogue. That minimum is provisional. |
| Exclude previously rated movies from each user's candidate set | The ranking task targets movies the user has yet to rate at the recommendation date. For the 2017 label build, the exclusion used pre-2017 history. |
| Treat a 2017 rating greater than 3 as an observed positive | This working threshold captures favorable explicit feedback, starting at 3.5 on the dataset's half-star scale. An unrated pair has unknown relevance. Future positives provide evaluation labels and determine which users have a defined ranking-metric denominator. Historical eligibility and candidate generation are separate from that scoring subset. |

## 2. Implementation and observed results

### Environment and ratings loader

The run used Python **3.13.2** from the project-local `.venv` on Windows PowerShell. Imports of NumPy, pandas, scikit-learn, Matplotlib, and ipykernel succeeded. Installed dependencies were recorded with `pip freeze` in `requirements.txt`.

`src/ratings.py` resolves `data/raw/ml-25m/ratings.csv` relative to the project root. It reads only the four rating fields, sets explicit numeric types to reduce memory use, and accepts an optional `nrows` argument for smoke checks.

| Check | Observed result |
|---|---|
| Five-row smoke test | Loaded five rows for user 1. The first row was movie 296, rating 5.0, timestamp 1147880044. |
| Full row count | 25,000,095 |
| Column types | userId and movieId int32, rating float32, timestamp int64 |
| Missing values | Zero in each of the four columns |
| Rating range | 0.5–5.0 |

### Chronological split

`src/splits.py` compares Unix timestamps against UTC boundaries. Each bounded window includes its start and excludes its end.

| Partition | UTC window | Observed dates | Rows |
|---|---|---|---:|
| Train | Before 2017-01-01 | 1995-01-09 to 2016-12-31 | 20,798,765 |
| Validation | 2017-01-01 to 2018-01-01 | 2017-01-01 to 2017-12-31 | 1,689,935 |
| Test | 2018-01-01 to 2020-01-01 | 2018-01-01 to 2019-11-21 | 2,511,395 |

The split accounted for all rows, with zero outside the windows. The test partition was inspected for dates and counts in this run.

### Fixed movie catalogue

`src/catalogue.py` counts pre-2017 rows per movie, retains counts of 10 or more, and sorts by movie ID. The output `data/processed/eligible_movies.csv` contains `movieId` and `pre_2017_rating_count`.

| Measure | Observed result |
|---|---:|
| Movies with pre-2017 ratings | 37,363 |
| Eligible movies | 17,965 |
| Excluded movies | 19,398 |
| Total pre-2017 ratings | 20,798,765 |
| Ratings for eligible movies | 20,740,259 |
| Ratings for excluded movies | 58,506 |
| Share of pre-2017 ratings retained | 99.7187% |

The eligible title count and retention reproduce the earlier feasibility figures.

### 2017 observed positives

`src/validation.py` first forms distinct 2017 user–movie pairs with ratings above 3. It then filters them by the historical user minimum, the fixed catalogue, and an exclusion join against pre-2017 rated pairs. The output `data/processed/validation_positives.csv` contains `userId` and `movieId`, sorted by both fields.

| Measure | Observed result |
|---|---:|
| Users with at least one pre-2017 rating | 142,184 |
| Distinct positive pairs in 2017 | 1,115,965 |
| Pairs removed by historical user eligibility | 923,439 |
| Pairs removed by catalogue eligibility | 23,158 |
| Pairs removed as previously rated | 0 |
| Eligible positive pairs | 169,368 |
| Users with eligible positives | 4,778 |

Removal counts are sequential. The 4,778 users are the provisional denominator for mean user-level NDCG@10 and Recall@10 under the working rules. The zero overlap observed in the labels still leaves a separate candidate-generation check. Each user's recommendation list must exclude movies already rated in their available history.

## 3. Reproduce from the project root

With the four source modules, installed dependencies, and the extracted dataset in place, run the following in PowerShell. The script regenerates both processed CSVs.

```powershell
@'
from pathlib import Path

import pandas as pd

from src.catalogue import build_catalogue, pre_2017_movie_counts
from src.ratings import load_ratings
from src.splits import split_ratings
from src.validation import build_validation_positives

print(load_ratings(nrows=5).to_string(index=False))
ratings = load_ratings()
print("Rows", len(ratings))
print(ratings.dtypes)
print(ratings.isna().sum())
print("Rating range", ratings["rating"].min(), ratings["rating"].max())

train, validation, test = split_ratings(ratings)
for name, frame in zip(("Train", "Validation", "Test"), (train, validation, test)):
    dates = pd.to_datetime(
        [frame["timestamp"].min(), frame["timestamp"].max()], unit="s", utc=True
    )
    print(name, len(frame), dates[0].date(), dates[1].date())
print("Outside windows", len(ratings) - len(train) - len(validation) - len(test))
del frame, test

movie_counts = pre_2017_movie_counts(ratings)
del ratings
catalogue = build_catalogue(movie_counts)
total = int(movie_counts.sum())
retained = int(catalogue["pre_2017_rating_count"].sum())
print("Movies before cutoff / eligible / excluded",
      len(movie_counts), len(catalogue), len(movie_counts) - len(catalogue))
print("Ratings before cutoff / retained / excluded", total, retained, total - retained)
print(f"Ratings retained {retained / total:.4%}")

positives, counts = build_validation_positives(
    train, validation, catalogue, min_user_ratings=1, positive_threshold=3.0
)
for label, count in counts.items():
    print(f"{label}: {count:,}")

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)
catalogue.to_csv(output / "eligible_movies.csv", index=False)
positives.to_csv(output / "validation_positives.csv", index=False)
'@ | .\.venv\Scripts\python.exe -
```

## 4. Open items for Phase 1 review

- **P1.2** — Confirm the user-history minimum and the working positive threshold. Document which users enter each metric and how zero-positive users are handled.
- **P1.3** — Finalize recommendation dates, evaluation horizons, and the final-test refit and history policy.
- **P1.4** — Implement and verify shared per-user candidate generation for every ranker.
- **P1.7** — Add user and movie ID mappings, record dataset and code versions in the manifest, and have a teammate reproduce the outputs.

# MovieLens 25M P1.5 Evaluation Metrics

**Status** — Ready for teammate review. Metric checks use small examples. The popularity audit results below come from the local MovieLens run.

## Evaluation inputs

At each recommendation date, use the fixed pre-2017 catalogue and remove movies each user already rated by that date under P1.4. Rank each user's eligible candidates by model score descending. For ties, use movie ID ascending as a reproducible default. Pass ordered movie IDs to the metric functions. Scores can come from rating predictions, popularity or a hybrid.

An eligible observed positive is a distinct candidate movie with a held-out rating greater than 3. A 3.5 and a 5 earn equal binary gain. Missing feedback has unknown relevance and contributes zero observed gain.

## Metric definitions

| Metric | Definition |
|---|---|
| Observed-positive NDCG@10 (primary) | Sum 1/log₂(r+1) for observed hits at ranks r from 1 to 10. Divide by the ideal gain from placing min(10, number of eligible positives) hits first. |
| Recall@10 | Divide hits in the first 10 slots by the user's total eligible observed positives. |
| Catalogue coverage@10 | Divide the number of distinct movies recommended across all evaluated users' top-10 lists by the 17,965 movies in the fixed catalogue. Report the count and percentage. |
| Less-popular exposure@10 | Divide returned top-10 slots occupied by less-popular movies by all returned top-10 slots. A movie shown to 100 users contributes 100 slots. With zero returned slots the metric is undefined. |
| RMSE | Take the square root of the mean squared prediction error over aligned held-out observed ratings. |
| MAE | Take the mean absolute prediction error over the same pairs. |

Average NDCG and Recall equally across a common cohort of users with at least one eligible observed positive. In the primary validation run, this is 4,778 users. Users in the cohort with zero hits score zero. Short lists earn credit for the hits they return. Use the same users for coverage and exposure. For the 1+, 5+, 10+ and 20+ user-history sensitivity checks, aggregate the saved lists and user scores over each subset with the fixed catalogue denominator.

For RMSE and MAE, align predictions with the same eligible held-out observed pairs across the rating predictors being compared, including ratings at or below 3. Report the common pair count. Supply finite ratings and predictions on the original rating scale. This pair set can differ from the ranking cohort.

## Less-popular movies

Sort the fixed catalogue by pre-2017 count descending, then movie ID ascending. The first 3,593 movies form the most-rated fifth. The remaining 14,372 movies form the less-popular group. Freeze the grouping across both phases.

| Popular group cutoff | Movies | Catalogue share | Historical rating share | Last included count | First excluded count |
|---|---:|---:|---:|---:|---:|
| Top 10% of titles | 1,797 | 10.00% | 76.87% | 2,583 | 2,582 |
| Top 20% of titles | 3,593 | 20.00% | 89.87% | 865 | 864 |
| Top 30% of titles | 5,390 | 30.00% | 94.88% | 380 | 380 |
| Smallest group reaching 80% of ratings | 2,077 | 11.56% | 80.01% | 2,089 | 2,088 |

The top fifth accounts for 89.87% of historical ratings. The exposure metric measures how much recommendation space goes to the remaining 80% of titles, which accounted for 10.13% of historical ratings. The selected boundary separates 865 from 864 ratings without splitting a tie. Historical rating share describes the grouping and does not enter the exposure formula.

## Reproduce the popularity audit

From the project root in PowerShell, with dependencies installed and the fixed catalogue CSV already generated:

```powershell
Test-Path .\data\processed\eligible_movies.csv
.\.venv\Scripts\python.exe -m src.audit_popularity
```

Compare the output to the table above. The input totals should be 17,965 movies and 20,740,259 pre-2017 ratings. Use the Phase 1 data-preparation record to regenerate the CSV if needed.

## Use and check the module

Save metrics.py as src/metrics.py. The following example checks all six metrics and the popularity classification. Run it from the project root.

```powershell
@'
from math import isclose

import pandas as pd

from src.metrics import (
    catalogue_coverage_at_10,
    less_popular_exposure_at_10,
    less_popular_movie_ids,
    mae,
    ndcg_at_10,
    recall_at_10,
    rmse,
)

catalogue = pd.DataFrame({
    "movieId": [1, 2, 3, 4, 5],
    "pre_2017_rating_count": [100, 50, 40, 30, 20],
})
ranked = [1, 5, 2]
positives = {1, 2, 3}
assert isclose(ndcg_at_10(ranked, positives), 0.7039180890341347)
assert isclose(recall_at_10(ranked, positives), 2 / 3)
assert ndcg_at_10([], positives) == recall_at_10([], positives) == 0

lists = [ranked, [1, 3]]
less_popular = less_popular_movie_ids(catalogue)
assert less_popular == {2, 3, 4, 5}
assert isclose(catalogue_coverage_at_10(lists, catalogue["movieId"]), 4 / 5)
assert isclose(
    less_popular_exposure_at_10(lists, catalogue["movieId"], less_popular), 3 / 5
)
assert isclose(rmse([5, 3], [4, 3]), 2 ** -0.5)
assert isclose(mae([5, 3], [4, 3]), 0.5)
print("P1.5 metric examples passed")
'@ | .\.venv\Scripts\python.exe -
```

## Run a popularity baseline on 2017 validation

This experiment measures how well the 10 most frequently rated unseen movies recover each user's eligible observed positives from 2017.

1. Sort the 17,965 eligible movies by pre-2017 rating count, highest first, breaking ties by ascending movie ID. Every historical rating contributes to popularity regardless of its star value.
2. For each user, remove movies they rated before 1 January 2017 UTC. Take the first 10 remaining movies. All users start from the same ranking, and their history determines which titles are skipped.
3. Compare each list with that user's eligible 2017 positives (ratings above 3). The 169,368 saved pairs determine the evaluation cohort of 4,778 users and which recommendations count as hits. They supply no popularity scores or candidate exclusions.
4. Average NDCG@10 and Recall@10 equally across those users. Compute coverage from the distinct movies across their lists and exposure from the share of returned slots given to the less-popular group.

This is an offline benchmark for users with eligible observed positives. An unrated recommendation earns zero observed gain. For a like-for-like comparison, evaluate other models on the same users and candidate sets.

Save the script as `src/run_popularity_validation.py`. From the project root, with the dependencies installed, the raw ratings available, and eligible_movies.csv and validation_positives.csv generated under data/processed/, run:

```powershell
.\.venv\Scripts\python.exe -m src.run_popularity_validation
```

Observed output from the full 2017 validation run.

```text
Evaluated users: 4,778
Mean NDCG@10: 0.075212
Mean Recall@10: 0.023637
Catalogue coverage@10: 287 of 17,965 (1.598%)
Less-popular exposure@10: 0.000%
```

The script writes `outputs/popularity_validation_user_metrics.csv` and `outputs/popularity_validation_top10.csv`. The displayed exposure is 0.000%, so the share of slots outside the most-rated fifth was zero to the reported precision.

## Score predictions from a rating model

Popularity counts are ranking scores. RMSE and MAE are applicable when a model predicts star ratings. Run the following example after your rating model exports `outputs/validation_rating_predictions.csv`. If you only ran the popularity baseline, use the worked metric checks above to verify RMSE and MAE.

```powershell
@'
import pandas as pd

from src.metrics import mae, rmse

pairs = pd.read_csv("outputs/validation_rating_predictions.csv")
actual = pairs["rating"]
predicted = pairs["predicted_rating"]
print(f"Pairs scored: {len(pairs):,}")
print(f"RMSE: {rmse(actual, predicted):.6f}")
print(f"MAE: {mae(actual, predicted):.6f}")
'@ | .\.venv\Scripts\python.exe -
```

The rating example requires a separate model-generated CSV with one row per common observed pair and both values on the original rating scale.

## Review checklist

- Reproduce the popularity audit and run the worked examples.
- Check that every model uses the same evaluation cohort and per-user candidate rules.
- Preserve per-user metrics and recommendation lists for paired comparisons and sensitivity checks.

"""P2.7 independent metric check for the saved Phase 2 validation outputs.

Recomputes every ranking metric from the saved top-10 lists with its own
formulas (it does not import ``src.metrics`` or ``EvaluationCohort``), then
compares per-user and aggregate values with the saved outputs. It reads files
only: no model is fitted and no output is modified.

Inputs
- data/raw/ml-25m/ratings.csv: re-derives the inputs below and the
  already-rated exclusions
- data/processed/eligible_movies.csv: catalogue and pre-2017 counts
- data/processed/validation_positives.csv: eligible 2017 positives
- outputs/phase2_validation_<model>_top10.csv: userId, rank, movieId
- outputs/phase2_validation_<model>_user_metrics.csv: userId, ndcg_at_10, recall_at_10
- outputs/phase2_validation_summary.csv: aggregate metrics per model

Definitions (P1.5)
- Eligible positive: a 2017 rating strictly above 3 on a catalogue movie, by a
  user with at least one pre-2017 rating.
- NDCG@10 = DCG / IDCG with binary gains, DCG = sum over hits of 1/log2(rank+1),
  IDCG = sum_{k=1}^{min(|P_u|, 10)} 1/log2(k+1). Recall@10 = hits / |P_u|.
  Both are averaged over the evaluated users (those with at least one positive).
- Coverage = distinct recommended movies / catalogue size.
- Less-popular exposure = returned slots outside the top ceil(N/5) catalogue
  movies by pre-2017 count (ties by ascending movieId) / all returned slots.

Run from the project root: python -m src.check_phase2_metrics
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW_RATINGS = Path("data/raw/ml-25m/ratings.csv")
CATALOGUE = Path("data/processed/eligible_movies.csv")
POSITIVES = Path("data/processed/validation_positives.csv")
OUTPUTS = Path("outputs")
MODELS = ("popularity", "bias", "item_cf", "matrix_factorization")

VALIDATION_START = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp())
TEST_START = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
K = 10
MIN_MOVIE_HISTORY = 10
EXPECTED_USERS = 4_778
EXPECTED_CATALOGUE = 17_965
EXPECTED_POPULAR = 3_593

# Values are recomputed from the same inputs, so they should agree to rounding
# error only. Saved CSV floats round-trip exactly, so 1e-12 is generous.
TOLERANCE = 1e-12

failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    if not condition:
        failures.append(message)
    return condition


def load_inputs():
    ratings = pd.read_csv(
        RAW_RATINGS,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    history = ratings.loc[ratings["timestamp"] < VALIDATION_START, ["userId", "movieId"]]
    year_2017 = ratings.loc[
        (ratings["timestamp"] >= VALIDATION_START) & (ratings["timestamp"] < TEST_START),
        ["userId", "movieId", "rating"],
    ]
    del ratings

    # Catalogue: re-derive pre-2017 counts and the 10-rating threshold.
    counts = history["movieId"].value_counts()
    derived = counts[counts >= MIN_MOVIE_HISTORY]
    catalogue = pd.read_csv(CATALOGUE)
    saved_counts = catalogue.set_index("movieId")["pre_2017_rating_count"]
    check(len(catalogue) == EXPECTED_CATALOGUE,
          f"catalogue has {len(catalogue):,} movies, expected {EXPECTED_CATALOGUE:,}")
    check(catalogue["movieId"].is_unique, "catalogue movieIds are not unique")
    check(set(derived.index) == set(saved_counts.index),
          "catalogue movies differ from movies with >=10 pre-2017 ratings")
    check(saved_counts.equals(derived.reindex(saved_counts.index).astype(saved_counts.dtype)),
          "saved pre-2017 counts differ from counts re-derived from ratings.csv")

    # Positives: 2017 ratings > 3 on catalogue movies by users with pre-2017 history.
    known_users = pd.Index(history["userId"].unique())
    derived_pos = year_2017.loc[
        (year_2017["rating"] > 3)
        & year_2017["movieId"].isin(saved_counts.index)
        & year_2017["userId"].isin(known_users),
        ["userId", "movieId"],
    ].drop_duplicates()
    positives = pd.read_csv(POSITIVES)
    check(not positives.duplicated().any(), "validation_positives.csv has duplicate pairs")
    same = positives.merge(derived_pos, how="outer", indicator=True)
    check((same["_merge"] == "both").all(),
          f"validation_positives.csv differs from re-derived positives "
          f"({(same['_merge'] != 'both').sum():,} mismatched pairs)")
    check(positives["userId"].nunique() == EXPECTED_USERS,
          f"{positives['userId'].nunique():,} users have positives, expected {EXPECTED_USERS:,}")

    users = np.sort(positives["userId"].unique())
    seen = history.loc[history["userId"].isin(users)].drop_duplicates()
    return catalogue, positives, users, seen


def less_popular_set(catalogue: pd.DataFrame) -> tuple[set[int], str]:
    n_popular = -(-len(catalogue) // 5)  # ceil(N / 5)
    check(n_popular == EXPECTED_POPULAR, f"popular head is {n_popular:,}, expected {EXPECTED_POPULAR:,}")
    movie = catalogue["movieId"].to_numpy()
    count = catalogue["pre_2017_rating_count"].to_numpy()
    order = np.lexsort((movie, -count))  # count descending, then movieId ascending
    boundary = count[order[n_popular - 1]]
    tied_across = int((count == boundary).sum())
    note = (
        f"boundary count {boundary}; {tied_across} movie(s) share it"
        + (" (tie-break by movieId applies)" if tied_across > 1 else " (no tie at boundary)")
    )
    return set(movie[order[n_popular:]].tolist()), note


def check_lists(name: str, top10: pd.DataFrame, users: np.ndarray,
                catalogue_ids: set[int], seen: pd.DataFrame) -> None:
    check(list(top10.columns) == ["userId", "rank", "movieId"], f"{name}: unexpected top-10 columns")
    check(not top10.isna().any().any(), f"{name}: missing values in top-10 file")
    check(set(top10["userId"]) == set(users), f"{name}: top-10 users differ from the evaluated users")
    sizes = top10.groupby("userId").size()
    check(sizes.max() <= K, f"{name}: a list has more than {K} rows")
    short = int((sizes < K).sum())
    check(not top10.duplicated(["userId", "rank"]).any(), f"{name}: duplicate ranks within a list")
    ranks_ok = top10.groupby("userId")["rank"].agg(lambda r: sorted(r) == list(range(1, len(r) + 1)))
    check(ranks_ok.all(), f"{name}: ranks are not 1..n for {int((~ranks_ok).sum())} user(s)")
    check(not top10.duplicated(["userId", "movieId"]).any(), f"{name}: repeated movie within a list")
    outside = ~top10["movieId"].isin(catalogue_ids)
    check(not outside.any(), f"{name}: {int(outside.sum())} slot(s) outside the catalogue")
    already = top10.merge(seen, on=["userId", "movieId"], how="inner")
    check(already.empty, f"{name}: {len(already)} slot(s) recommend a movie rated before 2017")
    if short:
        print(f"    note: {short} user(s) have fewer than {K} recommendations")


def independent_metrics(top10: pd.DataFrame, positives: pd.DataFrame, users: np.ndarray,
                        n_catalogue: int, less_popular: set[int]):
    n_pos = positives.groupby("userId").size().reindex(users)
    hits = top10.merge(positives.assign(hit=1), on=["userId", "movieId"], how="left")
    hits["hit"] = hits["hit"].fillna(0)
    hits["gain"] = hits["hit"] / np.log2(hits["rank"] + 1.0)

    discounts = 1.0 / np.log2(np.arange(2, K + 2, dtype=float))
    ideal = np.cumsum(discounts)  # ideal[k-1] = IDCG with k relevant items
    idcg = ideal[np.minimum(n_pos.to_numpy(), K) - 1]

    per_user = pd.DataFrame({
        "ndcg_at_10": hits.groupby("userId")["gain"].sum().reindex(users, fill_value=0.0).to_numpy() / idcg,
        "recall_at_10": hits.groupby("userId")["hit"].sum().reindex(users, fill_value=0.0).to_numpy()
        / n_pos.to_numpy(),
    }, index=pd.Index(users, name="userId"))

    distinct = top10["movieId"].nunique()
    aggregate = {
        "mean_ndcg_at_10": per_user["ndcg_at_10"].mean(),
        "mean_recall_at_10": per_user["recall_at_10"].mean(),
        "coverage_movies": distinct,
        "coverage_fraction": distinct / n_catalogue,
        "less_popular_exposure": top10["movieId"].isin(less_popular).sum() / len(top10),
    }
    return per_user, aggregate


def main() -> int:
    print(f"P2.7 independent metric check (tolerance {TOLERANCE:g})")
    print("Re-deriving catalogue counts, positives and pre-2017 exclusions from ratings.csv ...")
    catalogue, positives, users, seen = load_inputs()
    catalogue_ids = set(catalogue["movieId"])
    less_popular, boundary_note = less_popular_set(catalogue)
    print(f"  users {len(users):,} | positives {len(positives):,} | catalogue {len(catalogue):,} "
          f"| popular head {len(catalogue) - len(less_popular):,} ({boundary_note})")
    print(f"  inputs: {'PASS' if not failures else 'FAIL'}")

    summary = pd.read_csv(OUTPUTS / "phase2_validation_summary.csv").set_index("model")
    for name in MODELS:
        before = len(failures)
        top10 = pd.read_csv(OUTPUTS / f"phase2_validation_{name}_top10.csv")
        saved_users = pd.read_csv(OUTPUTS / f"phase2_validation_{name}_user_metrics.csv")
        check_lists(name, top10, users, catalogue_ids, seen)

        per_user, aggregate = independent_metrics(top10, positives, users, len(catalogue), less_popular)

        check(list(saved_users.columns) == ["userId", "ndcg_at_10", "recall_at_10"],
              f"{name}: unexpected user-metric columns")
        check(saved_users["userId"].is_unique and set(saved_users["userId"]) == set(users),
              f"{name}: user-metric users differ from the evaluated users")
        saved_users = saved_users.set_index("userId").reindex(users)
        user_diff = {
            col: float(np.nanmax(np.abs(per_user[col] - saved_users[col])))
            for col in ("ndcg_at_10", "recall_at_10")
        }
        for col, diff in user_diff.items():
            check(diff <= TOLERANCE, f"{name}: per-user {col} differs by up to {diff:.3g}")

        row = summary.loc[name]
        agg_diff = {}
        for key, value in aggregate.items():
            diff = abs(value - row[key])
            agg_diff[key] = diff
            limit = 0 if key == "coverage_movies" else TOLERANCE
            check(diff <= limit, f"{name}: {key} independent {value!r} vs saved {row[key]!r}")
        # The saved per-user file should also average to the saved summary.
        check(abs(saved_users["ndcg_at_10"].mean() - row["mean_ndcg_at_10"]) <= TOLERANCE,
              f"{name}: saved per-user NDCG does not average to the saved summary")

        status = "PASS" if len(failures) == before else "FAIL"
        print(
            f"  {name:<21} {status} | NDCG@10 {aggregate['mean_ndcg_at_10']:.6f} "
            f"| Recall@10 {aggregate['mean_recall_at_10']:.6f} "
            f"| coverage {aggregate['coverage_movies']:,} ({aggregate['coverage_fraction']:.4%}) "
            f"| less-popular {aggregate['less_popular_exposure']:.6f} "
            f"| max |diff| user {max(user_diff.values()):.1e}, "
            f"aggregate {max(v for k, v in agg_diff.items() if k != 'coverage_movies'):.1e}"
        )

    if failures:
        print(f"\nFAIL: {len(failures)} discrepancy(ies)")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nPASS: all independent metrics match the saved outputs; all lists are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

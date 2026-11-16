import pandas as pd
from src.catalogue import candidate_movie_ids
from src.ratings import load_ratings
from src.splits import VALIDATION_START, TEST_START, phase_ratings

ratings = pd.DataFrame({
    "userId": [1, 1, 1, 2, 1],
    "movieId": [10, 20, 30, 40, 99],
    "timestamp": [
        VALIDATION_START - 1,
        VALIDATION_START,
        TEST_START,
        VALIDATION_START - 1,
        VALIDATION_START - 1,
    ],
})
catalogue_ids = pd.Series([10, 20, 30, 40], name="movieId")

for phase, expected in (("validation", [20, 30, 40]), ("test", [30, 40])):
    history, _ = phase_ratings(ratings, phase)
    rated = set(history.loc[history["userId"] == 1, "movieId"])
    actual = candidate_movie_ids(catalogue_ids, rated).tolist()
    assert actual == expected, (phase, actual, expected)
    print(f"{phase}: {actual} - passed")



pair_columns = ["userId", "movieId"]
positives = pd.read_csv(
    "data/processed/validation_positives.csv",
    dtype={"userId": "int32", "movieId": "int32"},
)
catalogue = pd.read_csv(
    "data/processed/eligible_movies.csv", dtype={"movieId": "int32"},
)

assert len(catalogue) == catalogue["movieId"].nunique() == 17_965
assert len(positives) == 169_368
assert positives["userId"].nunique() == 4_778
assert not positives.duplicated(pair_columns).any()

outside_catalogue = int((~positives["movieId"].isin(catalogue["movieId"])).sum())
ratings = load_ratings()
history_pairs = ratings.loc[
    (ratings["timestamp"] < VALIDATION_START)
    & ratings["userId"].isin(positives["userId"].unique())
    & ratings["movieId"].isin(catalogue["movieId"]),
    pair_columns,
].drop_duplicates()
del ratings

previously_rated = len(positives.merge(history_pairs, on=pair_columns, how="inner"))

print(f"Validation positive pairs checked: {len(positives):,}")
print(f"Outside the fixed catalogue: {outside_catalogue:,}")
print(f"Previously rated by the same user: {previously_rated:,}")

assert outside_catalogue == 0
assert previously_rated == 0
print("P1.4 validation candidate consistency - passed")
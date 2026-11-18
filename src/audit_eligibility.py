import pandas as pd
from src.catalogue import pre_2017_movie_counts
from src.splits import phase_ratings
from src.validation import build_validation_positives
from src.ratings import load_ratings
from src.splits import VALIDATION_START


def audit_user_thresholds() -> None:
    ratings = load_ratings()
    history_counts = ratings.loc[
        ratings["timestamp"] < VALIDATION_START, "userId"
    ].value_counts()
    del ratings

    positives = pd.read_csv(
        "data/processed/validation_positives.csv",
        dtype={"userId": "int32", "movieId": "int32"},
    )
    positive_counts = positives.groupby("userId").size()
    validation_history = positive_counts.index.map(history_counts)

    if validation_history.isna().any():
        raise ValueError("Validation positives include users without pre-2017 history.")

    rows = []
    for minimum in (1, 5, 10, 20):
        included = validation_history >= minimum
        rows.append({
            "min_history": minimum,
            "eligible_users": int((history_counts >= minimum).sum()),
            "users_with_2017_positives": int(included.sum()),
            "eligible_2017_positives": int(positive_counts[included].sum()),
        })

    print(pd.DataFrame(rows).to_string(index=False))

def audit_movie_thresholds() -> None:
    ratings = load_ratings()
    movie_counts = pre_2017_movie_counts(ratings)
    history, outcomes = phase_ratings(ratings, "validation")
    del ratings

    historical_catalogue = movie_counts.rename_axis("movieId").reset_index()
    positives, _ = build_validation_positives(
        history,
        outcomes,
        historical_catalogue,
        min_user_ratings=1,
        positive_threshold=3.0,
    )
    del history, outcomes

    positive_movie_support = positives["movieId"].map(movie_counts)
    assert positive_movie_support.notna().all()
    total_historical_ratings = int(movie_counts.sum())

    rows = []
    for minimum in (1, 5, 10, 20):
        eligible_counts = movie_counts[movie_counts >= minimum]
        retained_ratings = int(eligible_counts.sum())
        eligible_positives = positives.loc[positive_movie_support >= minimum]

        rows.append({
            "min_movie_history": minimum,
            "eligible_movies": len(eligible_counts),
            "pre_2017_ratings_retained": retained_ratings,
            "pre_2017_retained_pct": (
                f"{retained_ratings / total_historical_ratings:.4%}"
            ),
            "users_with_2017_positives": eligible_positives["userId"].nunique(),
            "eligible_2017_positive_pairs": len(eligible_positives),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    at_ten = next(row for row in rows if row["min_movie_history"] == 10)
    assert at_ten["eligible_movies"] == 17_965
    assert at_ten["pre_2017_ratings_retained"] == 20_740_259
    assert at_ten["users_with_2017_positives"] == 4_778
    assert at_ten["eligible_2017_positive_pairs"] == 169_368
    print("10-rating cutoff matches the existing P1.2 and P1.4 results.")


def main() -> None:
    print("User threshold audit (movie minimum = 10)")
    audit_user_thresholds()

    print("\nMovie threshold audit (user minimum = 1)")
    audit_movie_thresholds()

if __name__ == "__main__":
    main()
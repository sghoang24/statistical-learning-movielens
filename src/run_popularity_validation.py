"""Run the MovieLens popularity baseline on 2017 validation."""

from pathlib import Path

import pandas as pd

from src.catalogue import candidate_movie_ids
from src.metrics import (
    catalogue_coverage_at_10,
    less_popular_exposure_at_10,
    less_popular_movie_ids,
    ndcg_at_10,
    recall_at_10,
)
from src.ratings import load_ratings
from src.splits import VALIDATION_START


def main() -> None:
    catalogue = pd.read_csv("data/processed/eligible_movies.csv")
    positives = pd.read_csv("data/processed/validation_positives.csv")
    assert len(catalogue) == 17_965
    assert len(positives) == 169_368
    assert positives["userId"].nunique() == 4_778

    # Popularity is the historical rating count. Pre-sort once for all users.
    ranked_ids = catalogue.sort_values(
        ["pre_2017_rating_count", "movieId"], ascending=[False, True]
    )["movieId"]

    ratings = load_ratings()
    history = ratings.loc[
        (ratings["timestamp"] < VALIDATION_START)
        & ratings["userId"].isin(positives["userId"])
        & ratings["movieId"].isin(ranked_ids),
        ["userId", "movieId"],
    ]
    del ratings
    seen_by_user = history.groupby("userId")["movieId"].agg(set).to_dict()

    user_rows = []
    recommendation_rows = []
    ranked_lists = []
    for user_id, group in positives.groupby("userId"):
        candidates = candidate_movie_ids(ranked_ids, seen_by_user.get(user_id, set()))
        top10 = candidates.head(10).tolist()
        relevant = set(group["movieId"])
        ranked_lists.append(top10)
        user_rows.append({
            "userId": user_id,
            "ndcg_at_10": ndcg_at_10(top10, relevant),
            "recall_at_10": recall_at_10(top10, relevant),
        })
        for rank, movie_id in enumerate(top10, start=1):
            recommendation_rows.append({
                "userId": user_id, "rank": rank, "movieId": movie_id,
            })

    user_scores = pd.DataFrame(user_rows)
    coverage = catalogue_coverage_at_10(ranked_lists, catalogue["movieId"])
    exposure = less_popular_exposure_at_10(
        ranked_lists, catalogue["movieId"], less_popular_movie_ids(catalogue)
    )
    distinct_movies = len({movie for ranked in ranked_lists for movie in ranked})
    print(f"Evaluated users: {len(user_scores):,}")
    print(f"Mean NDCG@10: {user_scores['ndcg_at_10'].mean():.6f}")
    print(f"Mean Recall@10: {user_scores['recall_at_10'].mean():.6f}")
    print(f"Catalogue coverage@10: {distinct_movies:,} of {len(catalogue):,} ({coverage:.3%})")
    print(f"Less-popular exposure@10: {exposure:.3%}")

    output = Path("outputs")
    output.mkdir(exist_ok=True)
    user_scores.to_csv(output / "popularity_validation_user_metrics.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(
        output / "popularity_validation_top10.csv", index=False
    )


if __name__ == "__main__":
    main()

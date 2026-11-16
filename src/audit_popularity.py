from math import ceil
from pathlib import Path

import pandas as pd


CATALOGUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "eligible_movies.csv"
)
COUNT_COLUMN = "pre_2017_rating_count"


def main() -> None:
    catalogue = pd.read_csv(CATALOGUE_PATH)
    ranked = catalogue.sort_values(
        [COUNT_COLUMN, "movieId"],
        ascending=[False, True],
    ).reset_index(drop=True)

    counts = ranked[COUNT_COLUMN]
    movie_count = len(ranked)
    total_ratings = int(counts.sum())
    cumulative_ratings = counts.cumsum()

    rows = []
    cutoffs = [
        (f"Top {percent}% of movies", ceil(movie_count * percent / 100))
        for percent in (10, 20, 30)
    ]
    # Find the first cumulative count that reaches 80% of ratings.
    target_ratings = (4 * total_ratings + 4) // 5
    top_for_80 = int(cumulative_ratings.searchsorted(target_ratings)) + 1
    cutoffs.append(("At least 80% of ratings", top_for_80))

    for label, top_n in cutoffs:
        ratings_in_top = int(cumulative_ratings.iloc[top_n - 1])
        last_included = int(counts.iloc[top_n - 1])
        first_excluded = int(counts.iloc[top_n]) if top_n < movie_count else None

        rows.append({
            "cutoff": label,
            "popular_movies": top_n,
            "catalogue_share": f"{top_n / movie_count:.2%}",
            "historical_rating_share": f"{ratings_in_top / total_ratings:.2%}",
            "last_included_count": last_included,
            "first_excluded_count": first_excluded,
        })

    print(f"Eligible movies: {movie_count:,}")
    print(f"Pre-2017 ratings for eligible movies: {total_ratings:,}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
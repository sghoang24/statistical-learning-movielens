import pandas as pd


def build_validation_positives(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalogue: pd.DataFrame,
    *,
    min_user_ratings: int,
    positive_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return eligible observed positives and counts for the 2017 evaluation."""
    if min_user_ratings < 1:
        raise ValueError("min_user_ratings must be at least 1")

    pair_columns = ["userId", "movieId"]
    movie_ids = catalogue["movieId"]

    user_counts = train["userId"].value_counts()
    eligible_users = user_counts.index[user_counts >= min_user_ratings]

    positive_pairs = validation.loc[
        validation["rating"] > positive_threshold, pair_columns
    ].drop_duplicates()

    known_user_pairs = positive_pairs.loc[
        positive_pairs["userId"].isin(eligible_users)
    ]
    catalogue_pairs = known_user_pairs.loc[
        known_user_pairs["movieId"].isin(movie_ids)
    ]

    # Only check history for users with candidate positives in 2017.
    seen_pairs = train.loc[
        train["userId"].isin(catalogue_pairs["userId"].unique())
        & train["movieId"].isin(movie_ids),
        pair_columns,
    ].drop_duplicates()

    checked = catalogue_pairs.merge(
        seen_pairs, on=pair_columns, how="left", indicator=True
    )
    positives = (
        checked.loc[checked["_merge"] == "left_only", pair_columns]
        .sort_values(pair_columns)
        .reset_index(drop=True)
    )

    counts = {
        "Users meeting pre-2017 history minimum": len(eligible_users),
        "Distinct positive pairs in 2017": len(positive_pairs),
        "Removed by user history requirement": len(positive_pairs) - len(known_user_pairs),
        "Removed by catalogue requirement": len(known_user_pairs) - len(catalogue_pairs),
        "Removed as previously rated": len(catalogue_pairs) - len(positives),
        "Eligible positive pairs": len(positives),
        "Users with eligible positives": positives["userId"].nunique(),
    }
    return positives, counts
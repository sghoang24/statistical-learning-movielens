import pandas as pd

from src.splits import VALIDATION_START

from collections.abc import Collection

MIN_PRE_2017_RATINGS = 10


def pre_2017_movie_counts(ratings: pd.DataFrame) -> pd.Series:
    """Count pre-2017 ratings per movie, indexed by movieId."""
    pre_2017 = ratings.loc[ratings["timestamp"] < VALIDATION_START, "movieId"]
    return pre_2017.value_counts().rename("pre_2017_rating_count")


def build_catalogue(movie_counts: pd.Series) -> pd.DataFrame:
    """Build the fixed catalogue in movieId order."""
    eligible = movie_counts[movie_counts >= MIN_PRE_2017_RATINGS]
    return eligible.sort_index().rename_axis("movieId").reset_index()

def candidate_movie_ids(
    catalogue_movie_ids: pd.Series,
    rated_movie_ids: Collection[int],
) -> pd.Series:
    """Return eligible unseen movie IDs in catalogue order."""
    return catalogue_movie_ids.loc[
        ~catalogue_movie_ids.isin(rated_movie_ids)
    ].reset_index(drop=True)
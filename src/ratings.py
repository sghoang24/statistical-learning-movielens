from pathlib import Path

import pandas as pd

DEFAULT_RATINGS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-25m" / "ratings.csv"
)

RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}


def load_ratings(
    path: str | Path = DEFAULT_RATINGS_PATH,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load MovieLens ratings, optionally reading only the first nrows."""
    return pd.read_csv(path, usecols=list(RATING_DTYPES), dtype=RATING_DTYPES, nrows=nrows)
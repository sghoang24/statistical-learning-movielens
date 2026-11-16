"""MovieLens P1.5 ranking and rating metrics."""

from collections.abc import Iterable, Sequence
from math import fsum, log2, sqrt

import pandas as pd

K = 10
DISCOUNTS = tuple(1 / log2(rank + 1) for rank in range(1, K + 1))


def _top10(ranked: Sequence[int]) -> Sequence[int]:
    top = ranked[:K]
    if len(set(top)) != len(top):
        raise ValueError("Top-10 recommendations must be distinct.")
    return top


def ndcg_at_10(ranked: Sequence[int], positives: Iterable[int]) -> float:
    """Binary observed-positive NDCG for one user."""
    top = _top10(ranked)
    relevant = set(positives)
    if not relevant:
        raise ValueError("NDCG requires at least one eligible positive.")
    gain = fsum(weight for movie, weight in zip(top, DISCOUNTS) if movie in relevant)
    return gain / fsum(DISCOUNTS[:min(len(relevant), K)])


def recall_at_10(ranked: Sequence[int], positives: Iterable[int]) -> float:
    """Fraction of one user's eligible positives retrieved."""
    top = _top10(ranked)
    relevant = set(positives)
    if not relevant:
        raise ValueError("Recall requires at least one eligible positive.")
    return len(relevant.intersection(top)) / len(relevant)


def _recommendation_slots(
    ranked_lists: Iterable[Sequence[int]], catalogue: set[int]
) -> Iterable[int]:
    for ranked in ranked_lists:
        for movie in _top10(ranked):
            if movie not in catalogue:
                raise ValueError(f"Movie {movie} is outside the fixed catalogue.")
            yield movie


def catalogue_coverage_at_10(
    ranked_lists: Iterable[Sequence[int]], catalogue_ids: Iterable[int]
) -> float:
    catalogue = set(catalogue_ids)
    if not catalogue:
        raise ValueError("Coverage requires a nonempty catalogue.")
    return len(set(_recommendation_slots(ranked_lists, catalogue))) / len(catalogue)


def less_popular_movie_ids(catalogue: pd.DataFrame) -> set[int]:
    """Classify movies using only the fixed catalogue's historical counts."""
    if catalogue.empty or catalogue["movieId"].duplicated().any():
        raise ValueError("Expected a nonempty catalogue with unique movie IDs.")
    ranked = catalogue.sort_values(
        ["pre_2017_rating_count", "movieId"], ascending=[False, True]
    )
    popular_count = (len(ranked) + 4) // 5
    return set(ranked["movieId"].iloc[popular_count:])


def less_popular_exposure_at_10(
    ranked_lists: Iterable[Sequence[int]],
    catalogue_ids: Iterable[int],
    less_popular_ids: Iterable[int],
) -> float:
    catalogue = set(catalogue_ids)
    less_popular = set(less_popular_ids)
    if not less_popular.issubset(catalogue):
        raise ValueError("Less-popular movies must belong to the catalogue.")
    total = less_popular_count = 0
    for movie in _recommendation_slots(ranked_lists, catalogue):
        total += 1
        less_popular_count += movie in less_popular
    if total == 0:
        raise ValueError("Exposure requires at least one returned slot.")
    return less_popular_count / total


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error over aligned observed rating pairs."""
    if len(actual) == 0:
        raise ValueError("RMSE requires at least one rating pair.")
    return sqrt(fsum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error over aligned observed rating pairs."""
    if len(actual) == 0:
        raise ValueError("MAE requires at least one rating pair.")
    return fsum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)

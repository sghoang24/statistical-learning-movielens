"""Self-checks for the Phase 2 standalone models and shared pipeline.

Run from the project root with ``python -m src.check_phase2``. These checks use
small synthetic data and verify interface contracts and leakage-safe behaviour;
they do not require the full MovieLens archive.
"""

from math import isclose

import numpy as np
import pandas as pd

from src.bias import BiasModel
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.models import RATING_MAX, RATING_MIN, clip_rating
from src.pipeline import EvaluationCohort
from src.popularity import PopularityModel


def synthetic_history() -> pd.DataFrame:
    """Three like-minded users, one contrarian, and two popular items."""
    rows = []
    for user, (a, b) in enumerate([(5, 5), (5, 4), (4, 5), (1, 5), (5, 1)], start=1):
        rows.append({"userId": user, "movieId": 1, "rating": float(a)})
        rows.append({"userId": user, "movieId": 2, "rating": float(b)})
    return pd.DataFrame(rows)


def check_popularity() -> None:
    history = synthetic_history()
    model = PopularityModel().fit(history)
    assert model.ranking_[0] in (1, 2)
    grid = model.score_candidates(np.array([1, 2]), np.array([1, 2, 99]))
    assert grid.shape == (2, 3)
    assert (grid[:, 2] == 0).all(), "Unknown movie scores zero."
    try:
        model.predict(pd.DataFrame({"userId": [1], "movieId": [1]}))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Popularity must not expose star predictions.")
    print("popularity pass")


def check_bias() -> None:
    history = synthetic_history()
    model = BiasModel(lambda_user=0.0, lambda_movie=0.0, max_passes=200).fit(history)
    predicted = model.predict(history)
    model_rmse = float(np.sqrt(np.mean((predicted - history["rating"]) ** 2)))
    baseline_rmse = float(
        np.sqrt(np.mean((history["rating"] - history["rating"].mean()) ** 2))
    )
    assert model_rmse <= baseline_rmse + 1e-6, "Bias model should beat the global mean."
    grid = model.score_candidates(np.array([1]), np.array([1, 2]))
    assert grid.shape == (1, 2)
    print("bias pass")


def item_cf_history() -> pd.DataFrame:
    # Users who like item 1 also like item 2; item 3 is disliked by everyone.
    pattern = {
        1: [(1, 5), (2, 4), (3, 1)],
        2: [(1, 5), (2, 5), (3, 2)],
        3: [(1, 4), (2, 4), (3, 1)],
        4: [(1, 5), (2, 4), (3, 1)],
        5: [(1, 4), (2, 5), (3, 2)],
    }
    rows = [
        {"userId": user, "movieId": movie, "rating": float(rating)}
        for user, items in pattern.items()
        for movie, rating in items
    ]
    return pd.DataFrame(rows)


def check_item_cf() -> None:
    model = ItemCFModel(top_k=5, min_similarity=0.0).fit(item_cf_history())
    index = model.movie_index_
    assert model.similarity_[index[1], index[2]] > 0, "Co-liked items are similar."
    grid = model.score_candidates(np.array([1]), np.array([2]))
    assert np.isfinite(grid[0, 0]), "Unseen co-liked item should score finitely."
    unseen = model.score_candidates(np.array([999]), np.array([1, 2]))
    assert np.isnan(unseen).all(), "Unknown user yields no scores."
    print("item_cf pass")


def low_rank_history(n_u: int = 50, n_i: int = 30, f: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pu = rng.normal(0, 0.6, (n_u, f))
    qi = rng.normal(0, 0.6, (n_i, f))
    rows = []
    for user in range(n_u):
        for item in rng.choice(n_i, size=15, replace=False):
            rating = float(np.clip(3.0 + pu[user] @ qi[item], 0.5, 5.0))
            rows.append({"userId": user, "movieId": item, "rating": rating})
    return pd.DataFrame(rows)


def check_matrix_factorization() -> None:
    f = 3
    model = MatrixFactorizationModel(
        n_factors=f, regularization=1.0, max_iter=30, tolerance=1e-9, seed=1
    ).fit(low_rank_history(f=f))
    assert model.training_rmse_ < 0.4, "Should fit low-rank data."
    grid = model.score_candidates(np.array([0, 1]), np.array([0, 1, 2]))
    assert grid.shape == (2, 3)
    assert np.isfinite(grid).all()
    print("matrix_factorization pass")


def check_pipeline_excludes_rated() -> None:
    movie_ids = list(range(1, 13))
    catalogue = pd.DataFrame(
        {"movieId": movie_ids, "pre_2017_rating_count": [100 - i for i in movie_ids]}
    )
    positives = pd.DataFrame({"userId": [1, 1], "movieId": [3, 2]})
    history = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]})
    cohort = EvaluationCohort(catalogue, positives)
    cohort.set_history(history)
    # Score movie 1 highest, but it is already rated and must be excluded.
    scores = np.full((1, len(movie_ids)), 1.0)
    scores[0, 0] = 10.0
    ranked = cohort.rank_top_k(scores)
    assert 1 not in ranked[0], "Rated movie must be excluded."
    assert len(ranked[0]) == 10, "Top-10 should be full of the remaining candidates."
    print("pipeline exclusion pass")


def _check_clip_distinction(model, movie_attr: str, user: int, a: int, b: int) -> None:
    """Push fitted per-movie terms out of range; ranking stays raw, ratings clip.

    ``movie_attr`` is the fitted per-movie array added to every score
    (``bias_movie_`` or ``item_mean_``); shifting it moves the raw score of that
    movie by the same amount. Requires ``a < b``.
    """
    assert a < b
    base = getattr(model, movie_attr).copy()
    pos_a, pos_b = model.movie_index_[a], model.movie_index_[b]
    pairs = pd.DataFrame({"userId": [user, user], "movieId": [a, b]})

    # High end: both movies above 5, with b strictly higher than a.
    high = base.copy()
    high[pos_a] += 10
    high[pos_b] += 20
    setattr(model, movie_attr, high)
    raw = model.score_candidates(np.array([user]), np.array([a, b]))
    assert np.isfinite(raw).all(), f"{model.name}: synthetic scores must be finite."
    assert raw[0, 0] > RATING_MAX and raw[0, 1] > raw[0, 0], (
        f"{model.name}: ranking scores must stay unclipped, got {raw}."
    )
    assert np.allclose(model.predict(pairs), [RATING_MAX, RATING_MAX]), (
        f"{model.name}: predict must clip to {RATING_MAX}."
    )
    catalogue = pd.DataFrame({"movieId": [a, b], "pre_2017_rating_count": [2, 1]})
    cohort = EvaluationCohort(catalogue, pd.DataFrame({"userId": [user], "movieId": [a]}))
    cohort.set_history(pd.DataFrame({"userId": [], "movieId": [], "rating": []}))
    assert cohort.rank_top_k(raw)[0] == [b, a], "Raw scores keep b above a."
    # Clipping would tie them at 5 and the tie-break would hand a the top slot.
    assert cohort.rank_top_k(clip_rating(raw))[0] == [a, b]

    # Low end: a below 0.5 clips up to 0.5 in predict only.
    low = base.copy()
    low[pos_a] -= 10
    setattr(model, movie_attr, low)
    raw = model.score_candidates(np.array([user]), np.array([a]))
    assert raw[0, 0] < RATING_MIN, f"{model.name}: low raw score must stay unclipped."
    assert np.isclose(model.predict(pairs.iloc[:1])[0], RATING_MIN)
    setattr(model, movie_attr, base)


def check_rating_clipping() -> None:
    """RMSE/MAE inputs clip to [0.5, 5]; ranking scores never clip."""
    from src.run_phase2_validation import rating_predictions

    bias = BiasModel(lambda_user=0.0, lambda_movie=0.0).fit(synthetic_history())
    _check_clip_distinction(bias, "bias_movie_", user=1, a=1, b=2)
    item_cf = ItemCFModel(top_k=5, min_similarity=0.0).fit(item_cf_history())
    _check_clip_distinction(item_cf, "item_mean_", user=1, a=1, b=2)
    mf = MatrixFactorizationModel(n_factors=3, regularization=1.0, max_iter=5).fit(
        low_rank_history(f=3)
    )
    _check_clip_distinction(mf, "bias_movie_", user=0, a=0, b=1)

    # Unsupported Item-CF predictions stay NaN so the fallback can fill them.
    unknown = pd.DataFrame({"userId": [999], "movieId": [1]})
    assert np.isnan(item_cf.predict(unknown)).all()

    # Runner path: fallback-filled values are clipped too, even if the fallback
    # itself returned raw out-of-range values.
    class Stub:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def predict(self, pairs):
            return self.values.copy()

    pairs = pd.DataFrame({"userId": [1, 1, 1, 1], "movieId": [1, 2, 3, 4]})
    model = Stub([6.0, np.nan, 3.2, np.nan])
    fallback = Stub([0.0, 7.5, 0.0, -1.0])
    got = rating_predictions(model, fallback, pairs)
    assert np.allclose(got, [5.0, 5.0, 3.2, 0.5]), got
    print("rating clipping pass")


def main() -> None:
    check_popularity()
    check_bias()
    check_item_cf()
    check_matrix_factorization()
    check_pipeline_excludes_rated()
    check_rating_clipping()
    print("P2 self-checks passed")


if __name__ == "__main__":
    main()

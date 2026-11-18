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


def check_item_cf() -> None:
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
    history = pd.DataFrame(rows)
    model = ItemCFModel(top_k=5, min_similarity=0.0).fit(history)
    index = model.movie_index_
    assert model.similarity_[index[1], index[2]] > 0, "Co-liked items are similar."
    grid = model.score_candidates(np.array([1]), np.array([2]))
    assert np.isfinite(grid[0, 0]), "Unseen co-liked item should score finitely."
    unseen = model.score_candidates(np.array([999]), np.array([1, 2]))
    assert np.isnan(unseen).all(), "Unknown user yields no scores."
    print("item_cf pass")


def check_matrix_factorization() -> None:
    rng = np.random.default_rng(0)
    n_u, n_i, f = 50, 30, 3
    pu = rng.normal(0, 0.6, (n_u, f))
    qi = rng.normal(0, 0.6, (n_i, f))
    rows = []
    for user in range(n_u):
        for item in rng.choice(n_i, size=15, replace=False):
            rating = float(np.clip(3.0 + pu[user] @ qi[item], 0.5, 5.0))
            rows.append({"userId": user, "movieId": item, "rating": rating})
    history = pd.DataFrame(rows)
    model = MatrixFactorizationModel(
        n_factors=f, regularization=1.0, max_iter=30, tolerance=1e-9, seed=1
    ).fit(history)
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


def main() -> None:
    check_popularity()
    check_bias()
    check_item_cf()
    check_matrix_factorization()
    check_pipeline_excludes_rated()
    print("P2 self-checks passed")


if __name__ == "__main__":
    main()

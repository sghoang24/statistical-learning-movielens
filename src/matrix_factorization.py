"""P2.4 regularized matrix factorization (explicit ALS with biases)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models import clip_rating, require_history_columns


def _solve_rows(
    factors: np.ndarray,
    neighbours: np.ndarray,
    rated: sp.csr_matrix,
    residual: sp.csr_matrix,
    regularization: float,
) -> np.ndarray:
    """Solve one ALS half-step for every row with at least one rating."""
    penalty = regularization * np.eye(factors.shape[1])
    for row in range(factors.shape[0]):
        start, stop = rated.indptr[row], rated.indptr[row + 1]
        if start == stop:
            continue
        columns = rated.indices[start:stop]
        target = residual.data[start:stop]
        neighbour_rows = neighbours[columns]
        gram = neighbour_rows.T @ neighbour_rows + penalty
        rhs = neighbour_rows.T @ target
        factors[row] = np.linalg.solve(gram, rhs)
    return factors


@dataclass
class MatrixFactorizationModel:
    """Explicit-feedback ALS with global, user and movie biases.

    Predicts ``mu + b_u + b_i + p_u . q_i``. Each iteration updates user biases,
    movie biases, then the latent factors, and reports the training RMSE. The
    factor count, regularization and pass count are selected on 2017 validation
    only. Ranking scores are raw; :meth:`predict` clips to the shared rating
    range [0.5, 5] (``src.models.clip_rating``) for RMSE/MAE.

    ``monitor_clip_min``/``monitor_clip_max`` only clip the training RMSE used
    by the tolerance stop. They stay at [1, 5], as in the original fits, so the
    stopping iteration, fitted factors and top-10 lists are unchanged.
    """

    n_factors: int = 24
    regularization: float = 15.0
    lambda_user: float = 10.0
    lambda_movie: float = 10.0
    max_iter: int = 20
    tolerance: float = 1e-4
    seed: int = 0
    monitor_clip_min: float = 1.0
    monitor_clip_max: float = 5.0
    name: str = field(default="matrix_factorization", init=False)

    def __post_init__(self) -> None:
        if self.n_factors < 1:
            raise ValueError("n_factors must be at least 1.")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive.")

    def fit(self, history: pd.DataFrame) -> "MatrixFactorizationModel":
        require_history_columns(history)
        user_codes, user_ids = pd.factorize(history["userId"].to_numpy(), sort=True)
        movie_ids = np.sort(history["movieId"].unique())
        self.user_index_ = pd.Series(np.arange(len(user_ids)), index=user_ids)
        self.movie_index_ = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
        movie_codes = self.movie_index_.reindex(history["movieId"]).to_numpy()
        ratings = history["rating"].to_numpy(dtype=np.float64)

        n_users, n_items = len(user_ids), len(movie_ids)
        rated = sp.csr_matrix(
            (ratings, (user_codes, movie_codes)), shape=(n_users, n_items)
        ).tocsr()
        # CSR construction sorts by (row, column), so realign ratings to it.
        ratings = rated.data.astype(np.float64)
        rated_by_item = rated.T.tocsr()
        rows = np.repeat(np.arange(n_users), np.diff(rated.indptr))
        columns = rated.indices

        self.mu_ = float(ratings.mean())
        user_counts = np.diff(rated.indptr).astype(np.float64)
        item_counts = np.diff(rated_by_item.indptr).astype(np.float64)
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)
        rng = np.random.default_rng(self.seed)
        P = rng.normal(0, 0.01, (n_users, self.n_factors))
        Q = rng.normal(0, 0.01, (n_items, self.n_factors))

        previous_rmse = np.inf
        for iteration in range(self.max_iter):
            factor = np.einsum("ij,ij->i", P[rows], Q[columns])

            # Update user biases, then movie biases, holding factors fixed.
            user_residual = ratings - self.mu_ - factor - b_i[columns]
            b_u = np.bincount(rows, weights=user_residual, minlength=n_users) / (
                user_counts + self.lambda_user
            )
            item_residual = ratings - self.mu_ - factor - b_u[rows]
            b_i = np.bincount(
                columns, weights=item_residual, minlength=n_items
            ) / (item_counts + self.lambda_movie)

            # Update factors on the residual net of the current biases.
            residual = ratings - self.mu_ - b_u[rows] - b_i[columns]
            residual_by_user = sp.csr_matrix(
                (residual, columns, rated.indptr), shape=rated.shape
            )
            P = _solve_rows(
                P, Q, rated, residual_by_user, self.regularization
            )
            residual_by_item = residual_by_user.T.tocsr()
            Q = _solve_rows(
                Q, P, rated_by_item, residual_by_item, self.regularization
            )

            factor = np.einsum("ij,ij->i", P[rows], Q[columns])
            prediction = np.clip(
                self.mu_ + b_u[rows] + b_i[columns] + factor,
                self.monitor_clip_min,
                self.monitor_clip_max,
            )
            rmse = float(np.sqrt(np.mean((ratings - prediction) ** 2)))
            if abs(previous_rmse - rmse) < self.tolerance:
                break
            previous_rmse = rmse

        self.bias_user_ = b_u
        self.bias_movie_ = b_i
        self.factors_user_ = P.astype(np.float32)
        self.factors_movie_ = Q.astype(np.float32)
        self.iterations_ = iteration + 1
        self.training_rmse_ = rmse
        return self

    def _lookup(
        self, mapping: pd.Series, values: np.ndarray, keys: np.ndarray
    ) -> np.ndarray:
        position = mapping.reindex(keys).to_numpy()
        result = np.zeros(len(keys))
        found = ~pd.isna(position)
        result[found] = values[position[found].astype(int)]
        return result

    def score_candidates(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "factors_user_"):
            raise RuntimeError("Fit the model before predicting.")
        user_ids = np.asarray(user_ids)
        movie_ids = np.asarray(movie_ids)
        b_u = self._lookup(self.user_index_, self.bias_user_, user_ids)
        b_i = self._lookup(self.movie_index_, self.bias_movie_, movie_ids)
        user_pos = self.user_index_.reindex(user_ids).to_numpy()
        movie_pos = self.movie_index_.reindex(movie_ids).to_numpy()
        P = np.zeros((len(user_ids), self.n_factors), dtype=np.float32)
        Q = np.zeros((len(movie_ids), self.n_factors), dtype=np.float32)
        found_u = ~pd.isna(user_pos)
        found_i = ~pd.isna(movie_pos)
        P[found_u] = self.factors_user_[user_pos[found_u].astype(int)]
        Q[found_i] = self.factors_movie_[movie_pos[found_i].astype(int)]
        # Ranking uses full-precision raw predictions; clipping is rating output only.
        return self.mu_ + b_u[:, None] + b_i[None, :] + P @ Q.T

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "factors_user_"):
            raise RuntimeError("Fit the model before predicting.")
        user_ids = pairs["userId"].to_numpy()
        movie_ids = pairs["movieId"].to_numpy()
        user_pos = self.user_index_.reindex(user_ids).to_numpy()
        movie_pos = self.movie_index_.reindex(movie_ids).to_numpy()
        found_u = ~pd.isna(user_pos)
        found_i = ~pd.isna(movie_pos)
        P = np.zeros((len(pairs), self.n_factors), dtype=np.float32)
        Q = np.zeros((len(pairs), self.n_factors), dtype=np.float32)
        P[found_u] = self.factors_user_[user_pos[found_u].astype(int)]
        Q[found_i] = self.factors_movie_[movie_pos[found_i].astype(int)]
        b_u = self._lookup(self.user_index_, self.bias_user_, user_ids)
        b_i = self._lookup(self.movie_index_, self.bias_movie_, movie_ids)
        factor = np.einsum("ij,ij->i", P, Q)
        return clip_rating(self.mu_ + b_u + b_i + factor)

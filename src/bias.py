"""P2.2 regularized global/user/movie bias baseline."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models import require_history_columns


@dataclass
class BiasModel:
    """Predict stars as ``mu + b_u + b_i`` with regularized alternating updates.

    The global mean and user/movie biases are estimated jointly over observed
    training ratings by minimizing the regularized squared error documented in
    the project plan. A missing user or movie bias is treated as zero.
    """

    lambda_user: float = 10.0
    lambda_movie: float = 10.0
    max_passes: int = 20
    tolerance: float = 1e-6
    clip_min: float | None = None
    clip_max: float | None = None
    seed: int = 0
    name: str = field(default="bias", init=False)

    def __post_init__(self) -> None:
        if self.lambda_user < 0 or self.lambda_movie < 0:
            raise ValueError("Regularization strengths must be non-negative.")
        if self.max_passes < 1:
            raise ValueError("max_passes must be at least 1.")

    def fit(self, history: pd.DataFrame) -> "BiasModel":
        require_history_columns(history)
        users = history["userId"].to_numpy()
        movies = history["movieId"].to_numpy()
        ratings = history["rating"].to_numpy(dtype=np.float64)

        self.mu_ = float(ratings.mean())
        user_codes, user_ids = pd.factorize(users, sort=True)
        movie_codes, movie_ids = pd.factorize(movies, sort=True)
        self.user_ids_ = user_ids
        self.movie_ids_ = movie_ids

        user_counts = np.bincount(user_codes, minlength=len(user_ids)).astype(np.float64)
        movie_counts = np.bincount(movie_codes, minlength=len(movie_ids)).astype(np.float64)
        self.user_index_ = pd.Series(np.arange(len(user_ids)), index=user_ids)
        self.movie_index_ = pd.Series(np.arange(len(movie_ids)), index=movie_ids)

        b_u = np.zeros(len(user_ids))
        b_i = np.zeros(len(movie_ids))
        centered = ratings - self.mu_

        for _ in range(self.max_passes):
            previous = b_u.copy()

            user_residual = centered - b_i[movie_codes]
            user_sum = np.bincount(
                user_codes, weights=user_residual, minlength=len(user_ids)
            )
            b_u = user_sum / (user_counts + self.lambda_user)

            movie_residual = centered - b_u[user_codes]
            movie_sum = np.bincount(
                movie_codes, weights=movie_residual, minlength=len(movie_ids)
            )
            b_i = movie_sum / (movie_counts + self.lambda_movie)

            if np.max(np.abs(b_u - previous)) < self.tolerance:
                break

        self.bias_user_ = b_u
        self.bias_movie_ = b_i
        self.passes_ = _ + 1
        return self

    def _lookup(
        self, mapping: pd.Series, bias: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        position = mapping.reindex(values).to_numpy()
        result = np.zeros(len(values))
        found = ~pd.isna(position)
        result[found] = bias[position[found].astype(int)]
        return result

    def _close(self, prediction: np.ndarray) -> np.ndarray:
        if self.clip_min is not None or self.clip_max is not None:
            prediction = np.clip(prediction, self.clip_min, self.clip_max)
        return prediction

    def score_candidates(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "mu_"):
            raise RuntimeError("Fit the model before predicting.")
        user_ids = np.asarray(user_ids)
        movie_ids = np.asarray(movie_ids)
        b_u = self._lookup(self.user_index_, self.bias_user_, user_ids)
        b_i = self._lookup(self.movie_index_, self.bias_movie_, movie_ids)
        # Ranking uses full-precision raw predictions; clipping is rating output only.
        return self.mu_ + b_u[:, None] + b_i[None, :]

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mu_"):
            raise RuntimeError("Fit the model before predicting.")
        b_u = self._lookup(self.user_index_, self.bias_user_, pairs["userId"].to_numpy())
        b_i = self._lookup(
            self.movie_index_, self.bias_movie_, pairs["movieId"].to_numpy()
        )
        return self._close(self.mu_ + b_u + b_i)

"""P2.3 item-based collaborative filtering with adjusted-cosine similarity.

Similarity uses user-mean-centred ratings (adjusted cosine). Prediction combines
the candidate item's training mean with the similarity-weighted residuals of the
user's rated neighbours:

    pred(u, i) = b_i + sum_j s_ij (r_uj - b_j) / sum_j |s_ij|

Centring on item means keeps the score graded instead of saturating at the star
ceiling, which would flatten the ranking and hand ties to the lowest movie IDs.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models import require_history_columns


def _top_k_neighbours(similarity: sp.csr_matrix, top_k: int) -> sp.csr_matrix:
    """Keep each row's ``top_k`` strongest neighbours with a canonical structure."""
    similarity = similarity.tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    indptr, indices, data = similarity.indptr, similarity.indices, similarity.data
    new_indices: list[np.ndarray] = []
    new_data: list[np.ndarray] = []
    new_indptr = [0]
    for row in range(similarity.shape[0]):
        start, stop = indptr[row], indptr[row + 1]
        row_data = data[start:stop]
        row_indices = indices[start:stop]
        if row_data.size > top_k:
            keep = np.argpartition(row_data, -top_k)[-top_k:]
            row_data = row_data[keep]
            row_indices = row_indices[keep]
        order = np.argsort(row_indices)
        new_indices.append(row_indices[order])
        new_data.append(row_data[order])
        new_indptr.append(new_indptr[-1] + row_data.size)

    stacked_indices = np.concatenate(new_indices) if new_indices else np.empty(0, int)
    stacked_data = np.concatenate(new_data) if new_data else np.empty(0, np.float32)
    return sp.csr_matrix(
        (stacked_data, stacked_indices, np.asarray(new_indptr)), shape=similarity.shape
    )


@dataclass
class ItemCFModel:
    """Item-based CF: adjusted-cosine similarity with a top-K neighbourhood.

    Predicts a candidate item's training mean plus the similarity-weighted
    residuals of the user's rated neighbours, so scores stay graded instead of
    saturating at the star ceiling. A candidate with no positively-similar rated
    neighbour returns ``NaN`` so the shared fallback policy (P1.6) can supply a
    value.
    """

    top_k: int = 100
    min_similarity: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None
    name: str = field(default="item_cf", init=False)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

    def fit(self, history: pd.DataFrame) -> "ItemCFModel":
        require_history_columns(history)
        user_codes, user_ids = pd.factorize(history["userId"].to_numpy(), sort=True)
        movie_ids = np.sort(history["movieId"].unique())
        self.user_index_ = pd.Series(np.arange(len(user_ids)), index=user_ids)
        self.movie_index_ = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
        movie_codes = self.movie_index_.reindex(history["movieId"]).to_numpy()

        ratings = history["rating"].to_numpy(dtype=np.float32)
        n_users, n_items = len(user_ids), len(movie_ids)
        self.ratings_ = sp.csr_matrix(
            (ratings, (user_codes, movie_codes)), shape=(n_users, n_items)
        ).tocsr()
        self.binary_ = self.ratings_.copy()
        self.binary_.data[:] = 1.0

        user_counts = np.diff(self.ratings_.indptr).astype(np.float32)
        user_sums = np.asarray(self.ratings_.sum(axis=1)).ravel()
        user_mean = np.divide(
            user_sums, user_counts, out=np.zeros_like(user_sums), where=user_counts > 0
        )

        item_counts = np.diff(self.ratings_.tocsc().indptr).astype(np.float32)
        item_sums = np.asarray(self.ratings_.sum(axis=0)).ravel()
        self.item_mean_ = np.divide(
            item_sums, item_counts, out=np.zeros_like(item_sums), where=item_counts > 0
        )

        # Centre each user's ratings for adjusted-cosine similarity only.
        centered = self.ratings_.copy()
        for user in range(n_users):
            start, stop = centered.indptr[user], centered.indptr[user + 1]
            centered.data[start:stop] -= user_mean[user]

        gram = (centered.T @ centered).tocoo()
        norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=0)).ravel())
        denominator = norms[gram.row] * norms[gram.col]
        valid = denominator > 0
        similarity = gram.data[valid] / denominator[valid]
        sim_matrix = sp.csr_matrix(
            (similarity, (gram.row[valid], gram.col[valid])), shape=(n_items, n_items)
        )
        del gram, similarity, denominator, centered
        # Keep only positive neighbours so the weighted average stays stable.
        sim_matrix.data[sim_matrix.data <= self.min_similarity] = 0
        sim_matrix.eliminate_zeros()
        self.similarity_ = _top_k_neighbours(sim_matrix, self.top_k)

        # Item-mean-centred residuals reuse the raw rating structure.
        residuals = self.ratings_.copy()
        residuals.data = residuals.data - self.item_mean_[residuals.indices]
        self.residuals_ = residuals
        return self

    def _clip(self, prediction: np.ndarray) -> np.ndarray:
        if self.clip_min is not None or self.clip_max is not None:
            prediction = np.clip(prediction, self.clip_min, self.clip_max)
        return prediction

    def _user_positions(self, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = self.user_index_.reindex(np.asarray(user_ids)).to_numpy()
        found = ~pd.isna(position)
        return np.flatnonzero(found), position[found].astype(int)

    def score_candidates(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "similarity_"):
            raise RuntimeError("Fit the model before predicting.")
        movie_ids = np.asarray(movie_ids)
        candidate_positions = self.movie_index_.reindex(movie_ids).to_numpy()
        scores = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=np.float32)

        rows, positions = self._user_positions(user_ids)
        if rows.size == 0:
            return scores
        known = ~pd.isna(candidate_positions)
        if not known.any():
            return scores
        shard = self.similarity_[:, candidate_positions[known].astype(int)]
        # Similarity-weighted residuals plus the candidate item's training mean.
        numerator = np.asarray((self.residuals_[positions] @ shard).todense())
        weight = np.asarray((self.binary_[positions] @ shard).todense())
        with np.errstate(invalid="ignore", divide="ignore"):
            adjustment = numerator / weight
        adjustment[weight == 0] = np.nan
        item_mean = self.item_mean_[candidate_positions[known].astype(int)]
        values = adjustment + item_mean[None, :]
        scores[np.ix_(rows, np.flatnonzero(known))] = values.astype(np.float32)
        return self._clip(scores)

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "similarity_"):
            raise RuntimeError("Fit the model before predicting.")
        # Score the unique (user, movie) grid once and gather the requested pairs.
        unique_users, user_inverse = np.unique(
            pairs["userId"].to_numpy(), return_inverse=True
        )
        unique_movies, movie_inverse = np.unique(
            pairs["movieId"].to_numpy(), return_inverse=True
        )
        grid = self.score_candidates(unique_users, unique_movies)
        return grid[user_inverse, movie_inverse]

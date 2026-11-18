"""Shared rating-model interface for Phase 2 standalone models (P1.8)."""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class RatingModel(Protocol):
    """Common boundary for the Phase 2 standalone models.

    Fit consumes the approved phase training history from P1.3, keeping the
    original MovieLens ``userId``, ``movieId`` and ``rating`` columns. Ranking
    uses :meth:`score_candidates`, which returns predicted stars for the
    ``(user, movie)`` grid so the shared pipeline (P2.6) can apply the common
    per-user exclusions and ranking rules. :meth:`predict` returns predicted
    stars aligned to arbitrary explicit pairs for RMSE/MAE (P2.7).
    """

    name: str

    def fit(self, history: pd.DataFrame) -> "RatingModel":
        """Fit on the phase training history and return self."""

    def score_candidates(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        """Return dense predicted stars of shape (len(user_ids), len(movie_ids))."""

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return predicted stars aligned to ``pairs`` rows."""


def require_history_columns(history: pd.DataFrame) -> None:
    missing = {"userId", "movieId", "rating"}.difference(history.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")

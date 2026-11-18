"""P2.1 popularity ranking baseline."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models import require_history_columns


@dataclass
class PopularityModel:
    """Rank candidates by their rating count in the phase history.

    Popularity counts are ranking scores, not star predictions, so
    :meth:`predict` is unavailable. Every historical rating contributes to the
    count regardless of star value, matching the approved P2.1 specification.
    """

    name: str = field(default="popularity", init=False)

    def fit(self, history: pd.DataFrame) -> "PopularityModel":
        require_history_columns(history)
        counts = history["movieId"].value_counts()
        self.counts_ = counts
        self.ranking_ = counts.sort_values(
            ascending=False, kind="stable"
        ).index.to_numpy()
        return self

    def score_candidates(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "counts_"):
            raise RuntimeError("Fit the model before predicting.")
        movie_ids = np.asarray(movie_ids)
        scores = self.counts_.reindex(movie_ids).fillna(0.0).to_numpy(dtype=np.float32)
        return np.broadcast_to(scores, (len(user_ids), len(movie_ids))).copy()

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Popularity provides ranking scores, not star-rating predictions."
        )

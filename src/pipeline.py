"""P2.6/P2.7 shared ranking pipeline and standalone metric evaluation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.metrics import (
    catalogue_coverage_at_10,
    less_popular_exposure_at_10,
    less_popular_movie_ids,
    ndcg_at_10,
    recall_at_10,
)

K = 10


@dataclass
class EvaluationCohort:
    """Fixed evaluation inputs shared by every standalone model.

    Holds the approved catalogue (P1.4), the evaluated users with eligible
    observed positives (P1.5), and their per-user positive sets. Candidate
    generation removes each user's previously rated catalogue movies.
    """

    catalogue: pd.DataFrame
    positives: pd.DataFrame
    users: np.ndarray = field(init=False)
    movie_ids: np.ndarray = field(init=False)
    positives_by_user: dict[int, set[int]] = field(init=False)
    _movie_pos: pd.Series = field(init=False, repr=False)
    _user_pos: pd.Series = field(init=False, repr=False)
    less_popular: set[int] = field(init=False)
    _seen: dict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.movie_ids = self.catalogue["movieId"].to_numpy()
        self.users = np.sort(self.positives["userId"].unique())
        self.positives_by_user = (
            self.positives.groupby("userId")["movieId"].agg(set).to_dict()
        )
        self._movie_pos = pd.Series(
            np.arange(len(self.movie_ids)), index=self.movie_ids
        )
        self._user_pos = pd.Series(np.arange(len(self.users)), index=self.users)
        self.less_popular = less_popular_movie_ids(self.catalogue)
        self._seen = {}

    def set_history(self, history: pd.DataFrame) -> None:
        """Precompute each user's rated catalogue positions at the cutoff."""
        catalogue_rated = history.loc[
            history["movieId"].isin(self._movie_pos.index)
        ]
        positions = self._movie_pos.reindex(catalogue_rated["movieId"]).to_numpy()
        user_slot = self._user_pos.reindex(catalogue_rated["userId"]).to_numpy()
        found = ~pd.isna(user_slot)
        for user, position in zip(user_slot[found].astype(int), positions[found]):
            self._seen.setdefault(user, []).append(position)
        self._seen = {u: np.asarray(v, dtype=np.int64) for u, v in self._seen.items()}

    def seen_mask(self) -> np.ndarray:
        mask = np.zeros((len(self.users), len(self.movie_ids)), dtype=bool)
        for user_slot, positions in self._seen.items():
            mask[user_slot, positions] = True
        return mask

    def rank_top_k(self, scores: np.ndarray) -> list[list[int]]:
        """Rank candidates per user, excluding rated movies, and take top K."""
        if scores.shape != (len(self.users), len(self.movie_ids)):
            raise ValueError(
                f"Expected scores of shape {(len(self.users), len(self.movie_ids))}."
            )
        if not self._seen and not len(self.movie_ids):
            raise ValueError("No candidates available.")
        ranked: list[list[int]] = []
        for user_slot in range(len(self.users)):
            row = scores[user_slot].astype(np.float64, copy=True)
            positions = self._seen.get(user_slot)
            if positions is not None:
                row[positions] = -np.inf
            order = np.argsort(-row, kind="stable")[:K]
            # The catalogue is in ascending movieId order, so a stable sort on the
            # negated scores breaks exact ties by ascending movieId.
            ranked.append(self.movie_ids[order].tolist())
        return ranked

    def evaluate(self, ranked_lists: list[list[int]]) -> pd.DataFrame:
        rows = []
        for user, ranked in zip(self.users, ranked_lists):
            relevant = self.positives_by_user[user]
            rows.append(
                {
                    "userId": int(user),
                    "ndcg_at_10": ndcg_at_10(ranked, relevant),
                    "recall_at_10": recall_at_10(ranked, relevant),
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self, ranked_lists: list[list[int]]
    ) -> tuple[dict[str, float], pd.DataFrame]:
        user_metrics = self.evaluate(ranked_lists)
        distinct = len({movie for ranked in ranked_lists for movie in ranked})
        coverage = catalogue_coverage_at_10(ranked_lists, self.movie_ids)
        exposure = less_popular_exposure_at_10(
            ranked_lists, self.movie_ids, self.less_popular
        )
        return {
            "users": len(user_metrics),
            "mean_ndcg_at_10": float(user_metrics["ndcg_at_10"].mean()),
            "mean_recall_at_10": float(user_metrics["recall_at_10"].mean()),
            "coverage_movies": distinct,
            "coverage_fraction": float(coverage),
            "less_popular_exposure": float(exposure),
        }, user_metrics


def score_grid(
    model,
    cohort: EvaluationCohort,
    *,
    fallback=None,
) -> tuple[np.ndarray, int]:
    """Score the full candidate grid, applying the fallback for missing scores."""
    scores = model.score_candidates(cohort.users, cohort.movie_ids)
    missing = ~np.isfinite(scores)
    replaced = int(missing.sum())
    if replaced and fallback is not None:
        fallback_scores = fallback.score_candidates(cohort.users, cohort.movie_ids)
        scores = np.where(missing, fallback_scores, scores)
    return scores, replaced


def recommendations_frame(
    cohort: EvaluationCohort, ranked_lists: list[list[int]]
) -> pd.DataFrame:
    rows = []
    for user, ranked in zip(cohort.users, ranked_lists):
        for rank, movie in enumerate(ranked, start=1):
            rows.append({"userId": int(user), "rank": rank, "movieId": int(movie)})
    return pd.DataFrame(rows)

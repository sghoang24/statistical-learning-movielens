"""P3.1 parallel Item-CF + MF hybrid scoring.

For Item-CF weight ``w`` (0 <= w <= 1) every candidate is scored as

    hybrid(u, i) = w * item_cf(u, i) + (1 - w) * mf(u, i)

Protocol
- Alignment: both components are scored with ``score_grid`` against the same
  ``cohort.users`` and ``cohort.movie_ids`` arrays of original MovieLens IDs.
  Each model maps those IDs through its own index, so cell (r, c) means user
  ``users[r]`` and movie ``movie_ids[c]`` in both grids; internal model
  positions are never combined.
- Fallback: non-finite component scores (e.g. Item-CF with no rated
  neighbour) are filled by the lambda=10/10 bias fallback *before* blending,
  and the filled count is recorded per component. Blending only finite values
  avoids ``0 * NaN = NaN`` at the endpoints.
- Endpoints: with finite inputs, w=0 returns the MF grid and w=1 the filled
  Item-CF grid bit for bit, so they reproduce the standalone rankings.
- Scale: components are blended as raw, unclipped predicted stars. Their
  spreads can differ (Item-CF starts from unshrunk movie means; most Item-CF
  cells come from the fallback); no rescaling is applied.
- Exclusions: rated movies are removed only by ``cohort.rank_top_k`` after
  blending, which also breaks exact ties by ascending movieId.
- Rating metrics: hybrid star predictions blend the same raw, filled grids
  and are clipped to [0.5, 5] once, after blending. Ranking is never clipped.

Component models come from ``src.model_artifacts.load_or_fit_components``.
This module does not choose a weight; P3.2 sweeps weights using
``component_scores`` once and ``blend``/``rank`` per weight.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from src.models import clip_rating
from src.pipeline import EvaluationCohort, score_grid


@dataclass
class ComponentScores:
    """Fallback-filled, finite float64 component grids for one cohort."""

    users: np.ndarray
    movie_ids: np.ndarray
    item_cf: np.ndarray
    mf: np.ndarray
    fallback_counts: dict[str, int]
    # True where Item-CF scored the cell itself, False where the fallback filled
    # it. Taken from the raw Item-CF grid before filling; None if unknown.
    item_cf_direct: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.users = np.asarray(self.users)
        self.movie_ids = np.asarray(self.movie_ids)
        shape = (len(self.users), len(self.movie_ids))
        for name in ("item_cf", "mf"):
            grid = np.asarray(getattr(self, name), dtype=np.float64)
            if grid.shape != shape:
                raise ValueError(f"{name} grid has shape {grid.shape}, expected {shape}.")
            _require_finite(grid, name)
            setattr(self, name, grid)
        if self.item_cf_direct is not None:
            mask = np.asarray(self.item_cf_direct)
            if mask.dtype != bool or mask.shape != shape:
                raise ValueError(f"item_cf_direct must be a boolean {shape} mask.")
            if "item_cf" in self.fallback_counts and (
                int((~mask).sum()) != self.fallback_counts["item_cf"]
            ):
                raise ValueError("item_cf_direct disagrees with the Item-CF fallback count.")
            self.item_cf_direct = mask


def _require_finite(grid: np.ndarray, name: str) -> None:
    bad = ~np.isfinite(grid)
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum()):,} non-finite score(s); fill them with the "
            "fallback before blending."
        )


class _PrecomputedScores:
    """Feeds an already computed raw grid to ``score_grid`` (scores once)."""

    def __init__(self, scores: np.ndarray, users: np.ndarray, movie_ids: np.ndarray):
        self.scores, self.users, self.movie_ids = scores, users, movie_ids

    def score_candidates(self, user_ids, movie_ids):
        if not (np.array_equal(user_ids, self.users) and np.array_equal(movie_ids, self.movie_ids)):
            raise ValueError("Precomputed scores requested for different IDs.")
        return self.scores


def component_scores(item_cf, mf, fallback, cohort: EvaluationCohort) -> ComponentScores:
    """Score both components on the cohort grid and fill gaps with ``fallback``.

    Item-CF is scored once; its direct-score mask is taken from that raw grid
    before ``score_grid`` fills the gaps.
    """
    raw_item_cf = item_cf.score_candidates(cohort.users, cohort.movie_ids)
    item_cf_direct = np.isfinite(raw_item_cf)
    item_cf_grid, item_cf_missing = score_grid(
        _PrecomputedScores(raw_item_cf, cohort.users, cohort.movie_ids), cohort, fallback=fallback
    )
    del raw_item_cf
    mf_grid, mf_missing = score_grid(mf, cohort, fallback=fallback)
    return ComponentScores(
        users=cohort.users,
        movie_ids=cohort.movie_ids,
        item_cf=item_cf_grid,
        mf=mf_grid,
        fallback_counts={"item_cf": int(item_cf_missing), "mf": int(mf_missing)},
        item_cf_direct=item_cf_direct,
    )


def component_scores_from_models(models: dict, cohort: EvaluationCohort) -> ComponentScores:
    """Use the dict returned by ``load_or_fit_components``."""
    return component_scores(
        models["item_cf"], models["matrix_factorization"], models["fallback"], cohort
    )


def check_weight(w) -> float:
    if isinstance(w, bool) or not isinstance(w, Real):
        raise TypeError(f"Item-CF weight must be a real number, got {type(w).__name__}.")
    w = float(w)
    if not (0.0 <= w <= 1.0):  # also rejects NaN
        raise ValueError(f"Item-CF weight must satisfy 0 <= w <= 1, got {w}.")
    return w


def blend(components: ComponentScores, w) -> np.ndarray:
    """Raw hybrid scores ``w * item_cf + (1 - w) * mf`` (float64, unclipped)."""
    w = check_weight(w)
    _require_finite(components.item_cf, "item_cf")
    _require_finite(components.mf, "mf")
    scores = components.item_cf * w
    scores += components.mf * (1.0 - w)
    return scores


def _check_cohort(components: ComponentScores, cohort: EvaluationCohort) -> None:
    if not (
        np.array_equal(components.users, cohort.users)
        and np.array_equal(components.movie_ids, cohort.movie_ids)
    ):
        raise ValueError("Component scores were computed for a different cohort or catalogue.")


def rank(components: ComponentScores, cohort: EvaluationCohort, w) -> list[list[int]]:
    """Top-10 lists for weight ``w`` via the shared ranking path.

    ``cohort.rank_top_k`` excludes each user's rated movies (after blending)
    and breaks exact ties by ascending movieId.
    """
    _check_cohort(components, cohort)
    return cohort.rank_top_k(blend(components, w))


def rating_predictions(components: ComponentScores, w, pairs: pd.DataFrame) -> np.ndarray:
    """Hybrid star predictions for RMSE/MAE, clipped to [0.5, 5] after blending.

    Every pair must lie on the component grid (cohort user, catalogue movie).
    """
    w = check_weight(w)
    user_pos = pd.Series(np.arange(len(components.users)), index=components.users)
    movie_pos = pd.Series(np.arange(len(components.movie_ids)), index=components.movie_ids)
    rows = user_pos.reindex(pairs["userId"].to_numpy()).to_numpy()
    cols = movie_pos.reindex(pairs["movieId"].to_numpy()).to_numpy()
    outside = pd.isna(rows) | pd.isna(cols)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum()):,} rating pair(s) are outside the cohort/catalogue grid."
        )
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    item_cf = components.item_cf[rows, cols]
    mf = components.mf[rows, cols]
    _require_finite(item_cf, "item_cf")
    _require_finite(mf, "mf")
    raw = item_cf * w
    raw += mf * (1.0 - w)
    return clip_rating(raw)

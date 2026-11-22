"""Synthetic checks for src/hybrid.py (P3.1).

Run from the project root with ``python -m src.check_hybrid``. Needs no
MovieLens data and never touches ``artifacts/``:

- Stub models with known, ID-based scores give exact checks of alignment,
  fallback filling, ties, exclusions, clipping and input validation.
- The real selected Item-CF, MF and fallback models are fitted on synthetic
  data through ``load_or_fit_components`` (in a temporary directory) to check
  that w=0 and w=1 reproduce the standalone grids and top-10 lists exactly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import hybrid
from src.model_artifacts import load_or_fit_components
from src.pipeline import EvaluationCohort, score_grid
from src.run_phase2_validation import rating_predictions as standalone_rating_predictions


# --------------------------------------------------------------------------- #
# Stubs: scores are a known function of the original IDs
# --------------------------------------------------------------------------- #


class StubModel:
    """Scores from a {(userId, movieId): score} table; others use ``default``.

    Internally the stub keeps users and movies in *descending* ID order, so a
    caller that mixed up internal positions and original IDs would misalign.
    """

    def __init__(self, name, table, default=np.nan):
        self.name = name
        self.table = dict(table)
        self.default = default
        self.user_order = sorted({u for u, _ in self.table}, reverse=True)
        self.movie_order = sorted({m for _, m in self.table}, reverse=True)

    def score_candidates(self, user_ids, movie_ids):
        out = np.empty((len(user_ids), len(movie_ids)), dtype=np.float64)
        for r, u in enumerate(user_ids):
            for c, m in enumerate(movie_ids):
                out[r, c] = self.table.get((int(u), int(m)), self.default)
        return out


def make_cohort(users, movies, history_pairs, positives=None) -> EvaluationCohort:
    catalogue = pd.DataFrame({
        "movieId": sorted(movies),
        "pre_2017_rating_count": np.arange(len(movies), 0, -1),
    })
    if positives is None:
        positives = [(u, movies[0]) for u in users]
    cohort = EvaluationCohort(catalogue, pd.DataFrame(positives, columns=["userId", "movieId"]))
    history = pd.DataFrame(history_pairs, columns=["userId", "movieId"]).assign(rating=4.0)
    cohort.set_history(history)
    return cohort


def expect(exc, fn, label, contains=""):
    try:
        fn()
    except exc as error:
        assert contains in str(error), (label, str(error))
        print(f"ok  rejected {label}: {str(error)[:90]}")
    else:
        raise AssertionError(f"{label}: expected {exc.__name__}")


def check_alignment_and_fallback() -> None:
    users = [90210, 15, 4401]                 # unsorted, gapped original IDs
    movies = [7, 31, 120, 999, 5000, 12345]
    item_cf_table, mf_table, fb_table = {}, {}, {}
    for u in users:
        for m in movies:
            mf_table[(u, m)] = 1.0 + (u % 97) * 0.01 + (m % 89) * 0.001
            fb_table[(u, m)] = 2.0 + (u % 13) * 0.1 + (m % 7) * 0.01
            if (u + m) % 3:                    # about a third unsupported (NaN)
                item_cf_table[(u, m)] = 3.0 + (u % 31) * 0.01 + (m % 17) * 0.003
    cohort = make_cohort(users, movies, history_pairs=[(15, 999)])
    item_cf = StubModel("item_cf", item_cf_table)
    mf = StubModel("matrix_factorization", mf_table)
    fallback = StubModel("bias", fb_table)

    comp = hybrid.component_scores(item_cf, mf, fallback, cohort)
    n_nan = sum((u, m) not in item_cf_table for u in users for m in movies)
    assert comp.fallback_counts == {"item_cf": n_nan, "mf": 0}, comp.fallback_counts
    assert comp.item_cf.dtype == comp.mf.dtype == np.float64
    assert comp.item_cf.shape == (len(users), len(movies))

    for w in (0.0, 0.25, 0.5, 1.0):
        grid = hybrid.blend(comp, w)
        for r, u in enumerate(cohort.users):
            for c, m in enumerate(cohort.movie_ids):
                cf = item_cf_table.get((int(u), int(m)), fb_table[(int(u), int(m))])
                expected = w * cf + (1 - w) * mf_table[(int(u), int(m))]
                assert grid[r, c] == expected, (w, u, m, grid[r, c], expected)
    print(f"ok  ID alignment: every cell equals w*cf(u,i)+(1-w)*mf(u,i) by original IDs; "
          f"{n_nan} unsupported Item-CF cells filled from the fallback")

    # 0 * NaN: at w=0 the filled cells still give exactly MF.
    assert np.array_equal(hybrid.blend(comp, 0.0), comp.mf)
    print("ok  w=0 is exactly MF even where Item-CF was unsupported (no 0*NaN)")

    # Direct-score mask comes from the raw grid, before filling.
    raw = item_cf.score_candidates(cohort.users, cohort.movie_ids)
    assert np.array_equal(comp.item_cf_direct, np.isfinite(raw))
    assert int((~comp.item_cf_direct).sum()) == comp.fallback_counts["item_cf"]
    bad_mask = comp.item_cf_direct.copy()
    bad_mask[0, 0] = not bad_mask[0, 0]
    expect(ValueError, lambda: hybrid.ComponentScores(
        comp.users, comp.movie_ids, comp.item_cf, comp.mf, comp.fallback_counts, bad_mask),
        "mask disagreeing with the fallback count", "disagrees")
    print("ok  Item-CF direct mask matches the raw grid before filling")


def check_ties_exclusions_clipping() -> None:
    users, movies = [1, 2], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    cf = {(u, m): 1.0 for u in users for m in movies}
    mf = {(u, m): 1.0 for u in users for m in movies}
    # Movies 20 and 30 tie after blending (w=0.5): 0.5*7 + 0.5*3 = 0.5*3 + 0.5*7 = 5.
    cf[(1, 20)], mf[(1, 20)] = 7.0, 3.0
    cf[(1, 30)], mf[(1, 30)] = 3.0, 7.0
    # Movie 10 scores highest for both users but user 2 already rated it.
    cf[(1, 10)] = cf[(2, 10)] = 9.0
    mf[(1, 10)] = mf[(2, 10)] = 9.0
    cohort = make_cohort(users, movies, history_pairs=[(2, 10)])
    comp = hybrid.component_scores(StubModel("c", cf), StubModel("m", mf), StubModel("f", {}), cohort)

    lists = hybrid.rank(comp, cohort, 0.5)
    assert lists[0][:3] == [10, 20, 30], lists[0]
    print("ok  exact tie after blending (movies 20, 30) broken by ascending movieId")
    for w in (0.0, 0.3, 0.5, 1.0):
        lists = hybrid.rank(comp, cohort, w)
        assert 10 not in lists[1] and 10 in lists[0], (w, lists)
        assert all(len(l) == len(set(l)) == 10 for l in lists)
        assert np.isfinite(hybrid.blend(comp, w)).all(), "exclusion must not enter the blend"
    print("ok  rated movie excluded for its user at every weight; blend stays finite")

    raw = hybrid.blend(comp, 0.5)
    assert raw[0, 0] == 9.0 and raw.max() > 5.0, "ranking scores must stay unclipped"
    pairs = pd.DataFrame({"userId": [1, 1, 1], "movieId": [10, 20, 40]})
    pred = hybrid.rating_predictions(comp, 0.5, pairs)
    # 9 -> clipped 5; 0.5*7+0.5*3 = 5; 1 stays 1.
    assert np.array_equal(pred, [5.0, 5.0, 1.0]), pred
    # Clip after blending, not before: cf=7 (would clip to 5), mf=3 -> blend 5, not 4.
    assert pred[1] == 5.0
    low = hybrid.ComponentScores(comp.users, comp.movie_ids, comp.item_cf - 10, comp.mf - 10, {})
    assert np.all(hybrid.rating_predictions(low, 0.5, pairs) == 0.5)
    print("ok  ranking unclipped; rating predictions clipped to [0.5, 5] after blending")


def check_validation() -> None:
    users, movies = [1, 2], [10, 20, 30]
    table = {(u, m): 3.0 for u in users for m in movies}
    cohort = make_cohort(users, movies, history_pairs=[])
    comp = hybrid.component_scores(StubModel("c", table), StubModel("m", table),
                                   StubModel("f", table), cohort)

    for bad in (-0.1, 1.0000001, float("nan"), float("inf")):
        expect(ValueError, lambda: hybrid.blend(comp, bad), f"weight {bad}", "0 <= w <= 1")
    for bad in ("0.5", None, True):
        expect(TypeError, lambda: hybrid.blend(comp, bad), f"weight {bad!r}", "real number")
    assert np.array_equal(hybrid.blend(comp, 1), hybrid.blend(comp, 1.0))  # int accepted

    # Non-finite input is refused, never blended.
    nan_fallback = StubModel("f", {})            # fallback cannot fill -> NaN remains
    expect(ValueError, lambda: hybrid.component_scores(
        StubModel("c", {}), StubModel("m", table), nan_fallback, cohort),
        "unfillable NaN in Item-CF", "non-finite")
    for value in (np.nan, np.inf, -np.inf):
        broken = hybrid.ComponentScores(comp.users, comp.movie_ids, comp.item_cf, comp.mf, {})
        broken.item_cf = broken.item_cf.copy()
        broken.item_cf[0, 1] = value            # mutated after construction
        expect(ValueError, lambda: hybrid.blend(broken, 0.0), f"{value} in item_cf at w=0",
               "non-finite")
    expect(ValueError, lambda: hybrid.ComponentScores(
        comp.users, comp.movie_ids, comp.item_cf[:, :2], comp.mf, {}), "shape mismatch", "shape")

    other = make_cohort([1, 2], [10, 20, 40], history_pairs=[])
    expect(ValueError, lambda: hybrid.rank(comp, other, 0.5), "different catalogue",
           "different cohort")
    pairs = pd.DataFrame({"userId": [1, 3], "movieId": [10, 10]})
    expect(ValueError, lambda: hybrid.rating_predictions(comp, 0.5, pairs),
           "rating pair outside the grid", "outside")
    inside = pd.DataFrame({"userId": [1, 2], "movieId": [10, 20]})
    for name in ("item_cf", "mf"):
        broken = hybrid.ComponentScores(comp.users, comp.movie_ids, comp.item_cf, comp.mf, {})
        grid = getattr(broken, name).copy()
        grid[1, 1] = np.nan                     # mutated after construction
        setattr(broken, name, grid)
        expect(ValueError, lambda: hybrid.rating_predictions(broken, 0.5, inside),
               f"NaN in {name} for rating predictions", "non-finite")


# --------------------------------------------------------------------------- #
# Real selected components fitted on synthetic data
# --------------------------------------------------------------------------- #


def synthetic_data():
    rng = np.random.default_rng(21)
    users = rng.choice(np.arange(1_000, 60_000), 45, replace=False)
    movies = np.sort(rng.choice(np.arange(10, 150_000), 40, replace=False))
    rows = [
        (int(u), int(m), float(rng.choice(np.arange(0.5, 5.01, 0.5))))
        for u in users for m in rng.choice(movies, 12, replace=False)
    ]
    history = pd.DataFrame(rows, columns=["userId", "movieId", "rating"]).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32"})
    # Catalogue: all trained movies plus one never rated (unknown to every model).
    catalogue_ids = np.append(movies, movies.max() + 7)
    catalogue = pd.DataFrame({
        "movieId": catalogue_ids,
        "pre_2017_rating_count": history["movieId"].value_counts().reindex(catalogue_ids, fill_value=0).to_numpy(),
    })
    evaluated = users[:30]
    positives = pd.DataFrame({
        "userId": np.repeat(evaluated, 2),
        "movieId": rng.choice(catalogue_ids, 60),
    }).drop_duplicates()
    cohort = EvaluationCohort(catalogue, positives)
    cohort.set_history(history)
    return history, cohort


def check_endpoints_with_fitted_components() -> None:
    history, cohort = synthetic_data()
    with tempfile.TemporaryDirectory() as tmp:
        models = load_or_fit_components(history, directory=Path(tmp))
    item_cf, mf, fallback = models["item_cf"], models["matrix_factorization"], models["fallback"]
    print(f"    fitted Item-CF top_k={item_cf.top_k}, MF {mf.n_factors}/{mf.regularization}, "
          f"fallback lambda {fallback.lambda_user}/{fallback.lambda_movie}")

    comp = hybrid.component_scores_from_models(models, cohort)
    # Standalone grids exactly as the Phase 2 runner builds them.
    cf_standalone, cf_missing = score_grid(item_cf, cohort, fallback=fallback)
    mf_standalone, mf_missing = score_grid(mf, cohort, fallback=fallback)
    raw_cf = item_cf.score_candidates(cohort.users, cohort.movie_ids)
    assert cf_missing == int(np.isnan(raw_cf).sum()) > 0, "expected unsupported Item-CF pairs"
    assert np.isnan(raw_cf[:, -1]).all(), "never-rated catalogue movie is unsupported in Item-CF"
    assert comp.fallback_counts == {"item_cf": cf_missing, "mf": mf_missing} and mf_missing == 0
    fb = fallback.score_candidates(cohort.users, cohort.movie_ids)
    nan = np.isnan(raw_cf)
    assert np.array_equal(comp.item_cf[nan], fb[nan]), "filled cells must come from the fallback"
    print(f"ok  fitted components: {cf_missing} Item-CF cells filled by the lambda=10 fallback, "
          "MF needed none")

    w0, w1 = hybrid.blend(comp, 0.0), hybrid.blend(comp, 1.0)
    assert np.array_equal(w0, mf_standalone), "w=0 grid differs from standalone MF"
    assert np.array_equal(w1, cf_standalone), "w=1 grid differs from standalone Item-CF"
    assert hybrid.rank(comp, cohort, 0.0) == cohort.rank_top_k(mf_standalone)
    assert hybrid.rank(comp, cohort, 1.0) == cohort.rank_top_k(cf_standalone)
    print("ok  w=0 / w=1 reproduce the standalone MF / Item-CF grids and top-10 lists exactly")

    mid = hybrid.rank(comp, cohort, 0.5)
    seen = cohort.seen_mask()
    for slot, lst in enumerate(mid):
        positions = np.searchsorted(cohort.movie_ids, lst)
        assert not seen[slot, positions].any(), "rated movie in a hybrid list"
    print("ok  w=0.5 lists exclude every user's rated movies")

    # Rating metrics: w=1 / w=0 agree with the Phase 2 standalone rating path.
    pairs = pd.DataFrame({
        "userId": np.repeat(cohort.users[:10], 5),
        "movieId": np.tile(cohort.movie_ids[[0, 3, 9, 17, -1]], 10),
    })
    for w, model in ((1.0, item_cf), (0.0, mf)):
        ours = hybrid.rating_predictions(comp, w, pairs)
        theirs = standalone_rating_predictions(model, fallback, pairs)
        assert ours.min() >= 0.5 and ours.max() <= 5.0
        assert np.allclose(ours, theirs, rtol=0, atol=1e-5), (w, np.abs(ours - theirs).max())
    print("ok  w=1 / w=0 rating predictions match the standalone rating path (atol 1e-5)")


def main() -> None:
    check_alignment_and_fallback()
    check_ties_exclusions_clipping()
    check_validation()
    check_endpoints_with_fitted_components()
    print("hybrid checks passed")


if __name__ == "__main__":
    main()

"""Synthetic checks for the P3.1 endpoint runner and the component-grid cache.

Run from the project root with ``python -m src.check_hybrid_endpoints``. Uses
synthetic data and temporary directories only: it needs no MovieLens data and
never touches ``artifacts/`` or ``outputs/``. The models use the real selected
settings (Item-CF k=3, MF 24/30, fallback lambda 10/10).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import grid_cache as gc
from src import hybrid
from src import model_artifacts as ma
from src import run_hybrid_endpoints as runner
from src.bias import BiasModel
from src.check_hybrid import synthetic_data
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.pipeline import EvaluationCohort, score_grid
from src.splits import VALIDATION_START


def expect(fn, label, contains):
    try:
        fn()
    except gc.GridCacheError as error:
        assert contains in str(error), (label, str(error))
        print(f"ok  rejected stale cache ({label}): {str(error).splitlines()[-1].strip()[:80]}")
    else:
        raise AssertionError(f"{label}: expected GridCacheError")


def same_components(a: hybrid.ComponentScores, b: hybrid.ComponentScores) -> None:
    for name in ("users", "movie_ids", "item_cf", "mf", "item_cf_direct"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and np.array_equal(x, y), name
    assert a.fallback_counts == b.fallback_counts


def write_saved_top10(cohort, models, directory: Path, alter_item_cf_user: bool) -> None:
    """Phase 2-style saved top-10 files built from the same models."""
    for role, file_name in runner.SAVED_TOP10.items():
        scores, _ = score_grid(models[role], cohort, fallback=models["fallback"])
        lists = cohort.rank_top_k(scores)
        if role == "item_cf" and alter_item_cf_user:
            lists[0] = [lists[0][1], lists[0][0], *lists[0][2:]]  # swap two slots
        rows = [(int(u), r + 1, int(m)) for u, l in zip(cohort.users, lists)
                for r, m in enumerate(l)]
        pd.DataFrame(rows, columns=["userId", "rank", "movieId"]).to_csv(
            directory / file_name, index=False)


def check_runner(history, cohort, root: Path) -> dict:
    model_dir, grid_dir, out_dir = root / "models", root / "grids", root / "outputs"
    out_dir.mkdir()
    models = ma.load_or_fit_components(history, directory=model_dir)
    write_saved_top10(cohort, models, out_dir, alter_item_cf_user=True)
    output = out_dir / "phase3_hybrid_endpoint_check.json"
    output.write_text("STALE\n")

    report = runner.run(history, cohort, model_dir=model_dir, grid_dir=grid_dir,
                        output_path=output, saved_top10_dir=out_dir)
    saved = json.loads(output.read_text())
    assert saved == json.loads(json.dumps(report)), "written report differs from returned report"
    assert report["passed_in_process_endpoints"] is True
    for check in report["endpoints"].values():
        assert check["grid_equal"] and check["lists_equal"]
        assert check["fallback_count_matches_components"]
    cmp = report["saved_phase2_comparison"]
    assert cmp["item_cf_vs_w1"]["mismatched_users"] == 1, cmp
    assert cmp["item_cf_vs_w1"]["mismatched_users_same_movies_different_order"] == 1
    assert cmp["matrix_factorization_vs_w0"]["mismatched_users"] == 0, cmp
    assert report["grid_cache"]["status"] == "built"
    print("ok  runner: in-process endpoints exact; saved-file mismatches counted "
          "(1 altered Item-CF user, 0 MF); stale report replaced")

    # Direct share recomputed independently from the raw Item-CF grid.
    raw = models["item_cf"].score_candidates(cohort.users, cohort.movie_ids)
    w1, _ = score_grid(models["item_cf"], cohort, fallback=models["fallback"])
    lists = cohort.rank_top_k(w1)
    direct = sum(
        bool(np.isfinite(raw[r, np.searchsorted(cohort.movie_ids, m)]))
        for r, l in enumerate(lists) for m in l
    )
    share = report["w1_top10_direct_item_cf"]
    assert share["direct_item_cf_slots"] == direct and share["slots"] == 10 * len(cohort.users)
    assert report["fallback_counts"]["item_cf"] == int(np.isnan(raw).sum())
    assert report["model_settings"]["item_cf"]["top_k"] == 3
    for key in ("git_commit", "numpy", "created_utc"):
        assert key in report["provenance"], key
    print(f"ok  direct-slot share {share['direct_share']:.3f} matches an independent count; "
          "fallback counts, settings and provenance recorded")

    # Second run reuses the grid cache.
    report2 = runner.run(history, cohort, model_dir=model_dir, grid_dir=grid_dir,
                         output_path=output, saved_top10_dir=out_dir)
    assert report2["grid_cache"]["status"] == "loaded" and report2["passed_in_process_endpoints"]
    print("ok  runner second run reused the grid cache and still passed")

    # A missing saved file is reported, not fatal.
    (out_dir / runner.SAVED_TOP10["matrix_factorization"]).unlink()
    report3 = runner.run(history, cohort, model_dir=model_dir, grid_dir=grid_dir,
                         output_path=output, saved_top10_dir=out_dir)
    assert report3["saved_phase2_comparison"]["matrix_factorization_vs_w0"]["available"] is False
    print("ok  missing saved Phase 2 file reported as unavailable")
    return models


def check_cache(history, cohort, models, root: Path) -> None:
    model_dir, grid_dir = root / "models", root / "grids"
    path = gc.grid_path(VALIDATION_START, grid_dir)
    built = hybrid.component_scores_from_models(models, cohort)

    # Reload without fitting or scoring: make both impossible.
    blocked = {}
    for cls in (ItemCFModel, MatrixFactorizationModel, BiasModel):
        blocked[cls] = (cls.fit, cls.score_candidates)
        cls.fit = lambda *a, **k: (_ for _ in ()).throw(AssertionError("fit called"))
        cls.score_candidates = lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored"))
    try:
        loaded, loaded_models, status = gc.load_or_build_component_grids(
            history, cohort, model_dir=model_dir, grid_dir=grid_dir)
    finally:
        for cls, (fit, score) in blocked.items():
            cls.fit, cls.score_candidates = fit, score
    assert status == "loaded" and loaded_models is None
    same_components(built, loaded)
    print("ok  cache reload: grids, ordered IDs, direct mask and counts identical; "
          "no fitting or scoring")

    kwargs = dict(model_dir=model_dir, grid_dir=grid_dir)
    pristine = path.read_bytes()

    # Different training history.
    changed = history.copy()
    changed.loc[0, "rating"] = 5.0 if changed.loc[0, "rating"] != 5.0 else 4.0
    expect(lambda: gc.load_or_build_component_grids(changed, cohort, **kwargs),
           "different training history", "components")

    # Different selected settings.
    real_specs = ma.component_specs
    def other_specs(selection=None):
        specs = real_specs(selection)
        cls, params = specs["matrix_factorization"]
        specs["matrix_factorization"] = (cls, {**params, "regularization": 31.0})
        return specs
    ma.component_specs = other_specs
    try:
        expect(lambda: gc.load_or_build_component_grids(history, cohort, **kwargs),
               "different MF settings", "components")
    finally:
        ma.component_specs = real_specs

    # Model artifact replaced (valid file, different fitted values).
    fb_path = ma.artifact_path("fallback", VALIDATION_START, model_dir)
    fb_original = fb_path.read_bytes()
    altered = ma.load_model(fb_path, ma.read_metadata(fb_path), BiasModel)
    altered.bias_movie_ = altered.bias_movie_ + 1e-6
    ma.save_model(altered, fb_path, {k: ma.read_metadata(fb_path)[k] for k in ma.CHECKED_FIELDS})
    expect(lambda: gc.load_or_build_component_grids(history, cohort, **kwargs),
           "fallback model refit/replaced", "has changed")
    fb_path.write_bytes(fb_original)

    # Different cohort (one evaluated user fewer) and different catalogue.
    smaller = EvaluationCohort(cohort.catalogue,
                               cohort.positives[cohort.positives["userId"] != cohort.users[0]])
    smaller.set_history(history)
    expect(lambda: gc.load_or_build_component_grids(history, smaller, **kwargs),
           "different cohort users", "cohort")
    fewer_movies = EvaluationCohort(cohort.catalogue.iloc[:-1], cohort.positives)
    fewer_movies.set_history(history)
    expect(lambda: gc.load_or_build_component_grids(history, fewer_movies, **kwargs),
           "different catalogue", "cohort")

    # Changed scoring code.
    real_hash = ma._source_sha256
    ma._source_sha256 = lambda p: "changed" if Path(p).name == "hybrid.py" else real_hash(p)
    try:
        expect(lambda: gc.load_or_build_component_grids(history, cohort, **kwargs),
               "changed hybrid.py", "code")
    finally:
        ma._source_sha256 = real_hash

    # Damaged: truncated file; array altered with metadata kept.
    path.write_bytes(pristine[: len(pristine) // 3])
    expect(lambda: gc.load_or_build_component_grids(history, cohort, **kwargs),
           "truncated cache", "damaged")
    path.write_bytes(pristine)
    with np.load(path, allow_pickle=False) as archive:
        payload = {k: archive[k] for k in archive.files}
    payload["mf"] = payload["mf"] + 1e-9
    np.savez(path, **payload)
    expect(lambda: gc.load_or_build_component_grids(history, cohort, **kwargs),
           "altered grid values", "content hash")

    # Explicit rebuild replaces the stale cache; the result then loads.
    components, _, status = gc.load_or_build_component_grids(
        history, cohort, rebuild_on_mismatch=True, **kwargs)
    assert status == "rebuilt"
    same_components(built, components)
    again, _, status = gc.load_or_build_component_grids(history, cohort, **kwargs)
    assert status == "loaded"
    same_components(built, again)
    print("ok  rebuild_on_mismatch=True replaces a stale cache; the new cache loads")

    # Interrupted save leaves the previous cache intact.
    before = path.read_bytes()
    real_replace = ma.os.replace
    ma.os.replace = lambda src, dst: (_ for _ in ()).throw(OSError("interrupted"))
    try:
        gc.save_component_grids(built, path, gc.read_grid_metadata(path))
    except OSError:
        pass
    finally:
        ma.os.replace = real_replace
    assert path.read_bytes() == before and not list(path.parent.glob(".*.partial"))
    print("ok  interrupted cache save: previous cache intact, nothing partial")


def main() -> None:
    history, cohort = synthetic_data()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        models = check_runner(history, cohort, root)
        check_cache(history, cohort, models, root)
    print("hybrid endpoint and grid cache checks passed")


if __name__ == "__main__":
    main()

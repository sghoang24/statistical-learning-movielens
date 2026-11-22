"""Synthetic checks for the P3.2 weight sweep (src/run_hybrid_sweep.py).

Run from the project root with ``python -m src.check_hybrid_sweep``. Synthetic
data and temporary directories only; never touches ``artifacts/`` or
``outputs/``. Builds the P3.1 artifacts with the real endpoint runner first,
then checks:

- cache-only execution (fitting, model loading and scoring all disabled);
- every metric row against an independent recomputation;
- endpoint rows against Phase 2-style standalone runs, and the endpoint
  confirmation (including rejection of a tampered or missing P3.1 report);
- the selection rule and its ties;
- output files, provisional labelling and stale-output removal.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import grid_cache as gc
from src import model_artifacts as ma
from src import run_hybrid_endpoints as endpoints
from src import run_hybrid_sweep as sweep
from src.bias import BiasModel
from src.check_hybrid import synthetic_data
from src.check_hybrid_endpoints import write_saved_top10
from src.check_phase2_metrics import independent_metrics
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.metrics import less_popular_movie_ids
from src.pipeline import score_grid


def expect(exc, fn, label, contains):
    try:
        fn()
    except exc as error:
        assert contains in str(error), (label, str(error))
        print(f"ok  rejected {label}: {str(error)[:95]}")
    else:
        raise AssertionError(f"{label}: expected {exc.__name__}")


def check_selection_rule() -> None:
    def rows(values: dict[int, float]):
        base = {t: 0.1 for t in range(11)}
        base.update(values)
        return [{"tenths": t, "mean_ndcg_at_10": v} for t, v in base.items()]

    cases = [
        ("unique maximum", {7: 0.3}, 0.7, [0.7]),
        ("tie 0.4 / 0.6, equally close to 0.5 -> smaller", {4: 0.3, 6: 0.3}, 0.4, [0.4, 0.6]),
        ("tie 0.3 / 0.6 -> closer to 0.5", {3: 0.3, 6: 0.3}, 0.6, [0.3, 0.6]),
        ("tie including 0.5", {2: 0.3, 5: 0.3, 9: 0.3}, 0.5, [0.2, 0.5, 0.9]),
        ("tie 0.1 / 0.9 -> smaller", {1: 0.3, 9: 0.3}, 0.1, [0.1, 0.9]),
        ("endpoints higher than every interior weight", {0: 0.9, 10: 0.9, 8: 0.2}, 0.8, [0.8]),
        ("all interior equal -> 0.5", {}, 0.5, [t / 10 for t in range(1, 10)]),
    ]
    for label, values, want, tied in cases:
        chosen, got_tied = sweep.select_interior(rows(values))
        assert chosen["tenths"] / 10 == want and got_tied == tied, (label, chosen, got_tied)
        print(f"ok  selection: {label} -> w={want}")
    # Full precision: a 1e-15 difference decides, no rounding.
    chosen, tied = sweep.select_interior(rows({5: 0.3, 6: 0.3 + 1e-15}))
    assert chosen["tenths"] == 6 and tied == [0.6]
    print("ok  selection uses full precision (1e-15 difference decides)")
    incomplete = [r for r in rows({}) if r["tenths"] != 4]
    expect(sweep.SweepError, lambda: sweep.select_interior(incomplete),
           "missing interior weight", "missing")


def build_p31(history, cohort, root: Path) -> dict:
    model_dir, grid_dir, out_dir = root / "models", root / "grids", root / "outputs"
    out_dir.mkdir()
    models = ma.load_or_fit_components(history, directory=model_dir)
    write_saved_top10(cohort, models, out_dir, alter_item_cf_user=True)
    endpoints.run(history, cohort, model_dir=model_dir, grid_dir=grid_dir,
                  output_path=out_dir / "phase3_hybrid_endpoint_check.json",
                  saved_top10_dir=out_dir)
    return models


def run_cache_only(history, cohort, root: Path, output_dir: Path, report: Path | None = None):
    """Run the sweep with fitting, model loading and scoring all disabled."""
    blocked = {cls: (cls.fit, cls.score_candidates)
               for cls in (ItemCFModel, MatrixFactorizationModel, BiasModel)}
    real_load_model, real_load_or_fit = ma.load_model, ma.load_or_fit_components

    def boom(*args, **kwargs):
        raise AssertionError("component model fitted, loaded or scored during the sweep")

    for cls in blocked:
        cls.fit = cls.score_candidates = boom
    ma.load_model = ma.load_or_fit_components = boom
    try:
        return sweep.run(
            history, cohort, model_dir=root / "models", grid_dir=root / "grids",
            endpoint_report=report or root / "outputs" / "phase3_hybrid_endpoint_check.json",
            output_dir=output_dir,
        )
    finally:
        for cls, (fit, score) in blocked.items():
            cls.fit, cls.score_candidates = fit, score
        ma.load_model, ma.load_or_fit_components = real_load_model, real_load_or_fit


def check_sweep(history, cohort, models, root: Path) -> None:
    out = root / "sweep"
    out.mkdir()
    for name in sweep.FILES.values():
        (out / name).write_text("STALE\n")
    record = run_cache_only(history, cohort, root, out)
    print("ok  sweep ran from the cache with fitting, model loading and scoring disabled")

    # Outputs and labelling.
    table = pd.read_csv(out / sweep.FILES["table"])
    assert list(table["w"]) == [t / 10 for t in range(11)]
    assert table["selected"].sum() == 1 and not table.loc[table["w"].isin([0.0, 1.0]), "selected"].any()
    assert table.loc[table["selected"], "w"].item() == record["selected_w"]
    assert set(table["role"]) == {"interior", "reference: MF (w=0)", "reference: Item-CF (w=1)"}
    saved = json.loads((out / sweep.FILES["selection"]).read_text())
    assert saved["status"] == "provisional" and "P3.3" in saved["status_note"]
    assert 0.1 <= saved["selected_w"] <= 0.9
    for key in ("content_sha256", "model_artifact_sha256", "component_settings"):
        assert saved["grid_cache"][key], key
    for key in ("git_commit", "numpy"):
        assert key in saved["provenance"], key
    assert all(saved["endpoint_confirmation"]["checks"].values())
    for name in sweep.FILES.values():
        assert (out / name).read_bytes()[:6] != b"STALE\n", f"{name} not replaced"
    assert (out / sweep.FILES["plot"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    users = pd.read_csv(out / sweep.FILES["user_metrics"])
    assert len(users) == 11 * len(cohort.users) and users.groupby("w").size().eq(len(cohort.users)).all()
    print(f"ok  outputs written, stale copies replaced; provisional selection w={saved['selected_w']}")

    # Every metric row vs an independent recomputation from the cached arrays.
    cached = gc.load_existing_component_grids(history, cohort, model_dir=root / "models",
                                              grid_dir=root / "grids")
    raw_cf = models["item_cf"].score_candidates(cohort.users, cohort.movie_ids)
    less_popular = less_popular_movie_ids(cohort.catalogue)
    positives = cohort.positives[["userId", "movieId"]]
    for _, row in table.iterrows():
        w = row["w"]
        scores = w * cached.item_cf + (1 - w) * cached.mf
        lists = cohort.rank_top_k(scores)
        top10 = pd.DataFrame([(int(u), r + 1, int(m)) for u, l in zip(cohort.users, lists)
                              for r, m in enumerate(l)], columns=["userId", "rank", "movieId"])
        per_user, agg = independent_metrics(top10, positives, cohort.users,
                                            len(cohort.movie_ids), less_popular)
        for key in ("mean_ndcg_at_10", "mean_recall_at_10", "coverage_movies",
                    "less_popular_exposure"):
            assert abs(row[key] - agg[key]) <= 1e-12, (w, key, row[key], agg[key])
        direct = sum(bool(np.isfinite(raw_cf[r, np.searchsorted(cohort.movie_ids, m)]))
                     for r, l in enumerate(lists) for m in l)
        assert row["direct_item_cf_slots"] == direct and row["slots"] == 10 * len(cohort.users)
        mine = users[users["w"] == w].set_index("userId").reindex(cohort.users)
        assert np.allclose(mine["ndcg_at_10"], per_user["ndcg_at_10"], rtol=0, atol=1e-12)
    print("ok  all 11 metric rows (NDCG, recall, coverage, exposure, direct share, per-user) "
          "match an independent recomputation")

    # Endpoint rows equal Phase 2-style standalone evaluation.
    for w, role in ((0.0, "matrix_factorization"), (1.0, "item_cf")):
        grid, _ = score_grid(models[role], cohort, fallback=models["fallback"])
        summary, _ = cohort.summary(cohort.rank_top_k(grid))
        row = table[table["w"] == w].iloc[0]
        for key in ("mean_ndcg_at_10", "mean_recall_at_10", "coverage_movies",
                    "less_popular_exposure"):
            assert row[key] == summary[key], (w, key)
    print("ok  w=0 / w=1 rows equal standalone MF / Item-CF evaluation")

    # Comparison with MF.
    cmp = saved["comparison_with_mf_comparator"]
    sel = table[table["selected"]].iloc[0]
    mf = table[table["w"] == 0.0].iloc[0]
    assert abs(cmp["delta_mean_ndcg_at_10"] - (sel["mean_ndcg_at_10"] - mf["mean_ndcg_at_10"])) < 1e-15
    assert cmp["users_ndcg_higher"] + cmp["users_ndcg_lower"] + cmp["users_ndcg_equal"] == len(cohort.users)
    print("ok  comparison with the w=0 MF comparator is consistent with the table")

    # Selected top-10 file equals the selected weight's lists.
    w = saved["selected_w"]
    expected = cohort.rank_top_k(w * cached.item_cf + (1 - w) * cached.mf)
    got = pd.read_csv(out / sweep.FILES["selected_top10"]).sort_values(["userId", "rank"])
    assert got.groupby("userId")["movieId"].agg(list).reindex(cohort.users).tolist() == expected
    print("ok  selected top-10 file matches the selected weight")


def check_endpoint_confirmation(history, cohort, root: Path) -> None:
    report = root / "outputs" / "phase3_hybrid_endpoint_check.json"
    original = report.read_text()
    out = root / "sweep_rejected"

    def tampered(edit, label, contains):
        data = json.loads(original)
        edit(data)
        report.write_text(json.dumps(data))
        expect(sweep.SweepError, lambda: run_cache_only(history, cohort, root, out),
               label, contains)
        assert not (out / sweep.FILES["selection"]).exists(), "selection written despite failure"
        report.write_text(original)

    tampered(lambda d: d["grid_cache"].__setitem__("content_sha256", "0" * 64),
             "endpoint report for a different grid cache", "same_grid_cache")
    tampered(lambda d: d["saved_phase2_comparison"]["item_cf_vs_w1"].__setitem__("mismatched_users", 0),
             "endpoint report whose w=1 lists differ", "item_cf_vs_w1_lists")
    tampered(lambda d: d["w1_top10_direct_item_cf"].__setitem__("direct_item_cf_slots", -1),
             "different w=1 direct-slot count", "same_w1_direct_slots")
    tampered(lambda d: d.__setitem__("passed_in_process_endpoints", False),
             "failed endpoint report", "endpoint_report_passed")
    expect(sweep.SweepError, lambda: run_cache_only(history, cohort, root, out,
                                                    report=root / "missing.json"),
           "missing endpoint report", "not found")


def check_missing_or_stale_cache(history, cohort, root: Path) -> None:
    grid = gc.grid_path(directory=root / "grids")
    kept = grid.read_bytes()
    grid.unlink()
    expect(gc.GridCacheError, lambda: run_cache_only(history, cohort, root, root / "no_cache"),
           "sweep without a grid cache (no rebuild)", "No grid cache")
    assert not grid.exists(), "sweep must not build a cache"
    grid.write_bytes(kept)
    changed = history.copy()
    changed.loc[0, "rating"] = 5.0 if changed.loc[0, "rating"] != 5.0 else 4.0
    expect(gc.GridCacheError, lambda: run_cache_only(changed, cohort, root, root / "stale"),
           "sweep with a stale cache", "stale")


def main() -> None:
    check_selection_rule()
    history, cohort = synthetic_data()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        models = build_p31(history, cohort, root)
        check_sweep(history, cohort, models, root)
        check_endpoint_confirmation(history, cohort, root)
        check_missing_or_stale_cache(history, cohort, root)
    print("weight sweep checks passed")


if __name__ == "__main__":
    main()

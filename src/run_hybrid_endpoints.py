"""P3.1 real-data endpoint check for the Item-CF + MF hybrid (2017 validation).

Loads or fits the saved pre-2017 Item-CF, MF and lambda=10 fallback models,
scores the 2017 cohort once (or reuses a compatible grid cache), and checks:

- w=0 and w=1 hybrid grids and top-10 lists equal the in-process standalone
  MF and Item-CF grids and lists, built independently with ``score_grid`` and
  ``cohort.rank_top_k`` exactly as in Phase 2 (exact agreement required);
- how many users' w=0 / w=1 lists differ from the saved Phase 2 top-10 files
  (reported only; an earlier process may differ through refits or
  floating-point near-ties);
- the share of w=1 top-10 slots scored directly by Item-CF rather than the
  fallback, from the mask taken before fallback filling.

Writes ``outputs/phase3_hybrid_endpoint_check.json`` and saves the component
grids to ``artifacts/grids/`` for the P3.2 sweep. It does not select or sweep
weights and uses no 2018-2019 data.

Run from the project root:
    python -m src.run_hybrid_endpoints [--rebuild-grids] [--refit-models]
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src import grid_cache, hybrid
from src import model_artifacts as ma
from src.pipeline import score_grid
from src.splits import VALIDATION_START

OUTPUT_PATH = Path("outputs/phase3_hybrid_endpoint_check.json")
SAVED_TOP10_DIR = Path("outputs")
SAVED_TOP10 = {
    "item_cf": "phase2_validation_item_cf_top10.csv",
    "matrix_factorization": "phase2_validation_matrix_factorization_top10.csv",
}


def compare_with_saved(lists: list[list[int]], users: np.ndarray, path: Path) -> dict:
    """Users whose current list differs from a saved Phase 2 top-10 file."""
    if not path.is_file():
        return {"file": path.as_posix(), "available": False}
    saved = pd.read_csv(path).sort_values(["userId", "rank"])
    saved_lists = saved.groupby("userId")["movieId"].agg(list).to_dict()
    ordered = same_set = missing = 0
    for user, current in zip(users, lists):
        previous = saved_lists.get(int(user))
        if previous is None:
            missing += 1
        elif [int(m) for m in previous] != [int(m) for m in current]:
            ordered += 1
            same_set += set(previous) == set(current)
    return {
        "file": path.as_posix(),
        "available": True,
        "users_compared": int(len(users) - missing),
        "users_missing_from_file": int(missing),
        "users_in_file_not_in_cohort": int(len(set(saved_lists) - {int(u) for u in users})),
        "mismatched_users": int(ordered),
        "mismatched_users_same_movies_different_order": int(same_set),
    }


def direct_slot_share(lists, components: hybrid.ComponentScores) -> dict:
    """Share of returned slots whose Item-CF score was direct (not fallback)."""
    total = direct = 0
    for row, movies in enumerate(lists):
        positions = np.searchsorted(components.movie_ids, movies)
        direct += int(components.item_cf_direct[row, positions].sum())
        total += len(movies)
    return {"slots": total, "direct_item_cf_slots": direct,
            "direct_share": direct / total if total else None}


def _endpoint(components, cohort, w, model, fallback) -> tuple[dict, list[list[int]]]:
    """Compare one endpoint with an independently built standalone grid."""
    start = perf_counter()
    reference, missing = score_grid(model, cohort, fallback=fallback)
    blended = hybrid.blend(components, w)
    grid_equal = bool(np.array_equal(blended, reference))
    max_abs = float(np.max(np.abs(blended - reference))) if not grid_equal else 0.0
    del blended
    reference_lists = cohort.rank_top_k(reference)
    del reference
    hybrid_lists = hybrid.rank(components, cohort, w)
    mismatched = sum(a != b for a, b in zip(hybrid_lists, reference_lists))
    return {
        "w": w,
        "standalone_fallback_filled": int(missing),
        "grid_equal": grid_equal,
        "grid_max_abs_difference": max_abs,
        "lists_equal": mismatched == 0,
        "mismatched_users_vs_in_process": int(mismatched),
        "seconds": round(perf_counter() - start, 2),
    }, hybrid_lists


def _write_json_atomically(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    history: pd.DataFrame,
    cohort,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
    grid_dir: Path = grid_cache.GRID_DIR,
    output_path: Path = OUTPUT_PATH,
    saved_top10_dir: Path = SAVED_TOP10_DIR,
    rebuild_grids: bool = False,
    refit_models: bool = False,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()  # never leave an earlier report looking current
        print(f"Removed previous report {output_path}")
    started = perf_counter()

    components, models, grid_status = grid_cache.load_or_build_component_grids(
        history, cohort, cutoff=cutoff, model_dir=model_dir, grid_dir=grid_dir,
        rebuild_on_mismatch=rebuild_grids, refit_on_mismatch=refit_models,
    )
    if models is None:  # grids came from the cache; load models for the references
        models = ma.load_or_fit_components(history, cutoff=cutoff, directory=model_dir,
                                           refit_on_mismatch=refit_models)
    fallback = models["fallback"]

    item_cf_check, w1_lists = _endpoint(components, cohort, 1.0, models["item_cf"], fallback)
    mf_check, w0_lists = _endpoint(components, cohort, 0.0, models["matrix_factorization"], fallback)
    for check, role in ((item_cf_check, "item_cf"), (mf_check, "matrix_factorization")):
        check["fallback_count_matches_components"] = (
            check["standalone_fallback_filled"]
            == components.fallback_counts["item_cf" if role == "item_cf" else "mf"]
        )

    saved = {
        "item_cf_vs_w1": compare_with_saved(w1_lists, cohort.users,
                                            Path(saved_top10_dir) / SAVED_TOP10["item_cf"]),
        "matrix_factorization_vs_w0": compare_with_saved(
            w0_lists, cohort.users, Path(saved_top10_dir) / SAVED_TOP10["matrix_factorization"]),
    }
    direct = direct_slot_share(w1_lists, components)
    n_cells = components.item_cf.size

    passed = all(
        c["grid_equal"] and c["lists_equal"] and c["fallback_count_matches_components"]
        for c in (item_cf_check, mf_check)
    )
    path = grid_cache.grid_path(cutoff, grid_dir)
    report = {
        "task": "P3.1 hybrid endpoint check (2017 validation)",
        "passed_in_process_endpoints": passed,
        "endpoints": {"w1_item_cf": item_cf_check, "w0_matrix_factorization": mf_check},
        "saved_phase2_comparison": saved,
        "w1_top10_direct_item_cf": direct,
        "fallback_counts": {
            **components.fallback_counts,
            "grid_cells": int(n_cells),
            "item_cf_fallback_share": components.fallback_counts["item_cf"] / n_cells,
        },
        "model_settings": {role: ma.model_settings(m) for role, m in models.items()},
        "model_artifacts": {
            role: {
                "path": ma.artifact_path(role, cutoff, model_dir).as_posix(),
                "content_sha256": ma.read_metadata(ma.artifact_path(role, cutoff, model_dir))
                .get("content_sha256"),
            }
            for role in models
        },
        "grid_cache": {
            "path": path.as_posix(),
            "status": grid_status,
            "content_sha256": grid_cache.read_grid_metadata(path).get("content_sha256"),
        },
        "training_cutoff_utc": int(cutoff),
        "training_history": ma.history_fingerprint(history),
        "cohort": grid_cache.cohort_identity(cohort.users, cohort.movie_ids),
        "provenance": ma._informational(),
        "run_completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": round(perf_counter() - started, 1),
    }
    _write_json_atomically(output_path, report)

    print(f"\nEndpoint check {'PASSED' if passed else 'FAILED'} -> {output_path}")
    for name, check in report["endpoints"].items():
        print(f"  {name}: grid equal {check['grid_equal']}, lists equal {check['lists_equal']} "
              f"({check['mismatched_users_vs_in_process']} users differ)")
    for name, comparison in saved.items():
        if comparison["available"]:
            print(f"  saved Phase 2 {name}: {comparison['mismatched_users']} of "
                  f"{comparison['users_compared']} users differ")
        else:
            print(f"  saved Phase 2 {name}: file not found ({comparison['file']})")
    print(f"  Item-CF fallback-filled cells: {components.fallback_counts['item_cf']:,} of "
          f"{n_cells:,}; w=1 top-10 slots scored directly by Item-CF: "
          f"{direct['direct_item_cf_slots']:,} of {direct['slots']:,} ({direct['direct_share']:.2%})")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-grids", action="store_true",
                        help="replace a stale or damaged grid cache instead of stopping")
    parser.add_argument("--refit-models", action="store_true",
                        help="refit and replace stale or damaged model artifacts instead of stopping")
    args = parser.parse_args(argv)

    from src.run_phase2_validation import build_cohort, load_validation_history

    history = load_validation_history()
    print(f"Validation history rows: {len(history):,}")
    cohort = build_cohort(history)
    report = run(history, cohort, rebuild_grids=args.rebuild_grids, refit_models=args.refit_models)
    return 0 if report["passed_in_process_endpoints"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""P3.2 validation weight sweep for the Item-CF + MF hybrid (2017 cohort).

Evaluates Item-CF weights w = 0.0, 0.1, ..., 1.0 from the cached P3.1
component grids only: no component model is fitted, loaded or scored. w=0
(MF) and w=1 (Item-CF) are references; the provisional choice is the interior
weight (0.1-0.9) with the highest mean NDCG@10 at full precision. Exact ties go
to the weight closest to 0.5, then to the smaller weight (P1.6).

Ranking uses ``hybrid.rank`` (shared exclusions and ascending-movieId tie
break) and metrics use ``EvaluationCohort.summary``. Before selecting, the
sweep confirms its w=0 and w=1 lists against the saved P3.1 endpoint check:
same grid cache, cohort, training rows, fallback counts and direct-slot count,
and the same mismatch counts against the Phase 2 top-10 files that check used.

Outputs (in ``outputs/``; earlier copies are removed first):
- phase3_weight_sweep.csv: one row per weight;
- phase3_weight_sweep_user_metrics.csv: per-user NDCG@10 / Recall@10 per weight;
- phase3_selected_hybrid_top10.csv: top-10 lists for the selected weight;
- phase3_weight_sweep.png: NDCG@10 and coverage against w;
- phase3_provisional_selection.json: the provisional selection, written last.

The selection is provisional: P3.3 freezes it after review. No 2018-2019 data
is used. Run from the project root: python -m src.run_hybrid_sweep
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src import grid_cache, hybrid
from src import model_artifacts as ma
from src.run_hybrid_endpoints import (
    OUTPUT_PATH as ENDPOINT_REPORT,
    _write_json_atomically,
    compare_with_saved,
    direct_slot_share,
)
from src.splits import VALIDATION_START

OUTPUT_DIR = Path("outputs")
WEIGHT_TENTHS = tuple(range(11))          # w = tenths / 10, so 0.3 is exactly 3/10
INTERIOR_TENTHS = tuple(range(1, 10))
FILES = {
    "table": "phase3_weight_sweep.csv",
    "user_metrics": "phase3_weight_sweep_user_metrics.csv",
    "selected_top10": "phase3_selected_hybrid_top10.csv",
    "plot": "phase3_weight_sweep.png",
    "selection": "phase3_provisional_selection.json",
}
SELECTION_RULE = (
    "Highest mean 2017 validation NDCG@10 among interior weights 0.1-0.9 at full precision; "
    "exact ties go to the weight closest to 0.5, then to the smaller weight. "
    "w=0 (MF) and w=1 (Item-CF) are references and cannot be selected."
)


class SweepError(RuntimeError):
    """The sweep cannot proceed safely (e.g. endpoint confirmation failed)."""


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_interior(rows: list[dict]) -> tuple[dict, list[float]]:
    """Return (selected row, weights tied at the best NDCG) per the P1.6 rule.

    Rows carry ``tenths`` (int) and ``mean_ndcg_at_10``. Distance to 0.5 is
    computed in integer tenths, so no floating-point rounding can break a tie.
    """
    interior = {r["tenths"]: r for r in rows if r["tenths"] in INTERIOR_TENTHS}
    missing = sorted(set(INTERIOR_TENTHS) - set(interior))
    if missing:
        raise SweepError(f"Interior weights missing from the sweep: {[t / 10 for t in missing]}")
    best = max(r["mean_ndcg_at_10"] for r in interior.values())
    tied = sorted(t for t, r in interior.items() if r["mean_ndcg_at_10"] == best)
    chosen = min(tied, key=lambda t: (abs(t - 5), t))
    return interior[chosen], [t / 10 for t in tied]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_weight(components, cohort, tenths: int):
    start = perf_counter()
    w = tenths / 10
    lists = hybrid.rank(components, cohort, w)
    summary, user_metrics = cohort.summary(lists)
    direct = direct_slot_share(lists, components)
    role = {0: "reference: MF (w=0)", 10: "reference: Item-CF (w=1)"}.get(tenths, "interior")
    row = {
        "w": w,
        "tenths": tenths,
        "role": role,
        "users": summary["users"],
        "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
        "mean_recall_at_10": summary["mean_recall_at_10"],
        "coverage_movies": summary["coverage_movies"],
        "coverage_fraction": summary["coverage_fraction"],
        "less_popular_exposure": summary["less_popular_exposure"],
        "direct_item_cf_slots": direct["direct_item_cf_slots"],
        "slots": direct["slots"],
        "direct_item_cf_slot_share": direct["direct_share"],
        "seconds": round(perf_counter() - start, 2),
    }
    return row, user_metrics.assign(w=w), lists


def _hits(lists, cohort, k: int) -> np.ndarray:
    return np.array([
        len(set(lst[:k]) & cohort.positives_by_user[user])
        for user, lst in zip(cohort.users, lists)
    ])


def compare_with_mf(selected, mf, cohort) -> dict:
    """Selected interior weight versus the w=0 MF comparator (validation only)."""
    (s_row, s_users, s_lists), (m_row, m_users, m_lists) = selected, mf
    diff = s_users["ndcg_at_10"].to_numpy() - m_users["ndcg_at_10"].to_numpy()
    out = {
        "comparator": "w=0 (matrix factorization)",
        "selected_w": s_row["w"],
        "delta_mean_ndcg_at_10": s_row["mean_ndcg_at_10"] - m_row["mean_ndcg_at_10"],
        "delta_mean_recall_at_10": s_row["mean_recall_at_10"] - m_row["mean_recall_at_10"],
        "delta_coverage_movies": s_row["coverage_movies"] - m_row["coverage_movies"],
        "delta_less_popular_exposure": s_row["less_popular_exposure"] - m_row["less_popular_exposure"],
        "users_ndcg_higher": int((diff > 0).sum()),
        "users_ndcg_lower": int((diff < 0).sum()),
        "users_ndcg_equal": int((diff == 0).sum()),
        "mean_shared_top10_movies": float(np.mean(
            [len(set(a) & set(b)) for a, b in zip(s_lists, m_lists)])),
    }
    for k in (3, 10):
        s_hits, m_hits = _hits(s_lists, cohort, k), _hits(m_lists, cohort, k)
        out[f"total_hits_top{k}"] = {"selected": int(s_hits.sum()), "mf": int(m_hits.sum())}
        out[f"users_more_hits_top{k}"] = int((s_hits > m_hits).sum())
        out[f"users_fewer_hits_top{k}"] = int((s_hits < m_hits).sum())
    return out


# --------------------------------------------------------------------------- #
# Endpoint confirmation against the saved P3.1 check
# --------------------------------------------------------------------------- #


def confirm_endpoints(report_path: Path, cache_path: Path, components, cohort, history_fp,
                      lists_w0, lists_w1, w1_direct_slots: int) -> dict:
    report_path = Path(report_path)
    if not report_path.is_file():
        raise SweepError(f"P3.1 endpoint report not found at {report_path}; "
                         "run `python -m src.run_hybrid_endpoints` first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache_sha = grid_cache.read_grid_metadata(cache_path).get("content_sha256")
    checks = {
        "endpoint_report_passed": report.get("passed_in_process_endpoints") is True,
        "same_grid_cache": report.get("grid_cache", {}).get("content_sha256") == cache_sha,
        "same_cohort": report.get("cohort")
        == grid_cache.cohort_identity(cohort.users, cohort.movie_ids),
        "same_training_history": report.get("training_history") == history_fp,
        "same_fallback_counts": all(
            report.get("fallback_counts", {}).get(k) == v
            for k, v in components.fallback_counts.items()
        ),
        "same_w1_direct_slots": report.get("w1_top10_direct_item_cf", {})
        .get("direct_item_cf_slots") == w1_direct_slots,
    }
    saved = report.get("saved_phase2_comparison", {})
    for key, lists in (("item_cf_vs_w1", lists_w1), ("matrix_factorization_vs_w0", lists_w0)):
        recorded = saved.get(key, {})
        if not recorded.get("available"):
            checks[f"{key}_lists"] = False
            continue
        now = compare_with_saved(lists, cohort.users, Path(recorded["file"]))
        checks[f"{key}_lists"] = bool(now.get("available")) and all(
            now[field] == recorded[field]
            for field in ("users_compared", "mismatched_users",
                          "mismatched_users_same_movies_different_order")
        )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SweepError(f"w=0/w=1 do not match the saved P3.1 endpoint check: {failed}")
    with open(report_path, "rb") as stream:
        report_sha = hashlib.sha256(stream.read()).hexdigest()
    return {"report": report_path.as_posix(), "report_sha256": report_sha, "checks": checks}


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = "#2a78d6"  # categorical slot 1 (single series per panel)


def plot_sweep(table: pd.DataFrame, selected_w: float, path: Path, n_users: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.28})
    # (column, axis label, point-label format, tick format)
    panels = (("mean_ndcg_at_10", "Mean NDCG@10", "{:.4f}", "{:.3f}"),
              ("coverage_movies", "Catalogue coverage (distinct movies)", "{:,.0f}", "{:,.0f}"))
    interior = table[table["role"] == "interior"]
    ends = table[table["role"] != "interior"]
    chosen = table[table["w"] == selected_w].iloc[0]
    for ax, (column, label, fmt, tick_fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(table["w"], table[column], color=SERIES, linewidth=2, zorder=2)
        ax.scatter(interior["w"], interior[column], s=36, color=SERIES, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
        ax.scatter(ends["w"], ends[column], s=56, facecolors=SURFACE, edgecolors=SERIES,
                   linewidths=2, zorder=3)
        ax.scatter([chosen["w"]], [chosen[column]], s=110, color=SERIES, zorder=4,
                   edgecolors=INK, linewidths=1.5)
        span = table[column].max() - table[column].min() or 1.0
        ax.annotate(f"w={chosen['w']:.1f}: {fmt.format(chosen[column])}",
                    (chosen["w"], chosen[column]), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
        for _, end in ends.iterrows():
            name = "MF" if end["w"] == 0 else "Item-CF"
            # Put the label on the side away from the neighbouring segment.
            neighbour = table.loc[table["w"] == (0.1 if end["w"] == 0 else 0.9), column].item()
            above = neighbour < end[column]
            ax.annotate(f"{name}\n{fmt.format(end[column])}", (end["w"], end[column]),
                        xytext=(8 if end["w"] == 0 else -8, 8 if above else -8),
                        textcoords="offset points", ha="left" if end["w"] == 0 else "right",
                        va="bottom" if above else "top", fontsize=8, color=INK_2)
        ax.set_ylim(table[column].min() - 0.22 * span, table[column].max() + 0.22 * span)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _, f=tick_fmt: f.format(v)))
        ax.set_ylabel(label, color=INK_2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
    axes[1].set_xticks([t / 10 for t in WEIGHT_TENTHS])
    axes[1].set_xlabel("Item-CF weight w   (hybrid = w · Item-CF + (1 − w) · MF)",
                       color=INK_2, fontsize=9)
    fig.suptitle("P3.2 weight sweep, 2017 validation: relevance vs. catalogue coverage",
                 color=INK, fontsize=11, x=0.08, ha="left", y=0.97)
    fig.text(0.08, 0.915,
             f"{n_users:,} users. Hollow points: endpoints (references, not selectable). "
             f"Ringed point: provisional selection.",
             color=MUTED, fontsize=8, ha="left")
    fig.subplots_adjust(top=0.86, left=0.12, right=0.96, bottom=0.1)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run(
    history: pd.DataFrame,
    cohort,
    *,
    cutoff: int = VALIDATION_START,
    model_dir: Path = ma.ARTIFACT_DIR,
    grid_dir: Path = grid_cache.GRID_DIR,
    endpoint_report: Path = ENDPOINT_REPORT,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        if (output_dir / name).exists():
            (output_dir / name).unlink()
            print(f"Removed previous output {output_dir / name}")
    started = perf_counter()

    components = grid_cache.load_existing_component_grids(
        history, cohort, cutoff=cutoff, model_dir=model_dir, grid_dir=grid_dir)
    cache_path = grid_cache.grid_path(cutoff, grid_dir)
    history_fp = ma.history_fingerprint(history)

    results = {}

    def evaluate(tenths: int) -> None:
        results[tenths] = evaluate_weight(components, cohort, tenths)
        row = results[tenths][0]
        print(f"  w={row['w']:.1f}  NDCG@10 {row['mean_ndcg_at_10']:.6f}  "
              f"Recall@10 {row['mean_recall_at_10']:.6f}  coverage {row['coverage_movies']:,}  "
              f"less-popular {row['less_popular_exposure']:.4f}  "
              f"direct Item-CF {row['direct_item_cf_slot_share']:.3f}  ({row['seconds']}s)")

    # Endpoints first: confirm them against the P3.1 check before the interior.
    evaluate(0)
    evaluate(10)
    endpoint = confirm_endpoints(
        endpoint_report, cache_path, components, cohort, history_fp,
        lists_w0=results[0][2], lists_w1=results[10][2],
        w1_direct_slots=results[10][0]["direct_item_cf_slots"],
    )
    print("  endpoints confirmed against the saved P3.1 check")
    for tenths in INTERIOR_TENTHS:
        evaluate(tenths)
    rows = [results[t][0] for t in WEIGHT_TENTHS]
    selected_row, tied = select_interior(rows)
    s = selected_row["tenths"]
    comparison = compare_with_mf(results[s], results[0], cohort)

    table = pd.DataFrame(rows)
    table["selected"] = table["tenths"] == s
    table.drop(columns="tenths").to_csv(output_dir / FILES["table"], index=False)
    pd.concat([results[t][1] for t in WEIGHT_TENTHS])[
        ["w", "userId", "ndcg_at_10", "recall_at_10"]
    ].to_csv(output_dir / FILES["user_metrics"], index=False)
    pd.DataFrame(
        [(int(u), r + 1, int(m)) for u, lst in zip(cohort.users, results[s][2])
         for r, m in enumerate(lst)],
        columns=["userId", "rank", "movieId"],
    ).to_csv(output_dir / FILES["selected_top10"], index=False)
    plot_sweep(table, selected_row["w"], output_dir / FILES["plot"], len(cohort.users))

    cache_meta = grid_cache.read_grid_metadata(cache_path)
    record = {
        "task": "P3.2 validation weight sweep",
        "status": "provisional",
        "status_note": "Provisional validation selection. P3.3 freezes the weight after "
                       "team review; it must not be changed after any 2018-2019 scoring.",
        "selected_w": selected_row["w"],
        "selected_metrics": {k: selected_row[k] for k in (
            "mean_ndcg_at_10", "mean_recall_at_10", "coverage_movies", "coverage_fraction",
            "less_popular_exposure", "direct_item_cf_slot_share")},
        "selection_rule": SELECTION_RULE,
        "weights_tied_at_best_ndcg": tied,
        "weights_evaluated": [t / 10 for t in WEIGHT_TENTHS],
        "comparison_with_mf_comparator": comparison,
        "endpoint_confirmation": endpoint,
        "grid_cache": {
            "path": cache_path.as_posix(),
            "content_sha256": cache_meta.get("content_sha256"),
            "model_artifact_sha256": cache_meta.get("model_artifact_sha256"),
            "component_settings": {
                role: meta["settings"] for role, meta in cache_meta["components"].items()
            },
            "fallback_counts": cache_meta.get("fallback_counts"),
        },
        "training_cutoff_utc": int(cutoff),
        "training_history": history_fp,
        "cohort": grid_cache.cohort_identity(cohort.users, cohort.movie_ids),
        "outputs": {key: (output_dir / name).as_posix() for key, name in FILES.items()},
        "provenance": ma._informational(),
        "run_completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": round(perf_counter() - started, 1),
    }
    _write_json_atomically(output_dir / FILES["selection"], record)

    print(f"\nProvisional selection: w={selected_row['w']:.1f} "
          f"(NDCG@10 {selected_row['mean_ndcg_at_10']:.6f}; tied weights {tied})")
    print(f"  vs MF (w=0): delta NDCG@10 {comparison['delta_mean_ndcg_at_10']:+.6f}, "
          f"delta coverage {comparison['delta_coverage_movies']:+,} movies")
    print(f"  endpoint confirmation passed; outputs in {output_dir}")
    return record


def main() -> int:
    from src.run_phase2_validation import build_cohort, load_validation_history

    history = load_validation_history()
    print(f"Validation history rows: {len(history):,}")
    cohort = build_cohort(history)
    run(history, cohort)
    return 0


if __name__ == "__main__":
    sys.exit(main())

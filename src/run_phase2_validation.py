"""P2.6-P2.8 shared runner: fit standalone models, rank, score, and time them."""

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from src.bias import BiasModel
from src.metrics import mae, rmse
from src.models import RATING_MAX, RATING_MIN, clip_rating
from src.pipeline import (
    EvaluationCohort,
    recommendations_frame,
    score_grid,
)
from src.popularity import PopularityModel
from src.ratings import load_ratings
from src.splits import TEST_START, VALIDATION_START
from src.tune_models import FALLBACK_SETTINGS, MODEL_FACTORIES

PROCESSED = Path("data/processed")
OUTPUT_DIR = Path("outputs")

CATALOGUE_PATH = PROCESSED / "eligible_movies.csv"
POSITIVES_PATH = PROCESSED / "validation_positives.csv"

# P2.5 selection across the original and extension grids (src.tune_extension).
SELECTED_CONFIG_PATH = OUTPUT_DIR / "phase2_selected_config_extended.json"

# Settings each personalized model must take from the selection file. Other
# constructor arguments keep their defaults, exactly as in src.tune_models.
REQUIRED_SETTINGS = {
    "bias": ("lambda_user", "lambda_movie"),
    "item_cf": ("top_k", "min_similarity"),
    "matrix_factorization": ("n_factors", "regularization"),
}

PROCESS = psutil.Process()


def load_selected_config(path: Path | None = None) -> dict[str, dict]:
    """Read the P2.5 selection and check every required setting is present."""
    path = SELECTED_CONFIG_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Selected configuration file not found: {path}. "
            "Run `python -m src.tune_extension` to create it."
        )
    selection = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for model, keys in REQUIRED_SETTINGS.items():
        params = selection.get(model)
        if not isinstance(params, dict):
            problems.append(f"missing model '{model}'")
            continue
        missing = [key for key in keys if key not in params]
        unexpected = sorted(set(params).difference(keys))
        if missing:
            problems.append(f"'{model}' is missing {missing}")
        if unexpected:
            problems.append(f"'{model}' has unexpected settings {unexpected}")
    if problems:
        raise ValueError(f"Invalid selected configuration in {path}: " + "; ".join(problems))
    return {model: dict(selection[model]) for model in REQUIRED_SETTINGS}


def build_model_factories(selection: dict[str, dict]) -> dict:
    """Popularity plus each personalized model built as in src.tune_models."""
    factories = {"popularity": PopularityModel}
    for model in REQUIRED_SETTINGS:
        factories[model] = partial(MODEL_FACTORIES[model], **selection[model])
    return factories


def output_paths(model_names) -> list[Path]:
    """Every file this runner writes, so stale results can be cleared first."""
    paths = [
        OUTPUT_DIR / "phase2_validation_summary.csv",
        OUTPUT_DIR / "phase2_validation_environment.json",
    ]
    for name in model_names:
        paths.append(OUTPUT_DIR / f"phase2_validation_{name}_user_metrics.csv")
        paths.append(OUTPUT_DIR / f"phase2_validation_{name}_top10.csv")
    return paths


def git_state() -> dict[str, object]:
    """Best-effort code version for the experiment log (plan section 10.3)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_uncommitted_changes": None}
    return {"git_commit": commit, "git_uncommitted_changes": dirty}


def build_cohort(history: pd.DataFrame) -> EvaluationCohort:
    catalogue = pd.read_csv(CATALOGUE_PATH)
    positives = pd.read_csv(POSITIVES_PATH)
    cohort = EvaluationCohort(catalogue, positives)
    cohort.set_history(history)
    return cohort


def rss_mb() -> float:
    """Resident set size of this process in megabytes."""
    return PROCESS.memory_info().rss / (1024**2)


def load_validation_history() -> pd.DataFrame:
    ratings = load_ratings()
    history = ratings.loc[
        ratings["timestamp"] < VALIDATION_START, ["userId", "movieId", "rating"]
    ].copy()
    del ratings
    return history


def load_rating_pairs(cohort: EvaluationCohort) -> pd.DataFrame:
    """Observed 2017 ratings on eligible catalogue movies for evaluated users.

    These are the shared rating-prediction pairs for RMSE/MAE (P2.7). Only users
    in the evaluation cohort and catalogue movies are kept, so every rating model
    is scored on the same pairs.
    """
    ratings = load_ratings()
    timestamp = ratings["timestamp"]
    outcomes = ratings.loc[
        (timestamp >= VALIDATION_START) & (timestamp < TEST_START),
        ["userId", "movieId", "rating"],
    ]
    del ratings
    outcomes = outcomes.loc[
        outcomes["userId"].isin(cohort.users)
        & outcomes["movieId"].isin(cohort.movie_ids)
    ]
    return outcomes.drop_duplicates(["userId", "movieId"]).reset_index(drop=True)


def rating_predictions(model, fallback, pairs: pd.DataFrame) -> np.ndarray:
    """Star predictions for RMSE/MAE: fallback-filled, then clipped to [0.5, 5].

    Unsupported (non-finite) predictions take the fallback's value. The final
    clip covers model and fallback values alike; ranking never uses this path.
    """
    predicted = np.asarray(model.predict(pairs), dtype=np.float64)
    if not np.isfinite(predicted).all():
        fallback_predicted = fallback.predict(pairs)
        predicted = np.where(np.isfinite(predicted), predicted, fallback_predicted)
    return clip_rating(predicted)


def evaluate_all(
    history: pd.DataFrame,
    rating_pairs: pd.DataFrame,
    model_factories: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[list[int]]]]:
    cohort = build_cohort(history)
    # Same fallback as P2.5 tuning, used for both ranking scores and rating pairs.
    fallback = BiasModel(**FALLBACK_SETTINGS).fit(history)

    summary_rows = []
    user_frames: dict[str, pd.DataFrame] = {}
    ranked_lists_by_model: dict[str, list[list[int]]] = {}
    for name, factory in model_factories.items():
        model = factory()
        peak_before = rss_mb()
        fit_start = perf_counter()
        model = model.fit(history)
        fit_seconds = perf_counter() - fit_start

        start = perf_counter()
        scores, missing = score_grid(model, cohort, fallback=fallback)
        score_seconds = perf_counter() - start
        peak_mb = rss_mb()
        ranked_lists = cohort.rank_top_k(scores)
        summary, user_metrics = cohort.summary(ranked_lists)
        user_frames[name] = user_metrics
        ranked_lists_by_model[name] = ranked_lists

        row = {
            "model": name,
            "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
            "mean_recall_at_10": summary["mean_recall_at_10"],
            "coverage_movies": summary["coverage_movies"],
            "coverage_fraction": summary["coverage_fraction"],
            "less_popular_exposure": summary["less_popular_exposure"],
            "missing_scores": missing,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "peak_rss_mb": peak_mb,
            "rss_delta_mb": peak_mb - peak_before,
        }
        if hasattr(model, "predict"):
            try:
                predicted = rating_predictions(model, fallback, rating_pairs)
                actual = rating_pairs["rating"].to_numpy(dtype=np.float64)
                row["rmse"] = rmse(actual, predicted)
                row["mae"] = mae(actual, predicted)
            except NotImplementedError:
                pass
        summary_rows.append(row)
        print(
            f"{name:>22} | NDCG@10 {summary['mean_ndcg_at_10']:.6f} "
            f"| Recall@10 {summary['mean_recall_at_10']:.6f} "
            f"| cov {summary['coverage_movies']:,} "
            f"| fit {fit_seconds:.1f}s score {score_seconds:.1f}s "
            f"| rss {peak_mb:.0f}MB"
        )

    return pd.DataFrame(summary_rows), user_frames, ranked_lists_by_model


def main() -> None:
    # Validate the selection before any data loading or fitting.
    config_path = SELECTED_CONFIG_PATH
    config_bytes = config_path.read_bytes() if config_path.is_file() else None
    selection = load_selected_config(config_path)
    model_factories = build_model_factories(selection)
    print(f"Selected configurations from {config_path}:")
    print(json.dumps(selection, indent=2, sort_keys=True))
    print(f"Fallback for non-finite scores: bias {FALLBACK_SETTINGS}")

    # Remove earlier results first so a failed run cannot leave old benchmark
    # files that look like they came from this selection.
    for path in output_paths(model_factories):
        if path.exists():
            path.unlink()
            print(f"Removed previous output {path}")

    history = load_validation_history()
    print(f"Validation history rows: {len(history):,}")
    cohort = build_cohort(history)
    rating_pairs = load_rating_pairs(cohort)
    print(f"Rating-prediction pairs: {len(rating_pairs):,}")

    summary, user_frames, ranked_lists = evaluate_all(
        history, rating_pairs, model_factories
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "phase2_validation_summary.csv", index=False)
    for name, frame in user_frames.items():
        frame.to_csv(
            OUTPUT_DIR / f"phase2_validation_{name}_user_metrics.csv", index=False
        )
    for name, lists in ranked_lists.items():
        recommendations_frame(cohort, lists).to_csv(
            OUTPUT_DIR / f"phase2_validation_{name}_top10.csv", index=False
        )

    metadata = {
        "phase": "validation",
        "cutoff": "2017-01-01 UTC",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_memory_mb": round(psutil.virtual_memory().total / (1024**2), 1),
        "evaluated_users": int(len(next(iter(user_frames.values())))),
        "rating_pairs": int(len(rating_pairs)),
        "models": list(model_factories),
        "run_completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "selected_config_file": config_path.as_posix(),
        "selected_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "selected_configurations": selection,
        # Full constructor settings, including defaults (passes, tolerance,
        # clipping, seed), for each model as run.
        "model_settings": {
            name: dataclasses.asdict(factory())
            for name, factory in model_factories.items()
        },
        "fallback": {"model": "bias", **FALLBACK_SETTINGS},
        # RMSE/MAE use predictions clipped to this range; ranking is unclipped.
        "rating_metric_clip": [RATING_MIN, RATING_MAX],
    }
    with (OUTPUT_DIR / "phase2_validation_environment.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print("\nSaved outputs/phase2_validation_summary.csv")


if __name__ == "__main__":
    main()

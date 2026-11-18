"""P2.6-P2.8 shared runner: fit standalone models, rank, score, and time them."""

import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from src.bias import BiasModel
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.metrics import mae, rmse
from src.pipeline import (
    EvaluationCohort,
    recommendations_frame,
    score_grid,
)
from src.popularity import PopularityModel
from src.ratings import load_ratings
from src.splits import TEST_START, VALIDATION_START

PROCESSED = Path("data/processed")
OUTPUT_DIR = Path("outputs")

CATALOGUE_PATH = PROCESSED / "eligible_movies.csv"
POSITIVES_PATH = PROCESSED / "validation_positives.csv"

PROCESS = psutil.Process()

# Configurations frozen from the P2.5 validation selection
# (outputs/phase2_selected_config.json).
DEFAULT_MODELS = {
    "popularity": lambda: PopularityModel(),
    "bias": lambda: BiasModel(lambda_user=20.0, lambda_movie=20.0, max_passes=20),
    "item_cf": lambda: ItemCFModel(top_k=50, min_similarity=0.0),
    "matrix_factorization": lambda: MatrixFactorizationModel(
        n_factors=16, regularization=15.0, lambda_user=10.0, lambda_movie=10.0
    ),
}


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


def evaluate_all(
    history: pd.DataFrame,
    rating_pairs: pd.DataFrame,
    model_factories: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[list[int]]]]:
    cohort = build_cohort(history)
    fallback = BiasModel(lambda_user=20.0, lambda_movie=20.0, max_passes=20).fit(
        history
    )

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
                predicted = model.predict(rating_pairs)
                if not np.isfinite(predicted).all():
                    fallback_predicted = fallback.predict(rating_pairs)
                    predicted = np.where(
                        np.isfinite(predicted), predicted, fallback_predicted
                    )
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
    history = load_validation_history()
    print(f"Validation history rows: {len(history):,}")
    cohort = build_cohort(history)
    rating_pairs = load_rating_pairs(cohort)
    print(f"Rating-prediction pairs: {len(rating_pairs):,}")

    summary, user_frames, ranked_lists = evaluate_all(
        history, rating_pairs, DEFAULT_MODELS
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
        "models": list(DEFAULT_MODELS),
    }
    with (OUTPUT_DIR / "phase2_validation_environment.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print("\nSaved outputs/phase2_validation_summary.csv")


if __name__ == "__main__":
    main()

"""P2.5 comparable tuning budgets and validation selection for standalone models.

Runs a documented hyperparameter grid per personalized model, selects the
configuration with the highest mean 2017 validation NDCG@10, and writes the
full validation table. Selection uses 2017 validation only; no test-period
information is used. The grid and evaluation cohort match P2.6/P2.7.
"""

import json
from pathlib import Path

import pandas as pd

from src.bias import BiasModel
from src.item_cf import ItemCFModel
from src.matrix_factorization import MatrixFactorizationModel
from src.pipeline import EvaluationCohort, score_grid
from src.ratings import load_ratings
from src.splits import VALIDATION_START

PROCESSED = Path("data/processed")
OUTPUT_DIR = Path("outputs")

# Comparable search effort: each personalized model explores the same number of
# documented settings (three) for its principal capacity/regularization choice on
# 2017 validation. Bias and MF fix the second bias regularization at the shipped
# default so the grid length stays matched across models.
GRIDS = {
    "bias": [{"lambda_user": value, "lambda_movie": value} for value in (5.0, 10.0, 20.0, 40.0, 80.0)],
    "item_cf": [{"top_k": k, "min_similarity": 0.0} for k in (12, 25, 50, 100, 200)],
    "matrix_factorization": [
        {"n_factors": f, "regularization": r}
        for f, r in ((16, 15.0), (24, 15.0), (24, 10.0), (8, 15.0), (8, 30.0), (16, 30.0), (16, 45.0), (24, 30.0),)
    ],
}

MODEL_FACTORIES = {
    "bias": BiasModel,
    "item_cf": ItemCFModel,
    "matrix_factorization": MatrixFactorizationModel,
}

# Bias model that fills non-finite scores (P1.6 fallback). Shared with the
# P2.6 runner so tuning and benchmark runs use the same fallback; it is
# deliberately separate from the tuned standalone bias model.
FALLBACK_SETTINGS = {"lambda_user": 10.0, "lambda_movie": 10.0, "max_passes": 20}


def load_history() -> pd.DataFrame:
    ratings = load_ratings()
    history = ratings.loc[
        ratings["timestamp"] < VALIDATION_START, ["userId", "movieId", "rating"]
    ].copy()
    del ratings
    return history


def build_cohort(history: pd.DataFrame) -> EvaluationCohort:
    catalogue = pd.read_csv(PROCESSED / "eligible_movies.csv")
    positives = pd.read_csv(PROCESSED / "validation_positives.csv")
    cohort = EvaluationCohort(catalogue, positives)
    cohort.set_history(history)
    return cohort


def tune(history: pd.DataFrame,
         grids: dict[str, list[dict]] | None = None,
         ) -> pd.DataFrame:
    if grids is None:
        grids = GRIDS
    cohort = build_cohort(history)
    fallback = BiasModel(**FALLBACK_SETTINGS).fit(history)
    rows = []
    for model_name, grid in grids.items():
        factory = MODEL_FACTORIES[model_name]
        for params in grid:
            model = factory(**params).fit(history)
            scores, missing = score_grid(model, cohort, fallback=fallback)
            ranked = cohort.rank_top_k(scores)
            summary, _ = cohort.summary(ranked)
            rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, sort_keys=True),
                    "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
                    "mean_recall_at_10": summary["mean_recall_at_10"],
                    "coverage_movies": summary["coverage_movies"],
                    "less_popular_exposure": summary["less_popular_exposure"],
                    "missing_scores": missing,
                }
            )
            print(
                f"{model_name:>22} {params} -> "
                f"NDCG@10 {summary['mean_ndcg_at_10']:.6f} "
                f"cov {summary['coverage_movies']:,}",
                flush=True,
            )
    return pd.DataFrame(rows)


def select(table: pd.DataFrame) -> dict[str, dict]:
    selection = {}
    for model_name, group in table.groupby("model"):
        best = group.sort_values(
            ["mean_ndcg_at_10", "coverage_movies"], ascending=[False, False]
        ).iloc[0]
        selection[model_name] = json.loads(best["params"])
    return selection


def main() -> None:
    history = load_history()
    table = tune(history)
    selection = select(table)
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "phase2_tuning_grid.csv", index=False)
    with (OUTPUT_DIR / "phase2_selected_config.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print("\nSelected configurations:")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

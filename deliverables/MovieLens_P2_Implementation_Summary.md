# MovieLens 25M — Phase 2 Implementation Summary

This document summarizes **what was implemented** in Phase 2 (standalone models
and validation benchmarks). Benchmark numbers and the tuning grid live in
`deliverables/MovieLens_P2_Standalone_Benchmarks.md`; this file focuses on the
code, interfaces and how the pieces fit together.

## Modules added in Phase 2

| Module                         | Lines | Purpose                                                                   |
| ------------------------------ | ----: | ------------------------------------------------------------------------- |
| `src/models.py`                |    38 | Shared `RatingModel` protocol and history-column validation.              |
| `src/popularity.py`            |    43 | P2.1 popularity ranking baseline.                                         |
| `src/bias.py`                  |   112 | P2.2 regularized global/user/movie bias model.                            |
| `src/item_cf.py`               |   175 | P2.3 adjusted-cosine item-based CF with a top-K neighbourhood.            |
| `src/matrix_factorization.py`  |   185 | P2.4 explicit-feedback ALS with biases.                                   |
| `src/pipeline.py`              |   144 | P2.6/P2.7 shared cohort, candidate exclusion, ranking and metric summary. |
| `src/run_phase2_validation.py` |   197 | P2.6–P2.8 runner: fit, rank, score, time and measure.                     |
| `src/tune_models.py`           |   119 | P2.5 comparable tuning grids and validation selection.                    |
| `src/check_phase2.py`          |   133 | Synthetic self-checks for every model and the pipeline.                   |

No existing Phase 1 module was redesigned; the models and pipeline consume the
frozen P1.3–P1.5 protocol outputs unchanged.

## 1. Shared model interface (`src/models.py`)

- `RatingModel` — a `runtime_checkable` `typing.Protocol` with `name`, `fit(history)`,
  `score_candidates(user_ids, movie_ids)` and `predict(pairs)`.
- `require_history_columns(history)` — asserts the `userId`, `movieId`, `rating`
  schema on every `fit`.
- Contract: `score_candidates` returns a dense `(len(user_ids), len(movie_ids))`
  grid of predicted stars for ranking; `predict` returns stars aligned to explicit
  pairs for RMSE/MAE.

## 2. Models

### P2.1 Popularity (`src/popularity.py`)

- Ranks candidates by pre-2017 rating count, regardless of star value.
- `score_candidates` broadcasts the count; `predict` raises `NotImplementedError`
  because popularity produces ranking scores, not star predictions.

### P2.2 Regularized bias (`src/bias.py`)

- Predicts `mu + b_u + b_i`.
- Alternating updates: `b_u = sum(r - mu - b_i) / (count_u + lambda_u)`, then the
  symmetric movie update, iterated to a tolerance or pass cap.
- Missing user/movie biases default to zero. Selected config: lambda 20, 20 passes.

### P2.3 Item-CF (`src/item_cf.py`)

- Adjusted-cosine similarity from user-mean-centred ratings, positive neighbours
  only, reduced to each item's top-K rows (`_top_k_neighbours`, `argpartition`).
- Prediction blends the candidate item mean with similarity-weighted item-mean
  residuals: `pred(u,i) = b_i + Σ s_ij (r_uj − b_j) / Σ |s_ij|`.
- Items with no positively-similar rated neighbour return `NaN`, so the shared
  fallback policy (below) can fill them. Selected config: top_k 50.

### P2.4 Matrix factorization (`src/matrix_factorization.py`)

- Explicit-feedback ALS with global, user and movie biases; per-row
  `np.linalg.solve` over the latent Gram matrix (`_solve_rows`).
- Convergence reported via training RMSE with a tolerance stop.
- Selected config: n_factors 16, regularization 15, lambda_bias 10, 20 passes.

### Ranking vs. rating output

- `score_candidates` returns **unclipped** raw predictions for ranking, so scores
  do not saturate into ties.
- `predict` clips to the observed star range [1, 5] for RMSE/MAE.
- This split was essential: clipping inside ranking pushed the ascending-`movieId`
  tie-break onto the oldest obscure titles and collapsed MF's NDCG@10.

## 3. Shared evaluation pipeline (`src/pipeline.py`)

- `EvaluationCohort(catalogue, positives)` — fixed inputs shared by every model:
  the 17,965-movie catalogue (P1.4) and the user-to-positive map (P1.5).
- `set_history(history)` — precomputes each user's rated catalogue positions.
- `rank_top_k(scores)` — sets rated movies to `-inf`, negates scores, and uses a
  **stable** argsort so exact ties break by ascending `movieId` (the catalogue is
  stored in ascending ID order).
- `summary(ranked_lists)` — returns NDCG@10, Recall@10, coverage and less-popular
  exposure via `src/metrics.py`, plus a per-user metrics frame.
- `score_grid(model, cohort, fallback)` — scores the full grid and replaces
  non-finite entries with the bias fallback (P1.6 policy), reporting the count.
- `recommendations_frame(cohort, ranked_lists)` — flattens top-10 lists to a CSV.

## 4. P2.5 tuning (`src/tune_models.py`)

- Three documented settings per personalized model (matched search effort),
  evaluated on 2017 validation only:
  - bias: lambda 5/10/20 (equal on both sides),
  - item_cf: top_k 50/100/200,
  - MF: (16, reg 15), (24, reg 15), (24, reg 10).
- Selection: highest mean NDCG@10, ties broken by coverage.
- Writes `outputs/phase2_tuning_grid.csv` and `outputs/phase2_selected_config.json`.

## 5. P2.6–P2.8 runner (`src/run_phase2_validation.py`)

- Loads pre-2017 history and the shared 275,061-pair 2017 rating set.
- For each model: fits (timed), scores the grid (timed), ranks, summarises.
- Records fit/score seconds and peak RSS via `psutil`; computes RMSE/MAE with the
  same fallback for unsupported predictions.
- `DEFAULT_MODELS` holds the **P2.5-frozen** configurations.
- Writes `outputs/phase2_validation_summary.csv`, per-model user metrics and
  top-10 CSVs, and `phase2_validation_environment.json` (Python, OS, CPU/RAM).

## 6. Self-checks (`src/check_phase2.py`)

Deterministic synthetic tests, runnable without the full archive:

- popularity ranking + `predict` raises,
- bias beats the global-mean RMSE,
- item-CF similarity on co-liked items and `NaN` for unknown users,
- MF recovers low-rank synthetic data,
- pipeline excludes already-rated movies from top-10.

Run with `python -m src.check_phase2`; all checks pass.

## 7. How it fits the Phase 1 protocol

- Same catalogue, cohort, exclusions and tie-break as P1.3–P1.5.
- Popularity reproduces the published P1.5 reference exactly
  (NDCG@10 0.075212, coverage 287), confirming the shared pipeline is faithful.
- The bias model doubles as the P1.6 fallback for unsupported predictions.

## 8. Outputs produced

| File                                                 | Contents                                                 |
| ---------------------------------------------------- | -------------------------------------------------------- |
| `outputs/phase2_validation_summary.csv`              | One row per model: ranking, rating and resource metrics. |
| `outputs/phase2_validation_<model>_user_metrics.csv` | Per-user NDCG@10 / Recall@10.                            |
| `outputs/phase2_validation_<model>_top10.csv`        | Archived top-10 lists per user.                          |
| `outputs/phase2_validation_environment.json`         | Run metadata (P2.8).                                     |
| `outputs/phase2_tuning_grid.csv`                     | Full P2.5 grid.                                          |
| `outputs/phase2_selected_config.json`                | P2.5-selected settings.                                  |

## 9. Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets      # P1.7 datasets
.\.venv\Scripts\python.exe -m src.check_phase2          # self-checks
.\.venv\Scripts\python.exe -m src.tune_models           # P2.5 grids + selection
.\.venv\Scripts\python.exe -m src.run_phase2_validation # P2.6-P2.8 benchmarks
```

## 10. Status

- Implemented and locally verified: P2.1–P2.8.
- Remaining: **P2.9** independent reproduction and integration sign-off, plus
  carrying the MF comparator into the Phase 3 hybrid.

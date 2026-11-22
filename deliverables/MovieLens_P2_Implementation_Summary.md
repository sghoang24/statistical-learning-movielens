# MovieLens 25M — Phase 2 Implementation Summary

This document summarizes **what was implemented** in Phase 2 (standalone models
and validation benchmarks). Benchmark numbers and the tuning grid live in
`deliverables/MovieLens_P2_Standalone_Benchmarks.md`; this file focuses on the
code, interfaces and how the pieces fit together.

## Modules added in Phase 2

| Module                         | Lines | Purpose                                                                    |
| ------------------------------ | ----: | -------------------------------------------------------------------------- |
| `src/models.py`                |    52 | Shared `RatingModel` protocol, rating-range clip and history validation.   |
| `src/popularity.py`            |    43 | P2.1 popularity ranking baseline.                                          |
| `src/bias.py`                  |   105 | P2.2 regularized global/user/movie bias model.                             |
| `src/item_cf.py`               |   170 | P2.3 adjusted-cosine item-based CF with a top-K neighbourhood.             |
| `src/matrix_factorization.py`  |   185 | P2.4 explicit-feedback ALS with biases.                                    |
| `src/pipeline.py`              |   144 | P2.6/P2.7 shared cohort, candidate exclusion, ranking and metric summary.  |
| `src/run_phase2_validation.py` |   302 | P2.6–P2.8 runner: load selection, fit, rank, score, time and record.       |
| `src/tune_models.py`           |   126 | P2.5 base tuning grid, validation selection and shared fallback settings.  |
| `src/tune_extension.py`        |    90 | P2.5 targeted grid extensions combined with the base grid.                 |
| `src/check_phase2.py`          |   217 | Synthetic self-checks for every model, the pipeline and rating clipping.   |

No existing Phase 1 module was redesigned; the models and pipeline consume the
frozen P1.3–P1.5 protocol outputs unchanged.

## 1. Shared model interface (`src/models.py`)

- `RatingModel` — a `runtime_checkable` `typing.Protocol` with `name`, `fit(history)`,
  `score_candidates(user_ids, movie_ids)` and `predict(pairs)`.
- `require_history_columns(history)` — asserts the `userId`, `movieId`, `rating`
  schema on every `fit`.
- `clip_rating(prediction)` — clips star predictions to the MovieLens 25M rating
  range `[RATING_MIN, RATING_MAX] = [0.5, 5.0]`; `NaN` passes through so the
  fallback can still fill it.
- Contract: `score_candidates` returns a dense `(len(user_ids), len(movie_ids))`
  grid of **unclipped** predicted stars for ranking; `predict` returns stars
  aligned to explicit pairs, **clipped** to [0.5, 5], for RMSE/MAE.

## 2. Models

### P2.1 Popularity (`src/popularity.py`)

- Ranks candidates by pre-2017 rating count, regardless of star value.
- `score_candidates` broadcasts the count; `predict` raises `NotImplementedError`
  because popularity produces ranking scores, not star predictions.

### P2.2 Regularized bias (`src/bias.py`)

- Predicts `mu + b_u + b_i`.
- Alternating updates: `b_u = sum(r - mu - b_i) / (count_u + lambda_u)`, then the
  symmetric movie update, iterated to a tolerance (1e-6 on `b_u`) or a 20-pass cap.
- Missing user/movie biases default to zero.
- Selected and benchmarked configuration: **λ_user = λ_movie = 320**.

### P2.3 Item-CF (`src/item_cf.py`)

- Adjusted-cosine similarity from user-mean-centred ratings, positive neighbours
  only, reduced to each item's top-K neighbours (`_top_k_neighbours`, `argpartition`).
- Scoring uses each **candidate item's own** top-K row (`similarity_.T[:, candidates]`).
  The original benchmark read the matrix the other way round, scoring a candidate
  from the items that listed it among *their* top K; all current results use the
  corrected orientation.
- Prediction blends the candidate item mean with similarity-weighted item-mean
  residuals: `pred(u,i) = b_i + Σ s_ij (r_uj − b_j) / Σ |s_ij|`.
- Items with no positively-similar rated neighbour return `NaN`, so the shared
  fallback policy (below) can fill them.
- Selected and benchmarked configuration: **top_k = 3**, `min_similarity = 0`.
  With k=3, 95.1% of the validation score grid is filled by the fallback
  (see the benchmark record).

### P2.4 Matrix factorization (`src/matrix_factorization.py`)

- Explicit-feedback ALS with global, user and movie biases; per-row
  `np.linalg.solve` over the latent Gram matrix (`_solve_rows`).
- Fitting stops after 20 iterations or when the training RMSE changes by less
  than 1e-4. That training-RMSE monitor clips to [1, 5]
  (`monitor_clip_min`/`monitor_clip_max`); this was kept unchanged so the fitted
  factors match the tuning runs. It does not affect RMSE/MAE reporting.
- Selected and benchmarked configuration: **n_factors = 24, regularization = 30**,
  bias regularization λ_user = λ_movie = 10, seed 0.

### Ranking vs. rating output

- `score_candidates` returns **unclipped** raw predictions for ranking, so scores
  do not saturate into ties. This includes Item-CF.
- `predict` clips to [0.5, 5.0] for every rating model, and the runner clips
  again after fallback filling (`rating_predictions`), so fallback-filled values
  are always in range.
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

## 4. P2.5 tuning (`src/tune_models.py`, `src/tune_extension.py`)

- All settings were evaluated on the same 2017 validation cohort using pre-2017
  training history. Selection maximizes mean NDCG@10; coverage breaks exact ties.
  The targeted extensions produced unequal search budgets (bias 7, Item-CF 9,
  MF 8), so the earlier claim of three matched trials per model no longer applies.
- Bias tested equal user/movie regularization at 5, 10, 20, 40, 80, 160 and 320.
  Best: **λ_user = λ_movie = 320**, NDCG@10 **0.052504**, coverage **154**.
  This remains the upper edge of the explored grid; λ=160 scored 0.048267.
- Item-CF tested `top_k` 3, 4, 6, 8, 12, 25, 50, 100 and 200 with
  `min_similarity=0`. Best: **k=3**, NDCG@10 **0.059745**, coverage **4,627**.
  Larger k increases coverage but reduces NDCG in these runs; k=6 scores
  0.052409 with coverage 5,371. k=3 is the lower edge of the explored grid.
- MF tested (factors, regularization) (8,15), (8,30), (16,15), (16,30),
  (16,45), (24,10), (24,15) and (24,30). Best: **24 factors, reg 30**,
  NDCG@10 **0.095484**, coverage **1,182**. The (16,15) alternative scores
  0.095399 with coverage 1,506; its near-equal NDCG and greater coverage are
  useful context for the Phase 3 hybrid comparison.
- Tuning and the runner share one fallback definition,
  `tune_models.FALLBACK_SETTINGS` = bias λ_user = λ_movie = 10, 20 passes. It is
  distinct from the tuned standalone bias model (320/320). The reported Item-CF
  ranking therefore includes fallback scores; its coverage and NDCG are
  properties of this combined scoring policy.
- `src.tune_extension` combines the base grid with the extension probes and
  writes `outputs/phase2_selected_config_extended.json`. That file is the
  current selection; `outputs/phase2_selected_config.json` holds only the base
  grid's earlier choice (bias 80/80, Item-CF k=12, MF 24/30).

## 5. P2.6–P2.8 runner (`src/run_phase2_validation.py`)

- Loads the selection from `outputs/phase2_selected_config_extended.json` and
  fails before loading data if the file, a model or a required setting is
  missing, or if an unexpected setting is present. Models are built exactly as
  in tuning (`tune_models.MODEL_FACTORIES` with the selected settings and the
  remaining constructor defaults).
- Removes its own previous output files before fitting, so a failed run cannot
  leave earlier results that look like they came from the current selection.
- Loads pre-2017 history and the shared 275,061-pair 2017 rating set.
- For each model: fits (timed), scores the grid (timed), ranks, summarises.
  Missing ranking scores and missing rating predictions both use the same
  λ=10/10 fallback as tuning; RMSE/MAE use predictions clipped to [0.5, 5].
- Records fit/score seconds and a single RSS reading after scoring via `psutil`
  (not a per-model peak; see the benchmark record).
- Writes `outputs/phase2_validation_summary.csv`, per-model user metrics and
  top-10 CSVs, and finally `phase2_validation_environment.json`. The
  environment file records Python, OS, CPU/RAM, the selection file and its
  SHA-256, the selected settings, every model's full constructor settings
  (passes, tolerance, seed, MF monitor clip), the fallback settings, the
  rating-metric clip range, the git commit, an uncommitted-changes flag and the
  completion time.

## 6. Self-checks (`src/check_phase2.py`)

Deterministic synthetic tests, runnable without the full archive:

- popularity ranking + `predict` raises,
- bias beats the global-mean RMSE,
- item-CF similarity on co-liked items and `NaN` for unknown users,
- MF recovers low-rank synthetic data,
- pipeline excludes already-rated movies from top-10,
- rating clipping: for Bias, Item-CF and MF, out-of-range raw scores stay
  unclipped in `score_candidates` (and keep their ranking order), `predict`
  returns exactly 0.5 / 5.0, unsupported Item-CF predictions stay `NaN`, and the
  runner clips fallback-filled values.

Run with `python -m src.check_phase2`; all checks pass.

## 7. How it fits the Phase 1 protocol

- Same catalogue, cohort, exclusions and tie-break as P1.3–P1.5.
- Popularity reproduces the published P1.5 reference exactly
  (NDCG@10 0.075212, coverage 287), confirming the shared pipeline is faithful.
- The bias model doubles as the P1.6 fallback for unsupported predictions, with
  its own fixed settings (λ=10/10) separate from the tuned bias baseline.

## 8. Outputs produced

| File                                                 | Contents                                                              |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| `outputs/phase2_validation_summary.csv`              | One row per model: ranking, rating and resource metrics.              |
| `outputs/phase2_validation_<model>_user_metrics.csv` | Per-user NDCG@10 / Recall@10.                                         |
| `outputs/phase2_validation_<model>_top10.csv`        | Archived top-10 lists per user.                                       |
| `outputs/phase2_validation_environment.json`         | Run record: environment, settings, fallback, clip range, code version. |
| `outputs/phase2_tuning_grid.csv`                     | Corrected base grid.                                                  |
| `outputs/phase2_tuning_extension.csv`                | Extension probes run separately.                                      |
| `outputs/phase2_tuning_grid_extended.csv`            | Base and extension rows combined.                                     |
| `outputs/phase2_selected_config.json`                | Base-grid selection only; superseded.                                 |
| `outputs/phase2_selected_config_extended.json`       | Current selection, read by the runner.                                |

## 9. Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets      # P1.7 datasets
.\.venv\Scripts\python.exe -m src.check_phase2          # self-checks
.\.venv\Scripts\python.exe -m src.tune_models           # base P2.5 grid
.\.venv\Scripts\python.exe -m src.tune_extension        # targeted extensions + combined selection
.\.venv\Scripts\python.exe -m src.run_phase2_validation # P2.6-P2.8 benchmarks from the combined selection
```

## 10. Status

- P2.1–P2.8 have been run end to end with the current selections; the full
  validation run (22 November 2026) reproduced the tuning rows exactly. Results
  are in `deliverables/MovieLens_P2_Standalone_Benchmarks.md`.
- The run used uncommitted code on top of commit `2ddd23c`. Commit the code and
  outputs together so the run record identifies the exact version.
- Remaining: **P2.9** independent reproduction and integration sign-off, MF
  convergence evidence (iteration count and training RMSE are computed but not
  saved), and carrying the MF comparator into the Phase 3 hybrid.

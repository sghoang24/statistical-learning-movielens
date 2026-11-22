# MovieLens 25M Phase 2 Standalone Model Benchmarks

**Status — Full validation benchmark complete for the current selections (run 22 November 2026).** The Item-CF neighbour lookup was corrected and the tuning grids extended after the original benchmark. The full validation runner was then rerun with the selected configurations and the tuning fallback. Its ranking results match the tuning rows exactly, and it supplies the current rating-accuracy and runtime figures. Peer review and independent reproduction (P2.9) remain open.

This record covers the Phase 2 standalone models (P2.1–P2.8). The evaluation follows the specified Phase 1 data split and candidate rules; formal Phase 1 approval and integration are tracked separately.

## 1. Evaluation setup

- **History:** ratings strictly before 1 January 2017 UTC (20,798,765 rows).
- **Validation outcomes:** 2017 observations under the Phase 1 temporal protocol. No 2018–2019 data is used.
- **Cohort:** 4,778 users with at least one eligible observed positive (169,368 pairs).
- **Candidates:** the fixed 17,965-movie catalogue, excluding every movie each user rated before the cutoff.
- **Ranking:** unclipped scores, descending; exact ties break by ascending `movieId`; top 10.
- **Primary selection metric:** mean observed-positive NDCG@10 across the cohort; movie coverage is reported alongside it and breaks an exact metric tie in `select()`.
- **Rating accuracy:** 275,061 shared held-out 2017 rating pairs among cohort users on eligible catalogue movies. Predictions are clipped to the MovieLens range [0.5, 5] after any fallback filling.
- **Fallback:** a separately fitted bias model (`lambda_user = lambda_movie = 10`, 20 passes) fills missing ranking scores and missing rating predictions, in both tuning and the final run.

## 2. Models and current validation selections

The selections are in `outputs/phase2_selected_config_extended.json`, which the runner reads directly. They are **configuration values**, not saved fitted model objects. The earlier `outputs/phase2_selected_config.json` reflects only the base grid and is superseded.

| Model | Task | Method | Selected settings (as run) |
| --- | --- | --- | --- |
| Popularity (P2.1) | Ranking | Historical rating count per movie | Count of pre-2017 ratings |
| Bias (P2.2) | Rating | `mu + b_u + b_i`, alternating regularized updates | `lambda_user=320`, `lambda_movie=320`; ≤20 passes, tolerance 1e-6 |
| Item-CF (P2.3) | Rating | Adjusted-cosine item similarities, top-K neighbourhood, item-mean residual blend | `top_k=3`, `min_similarity=0` |
| MF (P2.4) | Rating | Explicit-feedback ALS with biases | `n_factors=24`, `regularization=30`; bias λ 10/10, ≤20 iterations, tolerance 1e-4, seed 0 |

The full constructor settings for each model are saved in `outputs/phase2_validation_environment.json` (`model_settings`).

Popularity counts serve as ranking scores. Rating models rank candidates using unclipped predictions; only RMSE/MAE inputs are clipped. Item-CF returns a missing score when a user has rated none of a candidate's retained neighbours, and the fallback above fills it. MF's training-RMSE stopping check still clips to [1, 5]; this was left unchanged so the fitted factors match the tuning runs, and it does not affect the reported RMSE/MAE.

### P2.5 tuning grid: corrected and extended

Selection uses highest mean 2017 validation NDCG@10, with coverage as the tie-breaker. The corrected base grid is in `outputs/phase2_tuning_grid.csv`; incremental probes are in `outputs/phase2_tuning_extension.csv`; their combined table and selections are in `outputs/phase2_tuning_grid_extended.csv` and `outputs/phase2_selected_config_extended.json`. No test-period data was used for this tuning.

**Bias model** — user and movie lambdas are equal within each probe.

| Lambda (user / movie) | Mean NDCG@10 | Coverage (movies) | Selected |
| --- | ---: | ---: | :---: |
| 5 / 5 | 0.033641 | 37 | |
| 10 / 10 | 0.033420 | 59 | |
| 20 / 20 | 0.038258 | 81 | |
| 40 / 40 | 0.043384 | 97 | |
| 80 / 80 | 0.049123 | 119 | |
| 160 / 160 | 0.048267 | 137 | |
| 320 / 320 | **0.052504** | 154 | ✓ |

**Item-CF** — all probes use `min_similarity=0` and the same `10/10` bias fallback.

| Top K neighbours | Mean NDCG@10 | Coverage (movies) | Selected |
| ---: | ---: | ---: | :---: |
| 3 | **0.059745** | 4,627 | ✓ |
| 4 | 0.055505 | 4,943 | |
| 6 | 0.052409 | 5,371 | |
| 8 | 0.048083 | 5,674 | |
| 12 | 0.044812 | 6,133 | |
| 25 | 0.033649 | 7,071 | |
| 50 | 0.026542 | 7,570 | |
| 100 | 0.018028 | 7,648 | |
| 200 | 0.013449 | 7,653 | |

**Matrix factorization** — only the factor/regularization pairs shown were tested.

| Factors | Regularization | Mean NDCG@10 | Coverage (movies) | Selected |
| ---: | ---: | ---: | ---: | :---: |
| 16 | 15 | 0.095399 | 1,506 | |
| 24 | 15 | 0.093660 | 1,584 | |
| 24 | 10 | 0.088820 | 1,806 | |
| 8 | 15 | 0.091685 | 1,195 | |
| 8 | 30 | 0.086962 | 953 | |
| 16 | 30 | 0.093215 | 1,122 | |
| 16 | 45 | 0.086405 | 942 | |
| 24 | 30 | **0.095484** | 1,182 | ✓ |

The Item-CF tuning values above use the corrected orientation: after retaining the top K similarities in each candidate item's row, scoring uses that row against a user's rated-item residuals. The old pre-fix values (such as 0.018548 at k=50) are obsolete and must not be mixed with this table. The current k=50 result is 0.026542. Smaller neighbourhoods have raised NDCG while lowering movie coverage. A k=1/2 check was discussed but has **not** been reported here.

Two selections sit at the edge of their explored grids: bias λ=320 is the largest value tried and Item-CF k=3 the smallest. MF's 24 factors is also the largest factor count tried. Better settings may lie beyond these ranges.

The MF winner at `(24, 30)` exceeds `(16, 15)` by only 0.000085 NDCG@10, while the latter covers 324 more movies. Treat these as a close validation tradeoff rather than evidence of a meaningful accuracy difference. For hybrid experiments, consider both MF settings and at least the k=3 and k=6 Item-CF settings; evaluate hybrids on the same validation cohort and report NDCG and coverage separately. Do not infer hybrid coverage by adding component coverages.

## 3. Validation ranking results (P2.6, P2.7)

Full validation run (`src.run_phase2_validation`) with the selected configurations on the 2017 cohort (4,778 users). Every selected model's NDCG@10, Recall@10, coverage, less-popular exposure and missing-score count is identical to its tuning row, and the popularity outputs are unchanged from the original benchmark.

| Model | Selected configuration | Mean NDCG@10 | Mean Recall@10 | Coverage (movies) | Coverage (% of catalogue) | Less-popular exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | Pre-2017 rating count | 0.075212 | 0.023637 | 287 | 1.60% | 0.0000 |
| Bias | lambda 320 / 320 | 0.052504 | 0.018621 | 154 | 0.86% | 0.0002 |
| Item-CF | top_k=3, min_similarity=0 | 0.059745 | 0.025483 | 4,627 | 25.76% | 0.1593 |
| Matrix factorization | 24 factors, reg 30 | **0.095484** | **0.040890** | 1,182 | 6.58% | 0.2726 |

Matrix factorization is the strongest standalone model on the agreed validation metric: NDCG@10 is 27.0% above the popularity baseline, and 40.3% of users have at least one hit in its top 10 (popularity 31.1%, Item-CF 31.2%, bias 23.4%). It is therefore the validation-selected comparator for the Phase 3 hybrid under P1.6.

Popularity still beats both the bias model (−30.2% NDCG@10) and Item-CF (−20.6%). Item-CF now ranks above the bias model, and it recommends by far the widest range of movies (25.8% of the catalogue). The bias model at λ=320 concentrates on a small head: 154 distinct movies, and less-popular titles fill only 0.02% of its slots. MF combines the best relevance with the highest less-popular exposure (27.3% of slots).

**Independent P2.7 check.** `python -m src.check_phase2_metrics` recalculated NDCG@10, Recall@10, coverage and less-popular exposure from the saved top-10 lists, with catalogue and positives re-derived from raw ratings. All four models matched the saved per-user and summary metrics within 1e-12; coverage counts matched exactly. All 4,778 users had 10 distinct, eligible, previously unrated recommendations. At the top-3,593 popularity boundary, three movies have a count of 865; ascending movie ID resolves the tie.

**Item-CF fallback share.** The score grid has 4,778 × 17,965 = 85,836,770 cells. Item-CF produced a direct score for 4,239,529 of them (4.94%); the remaining 81,597,241 (95.06%) came from the bias fallback. The grid includes movies a user already rated, which are excluded at ranking, so this is a share of all scored cells rather than of eligible candidates. Item-CF's ranking results therefore describe Item-CF scores mixed with a fallback that dominates the grid. How many of the top-10 slots come from direct Item-CF scores has not yet been measured.

## 4. Rating-prediction accuracy (P2.7 / P3.5 input)

Star predictions compared against the 275,061 shared held-out pairs, after fallback filling and clipping to [0.5, 5]:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Bias (320 / 320) | 0.8769 | 0.6596 |
| Item-CF (k=3) | 0.8796 | 0.6497 |
| Matrix factorization (24, 30) | **0.7991** | **0.5902** |

Matrix factorization is the best rating predictor. Item-CF has a slightly lower MAE than the bias model but a slightly higher RMSE. The number of rating pairs filled by the fallback is not recorded.

For comparison, the original benchmark (bias λ=20, pre-fix Item-CF k=50, MF 16/15) reported RMSE 0.8517 / 0.8702 / 0.7988. Those runs clipped only MF, and only to [1, 5]. Clipping to the valid rating range can only reduce each error, so the bias model's higher RMSE now (0.8769 vs 0.8517) comes from its configuration: the strong regularization chosen for NDCG@10 costs rating accuracy. MF's RMSE is essentially unchanged across the two configurations.

The earlier text reported a weakly regularized MF overfitting probe (`reg=0.05`, training RMSE 0.61, held-out RMSE 0.92), but its separate run record was not supplied. Preserve that claim as provisional until the supporting output is recovered or reproduced.

## 5. Ranking versus rating-prediction contrast

Matrix factorization is the only standalone model that is best on **both** ranking (NDCG@10) and rating accuracy (RMSE, MAE). The other two rating models disagree across the two views. Bias has the better RMSE but the worse NDCG@10. Item-CF has the better NDCG@10 and MAE. Both rank below the non-personalized popularity baseline on NDCG@10. This is the anticipated distinction between rating-prediction accuracy and ranked-list retrieval relevance (P3.5).

Within the tuning grids, the settings that raised NDCG@10 also moved coverage. For Item-CF, k=3 has higher NDCG and lower coverage than k=6 or k=12. For MF, `(16, 15)` gives nearly the same NDCG as `(24, 30)` with higher coverage. These are candidate component settings for a controlled Phase 3 hybrid comparison. Coverage counts distinct recommended movies across users; it does not by itself establish relevance or benefit for each user. Keep validation NDCG as the primary tuning metric while reporting coverage, recall and less-popular exposure alongside it. Changes to the final selection rule should be documented before using test-period outcomes.

## 6. Runtime and resource requirements (P2.8)

Measured in the full validation run on Windows 11, 20 logical CPUs, 16 GB RAM (16,088 MB), Python 3.13.2:

| Model | Fit time | Score time | RSS reading after scoring (MB) |
| --- | ---: | ---: | ---: |
| Popularity | 0.07 s | 0.15 s | 968 |
| Bias (320 / 320) | 4.7 s | 0.28 s | 1,299 |
| Item-CF (k=3) | 229.5 s | 1.53 s | 1,055 |
| Matrix factorization (24, 30) | 230.6 s | 0.57 s | 710 |

Item-CF and MF each take close to four minutes to fit; scoring the full grid takes under two seconds for every model. These timings come from a different machine than the original benchmark (16 logical CPUs, 32 GB RAM, Python 3.13.15), so the two sets of timings are not directly comparable.

The RSS column is a single reading taken after scoring, in one process shared by all models. It includes data already loaded in that process and can fall when memory is released; MF's reading is lower than popularity's. It is **not a per-model peak** and cannot support claims about each model's peak memory. Replace or relabel the metric in P2.8 before sign-off.

## 7. Reproduction and result files

From the project root, with the environment, MovieLens 25M raw ratings and processed datasets available:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets      # if datasets need preparation
.\.venv\Scripts\python.exe -m src.check_phase2          # interface and clipping self-checks
.\.venv\Scripts\python.exe -m src.tune_models           # corrected base grid; reruns every setting
.\.venv\Scripts\python.exe -m src.tune_extension        # only unrecorded extension settings
.\.venv\Scripts\python.exe -m src.run_phase2_validation # full benchmark from the extended selection
```

The extension script reads the base grid and appends only missing probes to its extension CSV. Do not rerun the base grid casually after further code or fallback changes: combining results across differing model versions, datasets or fallback policies would make the selection table incomparable. The runner reads `phase2_selected_config_extended.json` directly and removes its own earlier outputs before fitting.

Relevant outputs under `outputs/`:

- `phase2_tuning_grid.csv` — corrected base grid.
- `phase2_tuning_extension.csv` — incremental probes.
- `phase2_tuning_grid_extended.csv` and `phase2_selected_config_extended.json` — combined validation selection; the JSON does not contain fitted models.
- `phase2_validation_summary.csv` — one row per model with ranking, rating and resource metrics.
- `phase2_validation_<model>_user_metrics.csv` — per-user NDCG@10 / Recall@10.
- `phase2_validation_<model>_top10.csv` — archived top-10 lists.
- `phase2_validation_environment.json` — run record: environment, selection file and SHA-256 (`a38f255e…`), full model settings, fallback, rating clip range, git commit and completion time.

The run record shows the benchmark used uncommitted code on top of commit `2ddd23c` (`git_uncommitted_changes: true`). Commit the code and these outputs together so the record identifies the exact version that produced them.

## 8. Open items

- **Item-CF diagnostics:** measure how many top-10 slots come from direct Item-CF scores versus the fallback, and whether mixing the two score scales distorts the ranking. Optionally test k=1/2, since k=3 is the grid edge.
- **Grid edges:** bias λ=320 and Item-CF k=3 (and MF's 24 factors) are boundary selections; extend the grids or document why the search stops there.
- **Run records:** save MF convergence evidence (iteration count, final training RMSE and stopping reason) and the count of fallback-filled rating pairs.
- **Resources:** Measure per-model peak memory if required; the current table reports shared-process RSS readings after scoring.
- **P2.9 independent reproduction and integration:** obtain teammate review and reconcile Phase 1 sign-off, workflow and tracker state before claiming formal completion.

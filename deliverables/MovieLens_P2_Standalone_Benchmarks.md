# MovieLens 25M Phase 2 Standalone Model Benchmarks

**Status** — Local validation complete for all four standalone models. Ready for teammate review and independent reproduction (P2.9).

This record documents the Phase 2 individual models (P2.1–P2.8) evaluated on
2017 validation under the frozen Phase 1 protocol. All models use the fixed
17,965-movie catalogue (P1.4), the 4,778-user evaluation cohort (P1.5), and the
same per-user exclusions and ranking rules from P1.3–P1.5.

## 1. Evaluation setup

- **History:** all ratings strictly before 1 January 2017 UTC (20,798,765 rows).
- **Outcomes:** ratings from 1 January 2017 through 31 December 2017 exclusive.
- **Cohort:** 4,778 users with at least one eligible observed positive (169,368 pairs).
- **Candidates:** the fixed catalogue minus every movie each user rated before the cutoff.
- **Ranking:** score descending, break exact ties by ascending `movieId`, take top 10.
- **Primary metric:** mean observed-positive NDCG@10 across the cohort.
- **Rating pairs:** 275,061 observed 2017 ratings among cohort users on eligible
  catalogue movies, shared across rating models for RMSE/MAE.

## 2. Models and configuration

Configurations below are the P2.5 validation-selected settings
(`outputs/phase2_selected_config.json`).

| Model             | Task    | Method                                                                    | Selected settings                                     |
| ----------------- | ------- | ------------------------------------------------------------------------- | ----------------------------------------------------- |
| Popularity (P2.1) | Ranking | Historical rating count per movie                                         | count of pre-2017 ratings                             |
| Bias (P2.2)       | Rating  | `mu + b_u + b_i`, alternating regularized updates                         | lambda_u = lambda_i = 20, 20 passes                   |
| Item-CF (P2.3)    | Rating  | Adjusted-cosine similarity, top-K neighbourhood, item-mean residual blend | top_k = 50                                            |
| MF (P2.4)         | Rating  | Explicit-feedback ALS with biases                                         | n_factors = 16, reg = 15, lambda_bias = 10, 20 passes |

Popularity counts are ranking scores. Bias, Item-CF and MF predict star ratings
and rank candidates by their full-precision predictions. Rating output (RMSE/MAE
and any star prediction) is clipped to [1, 5]; ranking uses unclipped raw values
so predictions do not saturate into ties and hand the list to the lowest IDs.

### P2.5 tuning grids

Each personalized model explored three documented settings on 2017 validation;
selection is by highest mean NDCG@10, then coverage. Full table in
`outputs/phase2_tuning_grid.csv`.

| Model   | Probe              | Mean NDCG@10 | Coverage | Selected |
| ------- | ------------------ | -----------: | -------: | -------- |
| Bias    | lambda 5 / 5       |     0.033641 |       37 |          |
| Bias    | lambda 10 / 10     |     0.033420 |       59 |          |
| Bias    | lambda 20 / 20     |     0.038258 |       81 | ✓        |
| Item-CF | top_k 50           |     0.018548 |    7,456 | ✓        |
| Item-CF | top_k 100          |     0.012958 |    8,038 |          |
| Item-CF | top_k 200          |     0.007893 |    8,201 |          |
| MF      | 16 factors, reg 15 |     0.095399 |    1,506 | ✓        |
| MF      | 24 factors, reg 15 |     0.093660 |    1,584 |          |
| MF      | 24 factors, reg 10 |     0.088820 |    1,806 |          |

Stronger bias regularization, a smaller Item-CF neighbourhood and fewer factors
than the initial defaults all improved validation NDCG@10. Tuning uses 2017
validation only; no test-period information is used.

## 3. Validation ranking results (P2.6, P2.7)

Observed standalone results on the 2017 validation cohort (4,778 users):

| Model                | Mean NDCG@10 | Mean Recall@10 | Coverage (movies) | Less-popular exposure |
| -------------------- | -----------: | -------------: | ----------------: | --------------------: |
| Popularity           |     0.075212 |       0.023637 |               287 |                0.0000 |
| Bias                 |     0.038258 |       0.013524 |                81 |                0.3154 |
| Item-CF              |     0.018596 |       0.007146 |             7,467 |                0.4817 |
| Matrix factorization | **0.095399** |   **0.040035** |             1,506 |                0.2192 |

Matrix factorization is the strongest standalone model on the agreed validation
metric, improving NDCG@10 by 26.8% over the popularity baseline. The popularity
baseline reproduces the published P1.5 reference (NDCG@10 0.075212, coverage 287)
exactly, confirming the shared pipeline matches the Phase 1 definitions. Item-CF
gains coverage and less-popular exposure but scores far below the baselines on
ranking; the bias model collapses onto a small head (coverage 81).

## 4. Rating-prediction accuracy (P2.7 / P3.5 input)

Star predictions compared against the 275,061 shared held-out pairs, on the
original rating scale:

| Model                |       RMSE |        MAE |
| -------------------- | ---------: | ---------: |
| Bias                 |     0.8517 |     0.6346 |
| Item-CF              |     0.8702 |     0.6369 |
| Matrix factorization | **0.7988** | **0.5893** |

Matrix factorization is the best rating predictor once regularized strongly
enough. Weak regularization (reg = 0.05) overfits (train RMSE 0.61, test 0.92)
and is rejected. A validation probe selected n_factors = 16 and reg = 15.

## 5. Ranking versus rating-prediction contrast

Matrix factorization is the only standalone model that is best on **both**
ranking (NDCG@10) and rating accuracy (RMSE). The bias and Item-CF models reduce
squared error on observed pairs but do not translate that into top-10 retrieval:
their high less-popular exposure (0.32–0.48) comes with very low relevance. This
is the anticipated distinction between rating-prediction accuracy and ranked-list
retrieval relevance (P3.5) and motivates the Phase 3 hybrid.

## 6. Runtime and resource requirements (P2.8)

Measured on Windows 11, 16 logical CPUs, 32 GB RAM, Python 3.13.15:

| Model                | Fit time | Score time | Peak RSS (MB) |
| -------------------- | -------: | ---------: | ------------: |
| Popularity           |   0.09 s |     0.17 s |           969 |
| Bias                 |    5.8 s |     0.30 s |         1,300 |
| Item-CF              |   55.9 s |      2.7 s |         1,799 |
| Matrix factorization |  155.3 s |      0.7 s |         1,230 |

Matrix factorization is the most expensive to fit but cheap to score. Item-CF is
the most memory-hungry (dense similarity work). The tuned settings are also
faster than the initial defaults because the grids reduced model capacity.

## 7. Reproduction

From the project root, with the environment and processed datasets in place:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets     # P1.7 datasets
.\.venv\Scripts\python.exe -m src.check_phase2          # interface self-checks
.\.venv\Scripts\python.exe -m src.run_phase2_validation # P2.6-P2.8 benchmarks
.\.venv\Scripts\python.exe -m src.tune_models           # P2.5 tuning grids
```

Outputs are written under `outputs/`:

- `phase2_validation_summary.csv` — one row per model with ranking, rating and resource metrics.
- `phase2_validation_<model>_user_metrics.csv` — per-user NDCG/Recall.
- `phase2_validation_<model>_top10.csv` — archived top-10 lists.
- `phase2_validation_environment.json` — run metadata (P2.8).
- `phase2_tuning_grid.csv`, `phase2_selected_config.json` — P2.5 tuning records.

## 8. Open items

- **P2.9 independent reproduction** — a second member must rerun the benchmarks and confirm the summary table.
- **Integration** — carry the matrix-factorization comparator into the Phase 3 hybrid.

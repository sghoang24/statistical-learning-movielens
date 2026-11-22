# MovieLens Recommendation Strategies

## Project Execution Plan

**Project:** Statistical Learning for Data Science — Group Project  
**Team:** 3 members  
**Status:** Updated for team review  
**Planning horizon:** Approximately 4 weeks (adjust to actual submission deadline)  
**Document version:** 1.1 — Phase 1 reconciliation

---

# 1. Project Overview

## 1.1 Primary Research Question

**Can combining established rating-based movie recommendation methods meaningfully improve held-out top-10 recommendation performance compared with individual methods, and how does combining them affect catalogue coverage?**

## 1.2 Project Objectives

**Primary objective:** Evaluate whether combining item-based collaborative filtering (Item-CF) and matrix factorization (MF) provides a practically meaningful improvement in held-out observed-positive NDCG@10 over the stronger standalone personalized model selected on 2017 validation.

**Secondary objectives:**

- Evaluate catalogue coverage and exposure to less-popular movies.
- Investigate differences between rating-prediction accuracy and ranking performance.
- Examine model disagreement and performance differences across user-history groups.
- Assess whether any improvement justifies additional computational complexity.

## 1.3 Scope

**Required methods:**

- Popularity baseline
- Regularized user/item bias baseline
- Item-based collaborative filtering
- Matrix factorization
- Validation-selected weighted hybrid

**Required evaluation:**

- Observed-positive NDCG@10 (primary)
- Recall@10
- Catalogue coverage
- Less-popular recommendation share
- RMSE and MAE for models that predict star ratings on common held-out pairs
- Paired user-level uncertainty estimates
- Reproducibility and runtime measurements

**Optional extensions:** History-dependent weighting, popularity-aware reranking, popularity-penalized scoring, gradient boosting, and additional ranking-oriented methods.

Optional extensions must not delay the completion of the core experiment.

## 1.4 Initial Responsibility Allocation

| Role     | Initial owner | Main area                                              |
| -------- | ------------- | ------------------------------------------------------ |
| Member A | To assign     | Data preparation, eligibility and exploratory analysis |
| Member B | To assign     | Recommendation models and tuning                       |
| Member C | To assign     | Hybrid methods, evaluation and statistical inference   |

All members participate in research decisions, peer review, integration, interpretation, and final presentation.

**Review rotation:** A reviews B; B reviews C; C reviews A. Changes to the experimental protocol and final-test configuration require approval from all three members.

---

# 2. Project File Management

Use Google Drive for review packages while Git access is pending. Teammates work locally and place their additions in their own member workspaces. Keep development work and approved project files separate. Package the shared code after preliminary work is ready for handoff.

## 2.1 Folder Structure

```text
MovieLens Project/
├── 00_Project_Management/
│   ├── Experimental_Protocol/
│   ├── Project_Plan/
│   ├── Task_Tracker/
│   ├── Experiment_Log/
│   └── Meeting_Notes/
│
├── 01_Member_Workspaces/
│   ├── Member_A/
│   ├── Member_B/
│   └── Member_C/
│
├── 02_Review_Queue/
│   ├── Pending/
│   └── Revisions_Requested/
│
├── 03_Approved_Project/
│   ├── Code/
│   ├── Config/
│   ├── Data_Definitions/
│   └── Tests/
│
├── 04_Approved_Results/
│   ├── Validation/
│   └── Final_Test/
│
└── 05_Report/
    ├── Drafts/
    └── Final/
```

## 2.2 Review and Integration Rules

1. Each member develops and tests files in their own workspace.
2. Work submitted for review must include code or analysis, a brief explanation, dependencies, and test evidence.
3. A peer reviewer checks correctness, reproducibility, interface compatibility, and possible data leakage.
4. Approved work undergoes a basic integration check against the current approved project.
5. Only the integration owner publishes approved versions to `03_Approved_Project`.
6. Approved releases are archived rather than overwritten.
7. Results must identify the approved code version, dataset configuration, parameters and random seed that produced them.

**Important:** No experimental configuration may be changed after reviewing final test results in order to improve reported performance.

---

# 3. Phase 1 — Experimental Protocol and Data Preparation

**Suggested duration:** Week 1  
**Primary owner:** Member A  
**Supporting members:** B and C  
**Goal:** Establish a reproducible, leakage-controlled foundation for all subsequent experiments.

| Task                               | Responsible      | Action / Details                                                                                                          | Deliverable                                   | Acceptance Criteria                                                                        |
| ---------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| P1.1 Confirm research objectives   | All              | Agree on the primary question, primary comparison and secondary analyses                                                  | Approved objectives document                  | All members agree on the scope and can explain the primary decision                        |
| P1.2 Define data eligibility       | A; reviewed by C | Specify user-history minimums, movie eligibility and exclusions using historical information only                         | Eligibility specification                     | No future outcomes determine training or evaluation cohort eligibility                     |
| P1.3 Finalize temporal protocol    | A + C            | Confirm train before 2017, validation in 2017, final test in 2018–2019; define recommendation date and evaluation horizon | Data-splitting specification                  | Model training and feature construction cannot access future outcomes                      |
| P1.4 Define candidate catalogue    | A + C            | Determine eligible movies, already-rated exclusions and consistent candidate generation                                   | Candidate-selection specification             | Every model receives the same eligible candidate rules; no future-positive-based filtering |
| P1.5 Define evaluation metrics     | C; reviewed by A | Implement definitions for NDCG@10, Recall@10, catalogue coverage and less-popular exposure                                | Evaluation specification                      | Mathematical definitions, denominators and treatment of missing feedback are documented    |
| P1.6 Define meaningful improvement | All              | Set the decision rule and procedure for choosing a practical NDCG improvement margin before final testing                 | Decision-rule specification                   | The primary contrast, uncertainty method and interpretations are agreed upon               |
| P1.7 Prepare reproducible datasets | A                | Generate the fixed catalogue, validation positives, optional phase training mappings, and a checksum manifest             | Data-preparation scripts and dataset manifest | Another member can reproduce the datasets and verify counts                                |
| P1.8 Establish common interfaces   | B + C            | Agree on model scoring inputs, recommendation outputs and configuration formats                                           | Interface specification                       | Standalone models and hybrid can feed the same evaluation pipeline                         |
| P1.9 Integrate and review Phase 1  | All              | Peer-review protocol and execute data-generation checks                                                                   | Approved protocol and data pipeline           | No unresolved high-priority leakage or schema issues                                       |

### Phase 1 execution record

| Task | Current state and evidence                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1.1 | Primary comparison and secondary measures are documented. Team sign-off remains to be recorded.                                                                                                                                                                                                                                                                                                                                           |
| P1.2 | Locally verified. User eligibility requires at least one pre-2017 rating. Movie eligibility requires at least ten. Both rules use a 1 January 2017 UTC cutoff. Both threshold audits reproduced the 1, 5, 10, and 20 rows. Keep the user cutoffs as nested evaluation sensitivity views. The primary catalogue retains 17,965 titles and 99.7187% of pre-2017 ratings.                                                                    |
| P1.3 | Locally verified. Recommend as of 1 January 2017 UTC using pre-2017 history and score on 2017. After freezing choices, refit on all pre-2018 ratings and recommend as of 1 January 2018 UTC. Score through the exclusive cutoff of 1 January 2020 UTC. The last observed rating is on 21 November 2019.                                                                                                                                   |
| P1.4 | Locally verified. Start each user from the same fixed 17,965-title catalogue and remove movies they rated before the recommendation date. All 169,368 saved 2017 positive pairs across 4,778 users passed the catalogue and history checks. Both cutoff boundaries passed the synthetic check.                                                                                                                                            |
| P1.5 | Metric examples passed for all six metrics. Eligible observed positives are held-out ratings above 3. Average binary NDCG@10 and Recall@10 over the same users with eligible positives. Coverage is distinct titles across those users' top tens divided by 17,965. Less-popular exposure is the share of returned slots outside the top 3,593 movies by pre-2017 rating count. That top fifth accounts for 89.87% of historical ratings. |
| P1.6 | Protocol ready for review. Choose the stronger standalone by 2017 mean NDCG@10 and use that comparator for every test user. The provisional practical margin is 0.01 absolute mean NDCG@10. Confirm or revise it using validation hits, coverage, and runtime. Report a paired user-level bootstrap interval on test. The 10,000 resamples, seed 42, and 95% percentile interval remain proposed settings.                                |
| P1.7 | Full local preparation run passed. The command generates the catalogue, validation positives, four phase mapping CSVs, and a SHA-256 manifest. The validation fit maps 142,184 users and 37,363 movies, and the final refit maps 150,543 users and 45,074 movies. The mappings cover all phase training IDs and are optional for adapters with native ID handling. Independent reproduction remains pending.                              |
| P1.8 | **Blocked pending team selection of the first library to test.** Draft boundary uses original user and movie IDs and predicted stars. Smoke-test fit requirements, explicit pair scoring, unsupported predictions, and blend alignment before finalizing the adapter contract.                                                                                                                                                            |
| P1.9 | Pending. The tracker lists P1.5, P1.6, P1.7, P1.8, and P2.1 as dependencies. Local verification supplies review evidence. Record teammate acceptance and integration separately.                                                                                                                                                                                                                                                          |

Run the verified preparation command from the project root with the extracted MovieLens 25M data in place.

```powershell
.\.venv\Scripts\python.exe -m src.prepare_datasets
```

For the independent check, compare the source SHA-256 and six output hashes in the generated manifest with the reference run. Training mappings include movies below the recommendation threshold so their historical ratings can contribute to fitting.

### Phase 1 Deliverables

- Approved experimental protocol
- User and movie eligibility specification
- Chronological dataset preparation scripts
- Candidate catalogue definition
- Metric definitions and primary decision rule
- Shared interface and configuration specification
- Initial data-quality and reproducibility report

### Phase 1 Acceptance Criteria

The phase is complete when:

1. All members approve the protocol.
2. Dataset construction is reproducible from the raw MovieLens archive.
3. User and movie eligibility uses only information permitted at the recommendation date.
4. Model and evaluation interfaces are agreed upon.
5. The primary comparison and statistical decision rule are defined.
6. Test-period outcomes have not been used to tune the final study.

**Phase gate:** No final-model comparison begins until the shared data and evaluation specifications are approved.

---

# 4. Phase 2 — Individual Models and Validation Benchmarks

**Suggested duration:** Week 2  
**Primary owner:** Member B  
**Supporting members:** A and C  
**Goal:** Implement and fairly compare established standalone recommendation methods using the approved protocol.

| Task                                      | Responsible | Action / Details                                                                                                                                | Deliverable                             | Acceptance Criteria                                                  |
| ----------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------- |
| P2.1 Implement popularity baseline        | B           | Rank eligible movies using historical popularity. Complete shared-runner integration and review the end-to-end validation evidence before P1.9. | Popularity model and validation results | Historical counts only; model produces valid top-10 lists            |
| P2.2 Implement bias baseline              | B           | Fit regularized global, user and movie rating biases                                                                                            | Bias model and results                  | Deterministic predictions under fixed configuration                  |
| P2.3 Finalize Item-CF                     | B           | Review similarity computation, neighborhood construction and sparse-history handling                                                            | Item-CF module                          | Valid recommendations; documented approximation or sensitivity check |
| P2.4 Finalize matrix factorization        | B           | Fit latent factors with regularization and documented stopping criteria                                                                         | MF module                               | Convergence assessed; model parameters selected from validation only |
| P2.5 Establish tuning budgets             | B + C       | Define comparable search effort and selection criteria for standalone models                                                                    | Tuning configuration and logs           | Each model has a documented and justified tuning procedure           |
| P2.6 Generate common recommendation lists | B + C       | Score candidates and exclude previously rated movies                                                                                            | Validation recommendation lists         | All models use identical eligibility, ranking and exclusion rules    |
| P2.7 Calculate standalone metrics         | C           | Compute NDCG@10, Recall@10, coverage and exposure                                                                                               | Benchmark tables                        | Metric calculations pass independent checks                          |
| P2.8 Record resource requirements         | B           | Record fitting time, scoring time and peak memory where practical                                                                               | Runtime comparison table                | Measurements use documented hardware and settings                    |
| P2.9 Peer review and integration          | A + C       | Review implementations, rerun representative experiments and integrate approved code                                                            | Approved standalone-model release       | A second member successfully reproduces agreed benchmark outputs     |

### Baseline specifications

Both baselines use the approved phase history from P1.3 and the same per-user candidates from P1.4. The candidate catalogue is fixed from pre-2017 data. Exclude every movie the user rated before the phase's recommendation date. Rank scores descending, break exact ties by ascending movie ID, and return up to 10 titles.

#### P2.1 Popularity baseline

Score each candidate by the number of ratings that movie received in the phase history, regardless of star value. For 2017 validation, use the pre-2017 counts in eligible_movies.csv. For the final test, recompute counts using all pre-2018 ratings while keeping the candidate catalogue and less-popular classification fixed. The popularity order is shared across users. Each user's historical exclusions determine which titles remain in their top 10.

Report NDCG@10, Recall@10, catalogue coverage, less-popular exposure, and runtime using the common evaluation cohort. Popularity counts are ranking scores. RMSE and MAE apply to predicted star ratings.

#### P2.2 Regularized bias baseline

Predict a star rating as \(\hat r\_{ui}=\mu+b_u+b_i\). Compute \(\mu\) as the mean of all ratings in the phase training history. Estimate user and movie biases jointly over the observed training ratings by minimizing

\[
\sum*{(u,i)\in\mathcal{R}*{\rm train}}(r\_{ui}-\mu-b_u-b_i)^2
+\lambda_u\sum_u b_u^2+\lambda_i\sum_i b_i^2.
\]

Initialize biases to zero. In each pass, update all user biases with movie biases fixed, then update all movie biases with the new user biases fixed:

\[
b*u \leftarrow \frac{\sum*{i\in I*u}(r*{ui}-\mu-b*i)}{|I_u|+\lambda_u},
\qquad
b_i \leftarrow \frac{\sum*{u\in U*i}(r*{ui}-\mu-b_u)}{|U_i|+\lambda_i}.
\]

Here \(I_u\) is the set of movies rated by user \(u\) and \(U_i\) is the set of users who rated movie \(i\) in the phase history. Positive regularization shrinks biases toward zero, particularly when there are few observations. Use zero for a missing user or movie bias. Record the regularization values, stopping tolerance or fixed pass count, and clipping policy in the run configuration. Choose any tuned settings on 2017 validation and freeze them before the final refit.

For top-10 ranking, use full-precision raw predictions. The global mean and user bias are constant within a user's list, so movie bias determines the order and prior ratings determine exclusions. For RMSE and MAE, evaluate predicted stars against the same eligible observed pairs used for other rating models, including ratings at or below 3. Record any rating-output clipping consistently. This baseline is also the proposed fallback for unsupported Item-CF or MF predictions under P1.6.

**Existing P2.1 evidence.** The inline popularity validation run evaluated all 4,778 users. Mean NDCG@10 was 0.075212, mean Recall@10 was 0.023637, coverage was 287 of 17,965 movies (1.598%), and less-popular exposure displayed as 0.000%. This is validation evidence for the popularity baseline. P2.1 integration follows the tracker dependencies, including P1.8. Rating-error evaluation uses a separate model-generated prediction file.

### Phase 2 execution record

Full detail is recorded in `deliverables/MovieLens_P2_Standalone_Benchmarks.md`.

| Task | Current state and evidence |
| ---- | -------------------------- |
| P2.1 | Locally verified. `src/popularity.py` ranks eligible movies by pre-2017 rating count. Reproduces the P1.5 reference exactly (NDCG@10 0.075212, coverage 287). |
| P2.2 | Locally verified. `src/bias.py` fits `mu + b_u + b_i` with alternating regularized updates (λ 320/320, ≤20 passes). Validation NDCG@10 0.052504, RMSE 0.8769. |
| P2.3 | Locally verified after a fix. `src/item_cf.py` now scores each candidate from its own top-K neighbours (the original run read them the other way round). Adjusted-cosine top-3 neighbours with item-mean residual blending: validation NDCG@10 0.059745, coverage 4,627, RMSE 0.8796. 95.1% of the score grid is filled by the bias fallback. |
| P2.4 | Locally verified. `src/matrix_factorization.py` fits explicit ALS with biases (24 factors, reg 30, ≤20 iterations). Best standalone model: NDCG@10 0.095484, RMSE 0.7991. Convergence evidence (iterations, training RMSE) is computed but not yet saved. |
| P2.5 | Locally verified. `src/tune_models.py` (base grid) and `src/tune_extension.py` (targeted extensions) select by mean 2017 NDCG@10, coverage as tie-break. Budgets are unequal (bias 7, Item-CF 9, MF 8 settings). Selected: bias λ 320/320, Item-CF top_k 3, MF 24 factors / reg 30, in `outputs/phase2_selected_config_extended.json`. Bias λ=320 and Item-CF k=3 are grid edges. |
| P2.6 | Locally verified. `src/pipeline.py` scores the full candidate grid, excludes each user's pre-cutoff ratings, and takes top 10 with ascending movieId tie-break. Common to all models. Missing scores use a separate bias fallback (λ 10/10) shared by tuning and the runner. |
| P2.7 | Locally verified. NDCG@10, Recall@10, coverage, less-popular exposure, RMSE and MAE computed in `src/metrics.py` and written per model. RMSE/MAE use predictions clipped to [0.5, 5]; ranking uses unclipped scores. Independent ranking-metric check passed for all four models using src/check_phase2_metrics.py (tolerance 1e-12). |
| P2.8 | Locally verified. Fit time, score time and an RSS reading after scoring (not a per-model peak) recorded in `phase2_validation_summary.csv`; hardware, settings and code version in `phase2_validation_environment.json` (20 CPUs, 16 GB RAM). |
| P2.9 | Pending. The full validation run (22 November 2026) reproduced the tuning results exactly and local self-checks pass in `src/check_phase2.py`; independent reproduction and teammate review remain to be recorded. |

### Phase 2 ranking benchmark

| Model | Configuration | Mean NDCG@10 | Mean Recall@10 | Coverage | Less-popular exposure | RMSE | MAE |
| ----- | ------------- | -----------: | -------------: | -------: | --------------------: | ---: | --: |
| Popularity | Pre-2017 rating count | 0.075212 | 0.023637 | 287 | 0.0000 | — | — |
| Bias | λ 320 / 320 | 0.052504 | 0.018621 | 154 | 0.0002 | 0.8769 | 0.6596 |
| Item-CF | top_k 3 | 0.059745 | 0.025483 | 4,627 | 0.1593 | 0.8796 | 0.6497 |
| Matrix factorization | 24 factors, reg 30 | **0.095484** | **0.040890** | 1,182 | 0.2726 | **0.7991** | **0.5902** |

Results use the extended P2.5 selections from the full validation run. Matrix
factorization is the strongest standalone model on the agreed validation
metric (27.0% above popularity) and is the validation-selected comparator for
Phase 3. Bias and Item-CF both rank below popularity on NDCG@10.

### Phase 2 Deliverables

- Approved implementations of all four standalone models
- Model hyperparameter and tuning logs
- Validation recommendation lists
- Individual-model benchmark tables
- Initial runtime comparison
- Approved model release with reproducibility instructions

### Phase 2 Acceptance Criteria

The phase is complete when:

1. Every standalone model produces top-10 lists for the same eligible users and candidates.
2. Both Item-CF and MF have undergone documented tuning.
3. All models use approved historical inputs and are evaluated consistently.
4. Their validation results are reproducible.
5. The group can identify the best individual personalized model using the agreed validation metric.
6. An independent reviewer has verified the integrated benchmark pipeline.

**Phase gate:** Hybrid model selection begins only after the standalone benchmarks and evaluation pipeline are approved.

---

# 5. Phase 3 — Hybrid Methods and Statistical Evaluation

**Suggested duration:** Week 3  
**Primary owner:** Member C  
**Supporting members:** A and B  
**Goal:** Determine whether combining recommendation methods provides meaningful benefits under the approved evaluation criteria.

| Task                                                 | Responsible           | Action / Details                                                                                                                                | Deliverable                       | Acceptance Criteria                                                                    |
| ---------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| P3.1 Implement parallel hybrid                       | C + B                 | Align by original user and movie IDs, blend predicted star ratings, then rank                                                                   | Hybrid scoring module             | Endpoints reproduce component rankings where applicable; score treatment documented    |
| P3.2 Compare hybrid weights                          | C                     | Sweep Item-CF weights 0.1 through 0.9 by 0.1 on 2017 validation. Reuse component predictions. Check endpoints 0 and 1 against standalone lists. | Weight-performance table and plot | No test-period information used for selection                                          |
| P3.3 Freeze model choices                            | All                   | Freeze the validation-selected comparator, component settings, blend weight, practical margin, and inference settings                           | Frozen experiment configuration   | All choices recorded and approved before final testing                                 |
| P3.4 Analyze model disagreement (if time)            | C                     | Calculate top-10 overlap, candidate-score rank agreement and selected diagnostics                                                               | Model-disagreement results        | Analysis uses the same eligible candidates and user population                         |
| P3.5 Compare ranking and rating prediction (if time) | B + C                 | Analyze whether MAE/RMSE rankings align with NDCG@10 rankings                                                                                   | Diagnostic comparison             | Interpretations distinguish rating prediction from ranked-list retrieval               |
| P3.6 Analyze catalogue exposure                      | A + C                 | Compute catalogue coverage and less-popular slot share                                                                                          | Exposure analysis                 | Denominators, popularity thresholds and evaluated-user counts documented               |
| P3.7 Analyze user-history groups (if time)           | A + C                 | Compare methods across historically defined activity levels                                                                                     | Subgroup results                  | Groups use pre-cutoff information; sample sizes disclosed                              |
| P3.8 Quantify uncertainty                            | C                     | Calculate paired user-level bootstrap intervals for prespecified NDCG differences                                                               | Confidence-interval table         | User-level pairing preserved; uncertainty procedure reproducible                       |
| P3.9 Perform final held-out evaluation               | C; witnessed by A + B | Refit frozen component settings on pre-2018 ratings and score the fixed 2018–2019 holdout                                                       | Final test results                | No post-test retuning or selective exclusion; all outputs archived                     |
| P3.10 Review statistical conclusions                 | All                   | Compare results against meaningful-improvement criterion and inspect limitations                                                                | Approved analysis memo            | Claims follow prespecified criteria; inconclusive and negative outcomes are acceptable |

**Hybrid run protocol.** Fit both tuned components on the permitted history and score the same full candidate pairs. For Item-CF weight w, compute w × Item-CF predicted stars + (1 − w) × MF predicted stars. Sort descending and break movie-score ties by ascending movie ID. Select the interior weight by mean 2017 NDCG@10. P1.6 proposes resolving exact weight ties by proximity to 0.5, then the smaller Item-CF weight, and resolving a standalone tie in favor of Item-CF. Review the proposed regularized bias fallback and freeze a consistent clipping policy before the sweep. Report all three final NDCG means and estimate the paired difference between the hybrid and the frozen comparator. Evaluate coverage and computational cost alongside relevance.

P3.4, P3.5, and P3.7 are if-time diagnostics in the task tracker. The approved evaluation protocol still records the four user-history cutoffs for sensitivity analysis.

### Phase 3 execution record

| Task | Current state and evidence |
| ---- | -------------------------- |
| P3.1 | Locally verified. `src/hybrid.py` fills missing component scores with the λ=10/10 fallback, blends raw unclipped scores and ranks through the shared pipeline. The 2017 endpoint check (`outputs/phase3_hybrid_endpoint_check.json`) passed: w=0 and w=1 reproduce the standalone MF and Item-CF grids and top-10 lists exactly, and match the saved Phase 2 lists for all 4,778 users. |
| P3.2 | Provisional. The cache-based sweep of w = 0.0–1.0 selected w = 0.2 (NDCG@10 0.098614, +0.003131 over MF; coverage 1,375, +193). Records: `outputs/phase3_weight_sweep.csv` and `outputs/phase3_provisional_selection.json`. |
| P3.3 | P3.3 draft pending all-member approval. Draft memo and configuration: `deliverables/MovieLens_P3.3_Frozen_Experiment_Draft.md` and `deliverables/phase3_frozen_experiment_draft.json`. Every decision is pending, including P1.6 amendment A1 (clipping and blending rule) and the provisional 0.01 margin. |
| P3.4–P3.10 | Not started. P3.9 must not begin until P3.3 is approved and P1.9 and P2.9 approvals are recorded. |

### Phase 3 Deliverables

- Approved weighted-hybrid implementation
- Validation weight-selection results
- Frozen final model and evaluation configuration
- Model-agreement diagnostic figures if time permits
- Final held-out benchmark table
- Paired confidence intervals
- Catalogue coverage and popularity-exposure comparison
- User-history subgroup analysis if time permits
- Reviewed statistical interpretation memo

### Phase 3 Acceptance Criteria

The phase is complete when:

1. Hybrid parameters have been selected exclusively using permitted validation information.
2. The primary comparison is against the validation-selected individual personalized model.
3. Popularity remains an independently reported reference benchmark.
4. Final test evaluation is performed using frozen methods and rules.
5. The group reports uncertainty and distinguishes meaningful improvement, negligible differences, deterioration and inconclusive findings appropriately.
6. Coverage and popularity exposure use consistent denominators.
7. All claims are restricted to offline observed-positive retrieval rather than actual satisfaction.
8. Results can be reproduced using archived configurations.

**Phase gate:** The project must be able to answer its primary research question before optional experiments are considered.

---

# 6. Phase 4 — Report, Interpretation and Presentation

**Suggested duration:** Week 4  
**Primary responsibility:** Shared  
**Goal:** Produce a defensible final report and ensure every member understands the complete study.

| Task                                 | Responsible | Action / Details                                                                          | Deliverable                 | Acceptance Criteria                                                          |
| ------------------------------------ | ----------- | ----------------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| P4.1 Write data and EDA sections     | A           | Describe MovieLens, activity distribution, popularity concentration and filtering         | Data/EDA draft              | All statistics are traceable to reproducible outputs                         |
| P4.2 Write methodology               | B           | Explain baselines, Item-CF, MF, hybrid and tuning                                         | Methods draft               | Model assumptions, settings and limitations clearly stated                   |
| P4.3 Write evaluation methodology    | C           | Describe temporal design, metrics, uncertainty and decision rule                          | Evaluation draft            | Enough detail for another researcher to reproduce comparisons                |
| P4.4 Prepare results and figures     | B + C       | Select essential tables, ranking plots and coverage results                               | Figures and results section | Every figure uses approved final outputs and clear labels                    |
| P4.5 Interpret findings              | All         | Explain primary comparison, secondary outcomes and model disagreement                     | Discussion draft            | Conclusions distinguish evidence from conjecture                             |
| P4.6 Document limitations            | A + C       | Explain selective ratings, observational feedback, coverage interpretation and exclusions | Limitations section         | No unsupported claims about true user preferences or online performance      |
| P4.7 Finalize practical implications | All         | Assess whether additional hybrid complexity is justified                                  | Conclusion                  | Directly answers the approved research question                              |
| P4.8 Review report and references    | All         | Check consistency, citations, reproducibility and presentation quality                    | Final report                | No conflicting metrics, undocumented experimental changes or missing sources |
| P4.9 Prepare presentation            | All         | Create slides and assign speaking sections                                                | Presentation                | Every member contributes substantively                                       |
| P4.10 Conduct technical rehearsal    | All         | Review full pipeline, results and possible questions                                      | Rehearsal notes             | Every member can defend key methodological decisions and results             |

### Phase 4 Deliverables

- Completed project report
- Final figures and tables
- Methodology and reproducibility appendix
- Presentation slides
- Approved final experiment archive
- Group technical rehearsal notes

### Phase 4 Acceptance Criteria

The phase is complete when:

1. The report directly answers the approved research question.
2. The primary result includes appropriate uncertainty and interpretation.
3. All tables and figures match the approved experiment outputs.
4. Data selection, tuning and final evaluation procedures are documented.
5. Limitations and fallback decisions are transparently reported.
6. All members have reviewed the report and can explain the complete study.
7. Final files are placed in the approved Google Drive folders.

---

# 7. Work Allocation Summary

This is an initial allocation and can be adjusted by agreement.

| Phase                   | Member A                                                                   | Member B                                                                   | Member C                                                                   |
| ----------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1. Protocol and data    | Lead eligibility, data pipeline and EDA                                    | Define model interfaces and computational requirements                     | Lead metrics, comparison rules and inference plan                          |
| 2. Individual models    | Validate data and filtering                                                | Lead model implementation and tuning                                       | Lead common ranking evaluation                                             |
| 3. Hybrids and analysis | Catalogue, user-history and exposure analysis                              | Support hybrid integration and model diagnostics                           | Lead hybrid tuning, bootstrap inference and final evaluation               |
| 4. Reporting            | Data, EDA and limitations                                                  | Methods and computational results                                          | Evaluation, uncertainty and main results                                   |
| Shared activities       | Peer review, protocol decisions, integration, conclusions and presentation | Peer review, protocol decisions, integration, conclusions and presentation | Peer review, protocol decisions, integration, conclusions and presentation |

## Integration and Review Ownership

- **Integration owner:** Assign during kickoff.
- **Protocol changes:** Require unanimous approval.
- **Ordinary code changes:** Require one peer review plus an integration check.
- **Final test execution:** Require confirmation from all three members that configurations are frozen.

---

# 8. Optional Extensions and Fallback

## 8.1 Optional Extensions

Only consider these after the Phase 3 core experiment is functioning:

| Extension                        | Value                                                         | Constraint                                                   |
| -------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| Popularity-penalized scoring     | Explores relevance versus exposure trade-offs                 | Must not replace primary comparison                          |
| History-dependent hybrid weights | Tests whether user history helps choose weights               | Pilot improvement was very small                             |
| Reserved-slot reranking          | Quantifies relevance impact of enforced less-popular exposure | Exposure increase itself is mechanically guaranteed          |
| Gradient-boosting reranker       | Tests a nonlinear second-stage combination                    | Requires additional feature engineering and leakage controls |
| Ranking-oriented MF              | Tests whether training objective affects ranking performance  | Adds a model family and tuning workload                      |

**Rule:** An extension requires an explicit owner, estimated effort and group approval. No optional work should jeopardize report completion.

## 8.2 Fallback Plan

If full-catalogue ranking becomes impractical, use a documented, reproducible smaller candidate catalogue and disclose how that affects coverage interpretation.

If hybrid implementation or reliable hybrid evaluation proves infeasible, reduce the study to popularity, Item-CF and matrix factorization, focusing on their held-out ranking behavior and catalogue exposure.

This fallback **answers a narrower question about individual recommendation methods rather than the approved hybrid-effectiveness question**. Any such scope change must be documented and agreed upon by the team.

---

# 9. Key Risks and Controls

| Risk                           | Potential effect                                          | Control                                                           |
| ------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------- |
| Selectively observed ratings   | Offline ranking metrics may misrepresent user preferences | Interpret results as observed-positive retrieval only             |
| Data leakage                   | Inflated performance estimates                            | Historical-only eligibility; temporal validation; code review     |
| Nonrepresentative catalogue    | Distorted coverage and popularity comparisons             | Prefer full eligible catalogue or disclose documented sampling    |
| Unequal tuning effort          | Unfair model comparison                                   | Agreed tuning budget and selection protocol                       |
| Computational limitations      | Delayed experiments                                       | Profile models, prioritize essential methods, documented fallback |
| Small observed improvements    | Uncertain adoption conclusions                            | Predefined practical margin and paired intervals                  |
| Google Drive version conflicts | Nonreproducible results                                   | Member workspaces, review queue, versioned approved releases      |
| Scope expansion                | Delayed core completion                                   | Freeze mandatory scope before optional additions                  |
| Member availability            | Uneven workload                                           | Weekly review, task ownership and early escalation                |

---

# 10. Project Management and Monitoring

## 10.1 Task Statuses

Use the following status labels in the shared task tracker:

- Not Started
- In Progress
- Ready for Review
- Revisions Requested
- Approved
- Integrated
- Blocked

A task is considered **complete** only when its deliverables and acceptance criteria are satisfied, not merely when code has been written.

## 10.2 Weekly Review

Hold one scheduled group review each week covering:

1. Tasks completed and accepted.
2. Tasks currently blocked.
3. Review-queue items awaiting approval.
4. Changes affecting other members.
5. Upcoming phase-gate requirements.
6. Scope or schedule adjustments.

## 10.3 Required Experiment Log Fields

Each model run should record:

- Experiment ID and date
- Responsible member
- Approved code version
- Dataset and configuration version
- Training period and validation period
- Model parameters and random seed
- Evaluation population and candidate catalogue
- Metrics, runtime and output location
- Known limitations and review status

## 10.4 Definition of Project Completion

The project is complete when:

- All required models and evaluations have approved outputs.
- The primary comparison has been assessed using the predefined decision rule.
- Catalogue coverage and observed-positive ranking relevance are reported together.
- The main findings include appropriate uncertainty estimates.
- Every result is reproducible from archived code and configuration.
- The report acknowledges the limits of selectively observed ratings.
- All three members approve the final conclusions and can explain the complete methodology.

---

# 11. Remaining Kickoff and Phase 1 Actions

Confirm which kickoff items the team has already handled. The immediate technical actions are to choose the first P1.8 smoke-test library and reproduce P1.7 independently.

| Priority | Action                                                            | Owner             | Target        |
| -------- | ----------------------------------------------------------------- | ----------------- | ------------- |
| 1        | Confirm actual project deadline and adjust the four-week schedule | All               | Kickoff       |
| 2        | Assign Members A, B and C and select integration owner            | All               | Kickoff       |
| 3        | Create Google Drive folder structure                              | Integration owner | Kickoff       |
| 4        | Agree on review and approval procedures                           | All               | Kickoff       |
| 5        | Confirm Phase 1 ownership and validate the adapter interface      | All               | Kickoff       |
| 6        | Establish shared task tracker and experiment log                  | Integration owner | Early Phase 1 |
| 7        | Review the P1.2–P1.7 packages and record approvals                | Assigned owners   | Phase 1       |

**First major milestone:** An approved experimental protocol and reproducible dataset pipeline that all three members can use.

**Guiding principle:** Complete one statistically defensible, reproducible comparison that answers the approved research question before expanding the experiment set.

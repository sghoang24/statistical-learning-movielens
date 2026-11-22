# MovieLens 25M P3.3 Frozen Experiment — Draft for Team Review

**Status — DRAFT. Every decision below is pending approval by all three members. Nothing is frozen yet.** No 2018–2019 data has been read or scored, no model has been refit, and the P3.2 sweep has not been rerun. The machine-readable version of this draft is `deliverables/phase3_frozen_experiment_draft.json`; each decision there has `"status": "pending"` and blank approver fields.

## 1. What we are asked to approve

| # | Decision | Proposal | Status |
|---|---|---|---|
| 1 | Primary comparator | Matrix factorization (MF) | Pending |
| 2 | Hybrid weight | Item-CF w = 0.2, MF 0.8 | Pending |
| 3 | Component and fallback settings | As in section 4 | Pending |
| 4 | Scoring, ranking and clipping rules | As in section 4, including P1.6 amendment A1 | Pending |
| 5 | Practical margin δ | 0.01 absolute mean NDCG@10 (provisional) | Pending |
| 6 | Uncertainty method | Paired user bootstrap: 10,000 resamples, seed 42, two-sided 95% percentile interval | Pending |
| 7 | Final-test plan | Pre-2018 refit, 2018–2019 evaluation (section 5) | Pending |
| 8 | Phase gates | Record P1.9 and P2.9 approvals before any final-test scoring | Pending |

Once all eight are approved, P3.3 is complete and the configuration is frozen. After that, nothing in it may change on the basis of 2018–2019 results.

## 2. Validation evidence (2017 cohort, 4,778 users)

**Comparator.** P1.6 picks the stronger standalone personalized model by mean 2017 NDCG@10. MF scored 0.095484 and Item-CF 0.059745, so the rule gives MF.

**Weight sweep (P3.2).** The hybrid scores each candidate as w × Item-CF + (1 − w) × MF. The endpoints (w = 0 is MF alone, w = 1 is Item-CF alone) are references and cannot be selected. The rule picks the interior weight with the highest NDCG@10 at full precision; exact ties go to the weight closest to 0.5, then to the smaller weight. There was no tie: w = 0.2 is the unique best.

| w | Mean NDCG@10 | Mean Recall@10 | Coverage (movies) | Less-popular exposure | Top-10 slots scored directly by Item-CF |
|---:|---:|---:|---:|---:|---:|
| 0.0 (MF, reference) | 0.095484 | 0.040890 | 1,182 | 0.2726 | 42.7% |
| 0.1 | 0.097226 | 0.041153 | 1,198 | 0.2958 | 45.4% |
| **0.2 (proposed)** | **0.098614** | **0.041787** | **1,375** | **0.3064** | **51.0%** |
| 0.3 | 0.098127 | 0.041804 | 1,801 | 0.3017 | 58.9% |
| 0.4 | 0.096362 | 0.041336 | 2,336 | 0.2848 | 67.6% |
| 0.5 | 0.092508 | 0.039751 | 2,828 | 0.2596 | 75.4% |
| 0.6 | 0.086892 | 0.037534 | 3,280 | 0.2332 | 81.7% |
| 0.7 | 0.079696 | 0.034559 | 3,701 | 0.2084 | 86.5% |
| 0.8 | 0.072998 | 0.031932 | 4,048 | 0.1881 | 90.0% |
| 0.9 | 0.066133 | 0.028522 | 4,413 | 0.1728 | 92.5% |
| 1.0 (Item-CF, reference) | 0.059745 | 0.025483 | 4,627 | 0.1593 | 94.2% |

Source: `outputs/phase3_weight_sweep.csv`; plot in `outputs/phase3_weight_sweep.png`. "Scored directly" means the cell had an Item-CF score rather than a fallback score. At w = 0 that column describes MF's picks, not Item-CF's influence.

NDCG@10 is nearly flat between w = 0.1 and 0.3 (0.0972, 0.0986, 0.0981), while coverage rises steeply from w = 0.3 onward. The sweep does not measure how stable the choice of 0.2 over 0.3 is. Both are selected on the same 2017 cohort, so the 2017 gain of the selected weight is likely somewhat optimistic.

**Selected weight (w = 0.2) compared with MF (w = 0), 2017 validation:**

| Measure | MF | Hybrid w = 0.2 | Change |
|---|---:|---:|---:|
| Mean NDCG@10 | 0.095484 | 0.098614 | **+0.003131** (+3.3% relative) |
| Mean Recall@10 | 0.040890 | 0.041787 | +0.000897 |
| Coverage (distinct movies) | 1,182 | 1,375 | **+193** (+16.3%) |
| Less-popular exposure | 0.2726 | 0.3064 | +0.0338 |
| Total hits in top 10 | 3,949 | 4,062 | +113 |
| Total hits in top 3 | 1,441 | 1,499 | +58 |

| Per-user change | Count |
|---|---:|
| Users with higher NDCG@10 / lower / equal | 1,000 / 873 / 2,905 |
| Users with more / fewer hits in top 10 | 526 / 451 |
| Users with more / fewer hits in top 3 | 264 / 200 |
| Average top-10 movies shared with MF | 7.65 of 10 |

Source: `outputs/phase3_provisional_selection.json` (`comparison_with_mf_comparator`).

## 3. Practical margin — for team decision

**Proposal on the table: keep the provisional P1.6 margin δ = 0.01 absolute mean NDCG@10.** This draft presents it unchanged. It does not revise δ to fit the validation result.

What the team should weigh, per P1.6:

- **Relevance:** the observed 2017 gain is +0.003131, about a third of δ. Under the P1.6 decision table, a final-test gain of similar size would be classified by its confidence interval; it would not count as clearing the margin.
- **Coverage and exposure:** +193 distinct movies (+16.3%) and +3.4 percentage points of less-popular slots.
- **Hits:** 113 more top-10 hits in total. 526 users gained at least one hit and 451 lost at least one; the net change is modest and in both directions.
- **Cost (provisional figures):** the hybrid needs both components plus the fallback.
  - In the Phase 2 validation run, on a 20-CPU / 16 GB machine, MF took 230.6 s to fit and Item-CF 229.5 s. The λ=10 fallback fit was not timed separately; the λ=320 bias baseline took 4.7 s. Fitting the hybrid therefore costs roughly double what MF alone costs.
  - The 61.9 s P3.2 sweep evaluated all eleven weights from the cached component grids, without fitting or scoring any model. It is **not** a single hybrid's end-to-end runtime.
  - A configured hybrid's full runtime, from fitting both components and the fallback through to its top-10 lists, has not been measured. The final evaluation must measure and report it (section 5).

P1.6 requires the team to choose the smallest gain it would consider worthwhile given this coverage and runtime trade-off. It also warns against setting the margin to match the hybrid's observed gain. Any revision must be justified on those grounds and recorded with its reasons before any 2018–2019 scoring. Record one of:

- [ ] Keep δ = 0.01. Reason: ____________________
- [ ] Revise δ to ______. Reason (independent of the observed +0.003131): ____________________

## 4. Proposed frozen configuration

**Components** (fitted on the phase history; settings as recorded in the grid cache and endpoint report):

| Component | Settings |
|---|---|
| Item-CF | adjusted-cosine similarity, `top_k = 3`, `min_similarity = 0` |
| MF (comparator and component) | explicit ALS with biases, `n_factors = 24`, `regularization = 30`, bias λ_user = λ_movie = 10, `max_iter = 20`, `tolerance = 1e-4`, `seed = 0`. Fitting setting only: the training-RMSE stopping monitor clips to [1, 5] (`monitor_clip_min/max`). It decides when fitting stops and is not applied to predictions, ranking or the hybrid. |
| Fallback | bias model, λ_user = λ_movie = 10, `max_passes = 20`, `tolerance = 1e-6` (separate from the tuned λ = 320 bias baseline) |
| Hybrid | w = 0.2 Item-CF + 0.8 MF |

**Scoring, ranking and clipping rules** (P1.6 amendment A1, pending approval):

- Both components are scored on the same ordered arrays of original user IDs and movie IDs.
- Missing or non-finite component scores are filled with the shared fallback before blending, with counts recorded per component. In 2017, 81,597,241 of 85,836,770 Item-CF cells (95.1%) were filled and MF needed none.
- Blend the raw, unclipped component scores.
- Rank the blended scores **without clipping**. Movies the user rated before the recommendation date are excluded after blending, exact ties go to the lower movie ID, and 10 movies are returned.
- For RMSE and MAE, blend the raw scores first, then clip the resulting rating prediction once to [0.5, 5].
- MF's [1, 5] training-RMSE stopping monitor is a fitting setting (see the table above). It is not part of this rule.

**Protocol amendment (pending):** P1.6 step 2 previously said to "apply any clipping consistently before the blend." The P1.6 document now shows the rule above as proposed amendment A1, awaiting approval by all three members. It is the rule the P3.1 endpoint check and P3.2 sweep used. Clipping before the blend would stop w = 1 from reproducing the standalone Item-CF ranking. Approving decision 4 includes approving A1.

**Metrics:** as in P1.5. Positives are held-out ratings above 3. Binary NDCG@10 and Recall@10 are averaged over users with at least one positive. Coverage is distinct recommended movies divided by 17,965. Less-popular exposure is the share of slots outside the top 3,593 movies by pre-2017 count, with ties broken by movie ID.

## 5. Planned final test (P3.9, after approval only)

| Item | Plan |
|---|---|
| Refit history | All ratings before 1 January 2018 UTC, including 2017 (22,488,700 rows expected) |
| Refit | Item-CF, MF and fallback with the frozen settings and seeds; no retuning |
| Recommendation date | 1 January 2018 UTC |
| Scoring window | 1 January 2018 to 1 January 2020 UTC (exclusive) |
| Catalogue and less-popular classes | Unchanged: fixed pre-2017 catalogue of 17,965 movies and pre-2017 top 3,593 |
| User eligibility | At least one pre-2017 rating (P1.2 rule, frozen) |
| Positives | 2018–2019 ratings above 3 on catalogue movies the user had not rated before 2018 |
| Evaluated users | Eligible users with at least one test positive |
| Exclusions | Every movie the user rated before 2018, including 2017 ratings |
| Popularity reference | Counts recomputed on pre-2018 ratings; reported separately, not the comparator |
| Report | All three NDCG@10 means (hybrid, MF, Item-CF), the mean paired difference and its interval; Recall@10, coverage and less-popular exposure |
| Runtime (required) | Measure and report the configured hybrid's full runtime, from fitting Item-CF, MF and the fallback through scoring, blending and top-10 generation, under the same conditions as MF alone. Report the amortized sweep cost separately, per P1.6. |

The test positives file and the final-test runner do not exist yet and must be built under P3.9, after approval.

**Proposed paired user bootstrap (decision 6):**

- For each evaluated test user u, compute d_u = NDCG@10(hybrid) − NDCG@10(MF).
- Draw 10,000 resamples of the users with replacement, each the size of the original cohort, using `numpy.random.default_rng(42)`. Keep each user's pair together by resampling d_u.
- Compute the mean of d in each resample.
- Report the two-sided 95% percentile interval: the 2.5th and 97.5th percentiles, using NumPy's default linear interpolation.
- Interpret it with the P1.6 decision table, using strict comparisons:
  - lower bound > δ: the gain clears the margin;
  - lower bound > 0 and the interval contains δ: the gain is positive, but whether it clears the margin is uncertain;
  - upper bound < δ: a gain as large as δ is ruled out;
  - the interval includes both 0 and δ: the direction and size remain uncertain.
- Report inconclusive and negative outcomes as they are.

## 6. Identity and provenance of the validation evidence

| Item | Value |
|---|---|
| Training history (pre-2017) | 20,798,765 rows; fingerprint `25b04961…a309e8` |
| Raw ratings SHA-256 | `9c855257…6e385` |
| Cohort | 4,778 users (`2fad14e4…bd07`) × 17,965 movies (`6305a786…b954c`) |
| Component grid cache | `artifacts/grids/hybrid-components-before-2017-01-01.npz`, content `f732ba17…a71a` |
| Model artifacts | Item-CF `cd4dacb2…aed`, MF `7dfd8b7c…123f`, fallback `2bab97ff…782d` |
| P3.1 endpoint check | Passed; w = 0 and w = 1 match the saved Phase 2 lists for all 4,778 users; report `adcf2e47…2812` |
| P3.2 selection record | `outputs/phase3_provisional_selection.json` (`d9d035d9…`) |
| Libraries | Python 3.13.2, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1 |

Full hashes are in the draft JSON.

**Code provenance.**

- **The runs were not made from a clean checkout.** The P3.1 endpoint check (completed 14:21:41 UTC, 22 November 2026) and the P3.2 sweep (completed 14:38:16 UTC) both ran on base commit `1213c4d4af758665dbe9546b15edbe03c60a3d4a` with uncommitted changes. Their records say `git_uncommitted_changes: true`, and that is preserved here.
- **Commit `6df249b9c4bf14cea0977bc173f9290f67d82bd9`** (15:04:58 UTC) was made **after** both runs. It records the Phase 3 sources and outputs, and is the subsequently committed source snapshot.
- **What the snapshot is checked against:** the SHA-256 hashes recorded at run time for `src/hybrid.py`, `src/pipeline.py`, `src/item_cf.py`, `src/matrix_factorization.py`, `src/bias.py` and `src/models.py` all equal those files at `6df249b`. The runner and cache modules (`run_hybrid_endpoints.py`, `run_hybrid_sweep.py`, `grid_cache.py`, `model_artifacts.py`) were not hashed at run time. The commit records their state as committed afterwards, but it cannot independently prove that those exact file versions are the ones that ran.
- The committed sweep outputs match the working copies, apart from line endings in the CSV.

## 7. Known limitations to weigh before approving

- **Item-CF relies heavily on the fallback.** At `top_k = 3`, 95.1% of Item-CF grid cells are fallback bias scores. Even so, at w = 0.2, 51.0% of top-10 slots carry a direct Item-CF score. The hybrid's gain therefore mixes Item-CF and bias-model effects.
- **Grid-edge settings.** Item-CF `top_k = 3` and the λ = 320 bias baseline were selected at the edges of their Phase 2 grids, and MF's 24 factors is the largest tried. Phase 2 review (P2.9) is still pending; changing any of these would mean redoing P3.1–P3.2.
- **Selection optimism.** The weight and its +0.003131 gain both come from the same 2017 cohort.
- **Offline, observed-positive evaluation only.** The results say nothing about how users would respond to the lists.

## 8. Approval record

| Decision | Member A | Member B | Member C | Date |
|---|---|---|---|---|
| 1. Comparator: MF | Pending | Pending | Pending | |
| 2. Weight w = 0.2 | Pending | Pending | Pending | |
| 3. Component and fallback settings | Pending | Pending | Pending | |
| 4. Scoring/ranking/clipping rules and P1.6 amendment A1 | Pending | Pending | Pending | |
| 5. Practical margin δ | Pending | Pending | Pending | |
| 6. Bootstrap settings | Pending | Pending | Pending | |
| 7. Final-test plan | Pending | Pending | Pending | |
| 8. P1.9 and P2.9 approvals recorded | Pending | Pending | Pending | |

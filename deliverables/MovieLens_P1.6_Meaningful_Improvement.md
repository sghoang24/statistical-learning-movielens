# MovieLens 25M P1.6 Meaningful Improvement

**Status** — Protocol ready for teammate review. The numeric relevance margin is provisional pending validation. Proposed amendment A1 (clipping and blending, step 2) is **pending team approval**; see "Proposed amendments" below.

## Primary comparison

Compare the tuned Item-CF + MF hybrid with the stronger standalone personalized model, chosen by mean observed-positive NDCG@10 on 2017 validation. Freeze the hybrid configuration and the choice of comparator before final-test scoring. Use the same selected comparator for every user.

Score Item-CF, MF, and the hybrid on the same eligible test users and candidate sets under P1.3–P1.5. Keep one row per user with user ID, the three NDCG@10 scores, and the paired difference d = hybrid NDCG@10 − selected standalone NDCG@10. Report the three model means and the mean paired difference.

## Validation runs to freeze the models

The following grid is the proposed initial run plan for P3.1–P3.3. P2.3–P2.5 supply the tuned standalone models and their validation records. The hybrid owner can run this grid once those inputs are available.

1. **Freeze the standalone inputs to the search.** Use the selected Item-CF and MF settings from 2017 validation and fit both components on pre-2017 data. Record each component's configuration, seed, and training policy. Choose the standalone comparator by the higher mean validation NDCG@10 at full precision. Proposed tie rule for an exact tie is Item-CF.
2. **Prepare aligned rating predictions.** Use the 4,778 primary validation users and their P1.4 candidate sets. Produce one predicted star rating per component for each eligible user–movie pair and align by user ID and movie ID. Retain the full candidate predictions because the blend can promote movies outside both standalone top tens. The proposed fallback for a missing or non-finite prediction is the P2.2 regularized bias baseline fitted on the same phase history. Apply the same fallback in standalone tuning, standalone evaluation, and hybrid evaluation. Record the affected counts per component. Use full-precision predictions on the original rating scale. **[Proposed amendment A1, pending team approval — replaces "apply any clipping consistently before the blend":]** Fill missing or non-finite component scores with the shared fallback before blending, blend the raw, unclipped component scores, and rank the blended scores without clipping. For RMSE and MAE, blend the raw scores first, then clip the resulting rating prediction once to [0.5, 5]. Document and freeze the clipping policy.
3. **Blend predicted ratings and rank.** Use one global Item-CF weight w. For each candidate, compute w × Item-CF predicted rating + (1 − w) × MF predicted rating. Sort by the blended prediction descending, break exact ties by ascending movie ID, and take the first 10. Both components estimate stars, so direct blending preserves their predicted rating gaps. Their prediction spreads can differ even on the same scale. The validation grid measures which weighting works best for ranking.
4. **Run nine hybrid configurations.** Test Item-CF weights 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9. The MF weight is 1 − w. Also run w = 0 and w = 1 as checks that the blend reproduces the MF and Item-CF top tens, respectively. Fit each component once for this grid and reuse its full validation predictions across weights. Compare ordered top-10 movie IDs for every user at each endpoint using the same fallback, clipping, and tie rules.
5. **Select and record the winner.** Save a row per weight with mean NDCG@10, Recall@10, coverage, less-popular exposure, and timing. Keep the per-user scores and ranked lists for the selected weight and the standalone comparator. Choose the interior weight with the highest mean validation NDCG@10 at full precision. For exact ties, prefer the weight closest to 0.5, then the smaller Item-CF weight. Keep the selected hybrid in the final comparison even if its validation score trails a standalone. For runtime comparisons, measure a single configured hybrid including both components and blending, separately from the amortized cost of the validation sweep.
6. **Freeze the final-test configuration.** Save the selected weight, both standalone configurations, comparator, rating scale, clipping and missing-score policies, tie rules, code revision or snapshot, seeds, final refit policy, relevance margin, and bootstrap settings. Attach the validation grid results and the rationale for the margin. Refit the components on ratings before 1 January 2018 UTC, including 2017, using the frozen settings. Blend with the frozen weight and evaluate on 2018–2019.

## Provisional relevance margin

**δ = 0.01 absolute mean observed-positive NDCG@10**, equivalent to one percentage point on the 0–100 NDCG scale. This is a working minimum worthwhile relevance gain over the selected standalone. It is a team judgment to review using 2017 evidence.

Before final testing, review actual validation recommendation lists against observed 2017 positives. For the hybrid and selected standalone, inspect:

- How many users gained or lost at least one observed hit in the top 3 and top 10.
- The change in total observed hits and the rank changes for positives retrieved by both methods.
- The change in distinct titles and catalogue coverage percentage points.
- Fit and top-10 generation times measured under the same conditions, with the hybrid charged for both components and any sharing or reuse documented.

Use these comparisons to judge the smallest relevance gain the team would accept given the coverage and runtime trade-off. Record whether the team keeps or revises 0.01, the validation evidence, and the reason. Avoid setting the margin to match the hybrid's observed gain. Freeze the margin before any 2018–2019 model scoring or result inspection.

The feedback is based on real held-out MovieLens ratings and measures offline retrieval of observed positives. It does not estimate how users would respond to being shown those lists.

## Paired uncertainty and decision rule

Use a paired user-level bootstrap on the final-test cohort. Resample evaluated users with replacement, drawing the original number of users each time. Keep both scores together by resampling their difference. Compute the mean difference in each resample. Proposed reproducible settings are 10,000 resamples, seed 42, and a two-sided 95% percentile interval using the 2.5th and 97.5th percentiles. Confirm these settings in review.

| Interval result | Interpretation |
|---|---|
| Lower bound exceeds δ | Evidence that the mean relevance gain clears the practical margin. |
| Lower bound exceeds zero and the interval contains δ | Evidence of a positive mean gain. Whether it clears the margin remains uncertain. |
| Upper bound is below δ | The interval rules out a gain as large as δ at the stated confidence level. If the lower bound exceeds zero, the gain is positive but below the target. |
| Interval includes zero and reaches δ | The direction and practical size of the effect remain uncertain. |

Use strict comparisons for claims that a bound exceeds or is below a threshold. Assess the overall hybrid trade-off alongside the relevance decision using Recall@10, catalogue coverage, less-popular exposure, and measured computational cost.

## Proposed amendments

### A1 — Clipping and blending rule (step 2) — pending team approval

| | Text |
|---|---|
| Original wording | "Use full-precision predictions on the original rating scale and apply any clipping consistently before the blend." |
| Proposed wording | Fill missing or non-finite component scores with the shared fallback before blending, blend the raw, unclipped component scores, and rank the blended scores without clipping. For RMSE and MAE, blend the raw scores first, then clip the resulting rating prediction once to [0.5, 5]. |

**Reason.** This is the rule implemented in `src/hybrid.py` and used in the P3.1 endpoint check and the P3.2 validation sweep. Phase 2 ranked every standalone model on unclipped scores. Clipping before the blend would let scores saturate at 5 and break ties by movie ID, so w = 1 would no longer reproduce the standalone Item-CF ranking. MF's training-RMSE stopping monitor, which clips to [1, 5] while fitting, is a separate fitting setting. It is not part of this prediction or blending rule.

**Status.** Proposed, not yet approved. Protocol changes require approval from all three members. Until it is approved, the P3.1 and P3.2 results should be read as produced under the proposed rule.

| Member A | Member B | Member C | Date |
|---|---|---|---|
| Pending | Pending | Pending | |

## Review checklist

- Confirm the primary comparator selection rule, paired data format, and proposed bootstrap settings.
- Review the provisional 0.01 margin and the validation evidence required to confirm or revise it.
- Approve or reject amendment A1 (clipping and blending rule) and record each member's decision.
- Record the frozen comparator, model settings, margin, and inference settings before final testing.

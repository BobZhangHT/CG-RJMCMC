# Frozen E-M02 real-application checking protocol

Frozen before inspecting results on 2026-09-08. The nominal factorial-count-prior (`p(K) proportional to 1/K!`) joint-partition analysis remains primary; every extension below is a bounded sensitivity analysis and is not a candidate model-selection exercise.

## Data and sampling rule

The two randomized condition-order groups are fit separately after exactly reproducing `run_r61_pc_flanker_application.py`: source-study participant exclusions, correct visual Flanker observations, blocks 2--6 and 8--12, reaction time at least 100 ms, exclusion of block-first and post-catch trials, 64-bin mapping, log response centered by the frozen training mean, and the existing participant-stratified hashed holdout. The nominal sufficient statistics must reproduce the existing implementation before fitting is allowed.

Each model uses three chains, 15,000 sweeps, 2,000 warm-up sweeps, initial cut counts 0/2/4, minimum segment length 8, maximum four cuts, and `p(K) proportional to 1/K!`. The existing C sampler is used unchanged. To make the baseline an exact regeneration check, nominal seeds retain the original `R61-PC-FLANKER-CHAIN-v1` namespace; extension seeds occupy the new namespace `R61-PC-FLANKER-E-M02-v1`. All artifacts use new paths, so no existing chains or outputs are overwritten.

For contrast design `W` and nuisance design `Z`, nuisance coefficients have the proper conditional prior `nu | sigma^2 ~ N(0, sigma^2 D_nu^{-1})`. We pass the generalized collapsed summaries

`Q = I - Z (D_nu + Z'Z)^{-1} Z'`, `H = W'QW`, `h = W'Qy`, and `r = y'Qy`

to the unchanged sampler. All normal priors are conditional on `sigma^2`. The six cases are fixed as follows:

1. `nominal`: intercept precision 0.25, linear-bin-trend precision 1, participant-intercept precision 1.
2. `block10`: nominal columns plus one indicator for each of the ten retained experimental blocks, each with precision 4 (conditional variance `sigma^2/4`). The proper priors make the intentionally overcomplete intercept-plus-indicators parameterization nonsingular.
3. `ri_var_0p25`: participant-intercept variance ratio 0.25 (precision 4).
4. `ri_var_4`: participant-intercept variance ratio 4 (precision 0.25).
5. `slope_var_0p01`: nominal columns plus participant-specific congruency slopes with fixed mean-zero variance ratio 0.01 (precision 100).
6. `slope_var_0p1`: the same with variance ratio 0.1 (precision 10).

The two slope variances and all nuisance precisions are fixed in advance, not estimated or tuned to the observed conclusion.

## Comparisons and diagnostics

For every group and case we report `PIP(32)`, `PIP(12)`, `P(K)`, joint maximum-posterior partition, posterior mean curve, held-out RMSE conditional on existing participants, sampler diagnostics, and change relative to nominal. Phase summaries are computed per retained posterior trajectory: bins 1--32 and 33--64 are averaged within each draw, their after-minus-before difference is formed in that draw, and only then are posterior means and 95% intervals computed. Pointwise interval endpoints are never averaged to form a phase interval.

Nominal residual summaries are stratified by participant, congruency, and experimental phase and include mean, SD, quantiles, and tail rates. Participant-specific congruency contrasts and their dispersion are reported. Native-order lag-1 residual correlation is computed only within a participant and retained block; source `trialNum` distance is recorded. A strict distance-one estimate is reported separately. No pair crosses a block, the omitted practice block, or a phase boundary.

## Posterior predictive checks

For each nominal group, 1,000 posterior predictive replicates are drawn conditional on the existing participants. A retained partition is sampled from the fitted chains. For its segment design `S`, define `X_full=[S,Z]`, prior precision `D_full=blockdiag(4 I,D_nu)`, and posterior precision `Lambda_post=D_full+X_full'X_full`. Draw

`sigma^2 ~ InvGamma(3+n/2, 2 + (y'y-b' Lambda_post^{-1} b)/2)`

and then draw all segment effects and nuisance coefficients jointly from `N(Lambda_post^{-1}b, sigma^2 Lambda_post^{-1})`; generate replicated outcomes from that joint draw. Thus the PPC propagates structure, effect, nuisance, participant, and residual uncertainty rather than bootstrapping a constant residual distribution.

Paired observed/replicated checks cover residual location, dispersion and two-sided tails; phase-by-congruency residual location; maximum participant residual mean; participant congruency-contrast heterogeneity; and within-block native-order lag-1 dependence. Descriptive posterior-predictive folded-discrepancy tail areas and central predictive intervals are reported; these are diagnostics rather than calibrated frequentist tests. A small tail area is retained as evidence of mismatch, not optimized away.

## Post-review participant-level analysis

This analysis was added after editorial review and is not part of the frozen sensitivity protocol above. For each participant and phase, the code subtracts the nominal posterior-mean nuisance contribution from each training response, calculates the residual incongruent-minus-congruent mean difference, and adds the nominal phase-average effect. Each participant therefore contributes one adjusted contrast before and after the reversal. Their paired after-minus-before difference is the participant-level outcome.

Group estimates weight participants equally. The direct order-by-phase contrast is the low-first mean change minus the high-first mean change. Percentile intervals use 50,000 deterministic nonparametric bootstrap samples of complete participants within each randomized-order group; response rows are never resampled as independent units. The analysis supports or challenges the group-level direction of the phase change. It does not recalibrate posterior boundary probabilities or the nominal model's conditional segment intervals. Results are written to `participant_level_summary.csv`, with participant-phase inputs retained in `participant_contrasts.csv`.

The analytic Student-\(t\) intervals reported for the selected application partitions are produced by the separate precision check. The application stage retains empirical sampled endpoints, while `application_conditional_interval_check.csv` records the analytic values in fields beginning with `exact_` and the sampled values in fields beginning with `empirical_`.

## Stop and interpretation rules

Stop before scientific interpretation if preprocessing/input hashes or frozen sampler settings drift, nominal `H/h/r` fail to reproduce the original implementation, a required precision system is not positive definite/finite, posterior covariance tests fail, or an output would be overwritten. Failed convergence diagnostics remain visible and prevent a robustness claim but do not authorize silent reruns or tuning. Predictive mismatch or a changed bin-12/bin-32 conclusion is reported honestly. Agreement among these fixed-prior cases does not establish that every possible real-data misspecification has been repaired.

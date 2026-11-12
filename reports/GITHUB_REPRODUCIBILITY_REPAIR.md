# GitHub reproducibility repair

Date: 12 November 2026. Baseline commit: 57b7a48d22aadb6fd355b7e25f23e075a9b1f90a.

## Outcome

The local public-source candidate implements the revision calculations and passes the complete cached-input revision workflow. Native clean-environment acceptance remains subject to the host's Windows App Control block; this is not an unconditional reproducibility-ready declaration. No commit, push, release or remote CI dispatch was performed. The remote repository has not been changed.

## Repairs

- Made scripts/probe.c and the core synthetic tests Git-eligible. The probe includes the official kernel.c and calls its posterior/proposal functions. Added input-buffer, dimension, dtype, legal-boundary and integer-range checks; cache keys cover both C sources and build flags. No duplicated C posterior formula.
- Added small_sample_validation.py. Formal SN0/SN1 preserve the original cyclic21-layout-to35-participant design, independent synthetic intercepts/errors, masks, seeds, prior and15,000/2,000 budget. Cache mode regenerates outcome identities and aggregates1000 original fits. Smoke uses labelled synthetic layouts; it is not formal evidence.
- Added constrained_dp_validation.py. Independent unrestricted/count-capped DP and centered brute force evaluate the same weighted objective. Delayed PELT pruning remains separate from K<=4. Input/config/scientific-dependency fingerprints guard resume; the D5 input bridge works for fresh and cached workflows.
- Added exact_d5_500.py. Full mixture CDF inversion reuses public posterior/Student-t routines, all27,931 components and strict crossing certification. Workers, checkpoints and a wall budget are explicit. Cached verification is a separate mode and records root_evaluations_performed=false.
- Added revision_tables.py for Table2A/B/C, Table3 primary/N35/coverage/diagnostics and Table5. It reads actual outputs, handles public N35 CSV metadata/booleans, preserves conditional-level denominators and supports31 single-thread PELT repetitions plus IQR. Direct-module and CLI timing imports both work.
- Integrated revision stages and revision-smoke in reproduce.py. Default all explicitly excludes revision; all --include-revision runs the defined expensive workflow. Cache options compose without missing N35 records or D5 statistics. figures.py can call the revised table generator; final panels are in revision_tables/.
- Preserved the historical D5 namespace in the unified configuration, including outcome/holdout/MCAR/chain seeds. An explicit pilot --seed-namespace consistently overrides all DGPs; formal defaults do not change.
- Centered values before prefix SSE to prevent large-offset cancellation, with no clipping or objective change. Full archived objectives remained within tolerance; no selected-partition changes arose from this numerical-stability repair.
- Declared threadpoolctl3.6.0, exposed public pytest in Ubuntu CI, updated README and kept prior source-lock provenance. Normalized public text to Git's LF convention; immutable protocol records were excluded. Portable synthetic counterexample fixtures replace the last private test-file dependency.

## Numerical checks actually executed

The independent objective audit covered5600 datasets (3000 primary,1600 sensitivity,1000 N35) and two application groups. Primary count-cap changes remain21. The maximum difference from previously verified objectives was2.16e-12, below the stated tolerance. D1 replication66 gives102.143556537 to100.302134206; SN0 replication352 gives87.100979739 to84.925047559. Solver correction and feasible-set changes remain distinct.

All1000 N35 cached outcome/seed/layout identities passed. CG correct-count recovery remains1.000 and0.920; average coverage remains0.9705625 and0.93875. Both N35 diagnostic pass counts are500.

All500 exact caches and original input/endpoints were verified and reaggregated. Original coverage is93.634375%, exact-mixture coverage93.709375%, paired difference+0.075 percentage points, paired MCSE0.036859935 percentage points. There are147 endpoint discrepancies above0.002; maximum0.05728088195, median0.00128709436, q95=0.01366805164. These are computed aggregate checks, not a new full CDF run.

The current table generator reproduces manuscript-verified Tables2/3/5. Independent comparison checked90 Table2 numeric cells and the N35 coverage/diagnostic fields. The actual cached revision command completed all four stages with exit0. A new62-row timing execution (31 per group) also passed; its medians were0.0036168s and0.0036750s. These new functional-check timings do not replace the archived manuscript medians0.00485s and0.00480s; timings vary with the execution environment.

## Independent reviews

Four real reviewer roles examined statistics, numerical algorithms, reproducibility and CI. Confirmed issues included missing conditional-level metrics, dependency/fingerprint/certificate checks, cache composition, package imports and source-only test fixtures. Those issues were repaired and tested. The remaining explicit namespace override issue was also corrected with a dedicated test. Bayesian/numerical semantic reviews are same-family and provisional; source hashes do not constitute numerical proof.

## Preservation and remaining boundary

Formal kernel.c is byte-identical, SHA256 d36e15cb174012078649b489a361a71278b0fbec14cb717dc244bd9ab94a6171. The model, priors, DGP values, original formal seeds and budgets are retained. Historical caches and PLOS submission files were not overwritten; no manuscript result was changed.

The public full recomputation path is implemented, but no new3000+1600+1000 formal sampler campaign or fresh all500 posterior inversion was undertaken in this repair. Local results use explicit immutable cache inputs; fresh generation uses public code and OSF input instructions. Remote Linux CI was not executed. See TEST_RESULTS.md for actual command statuses and PUBLIC_RELEASE_CHECKLIST.md for the outstanding native execution gate.

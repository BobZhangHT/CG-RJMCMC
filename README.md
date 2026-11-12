# CG-RJMCMC: changing congruency effects in flanker tasks

CG-RJMCMC estimates a population difference in mean log reaction time across unknown task phases. Participant random intercepts and a shared linear progression term account for response-speed variation. The collapsed Gaussian sampler averages over partitions; PELT-BIC and Constrained DP-BIC provide single-partition comparisons. The code supports PLOS ONE manuscript PONE-D-26-45560.

## Requirements and installation

Use Python 3.12 and a GCC-compatible C compiler (GCC on Linux; MinGW-w64 on Windows). Exact dependency versions, including the direct `threadpoolctl` dependency, are in `pyproject.toml`. The sampler and the posterior-weight probe compile automatically. Generated libraries and results are not source files.

```bash
python -m pip install -e ".[test]"
gcc --version
python scripts/reproduce.py validate
python -m pytest -q
python scripts/reproduce.py smoke --workers 2
python scripts/reproduce.py revision-smoke --workers 2
```

`validate` checks SHA256 source integrity, installed dependency versions and GCC. The two smoke workflows use small synthetic data/budgets and are engineering tests, not manuscript results. Public tests include dense-reference collapsed weights, MH detailed balance, Student-t conditionals, weighted PELT/DP, mixture inversion, layouts/seeds and table aggregation. Real native tests compile and load C libraries; they are not replaced by mocks when a host blocks native execution. Windows App Control can reject a newly compiled DLL with error 4551; that is a blocked runtime test, not a numerical pass. Use a policy-permitted environment; the Ubuntu CI workflow executes the real tests. Remote CI success is not claimed until actually run.

## Model and fixed protocols

The formal model, `scripts/kernel.c`, priors, proposal transitions and MCMC budget are unchanged. Formal runs use T=64, minimum segment length 8, Kmax=4, three chains, 15,000 iterations/chain and 2,000 warm-up iterations. Primary D0-D5 have 500 independent datasets per scenario; the 16 sensitivity cells have 100 each. D5 is exploratory and result-informed.

`r61_formal.yaml` explicitly preserves the historical D5 namespace `R61-D5-BALANCED-v1`; D0-D4 retain `R61-FORMAL-v1`. This matters for outcome, holdout, missingness and chain seeds in a unified run. It does not choose new seeds. N35 uses its original namespace and cyclic assignment of the 21 low-first observation layouts, with independent synthetic participant intercepts/errors. The initial 200 and later 300 replications share that seed rule; the code does not present all500 as initially preregistered.

Weighted segmentation minimizes within-segment weighted SSE plus log(T) per segment. PELT defers pruning until the dominating prefix can form a legal segment. The independent DP does not prune; a capped version restricts K<=4. Prefix moments are formed after a common shift to avoid cancellation from large offsets. Independent centered-SSE enumeration checks tiny spaces and the two reported anomaly cases.

## External data

Download and extract `RawData_Experiment1.7z` from the [source investigators' OSF project](https://osf.io/6n597/), associated with [Braeutigam et al.](https://doi.org/10.3758/s13421-023-01447-x). Place the 50 original CSV files directly in:

```text
data/external/RawData_Experiment1/
```

The application runner records each file hash and the aggregate data hash. The manuscript input hash is `b61c1f400be01e79a34b633ede24f50cf0ec844c4f8b3d85076df4eb572ed480`; differing inputs must be investigated, not renamed to match. Processing and exclusions are in `scripts/config/application_protocol.md` and `application.prepare_group`.

Empirical participant records and their historical caches are not redistributed. The separate [Zenodo record](https://doi.org/10.5281/zenodo.14838065) concerns simulation condition sequences; the frozen geometry file in this repository supplies the simulation design. OSF empirical records and Zenodo sequences have different roles.

## Primary simulations and application

```bash
python scripts/reproduce.py simulation --workers 8
python scripts/reproduce.py application --application-input data/external/RawData_Experiment1
python scripts/reproduce.py checks --application-input data/external/RawData_Experiment1
```

The default output root is `results/reproduction`; assets go to `manuscript/reproduced`. `simulation` runs the 3,000 primary and 1,600 sensitivity datasets. `application` fits both groups; `checks` runs the five fixed precision cases and application model checks. The precision cases use chains independent of the main simulations. These stages are computationally substantial.

The default `all` command runs the original primary/sensitivity/application/checks/figures/export workflow and public tests. **It excludes the new revision validations and final added table panels.** To request the complete defined revision workflow explicitly:

```bash
python scripts/reproduce.py all --include-revision --workers 8 --application-input data/external/RawData_Experiment1 --output-root results/reproduction_final --assets manuscript/reproduced_final
```

This additionally runs N35, the full solver audit, all500 D5 CDF inversion and final table panels. It can take substantial time. It is not an ordinary CI job. Use `--dry-run` to inspect actual commands. `all` starts new campaigns; do not use it merely to postprocess historical caches.

## N=35 simulations

```bash
python scripts/small_sample_validation.py --smoke --output-root results/n35_smoke
python scripts/small_sample_validation.py --full --application-input data/external/RawData_Experiment1 --output-root results/n35_full --workers 8
```

SN0 is constant at0.08; SN1 changes from0.04 to0.14 after bin32. Formal mode runs 500 independent datasets per cell. The 35 synthetic participants inherit cyclic observation layouts, not repeated participant outcomes. Fresh effects/errors, preserved train/test assignments and original seeds are recorded. Smoke uses explicitly synthetic layouts without OSF records and is not application-calibrated validation.

Outputs include `small_sample_results.csv`, `small_sample_summary.csv/json`, `small_sample_coverage.csv` and `small_sample_manifest.json`. The manifest records scope, settings, seeds, input hashes and dependency versions. Existing matching outputs can resume; identity mismatches require a new directory. `--cache-root` with `--design-input` (or the raw application path) verifies and aggregates original N35 caches without new MCMC; it never treats copying rows as independent fitting.

## Corrected PELT and constrained DP

```bash
python scripts/constrained_dp_validation.py --smoke --output-root results/dp_smoke
python scripts/constrained_dp_validation.py --full --simulation-root results/reproduction/simulation --small-sample-root results/reproduction/revision/small_sample --application-input data/external/RawData_Experiment1 --output-root results/reproduction/revision/benchmark --workers 8
```

The full audit checks 3,000 primary, 1,600 sensitivity and 1,000 N35 datasets plus both application groups. It reconstructs hash-verified sufficient statistics from original outcomes/seeds when necessary, never reruns the sampler, and compares unrestricted PELT with an independent exact DP. Constrained DP and centered brute-force checks use the same penalty/indexing/minimum length. Solver corrections and count-cap changes are reported separately. A cap-induced partition change is not a solver error.

Outputs include per-dataset objectives, `solver_audit.json`, benchmark summaries, paired differences and a manifest. Reconstructed D5 statistics/provenance are exposed at `statistics/D5` and `replicates/D5` for the exact-mixture stage. `--cache-root` accepts previously saved audit inputs, recalculates objectives and verifies input identities. For legacy split campaigns use `--primary-root`, `--balanced-d5-root` and `--small-sample-root`; do not mix obsolete unbalanced D5 results into the final comparison. The two anomaly objectives are calculated, not embedded as returned results.

## Exact posterior intervals

```bash
python scripts/precision.py --smoke --output results/precision_smoke
python scripts/exact_d5_500.py --smoke --output results/exact_smoke
python scripts/precision.py --run --case d5_rep_000003 --iterations 100000 --warmup 2000 --simulation-results results/reproduction/simulation --output results/precision_long_case3
python scripts/exact_d5_500.py --full --simulation-results results/reproduction/simulation --statistics-root results/reproduction/revision/benchmark/statistics/D5 --records-root results/reproduction/revision/benchmark/replicates/D5 --output results/reproduction/revision/exact_d5 --workers 8
```

All500 mode consumes original saved draw quantiles and matching statistics; it does not create new outcomes or rerun main MCMC. The full posterior sums 27,931 legal partitions with log-sum-exp weights and conditional Student-t components. Marginal endpoints invert the complete mixture CDF, not averaged component quantiles. Auxiliary truncation is only a bracketing aid. Full-support CDF crossings at endpoint+/-1e-8 and strict sign margins >1e-13 certify location; low-density gaps can fail certification even with a small CDF residual.

`--preflight` validates all inputs without computing intervals; `--preflight --preflight-replications 1,3` checks bounded originals. Full mode supports process workers, checkpoints, a wall budget, resume and deterministic per-case outputs. No case is excluded to improve coverage. Output includes endpoints, per-dataset errors, pointwise/near/far coverage and widths, paired replication-level MCSEs and provenance. Existing independent-unit MCSE uses datasets, not the64 correlated positions.

To audit an existing exact cache, use `--reaggregate-cache --cached-exact-root ARCHIVE --output NEW_DIR`. That verifies saved input/output hashes, original endpoints, full support and saved certificates and recomputes aggregate statistics. It explicitly records `root_evaluations_performed=false`: it is not a fresh posterior inversion. Historical results must not be relabelled as a new run.

## Final figures and Tables 2/3/5

```bash
python scripts/reproduce.py figures
python scripts/reproduce.py revision --workers 8 --application-input data/external/RawData_Experiment1
python scripts/reproduce.py figures --include-revision --application-input data/external/RawData_Experiment1
```

`revision` runs N35, solver validation, all500 exact intervals and final tables, using existing primary/application results. Individual stages are `small-sample`, `benchmark`, `exact-d5` and `tables`. To process historical data without new MCMC, supply `--small-sample-cache`, `--design-input`, `--benchmark-cache`, `--exact-cache` and `--runtime-repeats` as appropriate. `--simulation-results` and `--application-results` select read-only existing roots. Results and caches are not bundled with the public source; full fresh commands above create the necessary inputs.

The six figures and original table layouts are under the asset root. **Final revised Tables 2/3/5 are under `revision_tables/`**, with CSV/Markdown panels and compact TeX. Table2 includes Panels A/B/C; Table3 includes primary structure, N35 structure, coverage/MCSE/width and diagnostics. The table generator reads analysis outputs, never constants transcribed from the manuscript.

```bash
python scripts/revision_tables.py --simulation results/reproduction/simulation --application results/reproduction/application/results.json --benchmark results/reproduction/revision/benchmark --small-sample results/reproduction/revision/small_sample --timing --application-input data/external/RawData_Experiment1 --output-root manuscript/reproduced/revision_tables
```

`--timing` performs one untimed warmup and31 single-thread corrected PELT repetitions per group. Table5 preserves the original CG three-chain execution times and records the different protocols. To retain an actual earlier timing run, replace `--timing --application-input ...` with `--runtime-repeats PATH_TO_REPEATS_CSV`. Timings depend on hardware/load and are not expected to reproduce published digits on another machine. `figures.py --revision-results REVISION_ROOT` also calls the final table generator; supply saved repeat timings or the raw application path.

## Result-to-code map

| Manuscript result | Entry | Main output |
|---|---|---|
| D0-D5 and sensitivity | simulation.py | replicate summaries, curves, diagnostics |
| SN0/SN1 | small_sample_validation.py | small_sample_results/summary/coverage |
| PELT repair, common-cap DP | constrained_dp_validation.py | objectives, solver_audit, paired summaries |
| Fixed precision/longer chains | precision.py | case summaries and trajectory/discrete diagnostics |
| D5 all500 exact-mixture | exact_d5_500.py | exact endpoints, coverage, endpoint discrepancies |
| Application predictions | application.py | results.json, heldout_rmse.csv |
| PPC/specification/bootstrap | checks.py | predictive and contrast summaries |
| Tables2/3/5 and31-repeat timing | revision_tables.py | revision_tables/*.csv/md/tex, manifest |
| Six figures | figures.py, simulation_figures.py | PDFs, separate TIFFs, source data |

## Integrity, resume and release status

Source-lock history retains the previous release's fingerprints. Current-lock validation is a source-integrity check, not proof of numerical accuracy. Always choose a fresh `--output-root` and `--assets` for a new software version; output guards protect manuscript sources and historical results. Original simulation resume requires matching fingerprints; revised validation stages bind inputs and scientific dependencies and reject mismatches. Failed/budget-limited exact runs retain completed cases for an identical-identity resume.

`python scripts/reproduce.py supporting` is an optional historical fixed-check evidence export; it copies existing outputs byte-for-byte and is not independent validation or a required manuscript supplement. `python scripts/reproduce.py export --export-dir NEW_EMPTY_DIRECTORY` exports eligible public source/tests/reports without Git commit or push.

See the executed-test status in [TEST_RESULTS](reports/TEST_RESULTS.md), the [reproduction matrix](reports/REPRODUCTION_MATRIX.md), [repair report](reports/GITHUB_REPRODUCIBILITY_REPAIR.md), and [release checklist](reports/PUBLIC_RELEASE_CHECKLIST.md). These distinguish Passed, Failed, Not executed, missing-data blocks and host-policy blocks. Pipeline implementation or cached verification alone does not mean a fresh all500 numerical rerun has passed.

## License and citation

Code is [MIT licensed](LICENSE). Citation metadata are in [CITATION.cff](CITATION.cff). External participant data retain their source terms; no participant records are redistributed here.

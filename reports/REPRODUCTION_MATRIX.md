# Reproduction matrix

Run from repository root after installation. RAW means the50 CSV files extracted from the public OSF Experiment1 archive. No participant records are redistributed. Use a new output root when source/input fingerprints differ.

| Result | Public entry | Inputs | Executable command | Main output |
|---|---|---|---|---|
| Primary/sensitivity | scripts/simulation.py | frozen geometry/config | `python scripts/reproduce.py simulation --workers 8` | simulation replicate JSON/curves/summary |
| Application | scripts/application.py | RAW | `python scripts/reproduce.py application --application-input data/external/RawData_Experiment1` | application/results.json, heldout_rmse.csv |
| N35 SN0/SN1 | scripts/small_sample_validation.py | RAW and its outcome-blind layout | `python scripts/reproduce.py small-sample --workers 8 --application-input data/external/RawData_Experiment1` | revision/small_sample/*.csv and manifest |
| PELT/unrestricted/capped DP | scripts/constrained_dp_validation.py | primary/sensitivity/N35 results and RAW | `python scripts/reproduce.py benchmark --workers 8 --application-input data/external/RawData_Experiment1` | revision/benchmark/solver_audit.json, objectives, statistics/D5 |
| Fixed precision/PPC | scripts/precision.py, scripts/checks.py | simulation/application results and RAW | `python scripts/reproduce.py checks --application-input data/external/RawData_Experiment1` | precision cases, predictive checks, bootstrap summaries |
| Extended third D5 precision | scripts/precision.py | original simulation inputs | `python scripts/precision.py --run --case d5_rep_000003 --iterations 100000 --warmup 2000 --simulation-results results/reproduction/simulation --output results/precision_long_case3` | independent longer-run case diagnostics |
| All500 exact-mixture | scripts/exact_d5_500.py | original draw endpoints and matching benchmark statistics | `python scripts/reproduce.py exact-d5 --workers 8` | revision/exact_d5 endpoints/coverage/errors/manifest |
| Six figures | scripts/figures.py, scripts/simulation_figures.py | simulation/application outputs | `python scripts/reproduce.py figures` | reproduced figures and separate TIFFs |
| Final Tables2/3/5 | scripts/revision_tables.py | simulation, benchmark, N35, application; RAW for timing | `python scripts/reproduce.py tables --application-input data/external/RawData_Experiment1` | manuscript/reproduced/revision_tables/*.tex/csv/md |
| Synthetic integration | precision/small_sample/DP/exact | no participant data | `python scripts/reproduce.py revision-smoke --workers 2` | explicitly non-manuscript smoke outputs |
| Full defined revision | orchestration above | RAW, primary/application outputs | `python scripts/reproduce.py revision --workers 8 --application-input data/external/RawData_Experiment1` | all revision stages and final panels |

## Cached-input routes

N35 `--cache-root` verifies original layouts/seeds/outcome hashes and aggregates existing fits without new chains. DP `--cache-root` recomputes independent objectives from identity-checked sufficient statistics. Exact `--reaggregate-cache` verifies saved artifacts/certificates and reaggregates them; it does not invert CDFs afresh. Table `--runtime-repeats` aggregates an actual31-repeat CSV instead of remeasuring. README gives flags and scope. The default `all` excludes revision unless --include-revision is supplied.

No rows in this matrix alone certify execution. Actual statuses are in TEST_RESULTS.md. Historical caches are optional inputs, not supplied private Python dependencies; the fresh generation path uses public code and public raw-data sources.

## Final verified execution scope

The cached-input `revision` command completed N35 identity validation, independent5600+2 objective recomputation,500-case exact-cache verification/reaggregation and final table generation with exit0. It did not run new formal chains or invert all500 CDFs. The fresh source-only directory installed declared dependencies and passed source validation and pure DP/mixture smoke. Native host-policy failures and exact final test counts are documented separately in TEST_RESULTS.md.

The final source-only public package needs no private Python modules. Archived caches are optional read-only inputs; users can generate inputs through the public fresh path after obtaining the original OSF records. Source-lock history and protocol files preserve their historical identity; current-source integrity alone is not numerical acceptance.

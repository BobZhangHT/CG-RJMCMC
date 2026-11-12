# Executed test results

Date: 12 November 2026. All statuses below describe actual executions. No tests were deleted or newly skipped to obtain a pass.

## Environment

Windows x64; Python3.12.10; GCC16.1.0 MinGW-w64; NumPy2.4.6; SciPy1.16.3; pandas2.3.3; matplotlib3.11.0; ArviZ1.2.0; pytest9.1.1; threadpoolctl3.6.0. An independent virtual environment installed the declared requirements, and a new source-only directory contained no historical data/results/build artifacts before acceptance execution. No system-site-packages or undeclared diagnostic shim was used. ArviZ rank/bulk APIs also executed successfully in that isolated installation.

## Final command results

| Check | Status | Actual scope |
|---|---|---|
| Python compile and strict GCC syntax/warnings | Passed | All27 public Python files; kernel.c/probe.c with -Wall -Wextra -Werror -fsyntax-only |
| Current source lock, pinned dependencies and compiler | Passed | 42 exact LF-source hashes; previous-release hashes retained |
| Independent source-only installation | Passed | Declared test extra and direct runtime dependencies; no historical build artifacts supplied |
| Latest complete isolated pytest | **Failed** | **167 passed,2 failed,0 errors,1 skipped;170 test cases** |
| Table/timing focused tests | Passed |12 tests, including actual synthetic solver executions for31 repetitions/group plus warmups |
| Independent full solver audit | Passed |5600 datasets+2 application groups; cap changes21; prior objective differences at most2.16e-12 |
| N35 cached identity validation | Passed |1000 original layout/seed/outcome identities, no new formal sampler campaign |
| Exact all500 preflight/cache verification | Passed | Original input/endpoint hashes,27931 weights/support and saved full-CDF certificates; aggregates recomputed |
| Current cached revision workflow | Passed | N35 -> solver audit -> exact-cache verification -> final tables, exit0 |
| Current final table generation/comparison | Passed | Tables2/3/5 identical to independently manuscript-checked outputs;90 Table2 cells and N35 diagnostic fields match |
| Actual new timing execution | Passed |62 rows,31 single-thread corrected PELT repetitions per group; published timing records not overwritten |
| Clean DP and exact-mixture synthetic smoke | Passed | Tiny synthetic inputs, no raw participant data or C DLL required |
| Clean primary/N35/combined revision native smoke | Failed, host-policy blocked | DLL load raised WinError4551; these commands are not counted as passes |
| Remote Ubuntu GitHub Actions | Not executed | Workflow configured for real pytest and lightweight smokes; no push/dispatch performed |
| New formal3000+1600+1000 sampler campaign | Not executed | Historical correct caches retained and verified instead |
| Fresh all500 exact posterior inversion | Not executed | Full public numerical path exists; this repair verified cached endpoints/certificates and reaggregated, with root_evaluations_performed=false |
| Missing-data block in local cached verification | None | All required historical cases and both application groups were present |

## Exact remaining failed tests

| Test | Status | Reason |
|---|---|---|
| `scripts.tests.test_kernel::test_finite_state_kernel` | Failed, host-policy blocked | `OSError: [WinError 4551] 应用程序控制策略已阻止此文件。` |
| `scripts.tests.test_simulation::test_parameterized_c_kernel_runs_kmax5_minseg6_without_retaining_files` | Failed, host-policy blocked | `OSError: [WinError 4551] 应用程序控制策略已阻止此文件。` |

The failed tests remain enabled in ordinary pytest and CI. The separate skipped test reports: `Full-study generated evidence not installed; fixture-based package tests still run.`. It is not an AppControl failure suppression. Earlier development executions did exercise the real C probe, dense log-weight comparison and188-state MH/full100k sampler checks successfully; those do not replace the final clean-suite failure above.

## Reproduced numerical values

D1 r66:102.143556537 ->100.302134206; SN0 r352:87.100979739 ->84.925047559. Corrected PELT agrees with independent unrestricted DP on5600+2 cases. The21 primary cap-induced partition changes remain distinct from solver repairs.

D5 original/exact coverage:93.634375% /93.709375%; paired difference+0.075 percentage points; MCSE0.036859935 percentage points;147 datasets with maximum endpoint discrepancy>0.002; maximum0.05728088195, median0.00128709436, q95=0.01366805164. These values come from verified original arrays and recomputation of summaries, not hardcoded program outputs or fresh full root finding.

N35 correct-count recovery remains1.000/0.920 and coverage0.9705625/0.93875, with500 diagnostically valid fits per cell. New timing medians were0.0036168/0.0036750seconds. Their difference from published0.00485/0.00480seconds reflects a new timing execution, not a change to the manuscript dataset or recorded results; timing protocols and source inputs are explicit.

## Corrections made during testing

Resolved missing native wrapper/tests, private-only regression inputs, source-only imports, incomplete presentation fixture, missing direct dependency, source-lock/line-ending mismatch, D5 namespace identity, explicit namespace override, cache bridge/fingerprints, conditional-level metrics and public CSV metadata/boolean handling. The presentation test now supplies a completed participant summary and additionally forbids numerical recalculation. Portable counterexamples contain only synthetic aggregate inputs, and objectives are independently recomputed.

The first isolated test invocation had an unquoted absolute JUnit path containing spaces and exited before collection; the harness was corrected to a relative path. It is not counted as test execution. Early failed transcripts are retained; latest results supersede their counts. No security policy, raw records or historical result files were changed to force success.

## Evidence and acceptance boundary

Machine evidence is retained locally under tmp/github_reproducibility_repair_20261112: pytest_final_current.xml/log/status; final_clean_acceptance.json; final_composed_acceptance.json; final_table_comparison.json; new_timing_check_complete/timing_acceptance.json; static_acceptance.json; review_A/B/C/D reports and worker run records. Source hashes in each result identify its actual execution; LF normalization was separately verified to preserve Python ASTs, and current table generation/timing used the final code.

Release status: implemented and cache-validated; native clean-environment acceptance remains blocked by host policy. Run the unchanged real native gates in a policy-permitted environment before declaring unconditional reproducibility readiness. Remote Linux execution and a fresh full numerical campaign remain distinct, explicitly unexecuted checks.

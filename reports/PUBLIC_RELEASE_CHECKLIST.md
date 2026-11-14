# Public release checklist

Updated: 14 November 2026. This source release adds the completed targeted reviewer diagnostics to the existing public repository. Only source, protocols, tests and public documentation are eligible for publication.

- [x] Original formal kernel, DGPs, seeds, budgets and historical results preserved.
- [x] N35 identity verification, independent solver audit, exact-mixture verification and table generation implemented and checked (12 November scope recorded separately).
- [x] Prior/oracle diagnostic completed on all 500 original D5 datasets, with six conditions and certified endpoints.
- [x] Sample-size and individual-timing diagnostic completed on all 500 paired replications, with all 3,000 new fits and no exclusions.
- [x] Six residual specifications fitted in both application groups, with exact Gaussian mixtures and diagnostic-qualified Student chains.
- [x] Diagnostic extension and historical execution identities documented without pooling overlapping runs.
- [x] Public table generator reads residual-sensitivity outputs for Table 5 Panel B; baseline generation remains available.
- [x] README distinguishes original orchestration from the three explicit targeted diagnostic commands.
- [x] Public source exporter excludes participant records, manuscript files, experiment outputs, private audits and binaries.
- [x] Regression tests cover the added numerical routines and table parsing; actual execution counts are recorded in TEST_RESULTS.md.
- [ ] Entire native clean-environment suite green on this Windows host: App Control blocks specific DLL loads; these tests remain enabled.
- [ ] Remote Linux CI acceptance: check the GitHub Actions run attached to the published commit; local tests do not establish its status.
- [ ] Fresh original 3,000 primary + 1,600 sensitivity + 1,000 N35 sampler campaign: not rerun in this source update.

The 12 November repair verified saved all-500 exact-mixture results rather than performing fresh inversion. The later prior/oracle and calibration diagnostics performed their stated new calculations; reuse of original-prior baseline endpoints is explicit. These are separate execution scopes. See TEST_RESULTS.md and REPRODUCTION_MATRIX.md for inputs, commands and remaining host limitations.

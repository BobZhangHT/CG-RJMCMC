# Public release checklist

Date: 12 November 2026. Local source repair complete; no remote publication performed.

- [x] probe.c is eligible for Git and reuses official kernel functions.
- [x] Original formal kernel, model, DGPs, budgets, historical seeds and results preserved.
- [x] N35 formal/cache/smoke paths and original observation-layout protocol implemented.
- [x] Independent PELT/unrestricted/capped DP audit executed for5600+2 cases.
- [x] Both original anomalies and21 primary cap changes reproduced.
- [x] All500 exact-mixture generation path, resumability and certificate checks implemented.
- [x] All500 saved exact results verified/reaggregated without claiming fresh CDF inversion.
- [x] Final Tables2/3/5 generated from actual outputs; public N35 CSV round trip tested.
- [x] Real31-repeat-per-group single-thread timing executed; original manuscript times retained separately.
- [x] Public synthetic unit tests and portable counterexample fixtures available.
- [x] CI configuration executes pytest, source validation and bounded self-contained smoke.
- [x] README commands implemented; default all versus revision opt-in is explicit.
- [x] Source lock validates current LF source and retains previous-release identities.
- [x] Independent source-only directory and isolated declared-dependency installation validated.
- [ ] Entire native clean-environment test/smoke suite green: Windows App Control currently blocks specific DLL loads; failures are retained.
- [ ] Remote Ubuntu CI executed and successful: not dispatched or pushed in this task.
- [ ] Fresh full sampler/500-case inversion campaign rerun: not executed and not claimed.
- [ ] Author-authorized publication: no commit/push/release performed; remote remains unchanged.

Release status: reproduction pipeline implemented and cache-validated; native acceptance blocked on this host; full fresh numerical rerun pending. Do not relabel as unconditionally reproducibility-ready until the real native gates run successfully in a policy-permitted environment. No security policy or failed-test selection was changed to force a pass.

# A Bayesian change-point mixed model for changing congruency effects in flanker tasks

Reproduction code for a Bayesian mixed model of changing congruency effects. Participant random intercepts and a common progression trend are integrated out when sampling the shared change-point partition. The comparison methods are NoEffect, ConstEffect, FixedSplit, PELT-BIC and CG-RJMCMC.

## Layout

Only two top-level directories are published. Manuscript sources, results, downloaded data and environments remain local.

```text
scripts/
  reproduce.py          # Single command-line entry point
  simulation.py         # Simulations, baselines and resumable records
  application.py        # PC-Flanker preprocessing and group fits
  precision.py          # Coverage, paired comparisons and exact-posterior checks
  checks.py             # Application sensitivity and predictive checks
  figures.py            # All main figure/table exports and submission TIFFs
  simulation_figures.py # Simulation plotting functions used by figures.py
  package.py            # Curated supporting-data archive and verification
  kernel.c              # Collapsed sampler and conditional draws
  config/               # Study settings, condition geometry and provenance
.github/workflows/      # Automated source-only checks
```

The public release contains eight Python scripts and one C source file. Plotting, computation and evidence packaging remain separate so that each module has a clear job. Regression tests and their C probe are retained locally and excluded from GitHub.

## Install and quick check

Python 3.12 and a GCC-compatible C compiler are required. The tested Windows setup uses Python 3.12.10 and MinGW-w64. Use an isolated environment if desired, then install the project:

```shell
python -m pip install -e .
gcc --version
python scripts/reproduce.py validate
python scripts/reproduce.py smoke --workers 2
```

Dependency versions are defined once in `pyproject.toml`; transitive dependencies are resolved by pip. `validate` checks the current source hashes and compiler. Public CI runs validation and a small reproduction smoke run, without requiring local-only tests. The smoke run uses one short replication per cell and is not the formal experiment.

## Reproduce the paper

Download `RawData_Experiment1.7z` from the [source OSF project](https://osf.io/6n597/) and extract its 50 participant CSV files to `data/external/RawData_Experiment1`. The source article is [Braeutigam et al.](https://doi.org/10.3758/s13421-023-01447-x). Check and retain the source data's access and reuse terms; MIT does not relicense these data.

```shell
python scripts/reproduce.py all --workers 12 --application-input data/external/RawData_Experiment1
```

The full study includes 3,000 primary and 1,600 sensitivity datasets and can take substantial time. Choose the worker count for your machine. To run stages separately:

| Stage | Inputs | Output |
|---|---|---|
| `simulation` | Included configuration and condition geometry | 4,600 replicate records and aggregate summaries |
| `application` | Raw Experiment 1 CSV files | Two order-group fits, trajectories and held-out prediction |
| `checks` | Full simulation/application results and raw files | Coverage, paired comparisons, five exact/MCMC cases, 12 sensitivity fits and predictive checks |
| `figures` | Full simulation and application results | Tables 1–5, six figures and six named submission TIFFs |
| `supporting` | Completed upstream outputs | Curated evidence ZIP, schema/index and SHA-256 manifest |

Example: `python scripts/reproduce.py figures`. Add `--dry-run` to print commands without executing them. Omitting the stage runs only `validate`.

Results default to `results/reproduction/`; assets default to `manuscript/reproduced/`. Use `--output-root` and `--assets` to select new subdirectories. The driver rejects protected historical locations. Simulation resume requires an identical run fingerprint; for changed code/settings choose a fresh root. Other stages rebuild their selected output files. The supporting stage requires result paths inside the repository so packaged paths stay relative.

The figures stage generates assets, not manuscript prose or the complete article PDF. The supporting-data ZIP is a curated subset, not a replacement for raw data and all replicate records. Verify it with:

```shell
python scripts/package.py --verify manuscript/reproduced/supporting_information/review_supporting_data
```

## Data and model conventions

The included `scripts/config/r61_congruency_geometry.csv` contains 49 observed condition sequences of length 64 from [Lost in Migration](https://doi.org/10.5281/zenodo.14838065), released under CC0. Its columns are `anon_id`, `nth_play`, `selection_hash` and `bits`. They describe condition geometry, not response-time outcomes. The simulation repeats this fixed geometry and independently generates participant effects and residuals. The original archive's SHA-256 is `066dde1ebc592581e7d1dfdc60614ef45760fe3d3dc002fde339f594f083eeb7`.

The empirical loader uses `vpNum` (participant), `blockNum`/`trialNum` (native order), `type` (flanker/catch), `corrCode` (correctness), `modality`, `rt` (milliseconds), `congruency` and `block_congruency` (condition order). It applies the fixed participant exclusions, omits practice blocks 1 and 7, and retains eligible correct visual responses. Native positions map to 64 bins without averaging responses. Incongruent/congruent trials are coded +0.5/−0.5; centering and fitting use training data only. The split is outcome-independent, and prediction concerns eligible responses from participants represented in training.

Numerical accuracy, model adequacy and interval calibration are different questions. Preserve the paper's D5 recovery/coverage limitations and empirical residual discrepancies when interpreting results. Neither runtime nor mixing diagnostics establish overall superiority. Timing varies across executions and is not expected to match the paper exactly.

## Provenance and public release

The two archival protocols in `scripts/config/` preserve their original bytes. Their embedded historical paths/hashes describe earlier runs; moving the files is not new preregistration. `reproduction_sources.json` instead locks the current executable release. Refresh that lock only after reviewing intentional source changes. New manifests record the sources actually executed. Historical precision-manifest rewriting requires both `--finalize-provenance` and `--rewrite-historical-manifest`.

List the files eligible for GitHub, or export them to a new empty directory:

```shell
python scripts/reproduce.py export
python scripts/reproduce.py export --export-dir results/reproduction/public_source
```

These commands do not stage, commit or push. Do not force ignored manuscript, results or raw data into Git. Publish the code and an appropriate persistent results archive before claiming public availability. Local testing does not certify remote GitHub CI or numerical identity on other platforms.

## License and citation

Code: [MIT License](LICENSE). Third-party data retain their own terms. [CITATION.cff](CITATION.cff) identifies the authors and accompanying manuscript; no publication DOI is invented.

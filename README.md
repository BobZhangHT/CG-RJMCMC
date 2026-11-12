# Bayesian change-point mixed model for changing congruency effects

This repository implements **CG-RJMCMC**, a Bayesian change-point mixed model for estimating how the incongruent-versus-congruent response-time effect changes during a flanker task. It accounts for participant heterogeneity and uncertainty in both the number and locations of change points. The paper compares it with NoEffect, ConstEffect, FixedSplit, and PELT-BIC.

## Quick start

Requirements: Python 3.12 and a GCC-compatible C compiler (for example, MinGW-w64 on Windows). The C sampler is compiled automatically on first use.

```bash
python -m pip install -e .
gcc --version
python scripts/reproduce.py validate
python scripts/reproduce.py smoke --workers 2
```

`validate` checks the source files and compiler. `smoke` is a small installation test, not the full experiment.

## Use CG-RJMCMC on the real data

Download `RawData_Experiment1.7z` from the [PC-Flanker OSF project](https://osf.io/6n597/) ([Braeutigam et al.](https://doi.org/10.3758/s13421-023-01447-x)) and extract its 50 CSV files directly into:

```text
data/external/RawData_Experiment1/
```

Run the analysis:

```bash
python scripts/reproduce.py application --application-input "data/external/RawData_Experiment1"
```

The fixed pipeline preprocesses the trials, creates a deterministic participant-stratified 80/20 train/test split, fits three CG-RJMCMC chains, and evaluates all five methods. Main outputs are written to `results/reproduction/application/`:

- `results.json`: posterior change-point summaries, effect curves, and diagnostics.
- `heldout_rmse.csv`: held-out prediction error for each method.
- `run_manifest.json`: data, source, and environment fingerprints.

For an exact reproduction, `results.json` should report input hash `b61c1f400be01e79a34b633ede24f50cf0ec844c4f8b3d85076df4eb572ed480`.

The public entry point currently expects the original PC-Flanker CSV schema. To use another dataset, map it to the arrays constructed in `prepare_group()` in [`scripts/application.py`](scripts/application.py): participant index, trial position (1–64), centered position, congruency contrast (`-0.5`/`0.5`), and centered log response.

## Primary simulation and application workflows

Run the complete workflow after downloading the real data:

```bash
python scripts/reproduce.py all --workers 2 --application-input "data/external/RawData_Experiment1"
```

This runs the full simulations (3,000 primary and 1,600 sensitivity datasets), real-data analysis, checks, figures, and supporting-data export. It can take substantial time; increase `--workers` according to available CPU cores.

To run stages separately:

```bash
python scripts/reproduce.py simulation --workers 2
python scripts/reproduce.py application --application-input "data/external/RawData_Experiment1"
python scripts/reproduce.py checks --application-input "data/external/RawData_Experiment1"
python scripts/reproduce.py figures
python scripts/reproduce.py supporting
```

Run these stages in order: later stages consume outputs from the earlier ones, and `checks` also needs the raw data path.

Results are saved under `results/reproduction/`, while reproduced figures, tables, and supporting files are saved under `manuscript/reproduced/`. Use `--dry-run` to inspect commands. For a clean rerun without overwriting existing outputs, add:

```bash
--output-root results/reproduction_v2 --assets manuscript/reproduced_v2
```

The detailed real-data protocol is in [`scripts/config/application_protocol.md`](scripts/config/application_protocol.md). Configuration and dependency versions are recorded in [`scripts/config/r61_formal.yaml`](scripts/config/r61_formal.yaml) and [`pyproject.toml`](pyproject.toml).

## Penalized segmentation benchmark

PELT-BIC minimizes the position-weighted within-segment sum of squares plus
`log(T)` per segment. Its minimum segment length applies to both candidate
segments and pruning: a dominated prefix remains eligible until the dominating
prefix can be extended by a legal segment. This delayed-pruning implementation
preserves the unrestricted number of changes; the Bayesian `Kmax` setting does
not cap PELT. The prior, data-generating processes and CG-RJMCMC kernel are
unchanged.

Source fingerprints are recorded in `scripts/config/reproduction_sources.json`.
Use a new output directory when running the updated source against an existing
experiment cache so the archived outputs retain their original provenance.

## License and citation

Code is released under the [MIT License](LICENSE). See [`CITATION.cff`](CITATION.cff) for citation metadata. The external PC-Flanker data are subject to their original terms.

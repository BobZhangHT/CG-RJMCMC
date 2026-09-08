# A Bayesian change-point mixed model for changing congruency effects in flanker tasks

Code accompanying the paper. CG-RJMCMC estimates how the response-time difference between incongruent and congruent trials changes across a task, allowing for participant differences and uncertainty about when changes occur. Comparisons include NoEffect, ConstEffect, FixedSplit and PELT-BIC.

## 1. Set up

Download or clone this repository and open a terminal in its top-level folder. Use **Python 3.12** and a GCC-compatible C compiler on `PATH` (tested on Windows with MinGW-w64). The sampler compiles automatically when an analysis starts.

```bash
python -m pip install -e .
gcc --version
python scripts/reproduce.py validate
```

The last command checks the analysis source files and compiler without running experiments. For a small installation check, run `python scripts/reproduce.py smoke --workers 2`; this is not the full paper analysis. Dependency versions are specified in [pyproject.toml](pyproject.toml).

## 2. Reproduce the real-data application

Use **Braeutigam et al.'s PC-Flanker Experiment 1**, the dataset analyzed in the paper ([original article](https://doi.org/10.3758/s13421-023-01447-x)). Download **`RawData_Experiment1.7z`** from the authors' [OSF project](https://osf.io/6n597/) and extract its 50 participant CSV files into:

```text
data/external/RawData_Experiment1/
```

The 50 CSVs must be directly inside this folder, not in another nested folder. Keep the original filenames and columns; do not preprocess the files manually. Run:

```bash
python scripts/reproduce.py application --application-input "data/external/RawData_Experiment1"
```

You can replace the quoted path with your extracted folder's location. This runs the paper's analysis, not a shortened demonstration: separate high-first and low-first group fits, each with three chains of 15,000 iterations and 2,000 warm-up iterations.

The code applies the paper's participant exclusions and retains correct visual flanker trials with response times of at least 100 ms. It removes practice blocks, the first trial of each block and flanker trials immediately following catch trials. Trial positions are mapped to 64 bins **without averaging response times**. Approximately 20% of each participant's eligible trials are held out using a deterministic split; centering uses training data only. Prediction therefore concerns new trials from the same participants, not new participants.

Check these sample sizes in `results/reproduction/application/results.json`:

| Group | Participants | Eligible trials | Training trials | Held-out trials |
|---|---:|---:|---:|---:|
| High-first | 24 | 8,373 | 6,698 | 1,675 |
| Low-first | 21 | 7,423 | 5,937 | 1,486 |

For the exact paper data, the `input_hash` in `results.json` must also match:

```text
b61c1f400be01e79a34b633ede24f50cf0ec844c4f8b3d85076df4eb572ed480
```

The loader checks the file count but does not automatically enforce this fingerprint. If it differs, check the downloaded files before interpreting the run as an exact reproduction.

Outputs in `results/reproduction/application/`:

| File | What it tells you |
|---|---|
| `heldout_rmse.csv` | Prediction errors by method and group; lower RMSE is better. Errors are on the centered log-response-time scale. |
| `results.json` | Sample counts, Markov chain effect summaries, change-point probabilities and sampling diagnostics. |
| `run_manifest.json` | Input and source hashes plus environment information, to identify the data and code used. |

See [application_protocol.md](scripts/config/application_protocol.md) for the detailed specification. Its historical paths describe the original run; use the commands here for a new reproduction. Good sampling diagnostics do not establish model adequacy, and uncertainty intervals remain conditional on the assumptions discussed in the paper.

The application stage saves Monte Carlo summaries. The paper's analytic Student-\(t\) intervals for the selected partitions are produced later by the `checks` stage and written to `results/reproduction/precision/application_conditional_interval_check.csv`; columns beginning with `exact_` contain the reported analytic values, while `empirical_` columns retain the corresponding sampled quantiles. The same stage writes `results/reproduction/checks/participant_level_summary.csv`, where complete participants are resampled to assess the two phase-change directions and their direct order-by-phase contrast.

## 3. Reproduce simulations and paper outputs

For the complete workflow, first obtain the Experiment 1 data above, then run:

```bash
python scripts/reproduce.py all --workers 2 --application-input "data/external/RawData_Experiment1"
```

The full study includes 3,000 primary and 1,600 sensitivity simulation datasets and can take substantial time. Increase `--workers` to use more CPU processes for simulations if resources permit. Individual stages are also available:

| Suffix after `python scripts/reproduce.py` | Purpose / prerequisites |
|---|---|
| `simulation --workers 2` | Full simulations using the included settings. |
| `application` | Real-data analysis; requires downloaded data. |
| `checks` | Analytic selected-partition intervals, participant-level resampling, uncertainty, sensitivity and predictive checks; requires full simulation and application results plus raw data. |
| `figures` | Main tables, figures and submission TIFFs; requires full simulation and application results. |
| `supporting` | Indexed supporting-data archive with replication-level metrics and reported table/figure source values; requires preceding analysis and check outputs. |

Results go to `results/reproduction/`; figures, tables and the supporting-data archive go to `manuscript/reproduced/`. Figure export does not compile the manuscript PDF. Add `--dry-run` to inspect an analysis command without running it.

For a repeat run, choose fresh directories by adding `--output-root results/reproduction_v2 --assets manuscript/reproduced_v2`, and use the same options for subsequent stages. The application refuses to overwrite a nonempty output folder; simulations resume only when stored settings and source fingerprints match. If you already ran the application alone, use the remaining individual stages or choose fresh directories for `all`. Runtime varies across computers.

## Data, code and citation

- **Real-data application:** the downloaded PC-Flanker Experiment 1 files above. External data are not covered by this repository's MIT license.
- **Simulation design only:** [r61_congruency_geometry.csv](scripts/config/r61_congruency_geometry.csv) contains 49 observed 64-position congruency sequences from [Lost in Migration](https://doi.org/10.5281/zenodo.14838065), distributed under CC0. They set simulated trial conditions; they are **not** the real-data response times. Simulated effects and responses are generated anew.
- **Code:** `scripts/reproduce.py` is the common entry point. Other files in `scripts/` implement sampling, analyses and exports; `scripts/config/` holds settings and provenance. Raw data, generated results, manuscript files, build products and local tests are excluded from the public source release.
- **Reuse:** see the [MIT license](LICENSE) and [CITATION.cff](CITATION.cff). To list public source files without committing or uploading anything, run `python scripts/reproduce.py export`.

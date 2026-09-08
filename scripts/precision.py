#!/usr/bin/env python
"""Interval coverage, paired comparisons and exact-versus-MCMC precision checks.

This script never writes into its formal/application input result trees.  It
re-summarizes the 3,000 main simulations and runs five prespecified,
independent original-prior checks in a caller-selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import ctypes as ct
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "reproduction" / "precision"
ASSETS = ROOT / "manuscript" / "reproduced"
PROTOCOL = ROOT / "scripts" / "config" / "original_precision_protocol.json"
SOURCE_LOCK = ROOT / "scripts" / "config" / "reproduction_sources.json"
SIMULATION_RESULTS = ROOT / "results" / "reproduction" / "simulation"
APPLICATION_RESULTS = ROOT / "results" / "reproduction" / "application"
FORMAL_PATH = ROOT / "scripts" / "simulation.py"
PC_PATH = ROOT / "scripts" / "application.py"
FORMAL_CONFIG = ROOT / "scripts" / "config" / "r61_formal.yaml"
PC_INPUT = ROOT / "data" / "external" / "RawData_Experiment1"
METHODS = ("B0", "B1", "B2", "PELT-BIC", "A3")
T = 64
KMAX = 4
MINSEG = 8
PRIOR_POWER = 1.0
PRECISION_NAMESPACE = "PLOS-PRECISION-ORIGINAL-ALPHA1-v1"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_module(FORMAL_PATH, "r61_formal_precision_dependency")


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(x) for x in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(native(rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_path(path: Path, root: Path = ROOT) -> str:
    """Return a stable repository-relative path, or an absolute external path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def configure_paths(
    simulation_results: Path, application_results: Path, output: Path, assets: Path
) -> None:
    """Bind caller-selected result and destination roots before running helpers."""
    global SIMULATION_RESULTS, APPLICATION_RESULTS, OUTPUT, ASSETS
    SIMULATION_RESULTS = simulation_results.expanduser().resolve()
    APPLICATION_RESULTS = application_results.expanduser().resolve()
    OUTPUT = output.expanduser().resolve()
    ASSETS = assets.expanduser().resolve()


def validate_protocol_sources(
    protocol_path: Path = SOURCE_LOCK, root: Path = ROOT
) -> dict[str, str]:
    """Strictly validate the current public release against its source lock.

    The archival original protocol is deliberately not used as a current-code
    lock: its hashes describe the sources used for the historical review run.
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol["source_sha256"]
    if not expected:
        raise RuntimeError(f"source lock contains no entries: {protocol_path}")
    observed = {rel: sha256_file(root / rel) for rel in expected}
    mismatch = {
        rel: {"expected": expected[rel], "observed": observed[rel]}
        for rel in expected
        if expected[rel].lower() != observed[rel].lower()
    }
    if mismatch:
        raise RuntimeError(f"current release source-hash mismatch: {mismatch}")
    return observed


def legal_partitions(
    kmax: int = KMAX, minseg: int = MINSEG, trials: int = T
) -> list[tuple[int, ...]]:
    """Enumerate complete legal partitions by count, including the empty one."""
    states = [()]
    for k in range(1, kmax + 1):
        for cuts in itertools.combinations(range(minseg, trials - minseg + 1), k):
            edges = (0,) + cuts + (trials,)
            if min(b - a for a, b in zip(edges[:-1], edges[1:])) >= minseg:
                states.append(cuts)
    return states


def _log_partition_count(k: int, minseg: int = MINSEG, trials: int = T) -> float:
    if k == 0:
        return 0.0
    top = trials - minseg * (k + 1) + k
    return math.lgamma(top + 1) - math.lgamma(k + 1) - math.lgamma(top - k + 1)


def student_t_scale(
    rate: float, shape: float, inverse_diagonal: np.ndarray | float
) -> np.ndarray:
    """Marginal Student-t scale after integrating sigma squared."""
    return np.sqrt((rate / shape) * np.asarray(inverse_diagonal, dtype=float))


class CWeightProbe:
    """Fresh exact-weight wrapper around the unchanged formal C implementation."""

    def __init__(self, output: Path | None = None):
        output = OUTPUT if output is None else output.expanduser().resolve()
        wrapper = ROOT / "scripts" / "probe.c"
        suffix = ".dll" if os.name == "nt" else ".so"
        tag = sha256_file(ROOT / "scripts" / "kernel.c")[:12]
        library_path = output / f"precision_weight_probe_{tag}{suffix}"
        output.mkdir(parents=True, exist_ok=True)
        if not library_path.exists():
            # Relative paths avoid MinGW failures on the non-ASCII workspace.
            subprocess.run(
                [
                    "gcc",
                    "-O3",
                    "-shared",
                    "-o",
                    provenance_path(library_path),
                    str(wrapper.relative_to(ROOT)),
                    "-lm",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        self.library_path = library_path
        self.lib = ct.CDLL(str(library_path))
        double_p, int_p = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)
        self.lib.probe_weight.argtypes = [
            double_p,
            double_p,
            ct.c_double,
            ct.c_int,
            int_p,
            ct.c_int,
            ct.c_int,
            ct.c_double,
        ]
        self.lib.probe_weight.restype = ct.c_double

    def weight(
        self, H: np.ndarray, h: np.ndarray, yqy: float, n: int, cuts: tuple[int, ...]
    ) -> float:
        ca = np.zeros(formal.MAX_K, dtype=np.int32)
        ca[: len(cuts)] = cuts
        return float(
            self.lib.probe_weight(
                np.ascontiguousarray(H).ctypes.data_as(ct.POINTER(ct.c_double)),
                np.ascontiguousarray(h).ctypes.data_as(ct.POINTER(ct.c_double)),
                float(yqy),
                int(n),
                ca.ctypes.data_as(ct.POINTER(ct.c_int)),
                len(cuts),
                MINSEG,
                PRIOR_POWER,
            )
        )


def partition_posterior_parameters(
    H: np.ndarray, h: np.ndarray, yqy: float, n: int, cuts: tuple[int, ...]
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    edges = (0,) + cuts + (T,)
    p = len(edges) - 1
    A = np.empty((p, p), dtype=float)
    g = np.empty(p, dtype=float)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        g[i] = h[a:b].sum()
        for j, (c, d) in enumerate(zip(edges[:-1], edges[1:])):
            A[i, j] = H[a:b, c:d].sum()
    A.flat[:: p + 1] += 4.0
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        raise RuntimeError(f"non-positive posterior precision for {cuts}")
    mu = np.linalg.solve(A, g)
    inverse = np.linalg.inv(A)
    rate = float(2.0 + 0.5 * yqy - 0.5 * g @ mu)
    shape = float(3.0 + n / 2.0)
    logw = (
        -PRIOR_POWER * math.lgamma(len(cuts) + 1)
        - _log_partition_count(len(cuts))
        + 0.5 * p * math.log(4.0)
        - 0.5 * logdet
        - shape * math.log(rate)
    )
    return logw, mu, np.diag(inverse), rate, shape


def mixture_quantiles(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    df: float,
    probabilities: Sequence[float] = (0.025, 0.975),
    iterations: int = 52,
) -> np.ndarray:
    """Vectorized bisection for coordinate-wise finite Student-t mixtures.

    ``locations`` and ``scales`` have shape components by coordinates.  The
    return value has shape probabilities by coordinates.
    """
    weights = np.asarray(weights, float)
    locations = np.asarray(locations, float)
    scales = np.asarray(scales, float)
    if locations.ndim == 1:
        locations = locations[:, None]
        scales = scales[:, None]
    if not (len(weights) == len(locations) == len(scales)):
        raise ValueError("mixture component dimensions disagree")
    if np.any(scales <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("invalid finite mixture")
    spread = 16.0 * scales.max(axis=0)
    lo0, hi0 = locations.min(axis=0) - spread, locations.max(axis=0) + spread
    result = []
    for probability in probabilities:
        lo, hi = lo0.copy(), hi0.copy()
        for _ in range(iterations):
            mid = (lo + hi) / 2
            cdf = (
                weights[:, None] * stdtr(df, (mid[None, :] - locations) / scales)
            ).sum(axis=0)
            lo = np.where(cdf < probability, mid, lo)
            hi = np.where(cdf >= probability, mid, hi)
        result.append((lo + hi) / 2)
    return np.asarray(result)


def exact_posterior(
    H: np.ndarray,
    h: np.ndarray,
    yqy: float,
    n: int,
    states: list[tuple[int, ...]],
    probe: CWeightProbe,
) -> dict[str, Any]:
    count = len(states)
    locations = np.empty((count, T), dtype=float)
    scales = np.empty((count, T), dtype=float)
    logw_c = np.empty(count, dtype=float)
    logw_py = np.empty(count, dtype=float)
    shape_value: float | None = None
    for index, cuts in enumerate(states):
        logw, mu, inverse_diag, rate, shape = partition_posterior_parameters(
            H, h, yqy, n, cuts
        )
        logw_py[index] = logw
        logw_c[index] = probe.weight(H, h, yqy, n, cuts)
        shape_value = shape
        edges = (0,) + cuts + (T,)
        seg_scale = student_t_scale(rate, shape, inverse_diag)
        for segment, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            locations[index, a:b] = mu[segment]
            scales[index, a:b] = seg_scale[segment]
    max_weight_error = float(np.max(np.abs(logw_c - logw_py)))
    if max_weight_error > 2e-8:
        raise RuntimeError(f"C/Python log-weight mismatch {max_weight_error:g}")
    stabilized = np.exp(logw_c - logw_c.max())
    weights = stabilized / stabilized.sum()
    exact_mean = weights @ locations
    quantiles = mixture_quantiles(weights, locations, scales, 2.0 * float(shape_value))
    p_k = np.asarray(
        [weights[[len(state) == k for state in states]].sum() for k in range(KMAX + 1)]
    )
    pips = np.asarray(
        [
            weights[[trial in state for state in states]].sum()
            for trial in range(1, T + 1)
        ]
    )
    order = np.argsort(weights)[::-1]
    return {
        "states": states,
        "weights": weights,
        "locations": locations,
        "scales": scales,
        "mean": exact_mean,
        "q025": quantiles[0],
        "q975": quantiles[1],
        "p_k": p_k,
        "pips": pips,
        "df": 2.0 * float(shape_value),
        "max_c_python_logweight_error": max_weight_error,
        "order": order,
        "pk_support": np.asarray(
            [any(len(state) == k for state in states) for k in range(KMAX + 1)]
        ),
        "pip_support": np.asarray(
            [any(trial in state for state in states) for trial in range(1, T + 1)]
        ),
    }


def replication_mcse(values: np.ndarray) -> float | None:
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize_coverage_matrix(
    cover: np.ndarray, width: np.ndarray, near: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cover, width, near = (
        np.asarray(cover, bool),
        np.asarray(width, float),
        np.asarray(near, bool),
    )
    if cover.shape != width.shape or cover.ndim != 2 or cover.shape[1] != len(near):
        raise ValueError("coverage inputs have incompatible shapes")
    n = cover.shape[0]
    point_rows = []
    for trial in range(cover.shape[1]):
        point_rows.append(
            {
                "trial": trial + 1,
                "replications": n,
                "near_boundary": bool(near[trial]),
                "coverage": float(cover[:, trial].mean()),
                "coverage_mcse": replication_mcse(cover[:, trial].astype(float)),
                "mean_width": float(width[:, trial].mean()),
                "width_mcse": replication_mcse(width[:, trial]),
            }
        )
    strata: dict[str, Any] = {}
    masks = {"all": np.ones(len(near), bool), "near_le_2": near, "far_gt_2": ~near}
    for label, mask in masks.items():
        if not mask.any():
            strata[label] = None
            continue
        per_rep_cover = cover[:, mask].mean(axis=1)
        per_rep_width = width[:, mask].mean(axis=1)
        strata[label] = {
            "positions": int(mask.sum()),
            "replications": n,
            "mean_pointwise_coverage": float(per_rep_cover.mean()),
            "coverage_mcse": replication_mcse(per_rep_cover),
            "mean_interval_width": float(per_rep_width.mean()),
            "width_mcse": replication_mcse(per_rep_width),
        }
    point = cover.mean(axis=0)
    strata["simultaneous_all64_coverage"] = float(cover.all(axis=1).mean())
    strata["simultaneous_all64_mcse"] = replication_mcse(
        cover.all(axis=1).astype(float)
    )
    strata["minimum_pointwise_coverage"] = float(point.min())
    strata["minimum_positions"] = (np.flatnonzero(point == point.min()) + 1).tolist()
    strata["maximum_pointwise_coverage"] = float(point.max())
    return point_rows, strata


def summarize_frozen_simulations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = json.loads(
        (SIMULATION_RESULTS / "resolved_config.json").read_text(encoding="utf-8")
    )["config"]
    pointwise_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    method_arrays: dict[str, dict[str, list[float]]] = {}
    base = SIMULATION_RESULTS / "replicates" / "main"
    for dgp, spec in config["dgps"].items():
        truth = np.asarray(spec["levels"], float)[
            np.searchsorted(spec["cuts"], np.arange(1, T + 1), side="left")
        ]
        near = np.asarray(
            [
                bool(spec["cuts"])
                and min(abs(trial - cut) for cut in spec["cuts"]) <= 2
                for trial in range(1, T + 1)
            ]
        )
        covers, widths = [], []
        method_arrays[dgp] = {method: [] for method in METHODS}
        paths = sorted((base / dgp).glob("rep_*/summary.json"))
        if len(paths) != 500:
            raise RuntimeError(
                f"expected 500 frozen {dgp} replications, found {len(paths)}"
            )
        for summary_path in paths:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            curve = pd.read_csv(summary_path.with_name("a3_bma_curve.csv"))
            lo, hi = curve["q025"].to_numpy(float), curve["q975"].to_numpy(float)
            covers.append((lo <= truth) & (truth <= hi))
            widths.append(hi - lo)
            by_method = {row["method"]: row for row in summary["methods"]}
            for method in METHODS:
                method_arrays[dgp][method].append(
                    float(by_method[method]["trajectory_RMSE"])
                )
        point, strata = summarize_coverage_matrix(
            np.asarray(covers), np.asarray(widths), near
        )
        pointwise_rows.extend({"dgp": dgp, **row} for row in point)
        for label in ("all", "near_le_2", "far_gt_2"):
            if strata[label] is not None:
                stratum_rows.append({"dgp": dgp, "stratum": label, **strata[label]})
        stratum_rows.append(
            {
                "dgp": dgp,
                "stratum": "simultaneous_all64",
                "positions": T,
                "replications": 500,
                "mean_pointwise_coverage": strata["simultaneous_all64_coverage"],
                "coverage_mcse": strata["simultaneous_all64_mcse"],
                "mean_interval_width": None,
                "width_mcse": None,
                "minimum_pointwise_coverage": strata["minimum_pointwise_coverage"],
                "minimum_positions": ";".join(map(str, strata["minimum_positions"])),
                "maximum_pointwise_coverage": strata["maximum_pointwise_coverage"],
            }
        )
    paired_rows: list[dict[str, Any]] = []
    for dgp in config["dgps"]:
        arrays = {m: np.asarray(v, float) for m, v in method_arrays[dgp].items()}
        for first, second in itertools.combinations(METHODS, 2):
            # Put A3 and then PELT first in interpretively important contrasts.
            method_a, method_b = (
                (second, first)
                if second == "A3" or (second == "PELT-BIC" and first != "A3")
                else (first, second)
            )
            diff = arrays[method_a] - arrays[method_b]
            paired_rows.append(
                {
                    "dgp": dgp,
                    "method_a": method_a,
                    "method_b": method_b,
                    "contrast": f"{method_a} minus {method_b}",
                    "replications": len(diff),
                    "mean_difference": float(diff.mean()),
                    "paired_mcse": replication_mcse(diff),
                    "ci95_mc": [
                        float(diff.mean() - 1.96 * replication_mcse(diff)),
                        float(diff.mean() + 1.96 * replication_mcse(diff)),
                    ],
                    "median_difference": float(np.median(diff)),
                    "fraction_method_a_lower": float(np.mean(diff < 0)),
                }
            )
    point_df, strata_df, paired_df = (
        pd.DataFrame(pointwise_rows),
        pd.DataFrame(stratum_rows),
        pd.DataFrame(paired_rows),
    )
    write_csv(OUTPUT / "frozen_pointwise_coverage.csv", pointwise_rows)
    write_csv(OUTPUT / "frozen_coverage_summary.csv", stratum_rows)
    write_csv(OUTPUT / "paired_rmse_differences.csv", paired_rows)
    return point_df, strata_df, paired_df


def prepare_cases(
    cfg: dict[str, Any], application_input: Path = PC_INPUT
) -> list[dict[str, Any]]:
    pc = _load_module(PC_PATH, "r61_pc_precision_dependency")
    application_input = application_input.expanduser().resolve()
    cases: list[dict[str, Any]] = []
    for first_pc in ("High", "Low"):
        train, _test, metadata = pc.prepare_group(application_input, first_pc)
        H, h, yqy = formal.collapsed(*train, int(metadata["participants"]))
        cases.append(
            {
                "case_id": f"application_{first_pc.lower()}_first",
                "kind": "application",
                "label": f"PC-Flanker {first_pc}-first",
                "H": H,
                "h": h,
                "yqy": yqy,
                "n": len(train[-1]),
                "metadata": metadata,
            }
        )
    resolved = json.loads(
        (SIMULATION_RESULTS / "resolved_config.json").read_text(encoding="utf-8")
    )["config"]
    base, _ = formal.load_frozen_geometry(ROOT / resolved["design"]["geometry_file"])
    for rep in (1, 2, 3):
        saved_path = (
            SIMULATION_RESULTS
            / "replicates"
            / "main"
            / "D5"
            / f"rep_{rep:06d}"
            / "summary.json"
        )
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        case = saved["case"]
        design = formal.design_for_n(base, int(case["participants"]))
        y, _, _ = formal.generate_outcome(
            resolved, case, design, int(saved["data_seed"])
        )
        if formal.sha256_array(y) != saved["data_hash"]:
            raise RuntimeError(f"D5 replicate {rep} data hash mismatch")
        label = f"{case['stage']}|{case['cell']}|R{rep:06d}"
        test_mask = formal.outcome_blind_holdout(
            design[0],
            design[1],
            resolved["design"]["holdout_trials_per_participant"],
            f"{resolved['namespace']}|{label}",
        )
        train_mask = formal.mcar_training_mask(
            ~test_mask, resolved["namespace"], label, case["training_mcar"]
        )
        if formal.sha256_array(np.packbits(train_mask)) != saved["training_mask_hash"]:
            raise RuntimeError(f"D5 replicate {rep} training-mask hash mismatch")
        train = formal.pack(design, y, train_mask)
        H, h, yqy = formal.collapsed(*train, int(case["participants"]))
        cases.append(
            {
                "case_id": f"d5_rep_{rep:06d}",
                "kind": "simulation",
                "label": f"D5 replicate {rep}",
                "H": H,
                "h": h,
                "yqy": yqy,
                "n": len(train[-1]),
                "metadata": {
                    "replicate": rep,
                    "data_seed": saved["data_seed"],
                    "data_hash": saved["data_hash"],
                    "training_mask_hash": saved["training_mask_hash"],
                },
            }
        )
    return cases


def numeric_diagnostic(values: np.ndarray, quantiles: bool = False) -> dict[str, Any]:
    values = np.asarray(values)
    if np.all(values == values.flat[0]):
        return {
            "status": "sample_constant",
            "rank_rhat": None,
            "bulk_ess": None,
            "tail_ess": None,
            "mcse_mean": 0.0,
            "mcse_q025": 0.0 if quantiles else None,
            "mcse_q975": 0.0 if quantiles else None,
        }
    if all(np.all(chain == chain[0]) for chain in values):
        return {
            "status": "between_chain_disagreement",
            "rank_rhat": None,
            "bulk_ess": None,
            "tail_ess": None,
            "mcse_mean": None,
            "mcse_q025": None,
            "mcse_q975": None,
        }
    result = {
        "status": "estimated",
        "rank_rhat": float(az.rhat(values, method="rank")),
        "bulk_ess": float(az.ess(values, method="bulk")),
        "tail_ess": float(az.ess(values, method="tail", prob=(0.05, 0.95))),
        "mcse_mean": float(az.mcse(values, method="mean")),
        "mcse_q025": None,
        "mcse_q975": None,
    }
    if quantiles:
        result["mcse_q025"] = float(az.mcse(values, method="quantile", prob=0.025))
        result["mcse_q975"] = float(az.mcse(values, method="quantile", prob=0.975))
    return result


def diagnostic_rows(
    chains: np.ndarray, exact: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kdraw = chains[:, :, 0]
    curve = chains[:, :, 1 + formal.MAX_K :]
    cuts = chains[:, :, 1 : 1 + formal.MAX_K]
    rows: list[dict[str, Any]] = []
    for trial in range(T):
        values = curve[:, :, trial]
        q = np.quantile(values, (0.025, 0.975))
        rows.append(
            {
                "estimand": f"delta[{trial + 1}]",
                "family": "trajectory",
                "index": trial + 1,
                "estimate": float(values.mean()),
                "exact": float(exact["mean"][trial]),
                "absolute_error": float(abs(values.mean() - exact["mean"][trial])),
                "q025": float(q[0]),
                "q975": float(q[1]),
                "exact_q025": float(exact["q025"][trial]),
                "exact_q975": float(exact["q975"][trial]),
                "q025_absolute_error": float(abs(q[0] - exact["q025"][trial])),
                "q975_absolute_error": float(abs(q[1] - exact["q975"][trial])),
                **numeric_diagnostic(values, quantiles=True),
            }
        )
    discrete: list[dict[str, Any]] = []
    k_item = {
        "estimand": "K",
        "family": "K",
        "index": None,
        "estimate": float(kdraw.mean()),
        "exact": float(np.dot(np.arange(KMAX + 1), exact["p_k"])),
        **numeric_diagnostic(kdraw),
    }
    if k_item["status"] == "sample_constant":
        structurally_variable = bool(np.count_nonzero(exact["pk_support"]) > 1)
        k_item["status"] = (
            "no_sampled_variation_exact_nonconstant"
            if structurally_variable
            else "not_applicable_exact_constant"
        )
        k_item["exact_constant_certified"] = not structurally_variable
        if structurally_variable:
            k_item["mcse_mean"] = None
    discrete.append(k_item)
    for k in range(KMAX + 1):
        values = (kdraw == k).astype(float)
        probability = float(exact["p_k"][k])
        item = {
            "estimand": f"Pr(K={k})",
            "family": "pK",
            "index": k,
            "estimate": float(values.mean()),
            "exact": probability,
            "absolute_error": float(abs(values.mean() - probability)),
            **numeric_diagnostic(values),
        }
        if item["status"] == "sample_constant":
            # Certify a constant from finite-state support, never from a
            # floating probability that may have rounded or underflowed.
            support = bool(exact["pk_support"][k])
            exact_constant = not support
            item["status"] = (
                "not_applicable_exact_constant"
                if exact_constant
                else "no_sampled_variation_exact_nonconstant"
            )
            item["exact_constant_certified"] = exact_constant
            if not exact_constant:
                item["mcse_mean"] = None
        discrete.append(item)
    for trial in range(1, T + 1):
        values = np.any(cuts == trial, axis=2).astype(float)
        probability = float(exact["pips"][trial - 1])
        item = {
            "estimand": f"PIP[{trial}]",
            "family": "PIP",
            "index": trial,
            "estimate": float(values.mean()),
            "exact": probability,
            "absolute_error": float(abs(values.mean() - probability)),
            **numeric_diagnostic(values),
        }
        if item["status"] == "sample_constant":
            support = bool(exact["pip_support"][trial - 1])
            exact_constant = not support
            item["status"] = (
                "not_applicable_exact_constant"
                if exact_constant
                else "no_sampled_variation_exact_nonconstant"
            )
            item["exact_constant_certified"] = exact_constant
            if not exact_constant:
                item["mcse_mean"] = None
        discrete.append(item)
    return rows, discrete


def top_partition_rows(
    chains: np.ndarray, exact: dict[str, Any], count: int = 10
) -> list[dict[str, Any]]:
    flat = chains.reshape(-1, chains.shape[-1])
    keys = [
        (int(row[0]),) + tuple(int(v) for v in row[1 : 1 + int(row[0])]) for row in flat
    ]
    empirical = Counter(keys)
    total = len(keys)
    exact_rank = {
        exact["states"][idx]: rank + 1 for rank, idx in enumerate(exact["order"])
    }
    union = set(exact["states"][idx] for idx in exact["order"][:count])
    union.update(key[1:] for key, _ in empirical.most_common(count))
    rows = []
    empirical_order = {
        key[1:]: rank + 1 for rank, (key, _) in enumerate(empirical.most_common())
    }
    state_index = {state: i for i, state in enumerate(exact["states"])}
    for state in sorted(union, key=lambda s: (exact_rank[s], s)):
        key = (len(state),) + state
        rows.append(
            {
                "partition": "none" if not state else ";".join(map(str, state)),
                "K": len(state),
                "exact_rank": exact_rank[state],
                "exact_probability": float(exact["weights"][state_index[state]]),
                "mcmc_rank": empirical_order.get(state),
                "mcmc_probability": empirical.get(key, 0) / total,
                "absolute_probability_error": abs(
                    empirical.get(key, 0) / total - exact["weights"][state_index[state]]
                ),
            }
        )
    return rows


def run_precision_case(
    case: dict[str, Any],
    states: list[tuple[int, ...]],
    probe: CWeightProbe,
    library: Any,
    iterations: int = 15000,
    warmup: int = 2000,
) -> dict[str, Any]:
    case_dir = OUTPUT / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    exact_started = time.perf_counter()
    exact = exact_posterior(case["H"], case["h"], case["yqy"], case["n"], states, probe)
    exact_seconds = time.perf_counter() - exact_started
    chain_list, stats, seeds = [], [], []
    sample_started = time.perf_counter()
    for index, initial_k in enumerate((0, 2, 4), 1):
        seed = formal.deterministic_seed(
            PRECISION_NAMESPACE, f"{case['case_id']}|I{iterations}|CHAIN{index}"
        )
        chain, chain_stats = formal.run_chain(
            library,
            case["H"],
            case["h"],
            case["yqy"],
            case["n"],
            iterations,
            warmup,
            seed,
            initial_k,
            KMAX,
            MINSEG,
            PRIOR_POWER,
        )
        chain_list.append(chain)
        stats.append(chain_stats)
        seeds.append(seed)
    sample_seconds = time.perf_counter() - sample_started
    chains = np.stack(chain_list)
    np.savez_compressed(
        case_dir / "full_three_chain_trajectories_and_cuts.npz",
        chains=chains,
        chain_stats=np.stack(stats),
        chain_seeds=np.asarray(seeds, dtype=np.uint64),
        iterations=np.asarray([iterations]),
        warmup=np.asarray([warmup]),
    )
    trajectory, discrete = diagnostic_rows(chains, exact)
    partitions = top_partition_rows(chains, exact)
    write_csv(case_dir / "trajectory_diagnostics.csv", trajectory)
    write_csv(case_dir / "discrete_diagnostics.csv", discrete)
    write_csv(case_dir / "top_partition_stability.csv", partitions)
    max_probability_error = max(
        row.get("absolute_error", 0.0) or 0.0
        for row in discrete
        if row["family"] in {"pK", "PIP"}
    )
    max_endpoint_error = max(
        max(row["q025_absolute_error"], row["q975_absolute_error"])
        for row in trajectory
    )
    max_endpoint_mcse = max(
        max(row["mcse_q025"] or 0.0, row["mcse_q975"] or 0.0) for row in trajectory
    )
    legal_variable = [
        row
        for row in discrete
        if row["family"] in {"pK", "PIP"}
        and row["status"] not in {"not_applicable_exact_constant"}
    ]
    summary = {
        "case_id": case["case_id"],
        "label": case["label"],
        "kind": case["kind"],
        "metadata": case["metadata"],
        "partitions": len(states),
        "prior_factorial_power": PRIOR_POWER,
        "summary_rule": "empirical joint-partition mode",
        "iterations_per_chain": iterations,
        "warmup_per_chain": warmup,
        "retained_per_chain": iterations - warmup,
        "chain_seeds": seeds,
        "sampling_seconds": sample_seconds,
        "exact_seconds": exact_seconds,
        "exact_df": exact["df"],
        "max_c_python_logweight_error": exact["max_c_python_logweight_error"],
        "maximum_absolute_pK_or_PIP_error": max_probability_error,
        "maximum_absolute_interval_endpoint_error": max_endpoint_error,
        "maximum_reported_quantile_mcse": max_endpoint_mcse,
        "probability_threshold_pass": bool(max_probability_error <= 0.01),
        "endpoint_threshold_pass": bool(max_endpoint_error <= 0.002),
        "variable_discrete_no_sampled_variation_count": sum(
            row["status"] == "no_sampled_variation_exact_nonconstant"
            for row in legal_variable
        ),
        "constant_discrete_count_exact_certified": sum(
            row["status"] == "not_applicable_exact_constant" for row in discrete
        ),
        "max_rank_rhat_trajectory": max(
            row["rank_rhat"] for row in trajectory if row["rank_rhat"] is not None
        ),
        "min_bulk_ess_trajectory": min(
            row["bulk_ess"] for row in trajectory if row["bulk_ess"] is not None
        ),
        "min_tail_ess_trajectory": min(
            row["tail_ess"] for row in trajectory if row["tail_ess"] is not None
        ),
        "exact_p_k": exact["p_k"],
        "mcmc_p_k": [float(np.mean(chains[:, :, 0] == k)) for k in range(KMAX + 1)],
        "exact_top_partition": list(exact["states"][exact["order"][0]]),
        "mcmc_top_partition": partitions[
            min(
                range(len(partitions)),
                key=lambda i: partitions[i]["mcmc_rank"] or 10**9,
            )
        ]["partition"],
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def make_assets(
    strata: pd.DataFrame, paired: pd.DataFrame, summaries: list[dict[str, Any]]
) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"all": "#0072B2", "near_le_2": "#D55E00", "far_gt_2": "#009E73"}
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(6)
    for offset, label, display in (
        (-0.18, "all", "All positions"),
        (0, "near_le_2", "Within 2 bins of a boundary"),
        (0.18, "far_gt_2", "More than 2 bins from a boundary"),
    ):
        frame = strata[strata["stratum"] == label].set_index("dgp")
        yy = np.asarray(
            [
                frame.loc[d, "mean_pointwise_coverage"] if d in frame.index else np.nan
                for d in [f"D{i}" for i in range(6)]
            ]
        )
        ee = np.asarray(
            [
                frame.loc[d, "coverage_mcse"] if d in frame.index else np.nan
                for d in [f"D{i}" for i in range(6)]
            ]
        )
        ax.errorbar(
            x + offset,
            yy,
            yerr=1.96 * ee,
            marker="o",
            linestyle="none",
            capsize=2.5,
            color=colors[label],
            label=display,
        )
    ax.axhline(0.95, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(x, [f"D{i}" for i in range(6)])
    ax.set_ylabel("Empirical pointwise coverage")
    ax.set_xlabel("Data-generating scenario")
    ax.set_ylim(0.72, 0.995)
    ax.legend(frameon=False, ncol=1, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ASSETS / f"precision_coverage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    case_rows = []
    for summary in summaries:
        case_rows.extend(
            [
                {
                    "case": summary["label"],
                    "metric": "P(K)/PIP",
                    "error": summary["maximum_absolute_pK_or_PIP_error"],
                    "threshold": 0.01,
                },
                {
                    "case": summary["label"],
                    "metric": "95% endpoint",
                    "error": summary["maximum_absolute_interval_endpoint_error"],
                    "threshold": 0.002,
                },
            ]
        )
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for ax, metric, ylabel in zip(
        axes,
        ("P(K)/PIP", "95% endpoint"),
        ("Maximum absolute probability error", "Maximum absolute error (log RT)"),
    ):
        subset = [row for row in case_rows if row["metric"] == metric]
        yy = np.arange(len(subset))
        values = [row["error"] for row in subset]
        ax.barh(yy, values, color="#0072B2")
        ax.axvline(subset[0]["threshold"], color="#D55E00", linestyle="--", linewidth=1)
        ax.set_yticks(
            yy,
            [row["case"].replace("PC-Flanker ", "") for row in subset]
            if ax is axes[0]
            else [],
        )
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            ASSETS / f"precision_exact_comparison.{ext}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)

    main_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Pointwise 95\\% model-averaged interval performance across 500 independent data sets per scenario. Overall, near, and far entries are mean coverages across their stated positions; near-boundary positions are at most two trial bins from a true boundary. Width is in log-response-time units and is shown as mean (Monte Carlo standard error). Minimum is the smallest empirical coverage among the 64 positions. Monte Carlo standard errors use the data set as the replication unit and retain within-trajectory dependence. D0 and D1 have no true boundaries.}",
        "\\label{tab:precision-coverage}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Scenario & Overall & MCSE & Near & Far & Width (MCSE) & Minimum \\\\ ",
        "\\midrule",
    ]
    for dgp in [f"D{i}" for i in range(6)]:
        frame = strata[
            (strata.dgp == dgp) & strata.stratum.isin(["all", "near_le_2", "far_gt_2"])
        ].set_index("stratum")
        allrow = frame.loc["all"]
        near = (
            "--"
            if "near_le_2" not in frame.index
            else f"{frame.loc['near_le_2', 'mean_pointwise_coverage']:.3f}"
        )
        far = f"{frame.loc['far_gt_2', 'mean_pointwise_coverage']:.3f}"
        sim = strata[
            (strata.dgp == dgp) & (strata.stratum == "simultaneous_all64")
        ].iloc[0]
        main_lines.append(
            f"{dgp} & {allrow.mean_pointwise_coverage:.3f} & {allrow.coverage_mcse:.3f} & {near} & {far} & {allrow.mean_interval_width:.4f} ({allrow.width_mcse:.4f}) & {sim.minimum_pointwise_coverage:.3f} \\\\"
        )
    main_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ASSETS / "precision_coverage_table.tex").write_text(
        "\n".join(main_lines) + "\n", encoding="utf-8"
    )

    key = paired[(paired.method_a == "A3") & (paired.method_b == "PELT-BIC")]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Paired trajectory-RMSE comparison of CG-RJMCMC model averaging (A3) with the PELT-BIC procedure over the same 500 data sets in each scenario. Negative differences favor A3; the uncertainty column is the Monte Carlo standard error of the paired mean.}",
        "\\label{tab:precision-rmse}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Scenario & Mean difference & Paired MCSE & A3 lower \\\\ ",
        "\\midrule",
    ]
    for _, row in key.iterrows():
        lines.append(
            f"{row.dgp} & {row.mean_difference:+.6f} & {row.paired_mcse:.6f} & {100 * row.fraction_method_a_lower:.1f}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ASSETS / "precision_paired_rmse_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    sup = [
        "\\begin{table}[p]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Independent original-prior precision checks for the two application groups and three prespecified D5 data sets. Each check uses three chains of 15,000 iterations (2,000 warm-up) and compares Monte Carlo summaries with enumeration of all 27,931 legal partitions. Probability discrepancies cover count and boundary probabilities; endpoint errors and extrema of $\\widehat R$, bulk ESS, and tail ESS cover all 64 trajectory coordinates.}",
        "\\label{tab:precision-exact}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Case & Max $|\\Delta p|$ & Max endpoint error & Max $\\widehat R$ & Min bulk ESS & Min tail ESS \\\\ ",
        "\\midrule",
    ]
    for row in summaries:
        sup.append(
            f"{latex_escape(row['label'])} & {row['maximum_absolute_pK_or_PIP_error']:.4f} & {row['maximum_absolute_interval_endpoint_error']:.5f} & {row['max_rank_rhat_trajectory']:.4f} & {row['min_bulk_ess_trajectory']:.0f} & {row['min_tail_ess_trajectory']:.0f} \\\\"
        )
    sup += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ASSETS / "precision_exact_table.tex").write_text(
        "\n".join(sup) + "\n", encoding="utf-8"
    )

    captions = (
        "Figure precision_coverage: Pointwise 95% model-averaged interval coverage over 500 independent data sets per scenario. "
        "Markers show averages across positions and data sets, and error bars are 95% Monte Carlo intervals based on data-set-level summaries. "
        "Near-boundary positions lie within two trial bins of a true boundary; D0 and D1 have no such positions. The dashed line marks 0.95.\n\n"
        "Figure precision_exact_comparison: Maximum absolute Monte Carlo discrepancies from exact finite-posterior enumeration in two application groups and three prespecified D5 data sets. "
        "The left panel covers all reported P(K) and boundary-probability estimands, and the right panel covers both endpoints at all 64 trajectory positions. "
        "Dashed lines mark the prespecified 0.01 probability and 0.002 log-response-time thresholds.\n"
    )
    (ASSETS / "precision_figure_captions.txt").write_text(captions, encoding="utf-8")


def make_results_md(
    strata: pd.DataFrame,
    paired: pd.DataFrame,
    summaries: list[dict[str, Any]],
    sources: dict[str, str],
) -> None:
    allrows = strata[strata.stratum == "all"].set_index("dgp")
    nearrows = strata[strata.stratum == "near_le_2"].set_index("dgp")
    farrows = strata[strata.stratum == "far_gt_2"].set_index("dgp")
    key = paired[(paired.method_a == "A3") & (paired.method_b == "PELT-BIC")].set_index(
        "dgp"
    )
    lines = [
        "# Precision verification for PLOS review E-M01, E-M03, and E-N04",
        "",
        "## Scope and protocol",
        "",
        "The check followed the archival `scripts/config/original_precision_protocol.json`, written before any new chain was run. Its public copy is a historical record, not a new preregistration. It used the original factorial prior (alpha=1), the original empirical joint-partition mode, both PC-Flanker order groups, and D5 replications 1, 2, and 3. The three D5 cases were fixed by replicate number rather than chosen from their results. No optimized candidate prior or two-stage rule entered this analysis.",
        "",
        "The existing interval assessment covers all 3,000 frozen main simulations (six scenarios by 500 independent data sets). The new chain assessment is deliberately a five-case precision check, not a claim that all 500 posterior computations in every scenario have independently passed every reported-quantity diagnostic.",
        "",
        "## Frozen interval performance",
        "",
        "| Scenario | Overall coverage (MCSE) | Near <=2 | Far >2 | Mean width | Minimum pointwise coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dgp in [f"D{i}" for i in range(6)]:
        row = allrows.loc[dgp]
        near = (
            "N/A"
            if dgp not in nearrows.index
            else f"{nearrows.loc[dgp, 'mean_pointwise_coverage']:.3f}"
        )
        far = f"{farrows.loc[dgp, 'mean_pointwise_coverage']:.3f}"
        sim = strata[
            (strata.dgp == dgp) & (strata.stratum == "simultaneous_all64")
        ].iloc[0]
        lines.append(
            f"| {dgp} | {row.mean_pointwise_coverage:.3f} ({row.coverage_mcse:.3f}) | {near} | {far} | {row.mean_interval_width:.5f} | {sim.minimum_pointwise_coverage:.3f} (bin {sim.minimum_positions}) |"
        )
    lines += [
        "",
        "The replication-level MCSE is computed after averaging within each 64-position trajectory, so it does not treat the 32,000 position-by-replication cells as independent. D5 undercoverage is substantive and is present both near and away from true boundaries; it is therefore not described as a single-boundary artifact. These are marginal, model-averaged credible intervals, not simultaneous bands and not intervals conditional on a selected partition.",
        "",
        "## Paired trajectory-RMSE comparisons",
        "",
        "The complete machine-readable table contains all ten unique method pairs for every scenario. The key A3-minus-PELT-BIC contrasts are:",
        "",
        "| Scenario | Mean paired difference | MCSE | Median | A3 lower |",
        "|---|---:|---:|---:|---:|",
    ]
    for dgp in [f"D{i}" for i in range(6)]:
        row = key.loc[dgp]
        lines.append(
            f"| {dgp} | {row.mean_difference:+.6f} | {row.paired_mcse:.6f} | {row.median_difference:+.6f} | {100 * row.fraction_method_a_lower:.1f}% |"
        )
    lines += [
        "",
        "A negative difference favors A3. The D5 mean and median answer different features of a skewed paired distribution and are reported together; a similar ratio of marginal medians must not be read as a universal paired advantage or as an equivalence test.",
        "",
        "## Independent posterior-precision checks",
        "",
        "| Prespecified case | Max P(K)/PIP error | Max 95% endpoint error | Max rank R-hat | Min bulk ESS | Min tail ESS | Exact top partition | MCMC top partition |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['maximum_absolute_pK_or_PIP_error']:.5f} | {row['maximum_absolute_interval_endpoint_error']:.6f} | {row['max_rank_rhat_trajectory']:.5f} | {row['min_bulk_ess_trajectory']:.0f} | {row['min_tail_ess_trajectory']:.0f} | {row['exact_top_partition']} | {row['mcmc_top_partition']} |"
        )
    probability_pass = all(row["probability_threshold_pass"] for row in summaries)
    endpoint_pass = all(row["endpoint_threshold_pass"] for row in summaries)
    extension_sentence = (
        "Consequently, the conditional 100,000-iteration extension was not invoked."
        if probability_pass and endpoint_pass
        else "At least one original-budget threshold was not met; the discrepancy and quantile MCSE are retained explicitly, and no automatic selective extension was performed."
    )
    lines += [
        "",
        f"All five cases passed the prespecified maximum absolute probability-error threshold: **{probability_pass}**. All five passed the maximum absolute 95% endpoint-error threshold: **{endpoint_pass}**. {extension_sentence}",
        "",
        "All 64 trajectory coordinates received rank-normalized split R-hat, bulk ESS, tail ESS, and quantile-MCSE checks. K, each P(K=k) indicator, and every boundary indicator received discrete diagnostics and mean MCSE where sampled variation existed. Structurally impossible boundary indicators were recorded as N/A only after exact enumeration certified posterior probability zero. A sampled all-zero indicator with nonzero exact probability was not assigned a fictitious ESS.",
        "",
        "Exact posterior probabilities, model-averaged means, and marginal interval endpoints were obtained by enumerating all 27,931 legal partitions. For each partition the marginal trajectory distribution is Student-t with df 2a_n and scale sqrt((b_n/a_n) diag(A_n^{-1})); finite-mixture CDFs were inverted numerically. C log weights were cross-checked against an independent matrix implementation in every case.",
        "",
        "## What is and is not covered",
        "",
        "Covered: all frozen main-simulation interval files; all paired main-simulation RMSE differences; complete retained three-chain trajectories/cuts for the five fixed checks; all-coordinate continuous diagnostics; all P(K) and boundary-indicator diagnostics; exact posterior p(K), PIP, mean, quantiles, and partition-rank comparisons for those five checks.",
        "",
        "Not covered: an exact-posterior rerun of all 500 D5 data sets or all 3,000 main simulations; a blanket requirement that every tail ESS exceed 1,500; conditional-on-selected-partition segment intervals beyond the original summary rule; or any validation of the optimized candidate prior. Accordingly, the evidence supports the numerical precision of the prespecified checks, not an unrestricted claim about every saved posterior summary.",
        "",
        "## Reproducibility and machine outputs",
        "",
        "- `frozen_pointwise_coverage.csv`: 384 DGP-by-position rows.",
        "- `frozen_coverage_summary.csv`: all/near/far/simultaneous summaries with correct replication denominators.",
        "- `paired_rmse_differences.csv`: 60 DGP-by-method-pair rows.",
        "- `cases/*/full_three_chain_trajectories_and_cuts.npz`: complete retained original-budget draws.",
        "- `cases/*/trajectory_diagnostics.csv`, `discrete_diagnostics.csv`, and `top_partition_stability.csv`: reported-quantity precision evidence.",
        "- `precision_manifest.json`: hashes, software versions, protocol, and case outcomes.",
        "",
        "Frozen source hashes were checked before running and matched the protocol:",
        "",
    ]
    lines.extend(f"- `{rel}`: `{digest}`" for rel, digest in sources.items())
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def application_conditional_interval_check(
    application_input: Path = PC_INPUT,
) -> list[dict[str, Any]]:
    """Check the published joint-MAP conditional segment intervals exactly."""
    pc = _load_module(PC_PATH, "r61_pc_conditional_precision_dependency")
    frozen = json.loads(
        (APPLICATION_RESULTS / "results.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for group in frozen["groups"]:
        first_pc = group["data"]["first_pc"]
        train, _test, metadata = pc.prepare_group(
            application_input.expanduser().resolve(), first_pc
        )
        if (
            metadata["participant_hash"] != group["data"]["participant_hash"]
            or metadata["split_hash"] != group["data"]["split_hash"]
        ):
            raise RuntimeError(
                f"{first_pc}-first application input does not reproduce the frozen analysis"
            )
        H, h, yqy = formal.collapsed(*train, int(metadata["participants"]))
        joint = tuple(int(x) for x in group["CG_RJMCMC"]["joint_map_partition"])
        cuts = joint[1:]
        _logw, mu, inverse_diag, rate, shape = partition_posterior_parameters(
            H, h, yqy, len(train[-1]), cuts
        )
        scales = student_t_scale(rate, shape, inverse_diag)
        exact_lo = mu + scales * stdtrit(2 * shape, 0.025)
        exact_hi = mu + scales * stdtrit(2 * shape, 0.975)
        saved = group["CG_RJMCMC"]["joint_map_conditional_levels"]
        edges = (0,) + cuts + (T,)
        if len(saved) != len(mu):
            raise RuntimeError(
                f"{first_pc}-first saved conditional-segment count mismatch"
            )
        for index, item in enumerate(saved):
            expected_edge = (edges[index] + 1, edges[index + 1])
            if (int(item["start_bin"]), int(item["end_bin"])) != expected_edge:
                raise RuntimeError(
                    f"{first_pc}-first saved conditional-segment edge mismatch"
                )
            rows.append(
                {
                    "group": f"{first_pc}-first",
                    "joint_map_cuts": ";".join(map(str, cuts)),
                    "segment": index + 1,
                    "start_bin": expected_edge[0],
                    "end_bin": expected_edge[1],
                    "df": 2 * shape,
                    "exact_mean": float(mu[index]),
                    "exact_q025": float(exact_lo[index]),
                    "exact_q975": float(exact_hi[index]),
                    "saved_mcmc_mean": float(item["mean"]),
                    "saved_mcmc_q025": float(item["q025"]),
                    "saved_mcmc_q975": float(item["q975"]),
                    "mean_absolute_error": float(abs(item["mean"] - mu[index])),
                    "q025_absolute_error": float(abs(item["q025"] - exact_lo[index])),
                    "q975_absolute_error": float(abs(item["q975"] - exact_hi[index])),
                    "definition": "posterior for segment level conditional on the empirical joint-MAP partition",
                }
            )
    write_csv(OUTPUT / "application_conditional_interval_check.csv", rows)
    lines = [
        "\\begin{table}[p]",
        "\\centering",
        "\\small",
        "\\caption{Exact checks of the application segment intervals conditional on each group's empirical joint-MAP partition. Exact endpoints use the conjugate Student-$t$ posterior; the saved endpoints are empirical quantiles of retained draws in the same partition.}",
        "\\label{tab:precision-conditional-application}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Group & Bins & Exact mean & Exact 95\\% interval & Lower error & Upper error \\\\ ",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group']} & {row['start_bin']}--{row['end_bin']} & {row['exact_mean']:.4f} & [{row['exact_q025']:.4f}, {row['exact_q975']:.4f}] & {row['q025_absolute_error']:.5f} & {row['q975_absolute_error']:.5f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ASSETS / "precision_conditional_interval_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def add_conditional_check_to_results(rows: list[dict[str, Any]]) -> None:
    path = OUTPUT / "RESULTS.md"
    text = path.read_text(encoding="utf-8")
    heading = "## Exact check of published conditional application intervals"
    if heading in text:
        text = (
            text[: text.index(heading)]
            + text[text.index("## What is and is not covered", text.index(heading)) :]
        )
    block = [
        heading,
        "",
        "The application reports segment-level intervals conditional on each group's empirical joint-MAP partition; these are distinct from the model-averaged trajectory intervals. The exact conjugate calculation is a Student-t posterior, not a conditional normal approximation. Its comparison with the saved empirical conditional-draw quantiles is:",
        "",
        "| Group | Bins | Exact mean | Exact 95% interval | Saved 95% interval | Endpoint absolute errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        block.append(
            f"| {row['group']} | {row['start_bin']}--{row['end_bin']} | {row['exact_mean']:.5f} | [{row['exact_q025']:.5f}, {row['exact_q975']:.5f}] | [{row['saved_mcmc_q025']:.5f}, {row['saved_mcmc_q975']:.5f}] | {row['q025_absolute_error']:.6f}, {row['q975_absolute_error']:.6f} |"
        )
    maximum = max(
        max(row["q025_absolute_error"], row["q975_absolute_error"]) for row in rows
    )
    block += [
        "",
        f"Across the five published application segments, the largest endpoint discrepancy is {maximum:.6f} log RT. This directly verifies the conditional intervals while preserving their conditional interpretation.",
        "",
    ]
    text = text.replace(
        "## What is and is not covered",
        "\n".join(block) + "\n## What is and is not covered",
        1,
    )
    text = text.replace(
        "Covered: all frozen main-simulation interval files; all paired main-simulation RMSE differences; complete retained three-chain trajectories/cuts for the five fixed checks; all-coordinate continuous diagnostics; all P(K) and boundary-indicator diagnostics; exact posterior p(K), PIP, mean, quantiles, and partition-rank comparisons for those five checks.",
        "Covered: all frozen main-simulation interval files; all paired main-simulation RMSE differences; complete retained three-chain trajectories/cuts for the five fixed checks; all-coordinate continuous diagnostics; all P(K) and boundary-indicator diagnostics; exact posterior p(K), PIP, mean, quantiles, and partition-rank comparisons for those five checks; and exact conditional Student-t checks of the five published application segment intervals.",
    )
    text = text.replace(
        "; conditional-on-selected-partition segment intervals beyond the original summary rule;",
        "; conditional-on-selected-partition intervals outside the two published application joint-MAP summaries;",
    )
    path.write_text(text, encoding="utf-8")


def refresh_coverage_table_from_machine_summary() -> None:
    """Refresh the compact table after a metadata-only finalization."""
    strata = pd.read_csv(OUTPUT / "frozen_coverage_summary.csv")
    pointwise = pd.read_csv(OUTPUT / "frozen_pointwise_coverage.csv")
    for column in (
        "minimum_pointwise_coverage",
        "minimum_positions",
        "maximum_pointwise_coverage",
    ):
        if column not in strata:
            strata[column] = None if column == "minimum_positions" else np.nan
    strata["minimum_positions"] = strata["minimum_positions"].astype(object)
    for dgp in [f"D{i}" for i in range(6)]:
        values = pointwise[pointwise.dgp == dgp]
        minimum = values.coverage.min()
        mask = (strata.dgp == dgp) & (strata.stratum == "simultaneous_all64")
        strata.loc[mask, "minimum_pointwise_coverage"] = minimum
        strata.loc[mask, "minimum_positions"] = ";".join(
            map(str, values.loc[values.coverage == minimum, "trial"].astype(int))
        )
        strata.loc[mask, "maximum_pointwise_coverage"] = values.coverage.max()
    write_csv(
        OUTPUT / "frozen_coverage_summary.csv",
        strata.replace({np.nan: None}).to_dict("records"),
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Pointwise 95\\% model-averaged interval performance across 500 independent data sets per scenario. Overall, near, and far entries are mean coverages across their stated positions; near-boundary positions are at most two trial bins from a true boundary. Width is in log-response-time units and is shown as mean (Monte Carlo standard error). Minimum is the smallest empirical coverage among the 64 positions. Monte Carlo standard errors use the data set as the replication unit and retain within-trajectory dependence. D0 and D1 have no true boundaries.}",
        "\\label{tab:precision-coverage}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Scenario & Overall & MCSE & Near & Far & Width (MCSE) & Minimum \\\\ ",
        "\\midrule",
    ]
    for dgp in [f"D{i}" for i in range(6)]:
        frame = strata[
            (strata.dgp == dgp) & strata.stratum.isin(["all", "near_le_2", "far_gt_2"])
        ].set_index("stratum")
        row = frame.loc["all"]
        near = (
            "--"
            if "near_le_2" not in frame.index
            else f"{frame.loc['near_le_2', 'mean_pointwise_coverage']:.3f}"
        )
        far = f"{frame.loc['far_gt_2', 'mean_pointwise_coverage']:.3f}"
        minimum = strata[
            (strata.dgp == dgp) & (strata.stratum == "simultaneous_all64")
        ].iloc[0]["minimum_pointwise_coverage"]
        lines.append(
            f"{dgp} & {row.mean_pointwise_coverage:.3f} & {row.coverage_mcse:.3f} & {near} & {far} & {row.mean_interval_width:.4f} ({row.width_mcse:.4f}) & {minimum:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ASSETS / "precision_coverage_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_all(application_input: Path = PC_INPUT) -> None:
    if not PROTOCOL.exists():
        raise RuntimeError(
            "precommitted protocol is missing; it must exist before running"
        )
    sources = validate_protocol_sources()
    protocol_hash = sha256_file(PROTOCOL)
    states = legal_partitions()
    if len(states) != 27931 or len(set(states)) != len(states):
        raise RuntimeError(f"legal partition contract failed: {len(states)}")
    pointwise, strata, paired = summarize_frozen_simulations()
    library, kernel_hash = formal.compile_kernel(ROOT / "build" / "r61_formal")
    if kernel_hash != sources["scripts/kernel.c"]:
        raise RuntimeError("compiled formal-kernel source hash mismatch")
    probe = CWeightProbe()
    summaries = [
        run_precision_case(case, states, probe, library)
        for case in prepare_cases(
            json.loads(
                (SIMULATION_RESULTS / "resolved_config.json").read_text(
                    encoding="utf-8"
                )
            )["config"],
            application_input,
        )
    ]
    extension_needed = [
        row["case_id"]
        for row in summaries
        if not row["probability_threshold_pass"] or not row["endpoint_threshold_pass"]
    ]
    # The protocol permits, but does not mandate, a fresh 100k run. Keep the
    # decision explicit rather than silently lengthening only unfavorable cases.
    extension = {
        "invoked": False,
        "cases_failing_original_budget_threshold": extension_needed,
        "reason": "not needed"
        if not extension_needed
        else "not automatically invoked; original-budget errors and MCSE are reported for audit",
    }
    make_assets(strata, paired, summaries)
    make_results_md(strata, paired, summaries, sources)
    conditional_rows = application_conditional_interval_check(application_input)
    add_conditional_check_to_results(conditional_rows)
    refresh_coverage_table_from_machine_summary()
    generated = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "precision_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "source_lock": str(SOURCE_LOCK.relative_to(ROOT)),
        "source_lock_sha256": sha256_file(SOURCE_LOCK),
        "source_hashes": sources,
        "kernel_probe": provenance_path(probe.library_path),
        "generation_script_sha256": sha256_file(Path(__file__)),
        "partitions": len(states),
        "cases": summaries,
        "conditional_extension": extension,
        "application_conditional_interval_check": {
            "segments": len(conditional_rows),
            "maximum_endpoint_absolute_error": max(
                max(row["q025_absolute_error"], row["q975_absolute_error"])
                for row in conditional_rows
            ),
            "definition": conditional_rows[0]["definition"],
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "arviz": az.__version__,
        },
        "output_hashes": {
            provenance_path(path): sha256_file(path) for path in generated
        },
    }
    write_json(OUTPUT / "precision_manifest.json", manifest)
    print(
        json.dumps(
            {
                "cases": len(summaries),
                "partitions": len(states),
                "extension": extension,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


def finalize_provenance(
    application_input: Path = PC_INPUT, *, rewrite_historical: bool = False
) -> None:
    """Enrich an old manifest only after an explicit historical-rewrite opt-in."""
    if not rewrite_historical:
        raise RuntimeError(
            "refusing to rewrite an existing historical manifest without explicit opt-in"
        )
    manifest_path = OUTPUT / "precision_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "precision_manifest.json does not exist; run the audit first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditional_rows = application_conditional_interval_check(application_input)
    add_conditional_check_to_results(conditional_rows)
    refresh_coverage_table_from_machine_summary()
    manifest["analysis_source_hashes"] = {
        "current_precision_script_sha256": sha256_file(Path(__file__)),
        "current_exact_weight_wrapper_source_sha256": sha256_file(
            ROOT / "scripts" / "probe.c"
        ),
    }
    manifest["analysis_source_note"] = (
        "These are current release hashes recorded during an explicitly requested enrichment. "
        "They do not claim to be the generation hashes of a historical run."
    )
    manifest["application_conditional_interval_check"] = {
        "segments": len(conditional_rows),
        "maximum_endpoint_absolute_error": max(
            max(row["q025_absolute_error"], row["q975_absolute_error"])
            for row in conditional_rows
        ),
        "definition": conditional_rows[0]["definition"],
    }
    manifest["asset_hashes"] = {
        provenance_path(path): sha256_file(path)
        for path in sorted(ASSETS.glob("precision_*"))
        if path.is_file()
    }
    manifest["output_hashes"] = {
        provenance_path(path): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["analysis_source_hashes"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="execute the prespecified audit"
    )
    parser.add_argument(
        "--finalize-provenance",
        action="store_true",
        help="record source provenance for an existing completed audit",
    )
    parser.add_argument(
        "--rewrite-historical-manifest",
        action="store_true",
        help="explicitly permit --finalize-provenance to modify an existing manifest",
    )
    parser.add_argument(
        "--simulation-results",
        type=Path,
        default=SIMULATION_RESULTS,
        help="formal simulation result root containing resolved_config.json and replicates/",
    )
    parser.add_argument(
        "--application-results",
        type=Path,
        default=APPLICATION_RESULTS,
        help="application result root containing results.json",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="precision output directory"
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=ASSETS,
        help="generated presentation-asset directory",
    )
    parser.add_argument(
        "--application-input",
        type=Path,
        default=Path(os.environ.get("R61_PC_FLANKER_INPUT", str(PC_INPUT))),
        help="extracted RawData_Experiment1 directory (or set R61_PC_FLANKER_INPUT)",
    )
    args = parser.parse_args()
    configure_paths(
        args.simulation_results, args.application_results, args.output, args.assets
    )
    if args.run == args.finalize_provenance:
        parser.error("choose exactly one of --run or --finalize-provenance")
    if args.rewrite_historical_manifest and not args.finalize_provenance:
        parser.error("--rewrite-historical-manifest requires --finalize-provenance")
    if args.finalize_provenance:
        finalize_provenance(
            args.application_input, rewrite_historical=args.rewrite_historical_manifest
        )
    else:
        run_all(args.application_input)

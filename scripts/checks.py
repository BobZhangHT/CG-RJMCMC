#!/usr/bin/env python
"""Frozen PLOS E-M02 checks for the PC-Flanker real-data application.

This runner leaves the nominal model and C kernel untouched.  It recomputes
collapsed sufficient statistics for six prespecified nuisance specifications,
fits each with the formal sampler, and performs a joint posterior predictive
check under the nominal model.
"""

from __future__ import annotations

import os

# The review protocol caps numerical-library threading independently of any
# process-level parallelism used by a caller.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APPLICATION_RUNNER = ROOT / "scripts" / "application.py"
FORMAL_RUNNER = ROOT / "scripts" / "simulation.py"
FORMAL_CONFIG = ROOT / "scripts" / "config" / "r61_formal.yaml"
DEFAULT_INPUT = Path(
    os.environ.get(
        "R61_PC_FLANKER_INPUT", ROOT / "data" / "external" / "RawData_Experiment1"
    )
)
DEFAULT_OUTPUT = ROOT / "results" / "reproduction" / "checks"
DEFAULT_ASSETS = ROOT / "manuscript" / "reproduced"
APPLICATION_PROTOCOL = ROOT / "scripts" / "config" / "application_protocol.md"
SENSITIVITY_NAMESPACE = "R61-PC-FLANKER-E-M02-v1"
PPC_NAMESPACE = "R61-PC-FLANKER-E-M02-PPC-v1"
CASE_ORDER = (
    "nominal",
    "block10",
    "ri_var_0p25",
    "ri_var_4",
    "slope_var_0p01",
    "slope_var_0p1",
)
READABLE_CASE_LABELS = {
    "nominal": "Nominal",
    "block10": "Block effects",
    "ri_var_0p25": "RI ratio 0.25",
    "ri_var_4": "RI ratio 4",
    "slope_var_0p01": "Slope ratio 0.01",
    "slope_var_0p1": "Slope ratio 0.1",
}
READABLE_CHECK_LABELS = {
    "residual_mean_signed": "Residual mean",
    "residual_sd_ratio": "Residual SD ratio",
    "lower_tail_rate": "Lower tail rate",
    "upper_tail_rate": "Upper tail rate",
    "max_abs_participant_mean": "Participant mean (max abs.)",
    "max_abs_phase_congruency_mean": "Phase/condition mean (max abs.)",
    "participant_contrast_sd": "Participant-phase\nresidual contrast SD",
    "lag1_analyzed_order_signed": "Within-block lag-1",
    "lag1_distance1_signed": "Strict native-distance-1 lag",
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


application = _load_module("r61_pc_application_e_m02_dependency", APPLICATION_RUNNER)
formal = _load_module("r61_formal_e_m02_dependency", FORMAL_RUNNER)


def case_spec(name: str) -> dict[str, Any]:
    """Return one of the six fixed nuisance-model specifications."""
    specs = {
        "nominal": {"participant_intercept_variance_ratio": 1.0},
        "block10": {
            "participant_intercept_variance_ratio": 1.0,
            "block_precision": 4.0,
        },
        "ri_var_0p25": {"participant_intercept_variance_ratio": 0.25},
        "ri_var_4": {"participant_intercept_variance_ratio": 4.0},
        "slope_var_0p01": {
            "participant_intercept_variance_ratio": 1.0,
            "participant_slope_variance_ratio": 0.01,
        },
        "slope_var_0p1": {
            "participant_intercept_variance_ratio": 1.0,
            "participant_slope_variance_ratio": 0.1,
        },
    }
    return {"name": name, **specs[name]}


def _selected_frame(
    input_dir: Path, first_pc: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproduce the frozen application rows while retaining native order fields."""
    kept: list[pd.DataFrame] = []
    excluded_ids: list[str] = []
    block_rank = {
        block: rank for rank, block in enumerate((2, 3, 4, 5, 6, 8, 9, 10, 11, 12))
    }
    for source in application._source_files(input_dir):
        frame = (
            pd.read_csv(source)
            .sort_values(["blockNum", "trialNum"])
            .reset_index(drop=True)
        )
        flanker, catch = frame["type"].eq("Flanker"), frame["type"].eq("Catch")
        flanker_error = 1.0 - float(frame.loc[flanker, "corrCode"].eq(1).mean())
        catch_error = 1.0 - float(frame.loc[catch, "corrCode"].eq(1).mean())
        uid = str(frame["vpNum"].iloc[0])
        if flanker_error > 0.20 or catch_error > 0.45:
            excluded_ids.append(uid)
            continue
        if str(frame["block_congruency"].iloc[0]) != first_pc:
            continue
        follows_catch = frame["type"].shift(1).eq("Catch")
        use = (
            flanker
            & ~frame["blockNum"].isin((1, 7))
            & frame["modality"].eq("Visual")
            & frame["corrCode"].eq(1)
            & frame["trialNum"].gt(1)
            & ~follows_catch
            & frame["rt"].ge(100)
        )
        selected = frame.loc[use].copy()
        selected["uid"] = uid
        kept.append(selected)
    data = pd.concat(kept, ignore_index=True)
    ids = sorted(data["uid"].unique())
    id_to_person = {uid: index for index, uid in enumerate(ids)}
    data["person"] = data["uid"].map(id_to_person).astype(int)
    data["block_rank"] = data["blockNum"].map(block_rank).astype(int)
    analyzed_position = data["block_rank"].to_numpy() * 84 + data["trialNum"].to_numpy(
        dtype=int
    )
    data["bin"] = np.minimum(64, np.ceil(analyzed_position * 64 / 840).astype(int))
    data["z"] = (data["bin"].to_numpy() - 32.5) / np.std(np.arange(1, 65))
    data["c"] = np.where(data["congruency"].eq("Incongruent"), 0.5, -0.5)
    data["congruency_label"] = np.where(data["c"].gt(0), "Incongruent", "Congruent")
    data["phase"] = np.where(data["bin"].le(32), "before", "after")
    test = np.zeros(len(data), dtype=bool)
    for uid in ids:
        indices = np.flatnonzero(data["uid"].to_numpy() == uid)
        count = max(1, round(0.20 * len(indices)))
        order = sorted(
            indices,
            key=lambda j: hashlib.sha256(
                f"{application.SPLIT_NAMESPACE}|{uid}|{int(data['blockNum'].iloc[j])}|{int(data['trialNum'].iloc[j])}".encode()
            ).digest(),
        )
        test[order[:count]] = True
    response = np.log(data["rt"].to_numpy(dtype=float))
    training_mean = float(response[~test].mean())
    data["y"] = response - training_mean
    data["is_test"] = test
    metadata = {
        "first_pc": first_pc,
        "participants": len(ids),
        "observations": len(data),
        "training_observations": int((~test).sum()),
        "test_observations": int(test.sum()),
        "training_mean_log_rt": training_mean,
        "excluded_participants": sorted(set(excluded_ids)),
        "participant_hash": hashlib.sha256("|".join(ids).encode()).hexdigest(),
        "split_hash": formal.sha256_array(np.packbits(test)),
    }
    return data.reset_index(drop=True), metadata


def _pack(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        frame["person"].to_numpy(dtype=int),
        frame["bin"].to_numpy(dtype=int),
        frame["z"].to_numpy(dtype=float),
        frame["c"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
    )


def prepare_group(
    input_dir: Path, first_pc: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data, metadata = _selected_frame(input_dir, first_pc)
    train = data.loc[~data["is_test"]].copy().reset_index(drop=True)
    test = data.loc[data["is_test"]].copy().reset_index(drop=True)
    ref_train, ref_test, ref_meta = application.prepare_group(input_dir, first_pc)
    for observed, reference, label in zip(
        _pack(train) + _pack(test),
        ref_train + ref_test,
        (
            "train_person",
            "train_bin",
            "train_z",
            "train_c",
            "train_y",
            "test_person",
            "test_bin",
            "test_z",
            "test_c",
            "test_y",
        ),
    ):
        if observed.dtype.kind in "iu":
            equal = np.array_equal(observed, reference)
        else:
            equal = np.allclose(observed, reference, atol=0.0, rtol=0.0)
        if not equal:
            raise RuntimeError(f"frozen preprocessing mismatch: {first_pc} {label}")
    for key in (
        "participants",
        "observations",
        "training_observations",
        "test_observations",
        "training_mean_log_rt",
        "participant_hash",
        "split_hash",
    ):
        if metadata[key] != ref_meta[key]:
            raise RuntimeError(f"frozen metadata mismatch: {first_pc} {key}")
    return train, test, metadata


def contrast_design(frame: pd.DataFrame) -> np.ndarray:
    """Construct the 64-column bin-by-congruency effect design."""
    W = np.zeros((len(frame), formal.T), dtype=float)
    W[np.arange(len(frame)), frame["bin"].to_numpy(dtype=int) - 1] = frame[
        "c"
    ].to_numpy(dtype=float)
    return W


def nuisance_design(
    frame: pd.DataFrame, participants: int, spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct nuisance columns and their fixed conditional precisions."""
    person = frame["person"].to_numpy(dtype=int)
    columns = [np.ones(len(frame)), frame["z"].to_numpy(dtype=float)]
    precision = [0.25, 1.0]
    names = ["intercept", "linear_bin_trend"]
    eye_p = np.eye(participants, dtype=float)[person]
    columns.extend(eye_p.T)
    ri_ratio = float(spec["participant_intercept_variance_ratio"])
    precision.extend([1.0 / ri_ratio] * participants)
    names.extend([f"participant_intercept_{j}" for j in range(participants)])
    if "block_precision" in spec:
        block = np.eye(10, dtype=float)[frame["block_rank"].to_numpy(dtype=int)]
        columns.extend(block.T)
        precision.extend([float(spec["block_precision"])] * 10)
        names.extend([f"retained_block_{j + 1}" for j in range(10)])
    if "participant_slope_variance_ratio" in spec:
        slopes = eye_p * frame["c"].to_numpy(dtype=float)[:, None]
        columns.extend(slopes.T)
        ratio = float(spec["participant_slope_variance_ratio"])
        precision.extend([1.0 / ratio] * participants)
        names.extend([f"participant_congruency_slope_{j}" for j in range(participants)])
    Z = np.column_stack(columns)
    D_nu = np.diag(np.asarray(precision, dtype=float))
    return Z, D_nu, names


def generalized_collapsed(
    W: np.ndarray, Z: np.ndarray, D_nu: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collapse proper-prior nuisance effects without forming the dense Q matrix."""
    """Compute W'QW, W'Qy, y'Qy without materializing the n by n Q."""
    if Z.shape[1] != D_nu.shape[0] or D_nu.shape[0] != D_nu.shape[1]:
        raise ValueError("nuisance precision dimension mismatch")
    V = np.column_stack((W, y))
    Lambda_nu = D_nu + Z.T @ Z
    chol = np.linalg.cholesky(Lambda_nu)
    projected = np.linalg.solve(chol.T, np.linalg.solve(chol, Z.T @ V))
    QV = V - Z @ projected
    H = W.T @ QV[:, :-1]
    h = W.T @ QV[:, -1]
    r = float(y @ QV[:, -1])
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(h)) or not math.isfinite(r):
        raise FloatingPointError("non-finite generalized collapsed summaries")
    return H, h, r


def dense_collapsed(
    W: np.ndarray, Z: np.ndarray, D_nu: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reference dense-Q implementation used to verify generalized collapse."""
    Q = np.eye(len(y)) - Z @ np.linalg.solve(D_nu + Z.T @ Z, Z.T)
    return W.T @ Q @ W, W.T @ Q @ y, float(y @ Q @ y)


def segment_design(frame: pd.DataFrame, cuts: tuple[int, ...]) -> np.ndarray:
    bins = frame["bin"].to_numpy(dtype=int)
    segment = np.searchsorted(np.asarray(cuts, dtype=int), bins, side="left")
    S = np.zeros((len(frame), len(cuts) + 1), dtype=float)
    S[np.arange(len(frame)), segment] = frame["c"].to_numpy(dtype=float)
    return S


class PosteriorSystem(NamedTuple):
    mean: np.ndarray
    lambda_post: np.ndarray
    chol_lambda: np.ndarray
    rate: float
    shape: float
    X_full: np.ndarray


def posterior_system(
    frame: pd.DataFrame, Z: np.ndarray, D_nu: np.ndarray, cuts: tuple[int, ...]
) -> PosteriorSystem:
    S = segment_design(frame, cuts)
    X_full = np.column_stack((S, Z))
    D_full = np.diag(np.r_[np.repeat(4.0, S.shape[1]), np.diag(D_nu)])
    Lambda_post = D_full + X_full.T @ X_full
    chol_lambda = np.linalg.cholesky(Lambda_post)
    y = frame["y"].to_numpy(dtype=float)
    b = X_full.T @ y
    mean = np.linalg.solve(chol_lambda.T, np.linalg.solve(chol_lambda, b))
    rate = float(2.0 + 0.5 * (y @ y - b @ mean))
    shape = float(3.0 + len(y) / 2.0)
    if rate <= 0 or not math.isfinite(rate):
        raise FloatingPointError(
            "posterior inverse-gamma rate is not positive and finite"
        )
    return PosteriorSystem(mean, Lambda_post, chol_lambda, rate, shape, X_full)


def draw_coefficients_given_sigma(
    system: PosteriorSystem, sigma2: float, rng: np.random.Generator
) -> np.ndarray:
    z = rng.normal(size=len(system.mean))
    return system.mean + math.sqrt(sigma2) * np.linalg.solve(system.chol_lambda.T, z)


def partition_key(row: np.ndarray) -> tuple[int, ...]:
    """Convert a sampler row to its canonical cut tuple."""
    k = int(row[0])
    return tuple(int(value) for value in row[1 : 1 + k])


def posterior_diagnostic(name: str, values: np.ndarray, wall: float) -> dict[str, Any]:
    rhat, ess, constant, status = formal.rank_diagnostic(values)
    return {
        "parameter": name,
        "split_Rhat": rhat,
        "bulk_ESS": ess,
        "ESS_per_second": None if ess is None else ess / wall,
        "constant_chain": constant,
        "diagnostic_status": status,
    }


def heldout_rmse(
    train: pd.DataFrame,
    test: pd.DataFrame,
    participants: int,
    spec: dict[str, Any],
    curve: np.ndarray,
) -> float:
    Z_train, D_nu, _ = nuisance_design(train, participants, spec)
    Z_test, _, _ = nuisance_design(test, participants, spec)
    residual = (
        train["y"].to_numpy(dtype=float)
        - train["c"].to_numpy() * curve[train["bin"].to_numpy(dtype=int) - 1]
    )
    Lambda_nu = D_nu + Z_train.T @ Z_train
    nuisance_mean = np.linalg.solve(Lambda_nu, Z_train.T @ residual)
    pred = (
        Z_test @ nuisance_mean
        + test["c"].to_numpy() * curve[test["bin"].to_numpy(dtype=int) - 1]
    )
    return float(np.sqrt(np.mean(np.square(test["y"].to_numpy(dtype=float) - pred))))


def summarize_draws(
    chains: list[np.ndarray], wall: float, kmax: int
) -> tuple[dict[str, Any], np.ndarray]:
    summary = formal.summarize_a3(
        chains, [np.zeros(8, dtype=np.int64)] * len(chains), wall, kmax
    )
    stack = np.stack(chains)
    curve_stack = stack[:, :, 1 + formal.MAX_K :]
    before = curve_stack[:, :, :32].mean(axis=2)
    after = curve_stack[:, :, 32:].mean(axis=2)
    change = after - before

    def interval(values: np.ndarray) -> dict[str, float]:
        flat = values.reshape(-1)
        return {
            "mean": float(flat.mean()),
            "q025": float(np.quantile(flat, 0.025)),
            "q975": float(np.quantile(flat, 0.975)),
        }

    summary["phase_before"] = interval(before)
    summary["phase_after"] = interval(after)
    summary["phase_after_minus_before"] = interval(change)
    summary["phase_direction"] = (
        "positive"
        if summary["phase_after_minus_before"]["q025"] > 0
        else "negative"
        if summary["phase_after_minus_before"]["q975"] < 0
        else "uncertain"
    )
    pip12 = np.mean(
        [12 in partition_key(row) for row in stack.reshape(-1, formal.DRAW_WIDTH)]
    )
    pip32 = np.mean(
        [32 in partition_key(row) for row in stack.reshape(-1, formal.DRAW_WIDTH)]
    )
    summary["PIP12"] = float(pip12)
    summary["PIP32"] = float(pip32)
    extra = [
        posterior_diagnostic(
            "PIP12_indicator",
            np.asarray(
                [[12 in partition_key(row) for row in chain] for chain in stack],
                dtype=float,
            ),
            wall,
        ),
        posterior_diagnostic(
            "PIP32_indicator",
            np.asarray(
                [[32 in partition_key(row) for row in chain] for chain in stack],
                dtype=float,
            ),
            wall,
        ),
        posterior_diagnostic("phase_after_minus_before", change, wall),
    ]
    summary["extended_diagnostics"] = summary["diagnostics"] + extra
    applicable = [
        x
        for x in summary["extended_diagnostics"]
        if x["diagnostic_status"] == "estimated"
    ]
    summary["extended_max_split_Rhat"] = max(
        (x["split_Rhat"] for x in applicable), default=None
    )
    summary["extended_min_bulk_ESS"] = min(
        (x["bulk_ESS"] for x in applicable), default=None
    )
    return summary, stack.reshape(-1, formal.DRAW_WIDTH)


def fit_case(
    train: pd.DataFrame,
    test: pd.DataFrame,
    participants: int,
    first_pc: str,
    spec: dict[str, Any],
    library: Any,
    cfg: dict[str, Any],
    iterations: int,
    warmup: int,
) -> tuple[dict[str, Any], list[np.ndarray], tuple[np.ndarray, np.ndarray, float]]:
    W = contrast_design(train)
    Z, D_nu, nuisance_names = nuisance_design(train, participants, spec)
    H, h, r = generalized_collapsed(W, Z, D_nu, train["y"].to_numpy(dtype=float))
    nominal = cfg["nominal_model"]
    chains: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    seeds: list[int] = []
    started = time.perf_counter()
    for index, initial_k in enumerate(cfg["sampler"]["initial_k"], 1):
        if spec["name"] == "nominal":
            seed = formal.deterministic_seed(
                application.CHAIN_NAMESPACE, f"{first_pc}|CHAIN{index}"
            )
        else:
            seed = formal.deterministic_seed(
                SENSITIVITY_NAMESPACE, f"{first_pc}|{spec['name']}|CHAIN{index}"
            )
        chain, chain_stats = formal.run_chain(
            library,
            H,
            h,
            r,
            len(train),
            iterations,
            warmup,
            seed,
            initial_k,
            nominal["kmax"],
            nominal["min_segment"],
            nominal["prior_factorial_power"],
        )
        chains.append(chain)
        stats.append(chain_stats)
        seeds.append(seed)
    wall = time.perf_counter() - started
    base_summary, flat = summarize_draws(chains, wall, nominal["kmax"])
    # Restore the actual proposal acceptance accounting (summarize_draws uses
    # zero placeholders only to avoid duplicating the formal summary logic).
    acceptance = {}
    for label, attempted, accepted in (
        ("birth", 0, 1),
        ("death", 2, 3),
        ("move", 4, 5),
    ):
        denominator = sum(int(x[attempted]) for x in stats)
        acceptance[label] = sum(int(x[accepted]) for x in stats) / max(1, denominator)
    base_summary["acceptance"] = acceptance
    curve = np.asarray(base_summary["bma_curve"], dtype=float)
    joint_state = list(base_summary["joint_map"])
    participant_level = participant_level_change_summary(all_contrast_rows)
    result = {
        "case": spec,
        "nuisance_columns": nuisance_names,
        "chain_seeds": seeds,
        "three_chain_walltime_seconds": wall,
        "heldout_RMSE": heldout_rmse(train, test, participants, spec, curve),
        "p_k": base_summary["p_k"],
        "PIP12": base_summary["PIP12"],
        "PIP32": base_summary["PIP32"],
        "joint_map_state_K_then_cuts": joint_state,
        "joint_map_K": int(joint_state[0]),
        "joint_map_cuts": joint_state[1:],
        "phase_before": base_summary["phase_before"],
        "phase_after": base_summary["phase_after"],
        "phase_after_minus_before": base_summary["phase_after_minus_before"],
        "phase_direction": base_summary["phase_direction"],
        "curve_mean": curve.tolist(),
        "curve_q025": np.asarray(base_summary["bma_q025"]).tolist(),
        "curve_q975": np.asarray(base_summary["bma_q975"]).tolist(),
        "boundary_pip": base_summary["boundary_pip"],
        "diagnostics": base_summary["extended_diagnostics"],
        "max_split_Rhat": base_summary["extended_max_split_Rhat"],
        "min_bulk_ESS": base_summary["extended_min_bulk_ESS"],
        "acceptance": acceptance,
    }
    return result, chains, (H, h, r)


def lag_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for (_, block), part in frame.groupby(["person", "blockNum"], sort=True):
        order = part.sort_values("trialNum").index.to_numpy(dtype=int)
        if len(order) < 2:
            continue
        native = frame.loc[order, "trialNum"].to_numpy(dtype=int)
        for left, right, distance in zip(order[:-1], order[1:], np.diff(native)):
            rows.append(
                {
                    "left": int(left),
                    "right": int(right),
                    "block": int(block),
                    "distance": int(distance),
                }
            )
    return pd.DataFrame(rows, columns=["left", "right", "block", "distance"])


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def check_statistics(
    frame: pd.DataFrame, residual: np.ndarray, sigma: float, pairs: pd.DataFrame
) -> dict[str, float]:
    person_means = pd.Series(residual).groupby(frame["person"]).mean().to_numpy()
    cell_means = (
        pd.Series(residual)
        .groupby([frame["phase"], frame["congruency_label"]])
        .mean()
        .to_numpy()
    )
    contrasts = []
    for (_, _), cell in frame.assign(_residual=residual).groupby(["person", "phase"]):
        plus = cell.loc[cell["c"].gt(0), "_residual"]
        minus = cell.loc[cell["c"].lt(0), "_residual"]
        if len(plus) and len(minus):
            contrasts.append(float(plus.mean() - minus.mean()))
    left, right = pairs["left"].to_numpy(dtype=int), pairs["right"].to_numpy(dtype=int)
    strict = pairs["distance"].to_numpy(dtype=int) == 1
    lag_all = safe_corr(residual[left], residual[right])
    lag_strict = safe_corr(residual[left[strict]], residual[right[strict]])
    return {
        "residual_mean_signed": float(np.mean(residual) / sigma),
        "residual_sd_ratio": float(np.std(residual, ddof=1) / sigma),
        "lower_tail_rate": float(np.mean(residual < -2.0 * sigma)),
        "upper_tail_rate": float(np.mean(residual > 2.0 * sigma)),
        "max_abs_participant_mean": float(np.max(np.abs(person_means)) / sigma),
        "max_abs_phase_congruency_mean": float(np.max(np.abs(cell_means)) / sigma),
        "participant_contrast_sd": float(np.std(contrasts, ddof=1) / sigma),
        "lag1_analyzed_order_signed": lag_all,
        "lag1_distance1_signed": lag_strict,
    }


SIGNED_CHECKS = {
    "residual_mean_signed",
    "lag1_analyzed_order_signed",
    "lag1_distance1_signed",
}


def run_ppc(
    train: pd.DataFrame,
    participants: int,
    first_pc: str,
    chains: list[np.ndarray],
    replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, float]:
    spec = case_spec("nominal")
    Z, D_nu, _ = nuisance_design(train, participants, spec)
    flat = np.stack(chains).reshape(-1, formal.DRAW_WIDTH)
    rng = np.random.default_rng(formal.deterministic_seed(PPC_NAMESPACE, first_pc))
    selected = rng.integers(0, len(flat), size=replicates)
    cache: dict[tuple[int, ...], PosteriorSystem] = {}
    pairs = lag_pairs(train)
    y = train["y"].to_numpy(dtype=float)
    mu_sum = np.zeros(len(train), dtype=float)
    sigma_draws = np.empty(replicates, dtype=float)
    draw_rows: list[dict[str, Any]] = []
    for rep, selected_index in enumerate(selected, 1):
        cuts = partition_key(flat[selected_index])
        if cuts not in cache:
            cache[cuts] = posterior_system(train, Z, D_nu, cuts)
        system = cache[cuts]
        sigma2 = system.rate / rng.gamma(system.shape)
        theta = draw_coefficients_given_sigma(system, sigma2, rng)
        mu = system.X_full @ theta
        replicated_residual = rng.normal(0.0, math.sqrt(sigma2), size=len(train))
        observed_residual = y - mu
        observed = check_statistics(train, observed_residual, math.sqrt(sigma2), pairs)
        replicated = check_statistics(
            train, replicated_residual, math.sqrt(sigma2), pairs
        )
        for name in observed:
            draw_rows.append(
                {
                    "group": first_pc + "-first",
                    "replicate": rep,
                    "check": name,
                    "observed": observed[name],
                    "replicated": replicated[name],
                }
            )
        mu_sum += mu
        sigma_draws[rep - 1] = math.sqrt(sigma2)
    summaries = []
    draw_frame = pd.DataFrame(draw_rows)
    for name, values in draw_frame.groupby("check", sort=False):
        obs = values["observed"].to_numpy(dtype=float)
        rep = values["replicated"].to_numpy(dtype=float)
        if name in SIGNED_CHECKS:
            exceed = np.abs(rep) >= np.abs(obs)
        else:
            exceed = rep >= obs
        p = float(np.mean(exceed))
        summaries.append(
            {
                "group": first_pc + "-first",
                "check": name,
                "observed_median": float(np.median(obs)),
                "replicated_median": float(np.median(rep)),
                "replicated_q025": float(np.quantile(rep, 0.025)),
                "replicated_q975": float(np.quantile(rep, 0.975)),
                "posterior_predictive_upper_tail_area": p,
                "folded_discrepancy_tail_area": float(min(1.0, 2.0 * min(p, 1.0 - p))),
                "descriptive_tail_flag_0p05": bool(2.0 * min(p, 1.0 - p) <= 0.05),
            }
        )
    result = {
        "replicates": replicates,
        "unique_partitions_used": len(cache),
        "checks": summaries,
        "conditional_population": "existing participants",
        "joint_draw": True,
    }
    return result, draw_rows, mu_sum / replicates, float(np.mean(sigma_draws))


def residual_summary_rows(
    train: pd.DataFrame, residual: np.ndarray, sigma: float, group: str
) -> list[dict[str, Any]]:
    work = train.copy()
    work["residual"] = residual
    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("participant", ["uid"]),
        ("congruency", ["congruency_label"]),
        ("phase", ["phase"]),
        ("congruency_phase", ["congruency_label", "phase"]),
        ("participant_congruency_phase", ["uid", "congruency_label", "phase"]),
    ]
    for stratum, keys in strata:
        groups: Iterable[tuple[Any, pd.DataFrame]] = (
            [((), work)] if not keys else work.groupby(keys, sort=True)
        )
        for key, cell in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(keys, key_tuple))
            values = cell["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "stratum": stratum,
                    **labels,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "lower_tail_rate_2sigma": float(np.mean(values < -2 * sigma)),
                    "upper_tail_rate_2sigma": float(np.mean(values > 2 * sigma)),
                }
            )
    return rows


def participant_contrast_rows(
    train: pd.DataFrame, residual: np.ndarray, curve: np.ndarray, group: str
) -> list[dict[str, Any]]:
    work = train.copy()
    work["residual"] = residual
    phase_effect = {
        "before": float(np.mean(curve[:32])),
        "after": float(np.mean(curve[32:])),
    }
    rows = []
    for (uid, phase), cell in work.groupby(["uid", "phase"], sort=True):
        plus = cell.loc[cell["c"].gt(0), "residual"]
        minus = cell.loc[cell["c"].lt(0), "residual"]
        if not len(plus) or not len(minus):
            continue
        residual_contrast = float(plus.mean() - minus.mean())
        rows.append(
            {
                "group": group,
                "participant": uid,
                "phase": phase,
                "n_incongruent": len(plus),
                "n_congruent": len(minus),
                "residual_contrast": residual_contrast,
                "adjusted_participant_contrast": phase_effect[phase]
                + residual_contrast,
                "nominal_population_phase_effect": phase_effect[phase],
            }
        )
    return rows


def participant_level_change_summary(
    rows: list[dict[str, Any]], bootstrap_replicates: int = 50_000
) -> dict[str, Any]:
    """Summarize phase changes with participants as the resampling units.

    Each participant contributes one adjusted incongruent-minus-congruent
    contrast before and after the experimental reversal.  The group summaries
    give equal weight to participants, and the order-by-phase contrast is the
    low-first mean change minus the high-first mean change.  Percentile
    intervals resample complete participants within randomized-order groups.
    """
    frame = pd.DataFrame(rows)
    wide = frame.pivot(
        index=["group", "participant"],
        columns="phase",
        values="adjusted_participant_contrast",
    )
    if not {"before", "after"}.issubset(wide.columns) or wide.isna().any().any():
        raise ValueError("each participant must have before and after contrasts")
    wide["change"] = wide["after"] - wide["before"]

    def seed_for(label: str) -> int:
        digest = hashlib.sha256(
            f"CG-RJMCMC-participant-bootstrap-v1|{label}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "little")

    def resampled_means(values: np.ndarray, label: str) -> tuple[int, np.ndarray]:
        seed = seed_for(label)
        rng = np.random.default_rng(seed)
        means = np.empty(bootstrap_replicates, dtype=float)
        chunk = 5_000
        for start in range(0, bootstrap_replicates, chunk):
            stop = min(start + chunk, bootstrap_replicates)
            draws = rng.integers(0, len(values), size=(stop - start, len(values)))
            means[start:stop] = values[draws].mean(axis=1)
        return seed, means

    changes = {
        group: part["change"].to_numpy(dtype=float)
        for group, part in wide.groupby(level="group", sort=True)
    }
    expected = {"High-first", "Low-first"}
    if set(changes) != expected:
        raise ValueError(f"unexpected order groups: {sorted(changes)}")

    summary_rows: list[dict[str, Any]] = []
    for group in ("High-first", "Low-first"):
        values = changes[group]
        seed, means = resampled_means(values, group)
        q025, q975 = np.quantile(means, [0.025, 0.975])
        summary_rows.append(
            {
                "contrast": "after_minus_before",
                "group_1": group,
                "group_2": "",
                "n_1": len(values),
                "n_2": 0,
                "estimate": float(values.mean()),
                "participant_sd": float(values.std(ddof=1)),
                "bootstrap_q025": float(q025),
                "bootstrap_q975": float(q975),
                "bootstrap_replicates": bootstrap_replicates,
                "seed": seed,
            }
        )

    interaction_seed = seed_for("interaction")
    interaction_rng = np.random.default_rng(interaction_seed)
    interaction = np.empty(bootstrap_replicates, dtype=float)
    low_values, high_values = changes["Low-first"], changes["High-first"]
    for start in range(0, bootstrap_replicates, 5_000):
        stop = min(start + 5_000, bootstrap_replicates)
        low_draws = interaction_rng.integers(
            0, len(low_values), size=(stop - start, len(low_values))
        )
        high_draws = interaction_rng.integers(
            0, len(high_values), size=(stop - start, len(high_values))
        )
        interaction[start:stop] = low_values[low_draws].mean(axis=1) - high_values[
            high_draws
        ].mean(axis=1)
    q025, q975 = np.quantile(interaction, [0.025, 0.975])
    summary_rows.append(
        {
            "contrast": "order_by_phase_low_minus_high",
            "group_1": "Low-first",
            "group_2": "High-first",
            "n_1": len(changes["Low-first"]),
            "n_2": len(changes["High-first"]),
            "estimate": float(
                changes["Low-first"].mean() - changes["High-first"].mean()
            ),
            "participant_sd": "",
            "bootstrap_q025": float(q025),
            "bootstrap_q975": float(q975),
            "bootstrap_replicates": bootstrap_replicates,
            "seed": interaction_seed,
        }
    )
    return {
        "estimand": (
            "equal-participant mean of adjusted training-response contrasts; "
            "interaction is low-first minus high-first change"
        ),
        "resampling_unit": "participant within randomized-order group",
        "interval": "deterministic nonparametric percentile bootstrap",
        "rows": summary_rows,
    }


def lag_summary_rows(
    train: pd.DataFrame, residual: np.ndarray, group: str
) -> list[dict[str, Any]]:
    pairs = lag_pairs(train)
    rows = []
    for block_label, cell in [("all", pairs)] + [
        (str(block), part) for block, part in pairs.groupby("block", sort=True)
    ]:
        for distance_label, chosen in [
            ("all_analyzed_adjacencies", cell),
            ("strict_1", cell.loc[cell["distance"].eq(1)]),
        ]:
            left, right = (
                chosen["left"].to_numpy(dtype=int),
                chosen["right"].to_numpy(dtype=int),
            )
            rows.append(
                {
                    "group": group,
                    "block": block_label,
                    "distance": distance_label,
                    "n_pairs": len(chosen),
                    "correlation": safe_corr(residual[left], residual[right]),
                    "native_distance_mean": float(chosen["distance"].mean())
                    if len(chosen)
                    else None,
                    "native_distance_median": float(chosen["distance"].median())
                    if len(chosen)
                    else None,
                    "native_distance_max": int(chosen["distance"].max())
                    if len(chosen)
                    else None,
                }
            )
        for distance, chosen in cell.groupby("distance", sort=True):
            left, right = (
                chosen["left"].to_numpy(dtype=int),
                chosen["right"].to_numpy(dtype=int),
            )
            rows.append(
                {
                    "group": group,
                    "block": block_label,
                    "distance": str(int(distance)),
                    "n_pairs": len(chosen),
                    "correlation": safe_corr(residual[left], residual[right]),
                    "native_distance_mean": float(distance),
                    "native_distance_median": float(distance),
                    "native_distance_max": int(distance),
                }
            )
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def plot_sensitivity(groups: list[dict[str, Any]], paths: list[Path]) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(6.3, 6.5), sharex=True, constrained_layout=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(CASE_ORDER)))
    for ax, group in zip(axes, groups):
        for color, name in zip(colors, CASE_ORDER):
            case = next(item for item in group["cases"] if item["case"]["name"] == name)
            ax.plot(
                np.arange(1, 65),
                case["curve_mean"],
                label=READABLE_CASE_LABELS[name],
                color=color,
                linewidth=2.2 if name in ("nominal", "block10") else 1.25,
                linestyle="--" if name == "block10" else "-",
                alpha=1.0 if name == "nominal" else 0.82,
            )
        ax.axvline(12.5, color="0.65", linestyle=":", linewidth=1)
        ax.axvline(32.5, color="0.25", linestyle="--", linewidth=1)
        ax.set_ylabel("log RT contrast")
        ax.set_title(group["group"])
    axes[-1].set_xlabel("analysis bin")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_ppc(groups: list[dict[str, Any]], paths: list[Path]) -> None:
    checks = [item["check"] for item in groups[0]["ppc"]["checks"]]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.3, 5.5), sharey=True, constrained_layout=True
    )
    for ax, group in zip(axes, groups):
        lookup = {item["check"]: item for item in group["ppc"]["checks"]}
        y = np.arange(len(checks))
        lo = np.array([lookup[x]["replicated_q025"] for x in checks])
        hi = np.array([lookup[x]["replicated_q975"] for x in checks])
        mid = np.array([lookup[x]["replicated_median"] for x in checks])
        obs = np.array([lookup[x]["observed_median"] for x in checks])
        ax.hlines(y, lo, hi, color="#4c78a8", linewidth=2)
        ax.scatter(
            mid, y, color="#4c78a8", s=22, label="replicated median/95% interval"
        )
        ax.scatter(obs, y, marker="x", color="#e45756", s=38, label="observed")
        ax.axvline(0, color="0.8", linewidth=0.8)
        ax.set_title(group["group"])
        ax.set_xlabel("check statistic")
    axes[0].set_yticks(
        np.arange(len(checks)), [READABLE_CHECK_LABELS[x] for x in checks], fontsize=8
    )
    # Symbol meanings are stated in the figure caption; omitting an in-panel
    # legend preserves the strict-lag row at journal column width.
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# E-M02 real-application model checks",
        "",
        "The nominal factorial-count-prior joint-partition model, p(K) proportional to 1/K!, remains the primary analysis. The five alternatives below are fixed-prior sensitivity checks, not candidates selected after viewing the results.",
        "",
    ]
    for group in result["groups"]:
        nominal = group["cases"][0]
        pip32 = [x["PIP32"] for x in group["cases"]]
        pip12 = [x["PIP12"] for x in group["cases"]]
        changes = [x["phase_after_minus_before"]["mean"] for x in group["cases"]]
        rmse = [x["heldout_RMSE"] for x in group["cases"]]
        lines.extend(
            [
                f"## {group['group']}",
                "",
                f"Nominal PIP(32)={nominal['PIP32']:.4f}, PIP(12)={nominal['PIP12']:.4f}, joint state has K={nominal['joint_map_K']} and cuts={nominal['joint_map_cuts']}, and held-out RMSE={nominal['heldout_RMSE']:.5f}.",
                f"Across the six prespecified cases, PIP(32) ranged {min(pip32):.4f}--{max(pip32):.4f}, PIP(12) ranged {min(pip12):.4f}--{max(pip12):.4f}, the posterior mean after-minus-before phase contrast ranged {min(changes):.4f}--{max(changes):.4f}, and held-out RMSE ranged {min(rmse):.5f}--{max(rmse):.5f}.",
                "",
            ]
        )
        if group["group"] == "High-first":
            block = next(x for x in group["cases"] if x["case"]["name"] == "block10")
            lines.extend(
                [
                    f"The high-first bin-12 PIP fell from {nominal['PIP12']:.4f} nominally to {block['PIP12']:.4f} with common block effects, so this early boundary is not robust to the checked common time structure. Bin 32 and the negative phase-change direction were retained in all six cases.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Bin 32 and the positive phase-change direction were retained in all six cases; bin 12 remained unsupported.",
                    "",
                ]
            )
        mismatch = [
            x["check"]
            for x in group["ppc"]["checks"]
            if x["descriptive_tail_flag_0p05"]
        ]
        lines.append(
            "Nominal PPC checks flagged by the descriptive folded-discrepancy tail area <=0.05: "
            + (", ".join(mismatch) if mismatch else "none")
            + ". These are posterior-predictive diagnostics, not calibrated frequentist tests."
        )
        lines.append("")
    participant = result.get("participant_level_analysis")
    if participant:
        lines.extend(
            [
                "## Participant-level phase changes",
                "",
                "Adjusted before/after contrasts were paired within each participant. "
                "Intervals used 50,000 deterministic nonparametric bootstrap resamples "
                "of complete participants within randomized-order groups.",
                "",
            ]
        )
        for row in participant["rows"]:
            lines.append(
                f"{row['contrast']} ({row['group_1']}"
                + (f" minus {row['group_2']}" if row["group_2"] else "")
                + f"): {row['estimate']:.6f} "
                f"[{row['bootstrap_q025']:.6f}, {row['bootstrap_q975']:.6f}]."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Agreement across these fixed cases supports only bounded robustness to the checked common block effects and variance ratios. It does not show that all residual dependence, tail behavior, nonlinear time trends, or participant heterogeneity has been repaired. Any PPC flags above are retained as model mismatch evidence.",
            "",
            "Raw tables: `case_summary.csv`, `curve_summary.csv`, `residual_summary.csv`, `participant_contrasts.csv`, `participant_level_summary.csv`, `lag1_summary.csv`, `ppc_check_summary.csv`, and `ppc_draws.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def supplement_tex(result: dict[str, Any]) -> str:
    rows = []
    for group in result["groups"]:
        for case in group["cases"]:
            phase = case["phase_after_minus_before"]
            rows.append(
                f"{group['group'].replace('-first', '')} & {READABLE_CASE_LABELS[case['case']['name']]} & "
                f"{case['PIP12']:.3f} & {case['PIP32']:.3f} & {phase['mean']:.3f} "
                f"[{phase['q025']:.3f}, {phase['q975']:.3f}] & {case['heldout_RMSE']:.5f} \\\\"
            )
    body = "\n".join(rows)
    high = result["groups"][0]
    low = result["groups"][1]
    high_nominal = high["cases"][0]
    high_block = high["cases"][1]
    high_pip32_min = min(x["PIP32"] for x in high["cases"])
    low_pip32_min = min(x["PIP32"] for x in low["cases"])
    participant_rows = result["participant_level_analysis"]["rows"]
    participant_body = "\n".join(
        (
            f"{row['group_1']}"
            + (f" minus {row['group_2']}" if row["group_2"] else "")
            + f" & {int(row['n_1'])}"
            + (f" + {int(row['n_2'])}" if int(row["n_2"]) else "")
            + f" & {float(row['estimate']):.3f} "
            f"[{float(row['bootstrap_q025']):.3f}, {float(row['bootstrap_q975']):.3f}] \\\\"
        )
        for row in participant_rows
    )
    header = r"""\section{Prespecified real-application model checks}\label{sec:application-checks}
 The nominal joint-partition analysis with factorial count prior $p(K)\propto 1/K!$ remains primary. We used the unchanged sampler for five bounded, fixed-prior sensitivities: ten common retained-block effects with precision 4; participant-intercept variance ratios 0.25 and 4; and mean-zero participant congruency-slope variance ratios 0.01 and 0.1. These alternatives were specified before inspecting their results and were not selected as replacements.

The design-aligned boundary after bin 32 was retained in every sensitivity (minimum PIP @HIGH_PIP32@ in the high-first group and @LOW_PIP32@ in the low-first group), and every 95\% posterior interval for the draw-wise after-minus-before phase contrast retained the nominal direction. In contrast, the high-first bin-12 PIP changed from @HIGH_NOMINAL_PIP12@ under the nominal specification to @HIGH_BLOCK_PIP12@ with common retained-block effects. The early high-first boundary is therefore not robust to the checked common time structure and should remain exploratory; the phase boundary and reversal-aligned direction are robust only within this bounded set of specifications.

\begin{table}[htbp]
\centering
\caption{Sensitivity in the high-first ($n=24$) and low-first ($n=21$) groups. PIP($b$) is the posterior probability of a cut after bin $b$. Phase change is the posterior mean log-RT congruency contrast after minus before the experimental reversal, with its 95\% credible interval computed within posterior draws. RMSE is held-out log-RT RMSE conditional on existing participants.}
\label{tab:application-checks}
\scriptsize
\begin{tabular}{@{}llrrrr@{}}
\toprule
Order & Model & PIP(12) & PIP(32) & Phase change (95\% CrI) & RMSE \\
\midrule
"""
    header = (
        header.replace("@HIGH_PIP32@", f"{high_pip32_min:.3f}")
        .replace("@LOW_PIP32@", f"{low_pip32_min:.3f}")
        .replace("@HIGH_NOMINAL_PIP12@", f"{high_nominal['PIP12']:.3f}")
        .replace("@HIGH_BLOCK_PIP12@", f"{high_block['PIP12']:.3f}")
    )
    footer = r"""
\bottomrule
\end{tabular}
\end{table}

As a participant-level robustness analysis, we computed one adjusted congruency contrast for each participant in each retained phase from the training responses. The adjustment subtracts the nominal fitted nuisance contribution and adds back the nominal phase-average population contrast. We then formed the within-participant after-minus-before difference. Group means give equal weight to participants, rather than to response rows, and percentile intervals use 50,000 deterministic nonparametric bootstrap samples of complete participants within randomized-order groups. The direct order-by-phase contrast is the low-first mean change minus the high-first mean change.

\begin{table}[htbp]
\centering
\caption{Participant-level robustness analysis of the phase change. Estimates are after-minus-before changes in the adjusted log-RT congruency contrast. The final row is the direct order-by-phase contrast. Intervals are 95\% participant-bootstrap percentile intervals. They assess variation across participants and do not recalibrate posterior boundary probabilities or the nominal model's segment intervals.}
\label{tab:participant-phase-change}
\begin{tabular}{lcc}
\toprule
Contrast & Participants & Estimate (95\% interval) \\
\midrule
@PARTICIPANT_ROWS@
\bottomrule
\end{tabular}
\end{table}

Posterior predictive checks used the training observations and 1000 joint posterior draws per group. At draw $s$, the observed residual vector was $\bm e^{(s)}=\bm y-\bm\mu^{(s)}$, where $\bm\mu^{(s)}$ is the fitted observation-level mean from that draw's partition and jointly sampled coefficients. Replicated responses used the same design, participants, fitted mean, and residual scale $\sigma^{(s)}$; their residuals were independent $N(0,\{\sigma^{(s)}\}^2)$ draws. Thus observed discrepancies also varied across posterior draws.

Location and dispersion were the overall residual mean and sample SD divided by $\sigma^{(s)}$. Tail rates were the proportions below $-2\sigma^{(s)}$ and above $2\sigma^{(s)}$. Participant and phase-by-congruency checks used the maximum absolute residual cell mean, divided by $\sigma^{(s)}$. Contrast dispersion was the sample SD of residual incongruent-minus-congruent mean differences across participant-by-phase cells, also divided by $\sigma^{(s)}$; cells missing either condition were omitted. The phase indicator separated bins 1--32 and 33--64. Lag checks pooled residual pairs within participants and retained blocks and calculated Pearson correlations. One used consecutive analyzed training records, whose native distances may exceed one; the other restricted those pairs to native distance one. No pair crossed a block or the omitted practice interval.

For each discrepancy, let $p$ be the proportion of replicated values at least as large as their draw-matched observed values. Absolute values were compared for the signed overall mean and the two correlations. The descriptive flag was $\min\{1,2\min(p,1-p)\}\leq0.05$. This folded tail-area rule summarizes disagreement within the posterior predictive check; it is not a calibrated hypothesis test or a multiplicity-adjusted decision rule.

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\linewidth]{revision_assets/application_sensitivity.pdf}
\caption{Posterior mean congruency-effect curves for the nominal analysis and five prespecified sensitivities. The nominal curve is a thick solid line and the common-block-effects curve a thick dashed line. RI denotes participant random intercept; RI and slope ratios express prior variance relative to residual variance. Vertical reference lines mark the boundaries after bins 12 and 32, the latter separating the retained experimental phases.}
\label{fig:application-sensitivity}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=.95\linewidth]{revision_assets/application_ppc.pdf}
\caption{Nominal posterior predictive checks on training observations. Residual means, residual SD, maximum absolute participant and phase/condition means, and participant-by-phase residual-contrast dispersion are standardized by the posterior draw of $\sigma$; tail rates use thresholds $\pm2\sigma$, and lag statistics are correlations. Crosses denote median observed discrepancies over posterior draws. Blue points and intervals are the median and 95\% posterior-predictive range of replicated discrepancies, not credible intervals for model effects.}
\label{fig:application-ppc}
\end{figure}

The nominal posterior predictive checks retained discrepancies in both groups for lower and upper tail rates, participant-by-phase residual-contrast dispersion, analyzed-order lag-1 dependence, and strict native-distance-one lag-1 dependence. Thus the sensitivity results do not repair all observed tail, individual-heterogeneity, or serial-dependence mismatch. These checks provide bounded evidence rather than a universal correction for misspecification; agreement among fixed prior ratios cannot establish robustness to every possible residual or random-effects structure.
"""
    return header + body + footer.replace("@PARTICIPANT_ROWS@", participant_body)


def flatten_case_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for case in group["cases"]:
            rows.append(
                {
                    "group": group["group"],
                    "case": case["case"]["name"],
                    "PIP12": case["PIP12"],
                    "PIP32": case["PIP32"],
                    "joint_map_K": case["joint_map_K"],
                    "joint_map_cuts": ",".join(map(str, case["joint_map_cuts"])),
                    "phase_before_mean": case["phase_before"]["mean"],
                    "phase_before_q025": case["phase_before"]["q025"],
                    "phase_before_q975": case["phase_before"]["q975"],
                    "phase_after_mean": case["phase_after"]["mean"],
                    "phase_after_q025": case["phase_after"]["q025"],
                    "phase_after_q975": case["phase_after"]["q975"],
                    "phase_change_mean": case["phase_after_minus_before"]["mean"],
                    "phase_change_q025": case["phase_after_minus_before"]["q025"],
                    "phase_change_q975": case["phase_after_minus_before"]["q975"],
                    "phase_direction": case["phase_direction"],
                    "heldout_RMSE": case["heldout_RMSE"],
                    "max_abs_curve_difference_from_nominal": case[
                        "max_abs_curve_difference_from_nominal"
                    ],
                    "max_split_Rhat": case["max_split_Rhat"],
                    "min_bulk_ESS": case["min_bulk_ESS"],
                    **{f"p_K{k}": p for k, p in enumerate(case["p_k"])},
                }
            )
    return rows


def run(
    input_dir: Path,
    output: Path,
    assets: Path,
    ppc_replicates: int,
    iterations: int = 15000,
    warmup: int = 2000,
) -> dict[str, Any]:
    input_dir, output, assets = input_dir.resolve(), output.resolve(), assets.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    asset_targets = [
        assets / "application_sensitivity.pdf",
        assets / "application_sensitivity.png",
        assets / "application_ppc.pdf",
        assets / "application_ppc.png",
        assets / "application_model_checks.tex",
    ]
    existing_assets = [str(path) for path in asset_targets if path.exists()]
    if existing_assets:
        raise FileExistsError(
            f"refusing to overwrite application assets: {existing_assets}"
        )
    cfg = formal.load_config(FORMAL_CONFIG)
    sampler, nominal_cfg = cfg["sampler"], cfg["nominal_model"]
    frozen = (
        sampler["chains"],
        sampler["iterations"],
        sampler["warmup"],
        sampler["initial_k"],
        nominal_cfg["kmax"],
        nominal_cfg["min_segment"],
        nominal_cfg["prior_factorial_power"],
    )
    if frozen != (3, 15000, 2000, [0, 2, 4], 4, 8, 1.0):
        raise RuntimeError(f"formal sampler settings drifted: {frozen}")
    if (iterations, warmup) != (15000, 2000) and ppc_replicates >= 500:
        raise ValueError("nonformal chain lengths are allowed only for smoke runs")
    library, kernel_hash = formal.compile_kernel(ROOT / "build" / "r61_formal")
    groups: list[dict[str, Any]] = []
    all_ppc_draw_rows: list[dict[str, Any]] = []
    all_residual_rows: list[dict[str, Any]] = []
    all_contrast_rows: list[dict[str, Any]] = []
    all_lag_rows: list[dict[str, Any]] = []
    for first_pc in ("High", "Low"):
        train, test, metadata = prepare_group(input_dir, first_pc)
        participants = int(metadata["participants"])
        case_results: list[dict[str, Any]] = []
        nominal_chains: list[np.ndarray] | None = None
        nominal_stats: tuple[np.ndarray, np.ndarray, float] | None = None
        for name in CASE_ORDER:
            fitted, chains, summaries = fit_case(
                train,
                test,
                participants,
                first_pc,
                case_spec(name),
                library,
                cfg,
                iterations,
                warmup,
            )
            if name == "nominal":
                original = formal.collapsed(*_pack(train), participants)
                errors = [
                    float(np.max(np.abs(a - b)))
                    if isinstance(a, np.ndarray)
                    else abs(float(a - b))
                    for a, b in zip(summaries, original)
                ]
                if max(errors) > 2e-12:
                    raise RuntimeError(
                        f"nominal sufficient statistics failed regeneration: {first_pc} {errors}"
                    )
                original_rmse = formal.test_rmse(
                    _pack(train),
                    _pack(test),
                    np.asarray(fitted["curve_mean"]),
                    participants,
                )
                if abs(original_rmse - fitted["heldout_RMSE"]) > 2e-12:
                    raise RuntimeError(
                        f"nominal heldout RMSE failed regeneration: {first_pc}"
                    )
                fitted["nominal_sufficient_stat_max_errors"] = {
                    "H": errors[0],
                    "h": errors[1],
                    "r": errors[2],
                }
                nominal_chains, nominal_stats = chains, summaries
            case_results.append(fitted)
        nominal_curve = np.asarray(case_results[0]["curve_mean"])
        for fitted in case_results:
            fitted["max_abs_curve_difference_from_nominal"] = float(
                np.max(np.abs(np.asarray(fitted["curve_mean"]) - nominal_curve))
            )
        if nominal_chains is None or nominal_stats is None:
            raise AssertionError("nominal fit missing")
        ppc, ppc_draw_rows, posterior_mean_mu, posterior_mean_sigma = run_ppc(
            train, participants, first_pc, nominal_chains, ppc_replicates
        )
        residual = train["y"].to_numpy(dtype=float) - posterior_mean_mu
        group_name = first_pc + "-first"
        all_ppc_draw_rows.extend(ppc_draw_rows)
        all_residual_rows.extend(
            residual_summary_rows(train, residual, posterior_mean_sigma, group_name)
        )
        all_contrast_rows.extend(
            participant_contrast_rows(train, residual, nominal_curve, group_name)
        )
        all_lag_rows.extend(lag_summary_rows(train, residual, group_name))
        groups.append(
            {
                "group": group_name,
                "data": metadata,
                "cases": case_results,
                "ppc": ppc,
                "nominal_residual_sigma_reference": posterior_mean_sigma,
            }
        )
    result = {
        "schema_version": 1,
        "protocol": str(APPLICATION_PROTOCOL.relative_to(ROOT)),
        "primary_model": "nominal alpha=1 joint partition",
        "sensitivity_status": "bounded prespecified checks; not candidate selection",
        "participant_level_analysis": participant_level,
        "groups": groups,
        "source_hashes": {
            "runner": formal.sha256_file(Path(__file__)),
            "application_runner": formal.sha256_file(APPLICATION_RUNNER),
            "formal_runner": formal.sha256_file(FORMAL_RUNNER),
            "formal_config": formal.sha256_file(FORMAL_CONFIG),
            "kernel": kernel_hash,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    formal.atomic_json(output / "application_results.json", result)
    case_rows = flatten_case_rows(groups)
    formal.atomic_csv(output / "case_summary.csv", list(case_rows[0]), case_rows)
    curve_rows = []
    for group in groups:
        for case in group["cases"]:
            for index, (mean, q025, q975, pip) in enumerate(
                zip(
                    case["curve_mean"],
                    case["curve_q025"],
                    case["curve_q975"],
                    case["boundary_pip"],
                ),
                1,
            ):
                curve_rows.append(
                    {
                        "group": group["group"],
                        "case": case["case"]["name"],
                        "bin": index,
                        "mean": mean,
                        "q025": q025,
                        "q975": q975,
                        "boundary_PIP": pip,
                    }
                )
    formal.atomic_csv(output / "curve_summary.csv", list(curve_rows[0]), curve_rows)
    residual_fields = sorted(set().union(*(row.keys() for row in all_residual_rows)))
    formal.atomic_csv(
        output / "residual_summary.csv", residual_fields, all_residual_rows
    )
    formal.atomic_csv(
        output / "participant_contrasts.csv",
        list(all_contrast_rows[0]),
        all_contrast_rows,
    )
    formal.atomic_csv(
        output / "participant_level_summary.csv",
        list(participant_level["rows"][0]),
        participant_level["rows"],
    )
    formal.atomic_csv(output / "lag1_summary.csv", list(all_lag_rows[0]), all_lag_rows)
    ppc_check_rows = [row for group in groups for row in group["ppc"]["checks"]]
    formal.atomic_csv(
        output / "ppc_check_summary.csv", list(ppc_check_rows[0]), ppc_check_rows
    )
    formal.atomic_csv(
        output / "ppc_draws.csv", list(all_ppc_draw_rows[0]), all_ppc_draw_rows
    )
    _atomic_text(output / "application_report.md", report_text(result))
    _atomic_text(assets / "application_model_checks.tex", supplement_tex(result))
    plot_sensitivity(
        groups,
        [
            output / "application_sensitivity.png",
            assets / "application_sensitivity.png",
            assets / "application_sensitivity.pdf",
        ],
    )
    plot_ppc(
        groups,
        [
            output / "application_ppc.png",
            assets / "application_ppc.png",
            assets / "application_ppc.pdf",
        ],
    )
    formal.atomic_json(
        output / "run_manifest.json",
        {
            "command": sys.argv,
            "created_unix": time.time(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "workers": 1,
            "iterations": iterations,
            "warmup": warmup,
            "ppc_replicates": ppc_replicates,
            "source_hashes": result["source_hashes"],
        },
    )
    return result


def refresh_presentation(output: Path, assets: Path) -> dict[str, Any]:
    """Render completed results without sampling or rewriting analysis records."""
    source = output / "application_results.json"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = json.loads(source.read_text(encoding="utf-8"))
    if "participant_level_analysis" not in result:
        contrast_rows = pd.read_csv(output / "participant_contrasts.csv").to_dict(
            orient="records"
        )
        result["participant_level_analysis"] = participant_level_change_summary(
            contrast_rows
        )
    _atomic_text(assets / "application_model_checks.tex", supplement_tex(result))
    plot_sensitivity(
        result["groups"],
        [
            assets / "application_sensitivity.png",
            assets / "application_sensitivity.pdf",
        ],
    )
    plot_ppc(
        result["groups"],
        [assets / "application_ppc.png", assets / "application_ppc.pdf"],
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    names = (
        "application_model_checks.tex",
        "application_sensitivity.png",
        "application_sensitivity.pdf",
        "application_ppc.png",
        "application_ppc.pdf",
    )
    manifest = {
        "operation": "presentation only; no fitting or numerical changes",
        "result_source": str(source.relative_to(ROOT))
        if source.is_relative_to(ROOT)
        else str(source),
        "result_sha256": source_hash,
        "presentation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "asset_sha256": {
            name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
            for name in names
        },
    }
    formal.atomic_json(assets / "application_presentation_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--assets", default=str(DEFAULT_ASSETS))
    parser.add_argument("--ppc-replicates", type=int, default=1000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="short chains/PPC for implementation validation only",
    )
    parser.add_argument(
        "--presentation-only",
        action="store_true",
        help="render completed JSON without fitting or changing analysis records",
    )
    args = parser.parse_args(argv)
    if args.ppc_replicates < 500 and not args.smoke:
        parser.error("formal PPC requires at least 500 replicates")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.presentation_only:
        refresh_presentation(Path(args.output), Path(args.assets))
        raise SystemExit(0)
    raise SystemExit(
        0
        if run(
            Path(args.input),
            Path(args.output),
            Path(args.assets),
            50 if args.smoke else args.ppc_replicates,
            400 if args.smoke else 15000,
            100 if args.smoke else 2000,
        )
        else 1
    )

#!/usr/bin/env python
"""Resumable R61 simulation runner for B0/B1/B2/PELT-BIC/A3.

The completion marker for each replicate is ``summary.json``. It is written
last and atomically; retained MCMC draws are never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "config" / "r61_formal.yaml"
KERNEL_SOURCE = ROOT / "scripts" / "kernel.c"
MAX_K = 5
T = 64
DRAW_WIDTH = 1 + MAX_K + T
METHODS = ("B0", "B1", "B2", "PELT-BIC", "A3")
_WORKER_STATE: dict[str, Any] = {}


def native(value: Any) -> Any:
    """Convert NumPy containers/scalars to strict, finite JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON value")
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Atomically publish JSON; callers use its final write as a completion signal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(native(list(rows)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(native(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_fingerprint(resolved: dict[str, Any], source_hashes: dict[str, str]) -> str:
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    payload += json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def process_peak_working_set() -> int | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def deterministic_seed(namespace: str, label: str) -> int:
    """Derive a platform-independent 64-bit seed from the declared run namespace."""
    return int.from_bytes(
        hashlib.sha256(f"{namespace}|{label}".encode()).digest()[:8], "big"
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load the frozen protocol and enforce kernel- and design-level invariants."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["design"]["trials"] != T:
        raise ValueError("the formal kernel requires exactly 64 trial positions")
    d5 = cfg["dgps"]["D5"]
    expected_d5 = {
        "cuts": [13, 25, 39, 51],
        "levels": [0.13125, 0.03125, 0.13125, 0.03125, 0.13125],
    }
    namespace = cfg.get("namespace")
    if namespace != "R61-FORMAL-v1" or d5 != expected_d5:
        raise ValueError(f"D5 does not match the registered design for {namespace!r}")
    sampler = cfg["sampler"]
    if sampler["chains"] != 3 or len(sampler["initial_k"]) != 3:
        raise ValueError("A3 requires exactly three chains and three initial states")
    if sampler["warmup"] >= sampler["iterations"]:
        raise ValueError("sampler warmup must be smaller than iterations")
    return cfg


def selected_geometry(raw: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select one complete play per participant without consulting outcomes."""
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(raw.glob("user*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    uid, play, trial = (
                        int(row["anon_id"]),
                        int(row["nth_play"]),
                        int(row["trial"]),
                    )
                    if not row["flanker_direction"] or not row["target_direction"]:
                        continue
                    response_time = float(row["response_time"])
                    if not math.isfinite(response_time) or response_time <= 0:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                row = dict(row)
                row["trial"] = str(trial)
                groups[(uid, play)].append(row)
    candidates: dict[int, list[tuple[str, int, list[dict[str, str]]]]] = defaultdict(
        list
    )
    for (uid, play), rows in groups.items():
        rows.sort(key=lambda r: int(r["trial"]))
        if len(rows) == T and [int(r["trial"]) for r in rows] == list(range(1, T + 1)):
            key = hashlib.sha256(
                f"PY-GRC-CP-PILOT-PLAY|{uid}|{play}".encode()
            ).hexdigest()
            candidates[uid].append((key, play, rows))
    geometry, manifest = [], []
    for uid in sorted(candidates):
        key, play, rows = min(candidates[uid], key=lambda x: x[0])
        geometry.append(
            [
                0.5 if r["flanker_direction"] != r["target_direction"] else -0.5
                for r in rows
            ]
        )
        manifest.append({"anon_id": uid, "nth_play": play, "selection_hash": key})
    if len(geometry) != 49:
        raise RuntimeError(
            f"expected 49 complete participant geometries, found {len(geometry)}"
        )
    return np.asarray(geometry, dtype=float), manifest


def load_frozen_geometry(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read the outcome-blind 49-play congruency geometry and its audit manifest."""
    # Historical result manifests retain the pre-cleanup repository path.
    if path == ROOT / "config/r61_congruency_geometry.csv" and not path.exists():
        path = DEFAULT_CONFIG.with_name("r61_congruency_geometry.csv")
    geometry, manifest = [], []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bits = row["bits"]
            if len(bits) != T or set(bits) - {"0", "1"}:
                raise ValueError("malformed frozen congruency geometry")
            geometry.append([0.5 if bit == "1" else -0.5 for bit in bits])
            manifest.append(
                {
                    "anon_id": int(row["anon_id"]),
                    "nth_play": int(row["nth_play"]),
                    "selection_hash": row["selection_hash"],
                }
            )
    if len(geometry) != 49:
        raise ValueError(f"frozen geometry must contain 49 rows, found {len(geometry)}")
    return np.asarray(geometry, dtype=float), manifest


def design_for_n(base: np.ndarray, participants: int) -> tuple[np.ndarray, ...]:
    """Return observation-level (person, trial, standardized trial, congruency) arrays."""
    person = np.repeat(np.arange(participants), T)
    trial = np.tile(np.arange(1, T + 1), participants)
    z = (trial - 32.5) / np.std(np.arange(1, T + 1))
    congruency = np.concatenate([base[i % len(base)] for i in range(participants)])
    return person, trial, z, congruency


def curve_from(cuts: Iterable[int], levels: Iterable[float]) -> np.ndarray:
    """Expand ordered cut positions and segment levels into a length-64 trial curve."""
    cuts, levels = tuple(cuts), np.asarray(tuple(levels), dtype=float)
    return np.asarray(
        [levels[np.searchsorted(cuts, t, side="left")] for t in range(1, T + 1)]
    )


def outcome_blind_holdout(
    person: np.ndarray, trial: np.ndarray, count: int, namespace: str
) -> np.ndarray:
    """Hash-select exactly ``count`` trial positions per person, without using outcomes."""
    test = np.zeros(len(person), dtype=bool)
    for participant in np.unique(person):
        indices = np.flatnonzero(person == participant)
        order = sorted(
            indices,
            key=lambda j: hashlib.sha256(
                f"{namespace}|HOLDOUT|{int(participant)}|{int(trial[j])}".encode()
            ).digest(),
        )
        test[order[:count]] = True
    return test


def mcar_training_mask(
    train: np.ndarray, namespace: str, label: str, rate: float
) -> np.ndarray:
    """Optionally remove a deterministic MCAR subset from the non-holdout rows."""
    if rate == 0:
        return train.copy()
    rng = np.random.default_rng(deterministic_seed(namespace, f"MCAR|{label}"))
    keep = train.copy()
    candidates = np.flatnonzero(train)
    drop_n = int(round(rate * len(candidates)))
    if drop_n:
        keep[rng.choice(candidates, size=drop_n, replace=False)] = False
    return keep


def pack(
    design: tuple[np.ndarray, ...], y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, ...]:
    return tuple(x[mask] for x in design) + (y[mask],)


def collapsed(
    person: np.ndarray,
    trial: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    participants: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Eliminate intercept, trial trend, and participant effects via a Schur complement.

    Returns H=X'QX, h=X'Qy, and yqy=y'Qy for the 64 congruency-by-trial
    coefficients; these sufficient statistics are the C kernel input schema.
    """
    n = len(y)
    X = np.zeros((n, T + 1))
    X[np.arange(n), trial - 1] = c
    X[:, -1] = y
    counts = np.bincount(person, minlength=participants)
    zsum = np.bincount(person, weights=z, minlength=participants)
    ZX = np.zeros((participants + 2, T + 1))
    ZX[0] = X.sum(0)
    ZX[1] = z @ X
    np.add.at(ZX[2:], person, X)
    diagonal = 1 + counts
    cross = np.vstack((counts, zsum))
    top = np.array([[0.25 + n, z.sum()], [z.sum(), 1 + z @ z]])
    schur = top - (cross / diagonal) @ cross.T
    beta = np.linalg.solve(schur, ZX[:2] - (cross / diagonal) @ ZX[2:])
    participant_coef = (ZX[2:] - cross.T @ beta) / diagonal[:, None]
    QX = X - beta[0] - z[:, None] * beta[1] - participant_coef[person]
    return X[:, :T].T @ QX[:, :T], X[:, :T].T @ QX[:, -1], float(y @ QX[:, -1])


def fixed_curve(
    H: np.ndarray, h: np.ndarray, yqy: float, n: int, cuts: tuple[int, ...]
) -> np.ndarray:
    """Fit segment levels at fixed cuts using the same ridge precision (4) as A3."""
    edges = (0,) + cuts + (T,)
    p = len(edges) - 1
    A = np.empty((p, p))
    g = np.empty(p)
    for i in range(p):
        g[i] = h[edges[i] : edges[i + 1]].sum()
        for j in range(p):
            A[i, j] = H[edges[i] : edges[i + 1], edges[j] : edges[j + 1]].sum()
    A.flat[:: p + 1] += 4
    levels = np.linalg.solve(A, g)
    return curve_from(cuts, levels)


def recover_nuisance(
    train: tuple[np.ndarray, ...], curve: np.ndarray, participants: int
) -> np.ndarray:
    """Refit nuisance intercept/trend/participant terms conditional on a trial curve."""
    person, trial, z, c, y = train
    residual = y - c * curve[trial - 1]
    counts = np.bincount(person, minlength=participants)
    zsum = np.bincount(person, weights=z, minlength=participants)
    diagonal = 1 + counts
    cross = np.vstack((counts, zsum))
    top = np.array([[0.25 + len(y), z.sum()], [z.sum(), 1 + z @ z]])
    participant_rhs = np.bincount(person, weights=residual, minlength=participants)
    top_rhs = np.array([residual.sum(), z @ residual])
    beta = np.linalg.solve(
        top - (cross / diagonal) @ cross.T,
        top_rhs - (cross / diagonal) @ participant_rhs,
    )
    participant_effect = (participant_rhs - cross.T @ beta) / diagonal
    return np.r_[beta, participant_effect]


def test_rmse(
    train: tuple[np.ndarray, ...],
    test: tuple[np.ndarray, ...],
    curve: np.ndarray,
    participants: int,
) -> float:
    """Outcome-scale RMSE on the preselected holdout after training-only nuisance recovery."""
    nuisance = recover_nuisance(train, curve, participants)
    person, trial, z, c, y = test
    pred = nuisance[0] + nuisance[1] * z + nuisance[person + 2] + c * curve[trial - 1]
    return float(np.sqrt(np.mean(np.square(y - pred))))


class WeightedL2(BaseCost):
    """Weighted constant-segment SSE cost used by the PELT-BIC comparator."""

    model = "weighted_l2"

    def __init__(self, min_size: int = 8):
        self.min_size = min_size

    def fit(self, signal: np.ndarray) -> "WeightedL2":
        values = np.asarray(signal, float)
        self.y, self.w = values[:, 0], values[:, 1]
        self.sw = np.r_[0.0, np.cumsum(self.w)]
        self.swy = np.r_[0.0, np.cumsum(self.w * self.y)]
        self.swy2 = np.r_[0.0, np.cumsum(self.w * self.y * self.y)]
        self.signal = values
        return self

    def error(self, start: int, end: int) -> float:
        if end - start < self.min_size:
            raise rpt.exceptions.NotEnoughPoints
        weight = self.sw[end] - self.sw[start]
        return float(
            self.swy2[end]
            - self.swy2[start]
            - (self.swy[end] - self.swy[start]) ** 2 / weight
        )


def pelt_bic(
    statistic: np.ndarray, weights: np.ndarray, minseg: int
) -> tuple[int, ...]:
    """Unrestricted PELT with log(T) per segment and length-safe pruning.

    A domination witness at endpoint t permits removal only at t+minseg,
    when the replacement segment is feasible. Near-equalities are retained;
    selection uses the actual objective and the first (smallest) predecessor.
    """
    y, w = np.asarray(statistic, float), np.asarray(weights, float)
    if y.ndim != 1 or w.shape != y.shape or not len(y):
        raise ValueError("statistic and weights must be nonempty equal-length vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("finite statistics and strictly positive finite weights required")
    if (isinstance(minseg, (bool, np.bool_))
            or not isinstance(minseg, (int, np.integer))
            or not 1 <= minseg <= len(y)):
        raise ValueError("minseg must be an integer between 1 and T")
    n, penalty = len(y), math.log(len(y))
    cost = WeightedL2(minseg).fit(np.c_[y, w])
    optimum, paths = {0: 0.0}, {0: ()}
    active, expiry = [], {}
    for end in range(minseg, n + 1):
        active = [s for s in active if expiry.get(s, n + 1) > end]
        new = end - minseg
        if new == 0 or new >= minseg:
            active.append(new)
        candidates = [(optimum[s] + cost.error(s, end) + penalty, s) for s in active]
        value, start = min(candidates, key=lambda item: item[0])
        optimum[end] = value
        paths[end] = paths[start] + ((start,) if start else ())
        for candidate, predecessor in candidates:
            guard = 1e-12 * max(1.0, abs(candidate), abs(value))
            if candidate - penalty - value > guard:
                expiry[predecessor] = min(expiry.get(predecessor, n + 1), end + minseg)
    return paths[n]


def pelt_fit(
    train: tuple[np.ndarray, ...],
    participants: int,
    H: np.ndarray,
    h: np.ndarray,
    yqy: float,
    minseg: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Form inverse-variance weighted trial contrasts, then segment them with PELT."""
    person, trial, z, c, y = train
    global_curve = fixed_curve(H, h, yqy, len(y), ())
    global_delta = float(global_curve[0])
    nuisance = recover_nuisance(train, global_curve, participants)
    residual = y - (
        nuisance[0] + nuisance[1] * z + nuisance[person + 2] + c * global_delta
    )
    sigma2 = float(residual @ residual / max(1, len(y) - (participants + 3)))
    diff = np.empty(T)
    variance = np.empty(T)
    for t in range(1, T + 1):
        index = trial == t
        rt, ct = residual[index], c[index]
        plus, minus = rt[ct > 0], rt[ct < 0]
        if not len(plus) or not len(minus):
            raise RuntimeError(f"training trial {t} lacks a congruency arm")
        diff[t - 1] = global_delta + plus.mean() - minus.mean()
        variance[t - 1] = sigma2 * (1 / len(plus) + 1 / len(minus))
    cuts = pelt_bic(diff, 1 / variance, minseg)
    return cuts, fixed_curve(H, h, yqy, len(y), cuts)


def hausdorff(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if not a and not b:
        return 0
    if not a or not b:
        return T
    return int(
        max(
            max(min(abs(x - y) for y in b) for x in a),
            max(min(abs(y - x) for x in a) for y in b),
        )
    )


def ordered_segment_error(
    curve: np.ndarray,
    estimated_cuts: tuple[int, ...],
    true_levels: list[float],
    true_k: int,
) -> tuple[float | None, list[float] | None]:
    """Report level error only when estimated and true segments have one-to-one order."""
    if len(estimated_cuts) != true_k:
        return None, None
    edges = (0,) + estimated_cuts + (T,)
    estimates = [float(curve[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
    bias = [estimate - truth for estimate, truth in zip(estimates, true_levels)]
    return float(np.sqrt(np.mean(np.square(bias)))), bias


def compile_kernel(build_dir: Path) -> tuple[ctypes.CDLL, str]:
    """Compile/load a hash-named native kernel so binaries track exact C source."""
    build_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(KERNEL_SOURCE)
    suffix = ".dll" if os.name == "nt" else ".so"
    library_path = build_dir / f"r61_formal_{source_hash[:12]}{suffix}"
    if not library_path.exists():
        # Relative paths avoid MinGW path-encoding failures in non-ASCII workspaces.
        try:
            output_arg = str(library_path.relative_to(ROOT))
        except ValueError:
            output_arg = str(library_path)
        cmd = [
            "gcc",
            "-O3",
            "-shared",
            "-o",
            output_arg,
            str(KERNEL_SOURCE.relative_to(ROOT)),
            "-lm",
        ]
        if os.name != "nt":
            cmd.insert(2, "-fPIC")
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(library_path))
    D = ctypes.POINTER(ctypes.c_double)
    L = ctypes.POINTER(ctypes.c_longlong)
    library.r61_formal_run.argtypes = [
        D,
        D,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        D,
        L,
    ]
    library.r61_formal_run.restype = ctypes.c_int
    return library, source_hash


def ptr(array: np.ndarray, typ: Any) -> Any:
    return np.ascontiguousarray(array).ctypes.data_as(ctypes.POINTER(typ))


def run_chain(
    library: ctypes.CDLL,
    H: np.ndarray,
    h: np.ndarray,
    yqy: float,
    n: int,
    iterations: int,
    warmup: int,
    seed: int,
    init_k: int,
    kmax: int,
    minseg: int,
    prior_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one seeded A3 chain; rows are [K, five cut slots, 64 expanded effects]."""
    output = np.empty((iterations - warmup, DRAW_WIDTH), dtype=float)
    stats = np.zeros(8, dtype=np.int64)
    rc = library.r61_formal_run(
        ptr(H, ctypes.c_double),
        ptr(h, ctypes.c_double),
        yqy,
        n,
        iterations,
        warmup,
        seed,
        init_k,
        kmax,
        minseg,
        prior_power,
        ptr(output, ctypes.c_double),
        ptr(stats, ctypes.c_longlong),
    )
    if rc:
        raise RuntimeError(f"r61 formal kernel returned {rc}")
    return output, stats


def rank_diagnostic(values: np.ndarray) -> tuple[float | None, float | None, bool, str]:
    if np.all(values == values.flat[0]):
        return None, None, True, "not_applicable_constant_all_chains"
    if all(np.all(chain == chain[0]) for chain in values):
        return None, None, False, "nonfinite_rhat_between_chain_disagreement"
    import arviz as az

    rhat, ess = (
        float(az.rhat(values, method="rank")),
        float(az.ess(values, method="bulk")),
    )
    if not math.isfinite(rhat):
        return (
            None,
            (ess if math.isfinite(ess) else None),
            False,
            "nonfinite_rhat_between_chain_disagreement",
        )
    if not math.isfinite(ess):
        return rhat, None, False, "nonfinite_bulk_ess"
    return rhat, ess, False, "estimated"


def summarize_a3(
    chains: list[np.ndarray], stats: list[np.ndarray], wall: float, kmax: int
) -> dict[str, Any]:
    """Pool retained chains into BMA/MAP estimates, boundary PIPs, and diagnostics."""
    stack = np.stack(chains)
    curve_offset = 1 + MAX_K
    flat = stack.reshape(-1, DRAW_WIDTH)
    keys = [
        (int(row[0]),) + tuple(int(v) for v in row[1 : 1 + int(row[0])]) for row in flat
    ]
    counts = Counter(keys)
    joint = min(counts, key=lambda key: (-counts[key], key))
    mask = np.asarray([key == joint for key in keys])
    curve_draws = flat[:, curve_offset:]
    bma = curve_draws.mean(0)
    bma_q025, bma_q975 = np.quantile(curve_draws, [0.025, 0.975], axis=0)
    map_curve = flat[mask, curve_offset:].mean(0)
    boundary_pip = [
        float(np.mean([t in key[1:] for key in keys])) for t in range(1, T + 1)
    ]
    monitors = {
        "K": stack[:, :, 0],
        "delta16": stack[:, :, curve_offset + 15],
        "delta48": stack[:, :, curve_offset + 47],
    }
    diagnostics = []
    for name, values in monitors.items():
        rhat, ess, constant, status = rank_diagnostic(values)
        diagnostics.append(
            {
                "parameter": name,
                "split_Rhat": rhat,
                "bulk_ESS": ess,
                "ESS_per_second": None if ess is None else ess / wall,
                "constant_chain": constant,
                "diagnostic_status": status,
                "diagnostic_convention": "not applicable for a constant estimand"
                if constant
                else "ArviZ rank Rhat and bulk ESS",
            }
        )
    acceptance = {}
    for label, attempted, accepted in (
        ("birth", 0, 1),
        ("death", 2, 3),
        ("move", 4, 5),
    ):
        denominator = sum(int(x[attempted]) for x in stats)
        acceptance[label] = sum(int(x[accepted]) for x in stats) / max(1, denominator)
    applicable = [
        item for item in diagnostics if item["diagnostic_status"] == "estimated"
    ]
    return {
        "bma_curve": bma,
        "bma_q025": bma_q025,
        "bma_q975": bma_q975,
        "boundary_pip": boundary_pip,
        "joint_map_curve": map_curve,
        "joint_map": joint,
        "joint_map_draws": int(counts[joint]),
        "p_k": [float(np.mean(flat[:, 0] == k)) for k in range(kmax + 1)],
        "diagnostics": diagnostics,
        "max_split_Rhat": max((x["split_Rhat"] for x in applicable), default=None),
        "min_bulk_ESS": min((x["bulk_ESS"] for x in applicable), default=None),
        "min_ESS_per_second": min(
            (x["ESS_per_second"] for x in applicable), default=None
        ),
        "diagnostic_failure_flags": [
            x["parameter"] + ":" + x["diagnostic_status"]
            for x in diagnostics
            if x["diagnostic_status"].startswith("nonfinite_")
        ],
        "acceptance": acceptance,
        "three_chain_walltime_seconds": wall,
    }


def expand_cases(cfg: dict[str, Any], stages: set[str]) -> list[dict[str, Any]]:
    nominal = cfg["nominal_model"]
    participants = cfg["design"]["participants"]
    out = []

    def add(stage: str, cell: str, dgp: str, reps: int, **changes: Any) -> None:
        case = {
            "stage": stage,
            "cell": cell,
            "dgp": dgp,
            "reps": reps,
            "participants": participants,
            "kmax": nominal["kmax"],
            "min_segment": nominal["min_segment"],
            "prior_factorial_power": nominal["prior_factorial_power"],
            "signal_multiplier": 1.0,
            "training_mcar": 0.0,
        }
        case.update(changes)
        out.append(case)

    if "main" in stages:
        for dgp in cfg["dgps"]:
            add("main", dgp, dgp, cfg["replications"]["main_reps"])
    if "sensitivity" in stages:
        reps = cfg["replications"]["sensitivity_reps"]
        for family, entries in cfg["sensitivity"].items():
            for item in entries:
                value = item.get("value", item.get("participants"))
                label = item.get("label", str(value).replace(".", "p"))
                changes = (
                    {family: value}
                    if family != "sample_size"
                    else {"participants": item["participants"]}
                )
                add(
                    "sensitivity",
                    f"{family}_{item['dgp']}_{label}",
                    item["dgp"],
                    reps,
                    **changes,
                )
    if "kmax" in stages:
        reps = cfg["replications"]["kmax_reps"]
        for item in cfg["kmax_sensitivity"]:
            add(
                "kmax",
                f"Kmax{item['value']}_{item['dgp']}",
                item["dgp"],
                reps,
                kmax=item["value"],
            )
    return out


def generate_outcome(
    cfg: dict[str, Any], case: dict[str, Any], design: tuple[np.ndarray, ...], seed: int
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Generate one replicate from its declared DGP with an isolated data RNG stream."""
    person, trial, z, c = design
    spec = cfg["dgps"][case["dgp"]]
    levels = np.asarray(spec["levels"], float)
    if case["signal_multiplier"] != 1:
        center = float(levels.mean())
        levels = center + case["signal_multiplier"] * (levels - center)
    truth = curve_from(spec["cuts"], levels)
    rng = np.random.default_rng(seed)
    participant_effect = rng.normal(
        0, cfg["design"]["participant_sd"], case["participants"]
    )
    y = (
        cfg["design"]["beta0"]
        + cfg["design"]["beta1"] * z
        + participant_effect[person]
        + c * truth[trial - 1]
    )
    y += rng.normal(0, cfg["design"]["residual_sd"], len(y))
    return y, truth, levels.tolist()


def fit_replicate(
    cfg: dict[str, Any],
    case: dict[str, Any],
    rep: int,
    base: np.ndarray,
    library: ctypes.CDLL,
    fingerprint: str,
    source_hashes: dict[str, str],
    output: Path,
) -> dict[str, Any]:
    """Fit all prespecified methods and atomically emit reproduction audit artifacts.

    Data, holdout, MCAR, and each A3 chain use separately derived seeds.  Metrics
    compare the trial curve, held-out outcomes, K, cut Hausdorff distance, and
    conditionally aligned segment levels against the known simulated truth.
    """
    label = f"{case['stage']}|{case['cell']}|R{rep:06d}"
    namespace = cfg["namespace"]
    data_seed = deterministic_seed(namespace, f"DATA|{label}")
    design = design_for_n(base, case["participants"])
    y, truth_curve, truth_levels = generate_outcome(cfg, case, design, data_seed)
    test_mask = outcome_blind_holdout(
        design[0],
        design[1],
        cfg["design"]["holdout_trials_per_participant"],
        f"{namespace}|{label}",
    )
    train_mask = mcar_training_mask(~test_mask, namespace, label, case["training_mcar"])
    train, test = pack(design, y, train_mask), pack(design, y, test_mask)
    H, h, yqy = collapsed(*train, case["participants"])
    true_cuts = tuple(cfg["dgps"][case["dgp"]]["cuts"])
    true_k = len(true_cuts)
    rows, curves = [], {}
    fixed = {"B0": ((), np.zeros(T)), "B1": ((), None), "B2": ((32,), None)}
    for method, (cuts, curve) in fixed.items():
        start = time.perf_counter()
        curve = (
            curve if curve is not None else fixed_curve(H, h, yqy, len(train[-1]), cuts)
        )
        runtime = time.perf_counter() - start
        curves[method] = curve
        seg_rmse, seg_bias = ordered_segment_error(curve, cuts, truth_levels, true_k)
        rows.append(
            {
                "method": method,
                "trajectory_RMSE": float(
                    np.sqrt(np.mean(np.square(curve - truth_curve)))
                ),
                "test_RMSE": test_rmse(train, test, curve, case["participants"]),
                "estimated_K": len(cuts),
                "estimated_cuts": list(cuts),
                "correct_K": int(len(cuts) == true_k),
                "Hausdorff": hausdorff(cuts, true_cuts),
                "segment_level_RMSE": seg_rmse,
                "segment_level_bias": seg_bias,
                "runtime_seconds": runtime,
            }
        )
    # PELT keeps its own nominal minseg and penalized search; A3-only
    # minseg/Kmax sensitivities do not constrain this comparator.
    start = time.perf_counter()
    pelt_cuts, pelt_curve = pelt_fit(
        train, case["participants"], H, h, yqy, cfg["nominal_model"]["min_segment"]
    )
    pelt_wall = time.perf_counter() - start
    curves["PELT-BIC"] = pelt_curve
    seg_rmse, seg_bias = ordered_segment_error(
        pelt_curve, pelt_cuts, truth_levels, true_k
    )
    rows.append(
        {
            "method": "PELT-BIC",
            "trajectory_RMSE": float(
                np.sqrt(np.mean(np.square(pelt_curve - truth_curve)))
            ),
            "test_RMSE": test_rmse(train, test, pelt_curve, case["participants"]),
            "estimated_K": len(pelt_cuts),
            "estimated_cuts": list(pelt_cuts),
            "correct_K": int(len(pelt_cuts) == true_k),
            "Hausdorff": hausdorff(pelt_cuts, true_cuts),
            "segment_level_RMSE": seg_rmse,
            "segment_level_bias": seg_bias,
            "runtime_seconds": pelt_wall,
        }
    )
    chains, chain_stats, chain_seeds = [], [], []
    sampler = cfg["sampler"]
    start = time.perf_counter()
    for index, configured_init in enumerate(sampler["initial_k"], 1):
        init_k = min(int(configured_init), int(case["kmax"]))
        chain_seed = deterministic_seed(namespace, f"CHAIN{index}|{label}")
        chain, stats = run_chain(
            library,
            H,
            h,
            yqy,
            len(train[-1]),
            sampler["iterations"],
            sampler["warmup"],
            chain_seed,
            init_k,
            case["kmax"],
            case["min_segment"],
            case["prior_factorial_power"],
        )
        chains.append(chain)
        chain_stats.append(stats)
        chain_seeds.append(chain_seed)
    a3 = summarize_a3(chains, chain_stats, time.perf_counter() - start, case["kmax"])
    a3_cuts = tuple(a3["joint_map"][1:])
    diagnostic_cfg = cfg["diagnostics"]
    k_ess = next(x["bulk_ESS"] for x in a3["diagnostics"] if x["parameter"] == "K")
    a3["diagnostics_valid"] = bool(
        not a3["diagnostic_failure_flags"]
        and a3["max_split_Rhat"] is not None
        and a3["min_bulk_ESS"] is not None
        and a3["max_split_Rhat"] <= diagnostic_cfg["max_split_Rhat"]
        and a3["min_bulk_ESS"] >= diagnostic_cfg["min_bulk_ESS"]
        and (
            case["dgp"] != "D4"
            or case["participants"] < 500
            or k_ess is None
            or k_ess >= diagnostic_cfg["D4_N500_K_min_bulk_ESS"]
        )
    )
    seg_rmse, seg_bias = ordered_segment_error(
        a3["joint_map_curve"], a3_cuts, truth_levels, true_k
    )
    rows.append(
        {
            "method": "A3",
            "trajectory_RMSE": float(
                np.sqrt(np.mean(np.square(a3["bma_curve"] - truth_curve)))
            ),
            "test_RMSE": test_rmse(train, test, a3["bma_curve"], case["participants"]),
            "estimated_K": len(a3_cuts),
            "estimated_cuts": list(a3_cuts),
            "correct_K": int(len(a3_cuts) == true_k),
            "Hausdorff": hausdorff(a3_cuts, true_cuts),
            "segment_level_RMSE": seg_rmse,
            "segment_level_bias": seg_bias,
            "runtime_seconds": a3["three_chain_walltime_seconds"],
        }
    )
    rep_dir = output / "replicates" / case["stage"] / case["cell"] / f"rep_{rep:06d}"
    atomic_csv(
        rep_dir / "a3_bma_curve.csv",
        ["trial", "estimate", "q025", "q975", "boundary_PIP"],
        [
            {
                "trial": t + 1,
                "estimate": a3["bma_curve"][t],
                "q025": a3["bma_q025"][t],
                "q975": a3["bma_q975"][t],
                "boundary_PIP": a3["boundary_pip"][t],
            }
            for t in range(T)
        ],
    )
    atomic_csv(
        rep_dir / "a3_p_k.csv",
        ["K", "probability"],
        [{"K": k, "probability": p} for k, p in enumerate(a3["p_k"])],
    )
    atomic_json(
        rep_dir / "a3_joint_map.json",
        {
            "partition": a3["joint_map"],
            "conditional_curve": a3["joint_map_curve"],
            "draw_count": a3["joint_map_draws"],
        },
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "case": case,
        "replicate": rep,
        "data_seed": data_seed,
        "chain_seeds": chain_seeds,
        "data_hash": sha256_array(y),
        "holdout_hash": sha256_array(np.packbits(test_mask)),
        "training_mask_hash": sha256_array(np.packbits(train_mask)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "truth": {"cuts": true_cuts, "levels": truth_levels},
        "methods": rows,
        "A3": {
            key: value
            for key, value in a3.items()
            if key not in {"bma_curve", "joint_map_curve"}
        },
    }
    atomic_json(rep_dir / "summary.json", summary)
    return summary


def worker_init(
    cfg: dict[str, Any],
    base: np.ndarray,
    fingerprint: str,
    source_hashes: dict[str, str],
    output: str,
) -> None:
    library, kernel_hash = compile_kernel(ROOT / "build" / "r61_formal")
    if kernel_hash != source_hashes["kernel"]:
        raise RuntimeError("worker kernel hash mismatch")
    _WORKER_STATE.update(
        cfg=cfg,
        base=base,
        fingerprint=fingerprint,
        source_hashes=source_hashes,
        output=Path(output),
        library=library,
    )


def worker_job(
    case: dict[str, Any], rep: int
) -> tuple[dict[str, Any], float, int, int | None]:
    state = _WORKER_STATE
    started = time.perf_counter()
    summary = fit_replicate(
        state["cfg"],
        case,
        rep,
        state["base"],
        state["library"],
        state["fingerprint"],
        state["source_hashes"],
        state["output"],
    )
    return (
        summary,
        time.perf_counter() - started,
        os.getpid(),
        process_peak_working_set(),
    )


def flatten_method_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = {
        "stage": summary["case"]["stage"],
        "cell": summary["case"]["cell"],
        "dgp": summary["case"]["dgp"],
        "participants": summary["case"]["participants"],
        "replicate": summary["replicate"],
    }
    rows = []
    for method in summary["methods"]:
        row = dict(prefix, **method)
        row["estimated_cuts"] = ",".join(map(str, row["estimated_cuts"]))
        row["segment_level_bias"] = (
            ""
            if row["segment_level_bias"] is None
            else ",".join(f"{x:.12g}" for x in row["segment_level_bias"])
        )
        row.pop("bic_scores_K0_to_Kmax", None)
        rows.append(row)
    return rows


def aggregate(output: Path, fingerprint: str) -> None:
    """Aggregate only summaries matching this exact resolved-config/source fingerprint."""
    summaries = []
    for path in output.glob("replicates/*/*/rep_*/summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("run_fingerprint") == fingerprint:
                summaries.append(value)
        except (OSError, ValueError):
            continue
    rows = [row for summary in summaries for row in flatten_method_rows(summary)]
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    atomic_csv(output / "per_method.csv", fields, rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["stage"], row["cell"], row["dgp"], row["participants"], row["method"])
        ].append(row)
    agg = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(("stage", "cell", "dgp", "participants", "method"), key))
        record["valid_replicates"] = len(values)
        correct = np.asarray(
            [v["correct_K"] for v in values if v.get("correct_K") is not None], float
        )
        record["correct_K_rate"] = float(np.mean(correct)) if len(correct) else ""
        record["correct_K_n"] = len(correct)
        for metric in (
            "trajectory_RMSE",
            "test_RMSE",
            "Hausdorff",
            "segment_level_RMSE",
            "runtime_seconds",
        ):
            x = np.asarray(
                [v[metric] for v in values if v.get(metric) is not None], float
            )
            record[f"{metric}_median"] = float(np.median(x)) if len(x) else ""
            record[f"{metric}_q25"] = float(np.quantile(x, 0.25)) if len(x) else ""
            record[f"{metric}_q75"] = float(np.quantile(x, 0.75)) if len(x) else ""
            record[f"{metric}_n"] = len(x)
        agg.append(record)
    atomic_csv(output / "summary.csv", list(agg[0]), agg)
    diag = []
    for summary in summaries:
        prefix = {
            "stage": summary["case"]["stage"],
            "cell": summary["case"]["cell"],
            "dgp": summary["case"]["dgp"],
            "replicate": summary["replicate"],
        }
        a3 = summary["A3"]
        diag.append(
            dict(
                prefix,
                max_split_Rhat=a3["max_split_Rhat"],
                min_bulk_ESS=a3["min_bulk_ESS"],
                min_ESS_per_second=a3["min_ESS_per_second"],
                birth_acceptance=a3["acceptance"]["birth"],
                death_acceptance=a3["acceptance"]["death"],
                move_acceptance=a3["acceptance"]["move"],
                three_chain_walltime_seconds=a3["three_chain_walltime_seconds"],
                diagnostics_valid=a3["diagnostics_valid"],
            )
        )
    atomic_csv(output / "a3_diagnostics.csv", list(diag[0]), diag)


def validate_resume_target(output: Path, fingerprint: str) -> None:
    """Refuse to overwrite records from a different source/configuration.

    Documentation changes also change source hashes. Use a new output directory
    for a new release; never relabel the historical run as if it used new code.
    """
    manifest = output / "run_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError(
                "Output belongs to a different run fingerprint; choose a new --output directory."
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Nonempty output has no run manifest; choose a new --output directory."
        )


def run(args: argparse.Namespace) -> int:
    """Coordinate resumable replicates; fingerprinted summaries permit safe skipping."""
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.seed_namespace is not None:
        cfg = json.loads(json.dumps(cfg))
        cfg["namespace"] = args.seed_namespace
    if args.smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["replications"].update(main_reps=1, sensitivity_reps=1, kmax_reps=1)
        cfg["design"]["participants"] = args.smoke_participants
        cfg["sampler"].update(
            iterations=args.smoke_iterations, warmup=args.smoke_warmup
        )
    stages = {"main", "sensitivity", "kmax"} if args.stage == "all" else {args.stage}
    output = Path(args.output).resolve() if args.output else ROOT / cfg["output_dir"]
    geometry_path = ROOT / cfg["design"]["geometry_file"]
    base, geometry_manifest = load_frozen_geometry(geometry_path)
    geometry_hash = sha256_array(base)
    source_hashes = {
        "runner": sha256_file(Path(__file__)),
        "kernel": sha256_file(KERNEL_SOURCE),
        "config_file": sha256_file(config_path),
        "geometry_file": sha256_file(geometry_path),
    }
    resolved = {
        "config": cfg,
        "stages": sorted(stages),
        "dgp_filter": args.dgp,
        "replications_override": args.replications,
        "geometry_hash": geometry_hash,
    }
    fingerprint = make_fingerprint(resolved, source_hashes)
    validate_resume_target(output, fingerprint)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "resolved_config.json", resolved)
    atomic_json(
        output / "run_manifest.json",
        {
            "run_fingerprint": fingerprint,
            "source_hashes": source_hashes,
            "geometry_hash": geometry_hash,
            "python": sys.version,
            "numpy": np.__version__,
            "command": sys.argv,
            "created_unix": time.time(),
        },
    )
    atomic_json(
        output / "geometry_manifest.json",
        {"geometry_hash": geometry_hash, "selected_plays": geometry_manifest},
    )
    _, kernel_hash = compile_kernel(ROOT / "build" / "r61_formal")
    assert kernel_hash == source_hashes["kernel"]
    cases = expand_cases(cfg, stages)
    if args.dgp is not None:
        cases = [case for case in cases if case["dgp"] == args.dgp]
    if args.replications is not None:
        for case in cases:
            case["reps"] = args.replications
    if not cases:
        raise ValueError("the requested stage and DGP filter produced no simulation cells")
    total = sum(case["reps"] for case in cases)
    completed = failed = skipped = 0
    overall = time.perf_counter()
    jobs: list[tuple[dict[str, Any], int]] = []
    peak_by_pid: dict[int, int] = {}
    for case in cases:
        for rep in range(1, case["reps"] + 1):
            summary_path = (
                output
                / "replicates"
                / case["stage"]
                / case["cell"]
                / f"rep_{rep:06d}"
                / "summary.json"
            )
            if summary_path.exists():
                try:
                    if (
                        json.loads(summary_path.read_text(encoding="utf-8")).get(
                            "run_fingerprint"
                        )
                        == fingerprint
                    ):
                        skipped += 1
                        continue
                except (OSError, ValueError):
                    pass
            jobs.append((case, rep))

    def record_success(summary: dict[str, Any], seconds: float) -> None:
        nonlocal completed
        completed += 1
        case, rep, a3 = summary["case"], summary["replicate"], summary["A3"]
        append_jsonl(
            output / "progress.jsonl",
            {
                "status": "complete",
                "stage": case["stage"],
                "cell": case["cell"],
                "replicate": rep,
                "seconds": seconds,
                "max_split_Rhat": a3["max_split_Rhat"],
                "min_bulk_ESS": a3["min_bulk_ESS"],
                "completed_this_run": completed,
                "failed_this_run": failed,
                "total_planned": total,
                "run_fingerprint": fingerprint,
                "unix_time": time.time(),
            },
        )
        print(
            f"[{completed + failed + skipped}/{total}] {case['stage']} {case['cell']} rep={rep} complete in {seconds:.2f}s",
            flush=True,
        )
        if (completed + failed) % args.aggregate_every == 0:
            aggregate(output, fingerprint)

    def record_failure(case: dict[str, Any], rep: int, exc: BaseException) -> None:
        nonlocal failed
        failed += 1
        append_jsonl(
            output / "failures.jsonl",
            {
                "status": "failed",
                "stage": case["stage"],
                "cell": case["cell"],
                "replicate": rep,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "run_fingerprint": fingerprint,
                "unix_time": time.time(),
            },
        )
        print(
            f"FAILED {case['stage']} {case['cell']} rep={rep}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    if args.workers == 1:
        library, _ = compile_kernel(ROOT / "build" / "r61_formal")
        for case, rep in jobs:
            started = time.perf_counter()
            try:
                summary = fit_replicate(
                    cfg, case, rep, base, library, fingerprint, source_hashes, output
                )
                peak = process_peak_working_set()
                if peak is not None:
                    peak_by_pid[os.getpid()] = max(
                        peak_by_pid.get(os.getpid(), 0), peak
                    )
                record_success(summary, time.perf_counter() - started)
            except Exception as exc:
                record_failure(case, rep, exc)
                if args.fail_fast:
                    aggregate(output, fingerprint)
                    raise
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(cfg, base, fingerprint, source_hashes, str(output)),
        ) as executor:
            future_to_job = {}
            for case, rep in jobs:
                future = executor.submit(worker_job, case, rep)
                future_to_job[future] = (case, rep)
            for future in as_completed(future_to_job):
                case, rep = future_to_job[future]
                try:
                    summary, worker_seconds, pid, peak = future.result()
                    if peak is not None:
                        peak_by_pid[pid] = max(peak_by_pid.get(pid, 0), peak)
                    record_success(summary, worker_seconds)
                except Exception as exc:
                    record_failure(case, rep, exc)
                    if args.fail_fast:
                        for pending in future_to_job:
                            pending.cancel()
                        aggregate(output, fingerprint)
                        raise
    aggregate(output, fingerprint)
    coordinator_peak = process_peak_working_set()
    if coordinator_peak is not None:
        peak_by_pid[os.getpid()] = max(
            peak_by_pid.get(os.getpid(), 0), coordinator_peak
        )
    atomic_json(
        output / "run_status.json",
        {
            "run_fingerprint": fingerprint,
            "planned": total,
            "completed_this_run": completed,
            "skipped_complete": skipped,
            "failed_this_run": failed,
            "walltime_seconds": time.perf_counter() - overall,
            "processes_observed": len(peak_by_pid),
            "max_process_peak_working_set_bytes": max(
                peak_by_pid.values(), default=None
            ),
            "sum_process_peak_working_set_upper_bound_bytes": sum(peak_by_pid.values())
            if peak_by_pid
            else None,
            "status": "complete"
            if failed == 0 and completed + skipped == total
            else "incomplete",
        },
    )
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage", choices=("all", "main", "sensitivity", "kmax"), default="all"
    )
    parser.add_argument(
        "--dgp",
        choices=tuple(f"D{i}" for i in range(6)),
        help="run only cells generated from the selected DGP",
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="override the registered replication count for each selected cell",
    )
    parser.add_argument(
        "--seed-namespace",
        help="override only the deterministic seed namespace for a separate pilot run",
    )
    parser.add_argument("--output")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--aggregate-every", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=1, help="independent replicate worker processes"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-participants", type=int, default=40)
    parser.add_argument("--smoke-iterations", type=int, default=400)
    parser.add_argument("--smoke-warmup", type=int, default=100)
    args = parser.parse_args()
    if args.aggregate_every < 1:
        parser.error("--aggregate-every must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.replications is not None and args.replications < 1:
        parser.error("--replications must be positive")
    if args.smoke_warmup >= args.smoke_iterations:
        parser.error("smoke warmup must be smaller than iterations")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

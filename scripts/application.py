#!/usr/bin/env python
"""Run the formal real-data analysis on the public proportion-congruency Flanker data.

Input is the extracted ``RawData_Experiment1`` directory from OSF project
https://osf.io/6n597/ (file ``RawData_Experiment1.7z``).  The two randomized
condition-order groups are analyzed separately under one frozen preprocessing
and fitting rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORMAL_RUNNER = ROOT / "scripts" / "simulation.py"
FORMAL_CONFIG = ROOT / "scripts" / "config" / "r61_formal.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "reproduction" / "application"
SPLIT_NAMESPACE = "R61-PC-FLANKER-HOLDOUT-v1"
CHAIN_NAMESPACE = "R61-PC-FLANKER-CHAIN-v1"
METHODS = ("NoEffect", "ConstEffect", "FixedSplit", "PELT-BIC", "CG-RJMCMC")


def _load_formal() -> Any:
    spec = importlib.util.spec_from_file_location(
        "r61_formal_pc_dependency", FORMAL_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FORMAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_formal()


def _source_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.csv")) if path.is_dir() else []
    if len(files) != 50:
        raise RuntimeError(
            f"expected 50 Experiment-1 CSV files in {path}; found {len(files)}"
        )
    return files


def prepare_group(
    path: Path, first_pc: str
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    """Apply the source-study exclusions, visual-task restriction, binning, and holdout."""
    kept: list[pd.DataFrame] = []
    excluded_ids: list[str] = []
    block_rank = {
        block: rank for rank, block in enumerate((2, 3, 4, 5, 6, 8, 9, 10, 11, 12))
    }
    for source in _source_files(path):
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
    person = data["uid"].map(id_to_person).to_numpy(dtype=int)
    analyzed_position = np.asarray(
        [
            block_rank[int(block)] * 84 + int(trial)
            for block, trial in zip(data["blockNum"], data["trialNum"])
        ]
    )
    trial = np.minimum(64, np.ceil(analyzed_position * 64 / 840).astype(int))
    z = (trial - 32.5) / np.std(np.arange(1, 65))
    contrast = np.where(data["congruency"].eq("Incongruent"), 0.5, -0.5)
    response = np.log(data["rt"].to_numpy(dtype=float))

    test = np.zeros(len(data), dtype=bool)
    for uid in ids:
        indices = np.flatnonzero(data["uid"].to_numpy() == uid)
        count = max(1, round(0.20 * len(indices)))
        order = sorted(
            indices,
            key=lambda j: hashlib.sha256(
                f"{SPLIT_NAMESPACE}|{uid}|{int(data['blockNum'].iloc[j])}|{int(data['trialNum'].iloc[j])}".encode()
            ).digest(),
        )
        test[order[:count]] = True
    train = ~test
    training_mean = float(response[train].mean())
    response -= training_mean

    def pack(mask: np.ndarray) -> tuple[np.ndarray, ...]:
        return person[mask], trial[mask], z[mask], contrast[mask], response[mask]

    packed_train, packed_test = pack(train), pack(test)
    for position in range(1, 65):
        for arm in (-0.5, 0.5):
            if not np.any((packed_train[1] == position) & (packed_train[3] == arm)):
                raise RuntimeError(
                    f"{first_pc}-first training bin {position} lacks contrast arm {arm}"
                )
    metadata = {
        "first_pc": first_pc,
        "second_pc": "Low" if first_pc == "High" else "High",
        "participants": len(ids),
        "observations": len(data),
        "training_observations": int(train.sum()),
        "test_observations": int(test.sum()),
        "training_mean_log_rt": training_mean,
        "excluded_participants": sorted(set(excluded_ids)),
        "participant_hash": hashlib.sha256("|".join(ids).encode()).hexdigest(),
        "split_hash": formal.sha256_array(np.packbits(test)),
    }
    return packed_train, packed_test, metadata


def conditional_levels(
    flat: np.ndarray, joint: tuple[int, ...]
) -> list[dict[str, float | int]]:
    keys = [
        (int(row[0]),) + tuple(int(v) for v in row[1 : 1 + int(row[0])]) for row in flat
    ]
    draws = flat[np.asarray([key == joint for key in keys]), 1 + formal.MAX_K :]
    edges = (0,) + tuple(joint[1:]) + (formal.T,)
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        values = draws[:, start:end].mean(axis=1)
        out.append(
            {
                "start_bin": start + 1,
                "end_bin": end,
                "mean": float(values.mean()),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return out


def fit_group(
    input_dir: Path, first_pc: str, library: Any, cfg: dict[str, Any]
) -> dict[str, Any]:
    train, test, metadata = prepare_group(input_dir, first_pc)
    participants = int(metadata["participants"])
    H, h, yqy = formal.collapsed(*train, participants)
    curves: dict[str, np.ndarray] = {}
    method_rows: list[dict[str, Any]] = []
    fixed = {"NoEffect": None, "ConstEffect": (), "FixedSplit": (32,)}
    for name, cuts in fixed.items():
        started = time.perf_counter()
        curves[name] = (
            np.zeros(formal.T)
            if cuts is None
            else formal.fixed_curve(H, h, yqy, len(train[-1]), cuts)
        )
        method_rows.append(
            {
                "group": first_pc + "-first",
                "method": name,
                "test_RMSE": formal.test_rmse(train, test, curves[name], participants),
                "K": 0 if cuts is None else len(cuts),
                "cuts": [] if cuts is None else list(cuts),
                "runtime_seconds": time.perf_counter() - started,
            }
        )

    started = time.perf_counter()
    pelt_cuts, curves["PELT-BIC"] = formal.pelt_fit(
        train, participants, H, h, yqy, cfg["nominal_model"]["min_segment"]
    )
    pelt_runtime = time.perf_counter() - started
    method_rows.append(
        {
            "group": first_pc + "-first",
            "method": "PELT-BIC",
            "test_RMSE": formal.test_rmse(
                train, test, curves["PELT-BIC"], participants
            ),
            "K": len(pelt_cuts),
            "cuts": list(pelt_cuts),
            "runtime_seconds": pelt_runtime,
        }
    )

    sampler, nominal = cfg["sampler"], cfg["nominal_model"]
    chains, stats, seeds = [], [], []
    started = time.perf_counter()
    for index, initial_k in enumerate(sampler["initial_k"], 1):
        seed = formal.deterministic_seed(CHAIN_NAMESPACE, f"{first_pc}|CHAIN{index}")
        chain, chain_stats = formal.run_chain(
            library,
            H,
            h,
            yqy,
            len(train[-1]),
            sampler["iterations"],
            sampler["warmup"],
            seed,
            initial_k,
            nominal["kmax"],
            nominal["min_segment"],
            nominal["prior_factorial_power"],
        )
        chains.append(chain)
        stats.append(chain_stats)
        seeds.append(seed)
    summary = formal.summarize_a3(
        chains, stats, time.perf_counter() - started, nominal["kmax"]
    )
    curves["CG-RJMCMC"] = summary["bma_curve"]
    method_rows.append(
        {
            "group": first_pc + "-first",
            "method": "CG-RJMCMC",
            "test_RMSE": formal.test_rmse(
                train, test, curves["CG-RJMCMC"], participants
            ),
            "K": int(summary["joint_map"][0]),
            "cuts": list(summary["joint_map"][1:]),
            "runtime_seconds": summary["three_chain_walltime_seconds"],
        }
    )
    if tuple(row["method"] for row in method_rows) != METHODS:
        raise AssertionError("unexpected method order")
    flat = np.stack(chains).reshape(-1, formal.DRAW_WIDTH)
    return {
        "data": metadata,
        "methods": method_rows,
        "PELT_BIC": {"cuts": list(pelt_cuts), "curve": curves["PELT-BIC"].tolist()},
        "CG_RJMCMC": {
            "p_k": summary["p_k"],
            "boundary_pip": summary["boundary_pip"],
            "bma_curve": summary["bma_curve"].tolist(),
            "bma_q025": summary["bma_q025"].tolist(),
            "bma_q975": summary["bma_q975"].tolist(),
            "joint_map_partition": list(summary["joint_map"]),
            "joint_map_curve": summary["joint_map_curve"].tolist(),
            "joint_map_conditional_levels": conditional_levels(
                flat, tuple(summary["joint_map"])
            ),
            "diagnostics": summary["diagnostics"],
            "max_split_Rhat": summary["max_split_Rhat"],
            "min_bulk_ESS": summary["min_bulk_ESS"],
            "min_ESS_per_second": summary["min_ESS_per_second"],
            "acceptance": summary["acceptance"],
            "three_chain_walltime_seconds": summary["three_chain_walltime_seconds"],
            "chain_seeds": seeds,
        },
    }


def run(input_dir: Path, output: Path) -> dict[str, Any]:
    input_dir, output = input_dir.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    cfg = formal.load_config(FORMAL_CONFIG)
    sampler, nominal = cfg["sampler"], cfg["nominal_model"]
    frozen = (
        sampler["chains"],
        sampler["iterations"],
        sampler["warmup"],
        sampler["initial_k"],
        nominal["kmax"],
        nominal["min_segment"],
        nominal["prior_factorial_power"],
    )
    if frozen != (3, 15000, 2000, [0, 2, 4], 4, 8, 1.0):
        raise ValueError(f"formal settings changed: {frozen}")
    library, kernel_hash = formal.compile_kernel(ROOT / "build" / "r61_formal")
    groups = [fit_group(input_dir, order, library, cfg) for order in ("High", "Low")]
    source_hashes = {
        "runner": formal.sha256_file(Path(__file__)),
        "formal_runner": formal.sha256_file(FORMAL_RUNNER),
        "kernel": kernel_hash,
        "config": formal.sha256_file(FORMAL_CONFIG),
    }
    file_hashes = {
        path.name: formal.sha256_file(path) for path in _source_files(input_dir)
    }
    result = {
        "schema_version": 1,
        "dataset": "Braeutigam et al. Experiment 1 visual Flanker",
        "article_doi": "10.3758/s13421-023-01447-x",
        "data_url": "https://osf.io/6n597/",
        "source_hashes": source_hashes,
        "input_hash": hashlib.sha256(
            json.dumps(file_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "groups": groups,
    }
    output.mkdir(parents=True)
    formal.atomic_json(output / "results.json", result)
    formal.atomic_csv(
        output / "heldout_rmse.csv",
        ["group", "method", "test_RMSE", "K", "cuts", "runtime_seconds"],
        [row for group in groups for row in group["methods"]],
    )
    formal.atomic_json(
        output / "run_manifest.json",
        {
            "source_hashes": source_hashes,
            "input_files": file_hashes,
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "command": sys.argv,
            "created_unix": time.time(),
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="extracted RawData_Experiment1 directory"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))

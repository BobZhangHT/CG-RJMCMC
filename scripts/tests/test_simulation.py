import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "r61_formal", ROOT / "scripts" / "simulation.py"
)
r61 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r61)


def toy(participants=6):
    person = np.repeat(np.arange(participants), 64)
    trial = np.tile(np.arange(1, 65), participants)
    z = (trial - 32.5) / np.std(np.arange(1, 65))
    c = np.where((person + trial) % 2, 0.5, -0.5)
    y = 2.0 + 0.05 * z + np.random.default_rng(8).normal(size=len(trial))
    return person, trial, z, c, y


def test_formal_config_and_replications():
    cfg = r61.load_config(ROOT / "scripts" / "config" / "r61_formal.yaml")
    assert cfg["dgps"]["D5"] == {
        "cuts": [13, 25, 39, 51],
        "levels": [0.13125, 0.03125, 0.13125, 0.03125, 0.13125],
    }
    assert cfg["replications"] == {
        "main_reps": 500,
        "sensitivity_reps": 100,
        "kmax_reps": 100,
        "status": "frozen_for_publication_run",
    }
    assert cfg["sampler"]["chains"] == 3
    cases = r61.expand_cases(cfg, {"main"})
    assert [case["dgp"] for case in cases] == [f"D{i}" for i in range(6)]


def test_geometry_hash_is_part_of_resume_fingerprint():
    sources = {"runner": "r", "kernel": "k", "config_file": "c"}
    a = r61.make_fingerprint(
        {"config": {}, "stages": ["main"], "geometry_hash": "a"}, sources
    )
    b = r61.make_fingerprint(
        {"config": {}, "stages": ["main"], "geometry_hash": "b"}, sources
    )
    assert a != b


def test_frozen_geometry_is_complete_and_binary():
    base, manifest = r61.load_frozen_geometry(
        ROOT / "scripts" / "config" / "r61_congruency_geometry.csv"
    )
    assert base.shape == (49, 64) and set(np.unique(base)) == {-0.5, 0.5}
    assert len(manifest) == 49 and len({row["anon_id"] for row in manifest}) == 49


def test_holdout_is_exact_participant_within_and_outcome_blind():
    person, trial, *_ = toy()
    a = r61.outcome_blind_holdout(person, trial, 13, "fixed")
    b = r61.outcome_blind_holdout(person, trial, 13, "fixed")
    assert np.array_equal(a, b)
    assert all(a[person == i].sum() == 13 for i in np.unique(person))
    assert not r61.outcome_blind_holdout.__code__.co_varnames[
        : r61.outcome_blind_holdout.__code__.co_argcount
    ].__contains__("y")


def test_arrow_collapse_matches_direct_dense_covariance():
    person, trial, z, c, y = toy(4)
    H, h, yqy = r61.collapsed(person, trial, z, c, y, 4)
    n = len(y)
    G = np.zeros((n, 64))
    G[np.arange(n), trial - 1] = c
    Z = np.column_stack((np.ones(n), z, np.eye(4)[person]))
    prior = np.diag(np.r_[0.25, 1.0, np.ones(4)])
    Q = np.eye(n) - Z @ np.linalg.solve(prior + Z.T @ Z, Z.T)
    assert np.max(np.abs(H - G.T @ Q @ G)) < 2e-10
    assert np.max(np.abs(h - G.T @ Q @ y)) < 2e-10
    assert abs(yqy - y @ Q @ y) < 5e-8


def test_weighted_cost_matches_direct_and_named_pelt_recovers_clear_cuts():
    y = np.r_[np.zeros(16), np.full(16, 4.0), np.zeros(16), np.full(16, 4.0)]
    weights = np.linspace(1, 2, 64)
    cost = r61.WeightedL2(8).fit(np.c_[y, weights])
    for start, end in ((0, 16), (7, 40), (48, 64)):
        mean = np.average(y[start:end], weights=weights[start:end])
        assert np.isclose(
            cost.error(start, end),
            np.sum(weights[start:end] * (y[start:end] - mean) ** 2),
        )
    assert r61.pelt_bic(y, weights, 8) == (16, 32, 48)


def test_structural_metric_empty_conventions_and_ordered_segment_rule():
    assert r61.hausdorff((), ()) == 0
    assert r61.hausdorff((), (13,)) == 64 and r61.hausdorff((13,), ()) == 64
    curve = r61.curve_from((20,), (0.12, 0.03))
    rmse, bias = r61.ordered_segment_error(curve, (20,), [0.10, 0.05], 1)
    assert np.allclose(bias, [0.02, -0.02]) and np.isclose(rmse, 0.02)
    assert r61.ordered_segment_error(curve, (), [0.10, 0.05], 1) == (None, None)


def test_atomic_summary_replacement_and_native_json(tmp_path):
    path = tmp_path / "summary.json"
    r61.atomic_json(path, {"x": np.float64(1), "ok": np.bool_(True)})
    assert json.loads(path.read_text()) == {"ok": True, "x": 1.0}
    assert not list(tmp_path.glob("*.tmp-*"))


def test_parameterized_c_kernel_runs_kmax5_minseg6_without_retaining_files(tmp_path):
    person, trial, z, c, y = toy(8)
    H, h, yqy = r61.collapsed(person, trial, z, c, y, 8)
    lib, _ = r61.compile_kernel(tmp_path)
    chains, stats = [], []
    for init, seed in zip((0, 2, 4), (11, 12, 13)):
        chain, stat = r61.run_chain(
            lib, H, h, yqy, len(y), 120, 20, seed, init, 5, 6, 2.0
        )
        chains.append(chain)
        stats.append(stat)
        assert chain.shape == (100, 70) and np.all(
            (chain[:, 0] >= 0) & (chain[:, 0] <= 5)
        )
        for row in chain:
            cuts = tuple(int(x) for x in row[1 : 1 + int(row[0])])
            edges = (0,) + cuts + (64,)
            assert all(b - a >= 6 for a, b in zip(edges[:-1], edges[1:]))
    summary = r61.summarize_a3(chains, stats, 1.0, 5)
    assert len(summary["p_k"]) == 6 and np.isclose(sum(summary["p_k"]), 1.0)
    assert len(summary["boundary_pip"]) == 64 and summary["boundary_pip"][-1] == 0
    assert summary["bma_q025"].shape == summary["bma_q975"].shape == (64,)
    assert set(summary["acceptance"]) == {"birth", "death", "move"}


def test_constant_chain_diagnostic_is_explicitly_flagged():
    rhat, ess, constant, status = r61.rank_diagnostic(np.ones((3, 20)))
    assert (rhat, ess, constant, status) == (
        None,
        None,
        True,
        "not_applicable_constant_all_chains",
    )


def test_between_chain_constant_disagreement_is_json_safe_failure_flag():
    values = np.vstack((np.zeros(50), np.ones(50), np.full(50, 2.0)))
    rhat, _, constant, status = r61.rank_diagnostic(values)
    assert (
        rhat is None
        and not constant
        and status == "nonfinite_rhat_between_chain_disagreement"
    )
    r61.atomic_json(
        Path(__file__).with_name("unused-do-not-write.json"),
        {"rhat": rhat, "status": status},
    )
    Path(__file__).with_name("unused-do-not-write.json").unlink()

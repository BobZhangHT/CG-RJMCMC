"""Finite-state regression checks for the sampler, using synthetic data only."""

import os
import ctypes as ct
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_finite_state_kernel(tmp_path):
    """Check algebra, reversibility, stationarity and a fixed-seed chain."""
    HERE = tmp_path
    spec = importlib.util.spec_from_file_location(
        "formal_r1", ROOT / "scripts/simulation.py"
    )
    formal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formal)
    subprocess.run(
        [
            "gcc",
            "-O2",
            *(["-fPIC"] if os.name != "nt" else []),
            "-shared",
            "-o",
            os.path.relpath(HERE / "kernel_probe.dll", ROOT),
            "scripts/probe.c",
            "-lm",
        ],
        cwd=ROOT,
        check=True,
    )
    lib = ct.CDLL(str(HERE / "kernel_probe.dll"))
    D = ct.POINTER(ct.c_double)
    I = ct.POINTER(ct.c_int)
    lib.probe_weight.argtypes = [
        D,
        D,
        ct.c_double,
        ct.c_int,
        I,
        ct.c_int,
        ct.c_int,
        ct.c_double,
    ]
    lib.probe_weight.restype = ct.c_double
    lib.probe_neighbors.argtypes = [
        D,
        D,
        ct.c_double,
        ct.c_int,
        I,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        D,
    ]
    lib.probe_neighbors.restype = ct.c_int
    dp = lambda a: a.ctypes.data_as(D)
    ip = lambda a: a.ctypes.data_as(I)
    rng = np.random.default_rng(8092026)
    N = 5
    T = 64
    person = np.repeat(np.arange(N), T)
    trial = np.tile(np.arange(1, T + 1), N)
    z = (trial - 32.5) / np.std(np.arange(1, 65))
    c = rng.choice([-0.5, 0.5], len(trial))
    y = (
        0.05 * z
        + rng.normal(0, 0.4, N)[person]
        + c * np.where(trial <= 32, 0.15, -0.03)
        + rng.normal(0, 0.4, len(trial))
    )
    keep = rng.random(len(y)) > 0.18
    person, trial, z, c, y = [a[keep] for a in (person, trial, z, c, y)]
    n = len(y)
    H, h, r = formal.collapsed(person, trial, z, c, y, N)
    Z = np.c_[np.ones(n), z, np.eye(N)[person]]
    C = np.eye(64)[trial - 1] * c[:, None]
    Dn = np.diag([4.0, 1.0] + [1.0] * N)
    V0 = np.eye(n) + Z @ Dn @ Z.T
    Q = np.linalg.inv(V0)
    result = {
        "seed": 8092026,
        "observations": n,
        "dense_sufficient_statistics_errors": {
            "H": float(np.max(abs(H - C.T @ Q @ C))),
            "h": float(np.max(abs(h - C.T @ Q @ y))),
            "r": float(abs(r - y @ Q @ y)),
        },
    }
    # m=16 permits complete exhaustive coverage of 188 legal partitions, including Kmax=3 and empty move neighborhoods.
    m = 16
    km = 3
    states = []
    for k in range(km + 1):
        states.extend(
            x
            for x in itertools.combinations(range(m, T - m + 1), k)
            if min(np.diff((0,) + x + (T,))) >= m
        )
    index = {x: i for i, x in enumerate(states)}

    def cuts_array(x):
        a = np.zeros(5, dtype=np.int32)
        a[: len(x)] = x
        return a

    weights = np.array(
        [
            lib.probe_weight(dp(H), dp(h), r, n, ip(cuts_array(x)), len(x), m, 1.0)
            for x in states
        ]
    )
    reference = []
    for cuts in states:
        k = len(cuts)
        B = np.eye(k + 1)[np.searchsorted(cuts, np.arange(1, 65), side="left")]
        A = 4 * np.eye(k + 1) + B.T @ H @ B
        g = B.T @ h
        rate = 2 + (r - g @ np.linalg.solve(A, g)) / 2
        reference.append(
            -math.lgamma(k + 1)
            - math.log(math.comb(T - (k + 1) * m + k, k))
            + 0.5 * (k + 1) * math.log(4)
            - 0.5 * np.linalg.slogdet(A)[1]
            - (3 + n / 2) * math.log(rate)
        )
    result["enumerated_partitions"] = len(states)
    result["count_by_K"] = {
        str(k): sum(len(x) == k for x in states) for k in range(km + 1)
    }
    result["max_C_vs_independent_logweight_error"] = float(
        np.max(abs(weights - reference))
    )
    nb = {}
    qerr = 0.0
    missing = 0
    for i, x in enumerate(states):
        for kind in range(3):
            out = np.zeros((512, 8))
            nn = lib.probe_neighbors(
                dp(H), dp(h), r, n, ip(cuts_array(x)), len(x), kind, km, m, 1.0, dp(out)
            )
            entries = {
                index[tuple(int(v) for v in row[1 : 1 + int(row[0])])]: row[7]
                for row in out[:nn]
            }
            assert len(entries) == nn
            nb[i, kind] = entries
            if nn:
                qerr = max(qerr, abs(sum(entries.values()) - 1))

    def pb(k):
        return 1.0 if k == 0 else (0.0 if k == km else 0.5)

    Pbd = np.zeros((len(states), len(states)))
    Pg = Pbd.copy()
    for i, x in enumerate(states):
        for kind in range(3):
            sf = pb(len(x)) if kind == 0 else (1 - pb(len(x)) if kind == 1 else 1.0)
            for j, qf in nb[i, kind].items():
                rev = 1 - kind if kind < 2 else 2
                qr = nb[j, rev].get(i, 0)
                missing += int(qr == 0)
                assert qr > 0
                sr = (
                    1 - pb(len(states[j]))
                    if kind == 0
                    else (pb(len(states[j])) if kind == 1 else 1.0)
                )
                alpha = min(
                    1.0, math.exp(weights[j] - weights[i]) * sr * qr / (sf * qf)
                )
                (Pbd if kind < 2 else Pg)[i, j] += sf * qf * alpha
    for mat in (Pbd, Pg):
        np.fill_diagonal(mat, 1 - mat.sum(1))
    pi = np.exp(weights - weights.max())
    pi /= pi.sum()
    round_ = Pbd @ Pg @ Pg
    full = round_ @ round_
    result.update(
        max_proposal_sum_error=qerr,
        missing_reverse_edges=missing,
        max_bd_detailed_balance_error=float(
            np.max(abs(pi[:, None] * Pbd - pi[None, :] * Pbd.T))
        ),
        max_global_detailed_balance_error=float(
            np.max(abs(pi[:, None] * Pg - pi[None, :] * Pg.T))
        ),
        max_full_sweep_stationarity_error=float(np.max(abs(pi @ full - pi))),
        min_transition_probability=float(min(Pbd.min(), Pg.min())),
    )
    # Check the actual exported C sampler against the exact K probabilities and trajectory posterior mean.
    lib.r61_formal_run.argtypes = [
        D,
        D,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_uint64,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        D,
        ct.POINTER(ct.c_longlong),
    ]
    lib.r61_formal_run.restype = ct.c_int
    draws = np.empty((100000, 70))
    stats = np.zeros(8, dtype=np.int64)
    rc = lib.r61_formal_run(
        dp(H),
        dp(h),
        r,
        n,
        101000,
        1000,
        8092026,
        3,
        km,
        m,
        1.0,
        dp(draws),
        stats.ctypes.data_as(ct.POINTER(ct.c_longlong)),
    )
    assert rc == 0
    exactpk = [float(pi[[len(x) == k for x in states]].sum()) for k in range(km + 1)]
    samplepk = [float(np.mean(draws[:, 0] == k)) for k in range(km + 1)]
    exactcurve = sum(p * formal.fixed_curve(H, h, r, n, x) for p, x in zip(pi, states))
    result.update(
        exact_p_K=exactpk,
        sampled_p_K=samplepk,
        max_trajectory_mean_difference=float(
            np.max(abs(draws[:, 6:].mean(0) - exactcurve))
        ),
    )
    # Deterministic algebra/probability checks are gates. Monte Carlo differences
    # are reported separately and are not treated as exact-equality assertions.
    for error in result["dense_sufficient_statistics_errors"].values():
        assert error < 1e-9, "dense/collapsed sufficient statistics disagree"
    assert result["max_C_vs_independent_logweight_error"] < 1e-9
    assert result["max_proposal_sum_error"] < 1e-12
    assert result["missing_reverse_edges"] == 0
    assert result["max_bd_detailed_balance_error"] < 1e-12
    assert result["max_global_detailed_balance_error"] < 1e-12
    assert result["max_full_sweep_stationarity_error"] < 1e-12
    assert result["min_transition_probability"] >= -1e-12
    assert np.isfinite(draws).all()
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

"""Regression coverage for the minimum-length-safe production PELT solver."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pelt_regression_sim", ROOT / "scripts/simulation.py")
sim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sim)


def centered_dp(y, w, m):
    """Independent unpruned DP using centered residuals, no prefix subtraction."""
    f, paths = {0: 0.0}, {0: ()}
    for end in range(m, len(y) + 1):
        choices = []
        for start in [0] + list(range(m, end - m + 1)):
            mu = np.average(y[start:end], weights=w[start:end])
            value = f[start] + np.sum(w[start:end] * (y[start:end] - mu)**2) + math.log(len(y))
            choices.append((value, start))
        f[end], start = min(choices)
        paths[end] = paths[start] + ((start,) if start else ())
    return f[len(y)], paths[len(y)]


def centered_objective(y, w, cuts):
    edges = (0,) + cuts + (len(y),)
    return sum(np.sum(w[a:b] * (y[a:b] - np.average(y[a:b], weights=w[a:b]))**2)
               + math.log(len(y)) for a, b in zip(edges, edges[1:]))


@pytest.mark.parametrize("name", ["primary_D1_R66", "N35_SN0_R352"])
def test_verified_late_pruning_counterexamples(name):
    import json
    path = Path(__file__).parent / "fixtures/pelt_counterexamples.json"
    data = json.loads(path.read_text(encoding="utf-8"))["cases"][name]
    y, w = np.asarray(data["statistic"]), np.asarray(data["weights"])
    assert y.shape == w.shape == (64,)
    assert np.isfinite(y).all() and np.isfinite(w).all() and np.all(w > 0)
    assert sim.pelt_bic(y, w, 8) == ()
    assert centered_dp(y, w, 8)[1] == ()


@pytest.mark.parametrize("n,m", [(11, 1), (24, 3), (64, 8)])
@pytest.mark.parametrize("seed", range(5))
def test_random_weighted_objectives_against_independent_dp(n, m, seed):
    rng = np.random.default_rng(seed + n)
    y, w = rng.normal(size=n), np.exp(rng.uniform(-3, 3, n))
    cuts = sim.pelt_bic(y, w, m)
    assert min(np.diff((0,) + cuts + (n,))) >= m
    assert centered_objective(y, w, cuts) == pytest.approx(centered_dp(y, w, m)[0], rel=1e-9)


def test_ties_and_single_feasible_segment():
    y, w = np.r_[np.zeros(8), np.ones(8)], np.ones(16) * (math.log(16) / 4)
    assert sim.pelt_bic(y, w, 8) == ()
    assert centered_objective(y, w, ()) == pytest.approx(centered_objective(y, w, (8,)), abs=1e-12)
    assert sim.pelt_bic(np.zeros(12), np.ones(12), 8) == ()
    assert sim.pelt_bic([1], [1], 1) == ()


@pytest.mark.parametrize("y,w,m", [([], [], 1), ([[1]], [[1]], 1), ([1], [1, 2], 1),
    ([np.nan], [1], 1), ([1], [np.inf], 1), ([1], [0], 1), ([1], [-1], 1),
    ([1], [1], 0), ([1], [1], 2), ([1], [1], 1.0), ([1], [1], True)])
def test_invalid_inputs(y, w, m):
    with pytest.raises(ValueError):
        sim.pelt_bic(y, w, m)

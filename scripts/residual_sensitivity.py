#!/usr/bin/env python
"""R3.M4: exact fixed-AR Gaussian and full-posterior Student-t sensitivities.

All nuisance, segment, variance and partition priors equal the original model.
Student-t errors are unit-variance t_df: c=(df-2)/df, w~Gamma(df/2,df/2),
epsilon|w,sigma2~N(0,sigma2*c/w). The partially collapsed Gibbs order is
(M,sigma2,delta)|w, then nuisance|M,sigma2,delta,w, then w|all coefficients.
No residual/IRLS plug-in weights are used. Native AR gaps are within retained
blocks and participants, measured before binning and before heldout removal.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
import arviz as az
import numpy as np
from scipy.special import stdtr
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "reproduction" / "residual_sensitivity"
PROTOCOL = ROOT / "scripts" / "config" / "residual_sensitivity_protocol.json"
NAMESPACE = "PLOS-RESIDUAL-SENSITIVITY-20261114"
PHASE = np.r_[np.full(32, -1 / 32), np.full(32, 1 / 32)]
DRAW_WIDTH = 74
RNG_DERIVATION = "SeedSequence(chain_seed).spawn(2): child 0 generates one uint64 structural/effect seed; child 1 initializes the nuisance/latent-weight Generator"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "residual_" + name, ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


precision = load("precision")
application = load("application")
checks = load("checks")


def seed_for(*labels):
    return int.from_bytes(
        hashlib.sha256("|".join((NAMESPACE, *map(str, labels))).encode()).digest()[:8],
        "little",
    )


def chain_streams(seed):
    """Independent spawned streams, never two NumPy streams with the same seed."""
    structural, augmented = np.random.SeedSequence(seed).spawn(2)
    structural_seed = int(structural.generate_state(1, dtype=np.uint64)[0])
    return structural_seed, np.random.default_rng(augmented)


def archived_statistic_check(archive_root, group, stats):
    if archive_root is None:
        return {"requested": False}
    archived = np.load(
        archive_root
        / f"revision_plosone/final_audit_20261112/pelt_refresh/{group}_application_statistics.npz"
    )
    errors = {
        name: float(np.max(np.abs(value - archived[name])))
        for name, value in zip(("H", "h", "yqy"), stats)
    }
    if max(errors.values()) > 1e-8:
        raise RuntimeError(f"archived Gaussian statistic mismatch: {errors}")
    return {
        "requested": True,
        "archive_root": str(archive_root),
        "statistic_max_errors": errors,
    }


def archived_exact_check(archive_root, group, exact):
    if archive_root is None:
        return {"requested": False}
    reference = json.loads(
        (
            archive_root
            / f"results/revision/plos_review_v1/precision/cases/application_{group.lower()}_first/summary.json"
        ).read_text(encoding="utf-8")
    )
    error = float(np.max(np.abs(exact["p_k"] - np.array(reference["exact_p_k"]))))
    if error > 1e-8 or list(exact["map_cuts"]) != reference["exact_top_partition"]:
        raise RuntimeError(f"archived baseline exact-posterior mismatch: {error}")
    return {"requested": True, "maximum_p_k_error": error, "map_partition_equal": True}


def weighted_stats(person, trial, z, c, y, participants, weights):
    """Sparse Schur sufficient statistics; weights are observation precision."""
    person, trial = np.asarray(person, int), np.asarray(trial, int) - 1
    w = np.asarray(weights, float)
    if w.shape != y.shape or np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite positive observation precisions")
    diag = 1 + np.bincount(person, weights=w, minlength=participants)
    cross = np.vstack(
        (diag - 1, np.bincount(person, weights=w * z, minlength=participants))
    )
    top = np.array([[0.25 + w.sum(), w @ z], [w @ z, 1 + w @ (z * z)]])
    schur = top - (cross / diag) @ cross.T
    B = np.zeros((participants + 2, 65))
    B[0, :64] = np.bincount(trial, weights=w * c, minlength=64)
    B[1, :64] = np.bincount(trial, weights=w * z * c, minlength=64)
    B[2:, :64] = np.bincount(
        person * 64 + trial, weights=w * c, minlength=participants * 64
    ).reshape(participants, 64)
    B[0, 64], B[1, 64] = w @ y, (w * z) @ y
    B[2:, 64] = np.bincount(person, weights=w * y, minlength=participants)
    beta = np.linalg.solve(schur, B[:2] - (cross / diag) @ B[2:])
    u = (B[2:] - cross.T @ beta) / diag[:, None]
    raw = np.zeros((65, 65))
    raw[np.arange(64), np.arange(64)] = np.bincount(
        trial, weights=w * c * c, minlength=64
    )
    raw[:64, 64] = np.bincount(trial, weights=w * c * y, minlength=64)
    raw[64, :64] = raw[:64, 64]
    raw[64, 64] = w @ (y * y)
    collapsed = raw - B[:2].T @ beta - B[2:].T @ u
    collapsed = (collapsed + collapsed.T) / 2
    return (
        collapsed[:64, :64].copy(),
        collapsed[:64, 64].copy(),
        float(collapsed[64, 64]),
        (diag, cross, schur, B),
    )


def draw_nuisance(cache, delta, sigma2, rng):
    if isinstance(cache, dict):
        B, L = cache["B"], cache["L"]
        r = B[:, -1] - B[:, :64] @ delta
        mean = np.linalg.solve(L.T, np.linalg.solve(L, r))
        return mean + math.sqrt(sigma2) * np.linalg.solve(
            L.T, rng.normal(size=len(mean))
        )
    diag, cross, schur, B = cache
    r = B[:, 64] - B[:, :64] @ delta
    mean = np.linalg.solve(schur, r[:2] - (cross / diag) @ r[2:])
    beta = mean + math.sqrt(sigma2) * np.linalg.solve(
        np.linalg.cholesky(schur).T, rng.normal(size=2)
    )
    u = (r[2:] - cross.T @ beta) / diag + np.sqrt(sigma2 / diag) * rng.normal(
        size=len(diag)
    )
    return np.r_[beta, u]


def innovation_design(train, participants, block, native_trial, rho):
    X, Z, y = designs(train, participants)
    X, Z, y = [ar_whiten(v, train[0], block, native_trial, rho) for v in (X, Z, y)]
    return hstack((csr_matrix(Z), csr_matrix(X), csr_matrix(y[:, None])), format="csr")


def weighted_sparse_stats(augmented, participants, weights):
    """Generic sparse innovation Gram; no dense n-by-n residual maker."""
    nuisance = participants + 2
    gram = (augmented.T @ augmented.multiply(weights[:, None])).toarray()
    A = gram[:nuisance, :nuisance] + np.diag(np.r_[0.25, np.ones(nuisance - 1)])
    L = np.linalg.cholesky(A)
    B = gram[:nuisance, nuisance:]
    collapsed = gram[nuisance:, nuisance:] - B.T @ np.linalg.solve(
        L.T, np.linalg.solve(L, B)
    )
    collapsed = (collapsed + collapsed.T) / 2
    return (
        collapsed[:64, :64].copy(),
        collapsed[:64, 64].copy(),
        float(collapsed[64, 64]),
        {"B": B, "L": L},
    )


def dense_stats(X, Z, y, weights=None):
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    prior = np.r_[0.25, np.ones(Z.shape[1] - 1)]
    A = Z.T @ (w[:, None] * Z) + np.diag(prior)
    D = np.column_stack((X, y))
    B = Z.T @ (w[:, None] * D)
    C = D.T @ (w[:, None] * D) - B.T @ np.linalg.solve(A, B)
    return C[:64, :64], C[:64, 64], float(C[64, 64])


def designs(train, participants):
    person, trial, z, c, y = train
    X = np.zeros((len(y), 64))
    X[np.arange(len(y)), trial - 1] = c
    Z = np.zeros((len(y), participants + 2))
    Z[:, 0] = 1
    Z[:, 1] = z
    Z[np.arange(len(y)), person + 2] = 1
    return X, Z, y


def ar_whiten(values, person, block, native_trial, rho):
    """Stationary AR(1), conditional across native gaps, resetting each block."""
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1)")
    values = np.asarray(values, float)
    out = values.copy()
    same = (person[1:] == person[:-1]) & (block[1:] == block[:-1])
    index = np.flatnonzero(same) + 1
    gaps = native_trial[index] - native_trial[index - 1]
    if np.any(gaps <= 0):
        raise ValueError("native trials must increase inside every participant block")
    correlation = rho**gaps
    if values.ndim == 2:
        correlation = correlation[:, None]
    out[index] = (values[index] - correlation * values[index - 1]) / np.sqrt(
        1 - correlation**2
    )
    return out


def prepare_group(path, first_pc):
    """The existing checks API asserts parity with original train/test and hashes."""
    frame, _test, metadata = checks.prepare_group(path, first_pc)
    train = checks._pack(frame)
    expected = {"High": (24, 6698, 1675), "Low": (21, 5937, 1486)}[first_pc]
    actual = tuple(
        metadata[k]
        for k in ("participants", "training_observations", "test_observations")
    )
    if actual != expected:
        raise RuntimeError(f"unexpected application cohort: {actual}")
    return (
        train,
        frame["blockNum"].to_numpy(int),
        frame["trialNum"].to_numpy(int),
        metadata,
    )


class NativeStepper:
    def __init__(self, output):
        flags = ["-O3", "-shared", *([] if os.name == "nt" else ["-fPIC"])]
        tag = hashlib.sha256(
            b"".join(
                (ROOT / "scripts" / f).read_bytes()
                for f in ("kernel.c", "residual_step.c")
            )
            + repr(flags).encode()
        ).hexdigest()[:20]
        self.path = output / (
            "residual_step_" + tag + (".dll" if os.name == "nt" else ".so")
        )
        output.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            subprocess.run(
                [
                    "gcc",
                    *flags,
                    "-o",
                    os.path.relpath(self.path, ROOT),
                    "scripts/residual_step.c",
                    "-lm",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        # Do not catch OS security rejection or move/copy a rejected binary.
        self.lib = ct.CDLL(str(self.path))
        dp, ip = ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)
        self.lib.residual_init.argtypes = [
            ct.c_int,
            ct.c_uint64,
            ip,
            ct.POINTER(ct.c_uint64),
        ]
        self.lib.residual_step.argtypes = [
            dp,
            dp,
            ct.c_double,
            ct.c_int,
            ip,
            ct.POINTER(ct.c_uint64),
            dp,
            ct.POINTER(ct.c_longlong),
        ]

    def initialize(self, initk, seed):
        state = np.zeros(6, np.int32)
        rng = ct.c_uint64(seed)
        rc = self.lib.residual_init(
            initk, seed, state.ctypes.data_as(ct.POINTER(ct.c_int)), ct.byref(rng)
        )
        if rc:
            raise RuntimeError(f"native init failed: {rc}")
        return state, rng, np.zeros(8, np.int64)

    def step(self, H, h, yqy, n, state):
        cuts, rng, stats = state
        out = np.empty(65)
        rc = self.lib.residual_step(
            np.ascontiguousarray(H).ctypes.data_as(ct.POINTER(ct.c_double)),
            np.ascontiguousarray(h).ctypes.data_as(ct.POINTER(ct.c_double)),
            yqy,
            n,
            cuts.ctypes.data_as(ct.POINTER(ct.c_int)),
            ct.byref(rng),
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
            stats.ctypes.data_as(ct.POINTER(ct.c_longlong)),
        )
        if rc:
            raise RuntimeError(f"native step failed: {rc}")
        return out[0], out[1:]


class PythonStepper:
    """Proper fallback MH kernel with uniform legal proposals and persistent M.

    It targets the identical collapsed posterior to kernel.c. Its proposals are
    uniform instead of locally informed, so mixing must be assessed separately.
    No shared library is loaded or copied by this fallback.
    """

    def initialize(self, initk, seed):
        state = np.zeros(6, np.int32)
        state[0] = initk
        state[1 : 1 + initk] = [64 * j // (initk + 1) for j in range(1, initk + 1)]
        return state, np.random.default_rng(seed), np.zeros(8, np.int64)

    @staticmethod
    def parameters(prefix, ph, yqy, n, cuts):
        edges = np.array((0,) + cuts + (64,))
        a, b = edges[:-1], edges[1:]
        A = (
            prefix[b[:, None], b]
            - prefix[a[:, None], b]
            - prefix[b[:, None], a]
            + prefix[a[:, None], a]
            + 4 * np.eye(len(b))
        )
        g = ph[b] - ph[a]
        L = np.linalg.cholesky(A)
        mu = np.linalg.solve(L.T, np.linalg.solve(L, g))
        rate = 2 + 0.5 * (yqy - g @ mu)
        shape = 3 + n / 2
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError("invalid conditional inverse-gamma rate")
        lw = (
            -math.lgamma(len(cuts) + 1)
            - precision._log_partition_count(len(cuts))
            + 0.5 * len(b) * math.log(4)
            - np.log(np.diag(L)).sum()
            - shape * math.log(rate)
        )
        return lw, mu, L, rate, shape

    @staticmethod
    def neighbors(cuts, kind):
        if kind == 0:
            return (
                [
                    tuple(sorted(cuts + (t,)))
                    for a, b in zip((0,) + cuts, cuts + (64,))
                    for t in range(a + 8, b - 7)
                ]
                if len(cuts) < 4
                else []
            )
        if kind == 1:
            return [cuts[:i] + cuts[i + 1 :] for i in range(len(cuts))]
        return [
            cuts[:i] + (t,) + cuts[i + 1 :]
            for i in range(len(cuts))
            for t in range(
                (cuts[i - 1] if i else 0) + 8,
                (cuts[i + 1] if i + 1 < len(cuts) else 64) - 7,
            )
            if t != cuts[i]
        ]

    @staticmethod
    def direction(k, kind):
        if kind == 2:
            return 1.0
        return (
            (1.0 if kind == 0 else 0.0)
            if k == 0
            else ((0.0 if kind == 0 else 1.0) if k == 4 else 0.5)
        )

    def step(self, H, h, yqy, n, state):
        array, rng, stats = state
        cuts = tuple(map(int, array[1 : 1 + array[0]]))
        prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        ph = np.r_[0, h.cumsum()]
        lw = self.parameters(prefix, ph, yqy, n, cuts)[0]  # refresh every weight update
        for _ in range(2):
            kind = (
                0 if not cuts else (1 if len(cuts) == 4 else int(rng.random() >= 0.5))
            )
            for kind in (kind, 2, 2):
                candidates = self.neighbors(cuts, kind)
                if not candidates:
                    continue
                proposal = candidates[rng.integers(len(candidates))]
                reverse = 1 - kind if kind < 2 else 2
                backward = self.neighbors(proposal, reverse)
                proposed = self.parameters(prefix, ph, yqy, n, proposal)[0]
                logratio = (
                    proposed
                    - lw
                    + math.log(self.direction(len(proposal), reverse) / len(backward))
                    - math.log(self.direction(len(cuts), kind) / len(candidates))
                )
                stats[2 * kind] += 1
                if math.log(rng.random()) < logratio:
                    cuts = proposal
                    lw = proposed
                    stats[2 * kind + 1] += 1
        _, mu, L, rate, shape = self.parameters(prefix, ph, yqy, n, cuts)
        sigma2 = rate / rng.gamma(shape)
        levels = mu + math.sqrt(sigma2) * np.linalg.solve(L.T, rng.normal(size=len(mu)))
        delta = np.repeat(levels, np.diff((0,) + cuts + (64,)))
        array[:] = 0
        array[0] = len(cuts)
        array[1 : 1 + len(cuts)] = cuts
        return sigma2, delta


class InformedPythonStepper(PythonStepper):
    """Batched version of kernel.c's 95% sqrt-target + 5% uniform proposals."""

    @staticmethod
    def logweights(prefix, ph, yqy, n, states):
        if not states:
            return np.empty(0)
        k = len(states[0])
        p = k + 1
        edges = np.array([(0,) + cuts + (64,) for cuts in states], int)
        a, b = edges[:, :-1], edges[:, 1:]
        A = (
            prefix[b[:, :, None], b[:, None, :]]
            - prefix[a[:, :, None], b[:, None, :]]
            - prefix[b[:, :, None], a[:, None, :]]
            + prefix[a[:, :, None], a[:, None, :]]
            + 4 * np.eye(p)
        )
        g = ph[b] - ph[a]
        sign, logdet = np.linalg.slogdet(A)
        if np.any(sign <= 0):
            raise RuntimeError("non-positive batched posterior precision")
        mu = np.linalg.solve(A, g[:, :, None])[:, :, 0]
        rate = 2 + 0.5 * (yqy - np.einsum("ij,ij->i", g, mu))
        if np.any(rate <= 0) or not np.isfinite(rate).all():
            raise RuntimeError("invalid batched inverse-gamma rate")
        return (
            -math.lgamma(k + 1)
            - precision._log_partition_count(k)
            + 0.5 * p * math.log(4)
            - 0.5 * logdet
            - (3 + n / 2) * np.log(rate)
        )

    @staticmethod
    def probabilities(logweights):
        weights = np.exp(0.5 * (logweights - logweights.max()))
        return 0.05 / len(weights) + 0.95 * weights / weights.sum()

    def step(self, H, h, yqy, n, state):
        array, rng, stats = state
        cuts = tuple(map(int, array[1 : 1 + array[0]]))
        prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        ph = np.r_[0, h.cumsum()]
        lw = self.parameters(prefix, ph, yqy, n, cuts)[0]
        for _ in range(2):
            direction = (
                0 if not cuts else (1 if len(cuts) == 4 else int(rng.random() >= 0.5))
            )
            for kind in (direction, 2, 2):
                candidates = self.neighbors(cuts, kind)
                if not candidates:
                    continue
                scores = self.logweights(prefix, ph, yqy, n, candidates)
                q = self.probabilities(scores)
                pick = int(rng.choice(len(candidates), p=q))
                proposal = candidates[pick]
                reverse = 1 - kind if kind < 2 else 2
                backward = self.neighbors(proposal, reverse)
                back_scores = self.logweights(prefix, ph, yqy, n, backward)
                back_q = self.probabilities(back_scores)
                back = back_q[backward.index(cuts)]
                logratio = (
                    scores[pick]
                    - lw
                    + math.log(
                        self.direction(len(proposal), reverse)
                        * back
                        / (self.direction(len(cuts), kind) * q[pick])
                    )
                )
                stats[2 * kind] += 1
                if math.log(rng.random()) < logratio:
                    cuts = proposal
                    lw = scores[pick]
                    stats[2 * kind + 1] += 1
        _, mu, L, rate, shape = self.parameters(prefix, ph, yqy, n, cuts)
        sigma2 = rate / rng.gamma(shape)
        levels = mu + math.sqrt(sigma2) * np.linalg.solve(L.T, rng.normal(size=len(mu)))
        array[:] = 0
        array[0] = len(cuts)
        array[1 : 1 + len(cuts)] = cuts
        return sigma2, np.repeat(levels, np.diff((0,) + cuts + (64,)))


def make_stepper(engine, output):
    if engine == "native":
        return NativeStepper(output)
    return PythonStepper() if engine == "uniform" else InformedPythonStepper()


def exact_gaussian(H, h, yqy, n, states=None):
    """Exact linear-functional t mixtures, using full segment covariance."""
    states = precision.legal_partitions() if states is None else states
    prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ph = np.r_[0, h.cumsum()]
    shape = 3 + n / 2
    locations = np.empty((len(states), 3))
    scales = np.empty_like(locations)
    logweights = np.empty(len(states))
    curve_means = np.empty((len(states), 64))
    for i, cuts in enumerate(states):
        edges = np.array((0,) + cuts + (64,))
        a, b = edges[:-1], edges[1:]
        A = (
            prefix[b[:, None], b]
            - prefix[a[:, None], b]
            - prefix[b[:, None], a]
            + prefix[a[:, None], a]
            + 4 * np.eye(len(b))
        )
        g = ph[b] - ph[a]
        inverse = np.linalg.inv(A)
        mu = inverse @ g
        rate = 2 + 0.5 * (yqy - g @ mu)
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError("invalid exact inverse-gamma rate")
        logweights[i] = (
            -math.lgamma(len(cuts) + 1)
            - precision._log_partition_count(len(cuts))
            + 0.5 * len(b) * math.log(4)
            - 0.5 * np.linalg.slogdet(A)[1]
            - shape * math.log(rate)
        )
        L = np.zeros((3, len(b)))
        L[0, np.searchsorted(cuts, 16, side="left")] = 1
        L[1, np.searchsorted(cuts, 48, side="left")] = 1
        L[2] = [PHASE[s:e].sum() for s, e in zip(a, b)]
        locations[i] = L @ mu
        variance = (rate / shape) * np.einsum("ij,jk,ik->i", L, inverse, L)
        if np.any(variance < 0):
            raise RuntimeError("negative linear-functional posterior variance")
        scales[i] = np.sqrt(variance)
        curve_means[i] = np.repeat(mu, b - a)
    weights = np.exp(logweights - logweights.max())
    weights /= weights.sum()
    # K=0 makes the phase contrast a point mass at zero; handle degenerate
    # component CDF explicitly instead of replacing it by an artificial scale.
    ci, certificates = exact_mixture_quantiles(
        weights, locations, scales, 2 * shape, return_certificates=True
    )
    probability_positive = np.sum(
        weights[:, None]
        * np.where(
            scales > 0,
            stdtr(2 * shape, locations / np.where(scales > 0, scales, 1)),
            (locations > 0).astype(float),
        ),
        axis=0,
    )
    order = np.argsort(weights)[::-1]
    return {
        "states": states,
        "weights": weights,
        "logweights": logweights,
        "locations": locations,
        "scales": scales,
        "mean": weights @ locations,
        "q025": ci[0],
        "q975": ci[1],
        "quantile_certificates": certificates,
        "probability_positive": probability_positive,
        "p_k": np.array(
            [weights[[len(s) == k for s in states]].sum() for k in range(5)]
        ),
        "pips": np.array(
            [weights[[t in s for s in states]].sum() for t in range(1, 65)]
        ),
        "map_cuts": states[order[0]],
        "map_probability": weights[order[0]],
        "curve_mean": weights @ curve_means,
        "df": 2 * shape,
        "enumerated_partitions": len(states),
    }


def mixture_cdf(weights, locations, scales, df, point, left=False):
    continuous = scales > 0
    atoms = (point > locations) if left else (point >= locations)
    return np.sum(
        weights[:, None]
        * np.where(
            continuous,
            stdtr(df, (point - locations) / np.where(continuous, scales, 1)),
            atoms.astype(float),
        ),
        axis=0,
    )


def exact_mixture_quantiles(
    weights,
    locations,
    scales,
    df,
    probabilities=(0.025, 0.975),
    return_certificates=False,
):
    from scipy.special import stdtrit

    result = []
    certificates = []
    for p in probabilities:
        lo = np.min(locations + scales * stdtrit(df, p), axis=0)
        hi = np.max(locations + scales * stdtrit(df, p), axis=0)
        for _ in range(65):
            mid = (lo + hi) / 2
            cdf = mixture_cdf(weights, locations, scales, df, mid)
            lo = np.where(cdf < p, mid, lo)
            hi = np.where(cdf >= p, mid, hi)
        quantile = (lo + hi) / 2
        # Exact point masses require a generalized inverse, with a CDF jump
        # certificate instead of a zero CDF residual at the requested p.
        for coordinate in range(locations.shape[1]):
            for atom in np.unique(locations[scales[:, coordinate] == 0, coordinate]):
                point = quantile.copy()
                point[coordinate] = atom
                left = mixture_cdf(weights, locations, scales, df, point, left=True)[
                    coordinate
                ]
                right = mixture_cdf(weights, locations, scales, df, point)[coordinate]
                if left <= p <= right:
                    quantile[coordinate] = atom
                    break
        left = mixture_cdf(weights, locations, scales, df, quantile, left=True)
        right = mixture_cdf(weights, locations, scales, df, quantile)
        certificates.append(
            {
                "probability": p,
                "left_cdf": left,
                "cdf": right,
                "is_atom": right > left,
                "max_continuous_cdf_error": float(
                    np.max(np.where(right == left, np.abs(right - p), 0))
                ),
            }
        )
        result.append(quantile)
    return (
        (np.asarray(result), certificates)
        if return_certificates
        else np.asarray(result)
    )


def run_student_chain(
    train, participants, df, iterations, warmup, seed, stepper, initk, innovations=None
):
    if df <= 2 or iterations <= warmup:
        raise ValueError("invalid Student-t run")
    person, trial, z, c, y = train
    structural_seed, rng = chain_streams(seed)
    state = stepper.initialize(initk, structural_seed)
    latent = np.ones(len(y))
    factor = (df - 2) / df
    draws = np.empty((iterations - warmup, DRAW_WIDTH))
    for it in range(iterations):
        H, h, yqy, cache = (
            weighted_stats(*train, participants, latent / factor)
            if innovations is None
            else weighted_sparse_stats(innovations, participants, latent / factor)
        )
        sigma2, delta = stepper.step(H, h, yqy, len(y), state)
        nuisance = draw_nuisance(cache, delta, sigma2, rng)
        residual = (
            y
            - (
                nuisance[0]
                + z * nuisance[1]
                + nuisance[2:][person]
                + c * delta[trial - 1]
            )
            if innovations is None
            else np.asarray(innovations @ np.r_[-nuisance, -delta, 1.0]).ravel()
        )
        latent = rng.gamma((df + 1) / 2, 2 / (df + residual**2 / (sigma2 * factor)))
        if it >= warmup:
            draws[it - warmup] = np.r_[
                state[0],
                delta,
                sigma2,
                latent.mean() / factor,
                np.log(latent).mean(),
                np.mean(latent < 0.25),
            ]
    return draws, state[2]


def chain_summary(draws):
    """K, five cuts, 64 levels, sigma2, mean(w/c), mean(log w), frac(w<.25)."""
    levels = draws[:, :, 6:70]
    cuts = draws[:, :, 1:6]
    k = draws[:, :, 0]
    pip = lambda t: np.any(cuts == t, axis=2).astype(float)
    phase = levels @ PHASE
    series = {
        "K": k,
        "delta16": levels[:, :, 15],
        "delta48": levels[:, :, 47],
        "phase_contrast": phase,
        "PIP32": pip(32),
        "PIP12": pip(12),
        "sigma2": draws[:, :, 70],
        "mean_effective_precision": draws[:, :, 71],
        "mean_log_latent_weight": draws[:, :, 72],
        "latent_weight_lower_tail_fraction": draws[:, :, 73],
    }
    diagnostics = {}
    for name, x in series.items():
        constant = np.ptp(x) == 0
        diagnostics[name] = {
            "rhat": None if constant else float(az.rhat(x, method="rank")),
            "ess_bulk": None if constant else float(az.ess(x, method="bulk")),
            "ess_tail": None
            if constant
            else float(az.ess(x, method="tail", prob=(0.05, 0.95))),
            "mcse_mean": 0 if constant else float(az.mcse(x, method="mean")),
            "constant_chain": constant,
        }
    diagnostics["phase_contrast"]["quantile_mcse"] = [
        0.0 if np.ptp(phase) == 0 else float(az.mcse(phase, method="quantile", prob=p))
        for p in (0.025, 0.975)
    ]
    pips = np.array([pip(t).mean() for t in range(1, 65)])
    key = lambda row: tuple(int(v) for v in row[: 1 + int(row[0])])
    from collections import Counter

    counts = Counter(key(row) for row in draws.reshape(-1, draws.shape[-1]))
    best, mass = counts.most_common(1)[0]
    summary = lambda x: {
        "mean": x.mean(),
        "q025": np.quantile(x, 0.025),
        "q975": np.quantile(x, 0.975),
        "probability_positive": np.mean(x > 0),
        "probability_negative": np.mean(x < 0),
        "probability_zero": np.mean(x == 0),
    }
    return {
        "diagnostics": diagnostics,
        "constant_monitor_interpretation": "No sampled variation; R-hat and ESS are undefined and do not certify posterior mass.",
        "p_k": [float((k == i).mean()) for i in range(5)],
        "pips": pips,
        "map_cuts": best[1:],
        "map_probability": mass / k.size,
        "mean": levels.mean(axis=(0, 1)),
        "delta16": summary(levels[:, :, 15]),
        "delta48": summary(levels[:, :, 47]),
        "phase_contrast": summary(phase),
        "variance_and_latent_weight_summaries": {
            name: summary(series[name])
            for name in (
                "sigma2",
                "mean_effective_precision",
                "mean_log_latent_weight",
                "latent_weight_lower_tail_fraction",
            )
        },
        "rng_derivation": RNG_DERIVATION,
        "needs_escalation": any(
            d["rhat"] is not None
            and (
                not all(math.isfinite(d[v]) for v in ("rhat", "ess_bulk", "ess_tail"))
                or d["rhat"] > 1.01
                or d["ess_bulk"] < 1500
                or d["ess_tail"] < 1500
            )
            for d in diagnostics.values()
        )
        or any(
            not math.isfinite(v) or v > 0.002
            for v in diagnostics["phase_contrast"]["quantile_mcse"]
        ),
    }


def validate_application_pilot(input_dir, output, archive_root=None):
    """Bounded correctness/runtime checks; no substantive robustness inference."""
    records = []
    for group in ("High", "Low"):
        started = time.perf_counter()
        train, block, native, metadata = prepare_group(input_dir, group)
        N = metadata["participants"]
        H, h, yqy, _ = weighted_stats(*train, N, np.ones(len(train[-1])))
        original = precision.formal.collapsed(*train, N)
        errors = {
            name: float(np.max(np.abs(value - ref)))
            for name, value, ref in zip(("H", "h", "yqy"), (H, h, yqy), original)
        }
        if max(errors.values()) > 1e-8:
            raise RuntimeError(f"independent Gaussian statistic mismatch: {errors}")
        archive = archived_statistic_check(archive_root, group, (H, h, yqy))
        conditions = []
        for df, rho in ((4, 0.0), (8, 0.0), (4, 0.2)):
            innovations = (
                None if rho == 0 else innovation_design(train, N, block, native, rho)
            )
            if innovations is not None:
                dense = innovations.toarray()
                q = N + 2
                for a, b in zip(
                    weighted_sparse_stats(innovations, N, np.ones(len(native))),
                    dense_stats(dense[:, q : q + 64], dense[:, :q], dense[:, -1]),
                ):
                    np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)
            before = time.perf_counter()
            draws, stats = run_student_chain(
                train,
                N,
                df,
                100,
                20,
                seed_for("application-pilot", group, df, rho),
                InformedPythonStepper(),
                2,
                innovations,
            )
            np.savez_compressed(
                output / f"{group.lower()}_pilot_t{df}_rho{rho:.1f}.npz",
                draws=draws,
                stats=stats,
            )
            conditions.append(
                {
                    "df": df,
                    "rho": rho,
                    "iterations": 100,
                    "warmup": 20,
                    "seconds": time.perf_counter() - before,
                    "finite": bool(np.isfinite(draws).all()),
                    "stats": stats,
                }
            )
        records.append(
            {
                "group": group,
                "metadata": metadata,
                "independent_gaussian_statistic_max_errors": errors,
                "archive_validation": archive,
                "conditions": conditions,
                "seconds": time.perf_counter() - started,
            }
        )
    result = {
        "purpose": "bounded validation only; pilot draws are not scientific sensitivity results",
        "rng_derivation": RNG_DERIVATION,
        "records": records,
        "source_sha256": {
            f: precision.sha256_file(ROOT / "scripts" / f)
            for f in ("residual_sensitivity.py", "residual_step.c", "kernel.c")
        },
    }
    precision.write_json(output / "application_pilot.json", result)
    return result


STUDENT_CONDITIONS = {
    "all": ((4, 0.0), (8, 0.0), (4, 0.2)),
    "t4-iid": ((4, 0.0),),
    "t8-iid": ((8, 0.0),),
    "t4-ar02": ((4, 0.2),),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            os.environ.get(
                "R61_PC_FLANKER_INPUT", ROOT / "data/external/RawData_Experiment1"
            )
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Optional historical repository root for archived input/exact-posterior parity checks.",
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "validate", "baseline", "exact", "student", "all"),
        default="pilot",
    )
    parser.add_argument("--group", choices=("High", "Low", "both"), default="both")
    parser.add_argument(
        "--student-condition", choices=tuple(STUDENT_CONDITIONS), default="all",
        help="Restrict Student fits to one fixed condition; default runs all three.",
    )
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument(
        "--engine", choices=("native", "python", "uniform"), default="python"
    )
    parser.add_argument("--warmup", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode not in ("pilot", "validate"):
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        expected = {
            ("Gaussian", None, 0.0),
            ("Gaussian", None, 0.2),
            ("Gaussian", None, 0.4),
            ("Student innovations", 4, 0.0),
            ("Student innovations", 8, 0.0),
            ("Student innovations", 4, 0.2),
        }
        observed = {
            (c["family"], c.get("degrees_of_freedom"), c["correlation"])
            for c in protocol["conditions"]
        }
        if observed != expected:
            raise RuntimeError("runner conditions differ from frozen design protocol")
        precision.write_json(
            args.output / f"run_manifest_{args.mode}_{args.group}.json",
            {
                "design_protocol_sha256": precision.sha256_file(args.protocol),
                "design_protocol_path": str(args.protocol),
                "source_sha256": {
                    f: precision.sha256_file(ROOT / "scripts" / f)
                    for f in ("residual_sensitivity.py", "residual_step.c", "kernel.c")
                },
                "engine": args.engine,
                "mode": args.mode,
                "student_condition": args.student_condition,
                "group": args.group,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed_namespace": NAMESPACE,
                "rng_derivation": RNG_DERIVATION,
                "archive_checks": {
                    "requested": args.archive_root is not None,
                    "archive_root": str(args.archive_root)
                    if args.archive_root is not None
                    else None,
                },
            },
        )
    start = time.perf_counter()
    if args.mode == "validate":
        validate_application_pilot(args.input, args.output, args.archive_root)
        return
    if args.mode == "pilot":
        rng = np.random.default_rng(seed_for("pilot"))
        n = 600
        participants = 8
        train = (
            rng.integers(participants, size=n),
            rng.integers(1, 65, size=n),
            rng.normal(size=n),
            rng.choice([-0.5, 0.5], size=n),
            rng.normal(size=n) * 0.2,
        )
        stepper = make_stepper(args.engine, args.output)
        draws, stats = run_student_chain(
            train, participants, 4, 30, 10, seed_for("pilot-chain"), stepper, 2
        )
        precision.write_json(
            args.output / "pilot.json",
            {
                "seconds": time.perf_counter() - start,
                "finite": np.isfinite(draws).all(),
                "draw_shape": draws.shape,
                "stats": stats,
                "kernel_sha256": precision.sha256_file(ROOT / "scripts/kernel.c"),
            },
        )
        return
    groups = ("High", "Low") if args.group == "both" else (args.group,)
    for group in groups:
        train, block, native, metadata = prepare_group(args.input, group)
        N = metadata["participants"]
        base = precision.formal.collapsed(*train, N)
        weighted = weighted_stats(*train, N, np.ones(len(train[-1])))[:3]
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(base, weighted)]
        if max(errors) > 1e-8:
            raise RuntimeError(f"baseline sufficient-statistic mismatch {errors}")
        archive = archived_statistic_check(args.archive_root, group, base)
        precision.write_json(
            args.output / f"{group.lower()}_provenance.json",
            {
                "metadata": metadata,
                "baseline_statistic_max_errors": errors,
                "archive_validation": archive,
                "kernel_sha256": precision.sha256_file(ROOT / "scripts/kernel.c"),
                "namespace": NAMESPACE,
            },
        )
        if args.mode in ("baseline", "exact", "all"):
            X, Z, y = designs(train, N)
            for rho in (0.0,) if args.mode == "baseline" else (0.0, 0.2, 0.4):
                transformed = [
                    ar_whiten(v, train[0], block, native, rho) for v in (X, Z, y)
                ]
                H, h, yqy = dense_stats(*transformed)
                if rho == 0:
                    for a, b in zip(base, (H, h, yqy)):
                        np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)
                exact = exact_gaussian(H, h, yqy, len(y))
                if rho == 0:
                    archive_exact = archived_exact_check(
                        args.archive_root, group, exact
                    )
                    exact["archive_exact_check_requested"] = archive_exact["requested"]
                    if archive_exact["requested"]:
                        exact["archived_baseline_max_p_k_error"] = archive_exact[
                            "maximum_p_k_error"
                        ]
                np.savez_compressed(
                    args.output / f"{group.lower()}_ar{rho:.1f}_exact.npz",
                    **{
                        k: v
                        for k, v in exact.items()
                        if k not in ("states", "quantile_certificates")
                    },
                    states=np.array(
                        [list(s) + [0] * (4 - len(s)) for s in exact["states"]]
                    ),
                    k=np.array(list(map(len, exact["states"]))),
                )
                precision.write_json(
                    args.output / f"{group.lower()}_ar{rho:.1f}_summary.json",
                    {
                        k: v
                        for k, v in exact.items()
                        if k
                        not in (
                            "states",
                            "weights",
                            "logweights",
                            "locations",
                            "scales",
                        )
                    },
                )
        if args.mode in ("student", "all"):
            stepper = make_stepper(args.engine, args.output)
            for df, rho in STUDENT_CONDITIONS[args.student_condition]:
                innovations = (
                    None
                    if rho == 0
                    else innovation_design(train, N, block, native, rho)
                )
                label = f"{group.lower()}_t{df}_rho{rho:.1f}"
                chains = []
                stats = []
                for chain in range(3):
                    draws, st = run_student_chain(
                        train,
                        N,
                        df,
                        args.iterations,
                        args.warmup,
                        seed_for(group, df, rho, chain),
                        stepper,
                        (0, 2, 4)[chain],
                        innovations,
                    )
                    chains.append(draws)
                    stats.append(st)
                chains = np.stack(chains)
                summary = chain_summary(chains)
                summary.update(
                    {
                        "df": df,
                        "rho": rho,
                        "iterations": args.iterations,
                        "warmup": args.warmup,
                        "retained_per_chain": args.iterations - args.warmup,
                        "engine": args.engine,
                        "working_model": "Student-t innovations with AR1 working correlation"
                        if rho
                        else "independent unit-variance Student-t errors",
                    }
                )
                np.savez_compressed(
                    args.output / f"{label}_draws.npz",
                    draws=chains,
                    stats=stats,
                    seeds=np.array(
                        [seed_for(group, df, rho, i) for i in range(3)], dtype=np.uint64
                    ),
                    structural_seeds=np.array(
                        [
                            chain_streams(seed_for(group, df, rho, i))[0]
                            for i in range(3)
                        ],
                        dtype=np.uint64,
                    ),
                    draw_columns=np.array(
                        [
                            "K",
                            *[f"cut{i}" for i in range(5)],
                            *[f"delta{i}" for i in range(1, 65)],
                            "sigma2",
                            "mean_effective_precision",
                            "mean_log_latent_weight",
                            "latent_weight_lower_tail_fraction",
                        ]
                    ),
                )
                precision.write_json(args.output / f"{label}_summary.json", summary)
                # Escalation uses fixed thresholds, never the scientific result.
                if (
                    summary["needs_escalation"]
                    and args.iterations == 15000
                    and args.warmup == 2000
                ):
                    chains = []
                    stats = []
                    for chain in range(3):
                        draws, st = run_student_chain(
                            train,
                            N,
                            df,
                            30000,
                            2000,
                            seed_for(group, df, rho, chain),
                            stepper,
                            (0, 2, 4)[chain],
                            innovations,
                        )
                        chains.append(draws)
                        stats.append(st)
                    chains = np.stack(chains)
                    np.savez_compressed(
                        args.output / f"{label}_escalated_draws.npz",
                        draws=chains,
                        stats=stats,
                    )
                    summary = chain_summary(chains)
                    summary.update(
                        {
                            "df": df,
                            "rho": rho,
                            "iterations": 30000,
                            "warmup": 2000,
                            "engine": args.engine,
                            "rng_derivation": RNG_DERIVATION,
                            "retained_per_chain": 28000,
                        }
                    )
                    precision.write_json(
                        args.output / f"{label}_escalated_summary.json", summary
                    )
    precision.write_json(
        args.output / f"runtime_{args.mode}_{args.group}.json",
        {"seconds": time.perf_counter() - start},
    )


if __name__ == "__main__":
    main()

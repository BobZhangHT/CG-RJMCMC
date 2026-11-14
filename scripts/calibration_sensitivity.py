"""Paired D5 sample-size and individual-timing calibration interventions.

Original N500 data and masks are reconstructed and checked, never regenerated
with a larger-N seed call. New roots are explicit and immutable. No MCMC/native
library is used. Full-support posterior weights and means remain exact finite
computations; only marginal CDF evaluation uses certified active-set bounds.
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import stdtr, stdtrit
from threadpoolctl import threadpool_limits

if __package__:
    from . import coverage_diagnostic as diagnostic
else:
    import coverage_diagnostic as diagnostic

exact = diagnostic.exact
formal = exact.core.f
ROOT = exact.ROOT
PROTOCOL = ROOT / "scripts/config/calibration_sensitivity_protocol.json"
CONDITIONS = ("N1000_shared", "N2000_shared", "N500_jitter2")
METHODS = ("working", "oracle")
CUTS = exact.CUTS
GRID = exact.GRID
LEVELS = np.array([0.13125, 0.03125, 0.13125, 0.03125, 0.13125])
BASE_TRUTH = exact.TRUTH
DISTANCE = np.min(abs(GRID[:, None] - CUTS), axis=1)
STRATA = {
    "all": np.ones(64, bool),
    "near": DISTANCE <= 2,
    "far": DISTANCE >= 3,
    "distance0": DISTANCE == 0,
    "distance1": DISTANCE == 1,
    "distance2": DISTANCE == 2,
    "distance_ge3": DISTANCE >= 3,
}
NAMESPACE = "PLOS-CALIBRATION-SENSITIVITY-20261114"
TAU = 1e-8
MARGIN = 1e-13
CDF_PAD = 2e-14
OMITTED_LIMIT = 1e-15
COARSE_BRACKET_MASS = 1e-4


def labelled_seed(rep, label):
    return formal.deterministic_seed(NAMESPACE, f"R{rep:06d}|{label}")


def array_hash(value):
    return formal.sha256_array(np.ascontiguousarray(value))


def geometry_path(cfg):
    path = ROOT / cfg["design"]["geometry_file"]
    if not path.exists() and path == ROOT / "config/r61_congruency_geometry.csv":
        path = ROOT / "scripts/config/r61_congruency_geometry.csv"
    return path


def configure(identity):
    diagnostic.configure(identity)
    threadpool_limits(limits=1)


def individual_phase(trial, person, offsets):
    return np.sum(trial[:, None] > CUTS[None, :] + offsets[person, None], axis=1)


def population_design(offsets):
    phase = np.sum(
        GRID[None, :, None] > CUTS[None, None, :] + offsets[:, None, None], axis=2
    )
    G = np.stack([(phase == s).mean(0) for s in range(5)], axis=1)
    if not np.array_equal(G.sum(1), np.ones(64)):
        np.testing.assert_allclose(G.sum(1), 1, rtol=0, atol=2e-16)
    return G


def jitter_signal(y, design, offsets):
    phase = individual_phase(design[1], design[0], offsets)
    return y + design[3] * (LEVELS[phase] - BASE_TRUTH[design[1] - 1])


def generic_collapsed(person, z, S, y, participants):
    """Five-column likelihood design, eliminating nuisance via a 2x2 Schur."""
    D = np.column_stack((S, y))
    p = D.shape[1]
    count = np.bincount(person, minlength=participants)
    zsum = np.bincount(person, weights=z, minlength=participants)
    diagonal = 1 + count
    cross = np.vstack((count, zsum))
    top = np.array([[0.25 + len(y), z.sum()], [z.sum(), 1 + z @ z]])
    schur = top - (cross / diagonal) @ cross.T
    B = np.empty((participants + 2, p))
    B[0] = D.sum(0)
    B[1] = z @ D
    B[2:] = 0
    np.add.at(B[2:], person, D)
    beta = np.linalg.solve(schur, B[:2] - (cross / diagonal) @ B[2:])
    u = (B[2:] - cross.T @ beta) / diagonal[:, None]
    C = D.T @ D - B[:2].T @ beta - B[2:].T @ u
    C = (C + C.T) / 2
    return C[:-1, :-1], C[:-1, -1], float(C[-1, -1])


def oracle_posterior(train, participants, offsets, G):
    person, trial, z, c, y = train
    phase = individual_phase(trial, person, offsets)
    S = np.zeros((len(y), 5))
    S[np.arange(len(y)), phase] = c
    H, h, yqy = generic_collapsed(person, z, S, y, participants)
    A = H + 4 * np.eye(5)
    L = np.linalg.cholesky(A)
    inverse = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(5)))
    mu = inverse @ h
    shape = 3 + len(y) / 2
    rate = 2 + 0.5 * (yqy - h @ mu)
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("Invalid oracle posterior rate")
    locations = G @ mu
    variance = (rate / shape) * np.einsum("ij,jk,ik->i", G, inverse, G)
    if np.any(variance <= 0):
        raise ValueError("Invalid full-covariance oracle target")
    return dict(
        weights=np.ones(1),
        locations=locations[None, :],
        scales=np.sqrt(variance)[None, :],
        df=2 * shape,
        logweights=np.zeros(1),
        oracle_mu=mu,
        oracle_inverse=inverse,
        oracle_rate=rate,
        oracle_shape=shape,
        target_matrix=G,
    )


def shared_posterior(case, states):
    """Reviewed frozen batched algebra, retaining every unnormalized log weight."""
    count = len(states)
    shape = 3 + case["n"] / 2
    out = dict(
        locations=np.empty((count, 64)),
        scales=np.empty((count, 64)),
        logweights=np.empty(count),
        df=2 * shape,
    )
    hp = np.r_[0.0, case["h"].cumsum()]
    prefix = np.pad(case["H"].cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sizes = np.array(list(map(len, states)))
    for k in np.unique(sizes):
        ids = np.flatnonzero(sizes == k)
        edges = np.array([(0,) + states[i] + (64,) for i in ids])
        a, b = edges[:, :-1], edges[:, 1:]
        matrices = (
            prefix[b[:, :, None], b[:, None, :]]
            - prefix[a[:, :, None], b[:, None, :]]
            - prefix[b[:, :, None], a[:, None, :]]
            + prefix[a[:, :, None], a[:, None, :]]
        )
        eigen, vectors = np.linalg.eigh((matrices + matrices.transpose(0, 2, 1)) / 2)
        denom = eigen + 4
        if np.any(denom <= 0):
            raise ValueError("Non-positive shared posterior precision")
        g = hp[b] - hp[a]
        projected = np.einsum("bij,bi->bj", vectors, g)
        mu = np.einsum("bij,bj->bi", vectors, projected / denom)
        diagonal = np.einsum("bij,bj->bi", vectors * vectors, 1 / denom)
        rate = 2 + 0.5 * case["yqy"] - 0.5 * np.einsum("bi,bi->b", g, mu)
        if np.any(rate <= 0):
            raise ValueError("Non-positive shared posterior rate")
        phase = np.sum(np.arange(64)[None, :, None] >= edges[:, None, 1:-1], axis=2)
        out["logweights"][ids] = (
            -math.lgamma(int(k) + 1)
            - exact.precision._log_partition_count(int(k))
            + 0.5 * (k + 1) * math.log(4)
            - 0.5 * np.log(denom).sum(1)
            - shape * np.log(rate)
        )
        out["locations"][ids] = np.take_along_axis(mu, phase, axis=1)
        out["scales"][ids] = np.take_along_axis(
            np.sqrt(rate[:, None] / shape * diagonal), phase, axis=1
        )
    scaled = np.exp(out["logweights"] - out["logweights"].max())
    normalizer = math.fsum(scaled)
    out["weights"] = scaled / normalizer
    return out


def conservative_omitted_mass(weights, ids):
    """Sum omitted weights directly, with rounding and underflow allowances.

    The zero-weight allowance is one smallest positive binary64 number per
    omitted component. This bounds a scaled exponential/division rounded to
    zero; the full normalizer is >=1 before normalization. Numerical CDF error
    is covered separately, not confused with omitted posterior mass.
    """
    if not len(ids):
        return 0.0
    values = weights[ids]
    mass = math.fsum(values)
    underflow = np.count_nonzero(values == 0) * np.nextafter(0.0, 1.0)
    return float(np.nextafter(mass * (1 + 8 * np.finfo(float).eps) + underflow, np.inf))


def active_set(weights, limit=OMITTED_LIMIT):
    order = np.argsort(weights)
    n = int(np.searchsorted(np.cumsum(weights[order]), limit, side="right"))
    n = min(n, len(weights) - 1)
    omitted = conservative_omitted_mass(weights, order[:n])
    while n and omitted > limit:
        n -= 1
        omitted = conservative_omitted_mass(weights, order[:n])
    return order[n:], order[:n], omitted


def cdf_sum(x, w, location, scale, df):
    return float(math.fsum(w * stdtr(df, (x - location) / scale)))


def crossing_certificate(q, p, cdf, epsilon=0.0, pad=CDF_PAD):
    middle = cdf(q)
    lower = p - (cdf(q - TAU) + epsilon + pad)
    upper = cdf(q + TAU) - pad - p
    residual = max(abs(middle - p), abs(middle + epsilon - p)) + pad
    return dict(
        accepted=residual < 1e-8 and lower > MARGIN and upper > MARGIN,
        residual_bound=residual,
        lower_margin=lower,
        upper_margin=upper,
    )


def validate_mixture(mixture):
    w, loc, scale = mixture["weights"], mixture["locations"], mixture["scales"]
    if w.ndim != 1 or loc.shape != scale.shape or loc.shape != (len(w), 64):
        raise ValueError("Invalid mixture dimensions")
    if not np.isfinite(w).all() or np.any(w < 0) or abs(math.fsum(w) - 1) > 1e-12:
        raise ValueError("Invalid full weights")
    if not np.isfinite(loc).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid continuous component distributions")
    if not math.isfinite(mixture["df"]) or mixture["df"] <= 0:
        raise ValueError("Invalid degrees of freedom")


def bounded_quantiles(mixture, independent_full_check=False):
    validate_mixture(mixture)
    w, loc, scale, df = (mixture[k] for k in ("weights", "locations", "scales", "df"))
    active, omitted, epsilon = active_set(w)
    coarse, _, coarse_epsilon = active_set(w, COARSE_BRACKET_MASS)
    roots = np.empty((2, 64))
    full_calls = partial_calls = coarse_calls = fallback = 0
    certs = []
    representatives = []
    mapping = []
    for t in range(64):
        representative = next(
            (
                r
                for r in representatives
                if np.array_equal(loc[:, t], loc[:, r])
                and np.array_equal(scale[:, t], scale[:, r])
            ),
            None,
        )
        if representative is None:
            representatives.append(t)
            representative = t
        mapping.append(representative)
    for t in representatives:
        full_cache = {}
        partial_cache = {}
        coarse_cache = {}
        wa, la, sa = w[active], loc[active, t], scale[active, t]
        wc, lc, sc = w[coarse], loc[coarse, t], scale[coarse, t]

        def full(x):
            nonlocal full_calls
            if x not in full_cache:
                full_calls += 1
                full_cache[x] = cdf_sum(x, w, loc[:, t], scale[:, t], df)
            return full_cache[x]

        def partial(x):
            nonlocal partial_calls
            if x not in partial_cache:
                partial_calls += 1
                partial_cache[x] = cdf_sum(x, wa, la, sa, df)
            return partial_cache[x]

        def coarse_cdf(x):
            nonlocal coarse_calls
            if x not in coarse_cache:
                coarse_calls += 1
                coarse_cache[x] = cdf_sum(x, wc, lc, sc, df)
            return coarse_cache[x]

        span = max(
            np.finfo(float).eps,
            float(loc[:, t].max() - loc[:, t].min() + 16 * scale[:, t].max()),
        )
        lo = float(loc[:, t].min() - span)
        hi = float(loc[:, t].max() + span)
        for _ in range(32):
            if (
                partial(lo) + epsilon + CDF_PAD < 0.025
                and partial(hi) - CDF_PAD > 0.975
            ):
                break
            lo -= span
            hi += span
            span *= 2
        else:
            raise ValueError("Mixture bracketing failed")
        for j, p in enumerate((0.025, 0.975)):
            try:
                # Coarse roots only propose a bracket; their omitted mass
                # never enters an accepted endpoint certificate.
                qa = (
                    brentq(
                        lambda x: coarse_cdf(x) - (p - coarse_epsilon),
                        lo,
                        hi,
                        xtol=1e-12,
                        rtol=1e-14,
                    )
                    - 1e-10
                )
                qb = (
                    brentq(lambda x: coarse_cdf(x) - p, lo, hi, xtol=1e-12, rtol=1e-14)
                    + 1e-10
                )
                if not partial(qa) <= p <= partial(qb):
                    qa, qb = lo, hi
                q = brentq(lambda x: partial(x) - p, qa, qb, xtol=1e-12, rtol=1e-14)
                certificate = crossing_certificate(q, p, partial, epsilon)
            except (ValueError, RuntimeError, FloatingPointError):
                certificate = {"accepted": False}
            solver = "bounded_active"
            if not certificate["accepted"]:
                fallback += 1
                solver = "full_fallback"
                q = brentq(lambda x: full(x) - p, lo, hi, xtol=1e-12, rtol=1e-14)
                certificate = crossing_certificate(q, p, full)
            if not certificate["accepted"]:
                raise exact.EndpointCertificationError(
                    "Endpoint crossing unresolved",
                    dict(bin=t + 1, probability=p, endpoint=q, certificate=certificate),
                )
            roots[j, t] = q
            certs.append(dict(bin=t + 1, probability=p, solver=solver, **certificate))
    for t, r in enumerate(mapping):
        roots[:, t] = roots[:, r]
    if not np.isfinite(roots).all() or np.any(roots[0] > roots[1]):
        raise ValueError("Invalid endpoint ordering")
    full_verification = None
    if independent_full_check:
        full_verification_started = time.perf_counter()
        residual = 0.0
        lower = upper = float("inf")
        for t in range(64):
            direct = lambda x: cdf_sum(x, w, loc[:, t], scale[:, t], df)
            for j, p in enumerate((0.025, 0.975)):
                certificate = crossing_certificate(roots[j, t], p, direct)
                full_calls += 3
                if not certificate["accepted"]:
                    raise exact.EndpointCertificationError(
                        "Independent full-CDF check failed",
                        dict(bin=t + 1, probability=p, certificate=certificate),
                    )
                residual = max(residual, certificate["residual_bound"])
                lower = min(lower, certificate["lower_margin"])
                upper = min(upper, certificate["upper_margin"])
        full_verification = dict(
            passed=True,
            endpoints=128,
            residual_max=residual,
            minimum_lower_margin=lower,
            minimum_upper_margin=upper,
            seconds=time.perf_counter() - full_verification_started,
        )
    return roots, dict(
        kind="bounded_active_cdf",
        full_support_count=len(w),
        active_count=len(active),
        omitted_count=len(omitted),
        omitted_mass_upper=epsilon,
        active_weights_renormalized=False,
        numerical_cdf_pad=CDF_PAD,
        location_tolerance=TAU,
        strict_cdf_margin=MARGIN,
        certified=True,
        max_cdf_residual_bound=max(c["residual_bound"] for c in certs),
        minimum_lower_margin=min(c["lower_margin"] for c in certs),
        minimum_upper_margin=min(c["upper_margin"] for c in certs),
        full_cdf_calls=full_calls,
        partial_cdf_calls=partial_calls,
        coarse_cdf_calls=coarse_calls,
        coarse_support_count=len(coarse),
        coarse_omitted_mass_upper=coarse_epsilon,
        full_fallback_endpoints=fallback,
        unique_coordinates=len(representatives),
        coordinate_representatives=mapping,
        independent_full_check=full_verification,
    )


def coverage_decisions(quantiles, mixture, truth):
    """Classify coverage with CDF bounds, resolving every ambiguous case fully."""
    w, loc, scale, df = (mixture[k] for k in ("weights", "locations", "scales", "df"))
    active, _, epsilon = active_set(w)
    below = np.zeros(64, bool)
    above = below.copy()
    full_calls = ambiguous = near = 0
    for t, x in enumerate(truth):
        near_endpoint = bool(np.any(abs(quantiles[:, t] - x) <= TAU))
        near += near_endpoint
        value = cdf_sum(x, w[active], loc[active, t], scale[active, t], df)
        low, high = value - CDF_PAD, value + epsilon + CDF_PAD
        clear_below = low > 0.975 + MARGIN
        clear_above = high < 0.025 - MARGIN
        clear_inside = low > 0.025 + MARGIN and high < 0.975 - MARGIN
        if near_endpoint or not (clear_below or clear_above or clear_inside):
            ambiguous += 1
            full_calls += 1
            value = cdf_sum(x, w, loc[:, t], scale[:, t], df)
            if min(abs(value - 0.025), abs(value - 0.975)) <= MARGIN + CDF_PAD:
                raise exact.EndpointCertificationError(
                    "Truth CDF lacks numerical separation",
                    dict(bin=t + 1, cdf_at_truth=value),
                )
            below[t] = value > 0.975
            above[t] = value < 0.025
        else:
            below[t] = clear_below
            above[t] = clear_above
    return dict(
        coverage=~(below | above), miscoverage_below=below, miscoverage_above=above
    ), dict(
        near_endpoint_count=near,
        full_cdf_resolutions=ambiguous,
        full_cdf_calls=full_calls,
        tie_handling="Unresolved within margin+pad of a threshold blocks completion",
    )


def oracle_quantiles(mixture):
    q = (
        mixture["locations"]
        + stdtrit(mixture["df"], np.array([0.025, 0.975]))[:, None] * mixture["scales"]
    )
    checks = []
    for t in range(64):
        cdf = lambda x: cdf_sum(
            x,
            mixture["weights"],
            mixture["locations"][:, t],
            mixture["scales"][:, t],
            mixture["df"],
        )
        for j, p in enumerate((0.025, 0.975)):
            cert = crossing_certificate(q[j, t], p, cdf)
            if not cert["accepted"]:
                raise exact.EndpointCertificationError(
                    "Analytic oracle endpoint unresolved",
                    dict(bin=t + 1, certificate=cert),
                )
            checks.append(cert)
    return q, dict(
        kind="analytic_full_covariance_oracle",
        full_support_count=1,
        full_cdf_calls=384,
        certified=True,
        max_cdf_residual_bound=max(c["residual_bound"] for c in checks),
        minimum_lower_margin=min(c["lower_margin"] for c in checks),
        minimum_upper_margin=min(c["upper_margin"] for c in checks),
        location_tolerance=TAU,
        strict_cdf_margin=MARGIN,
        numerical_cdf_pad=CDF_PAD,
    )


def design_hashes(design, y, train, test):
    return {
        **{
            name: array_hash(a)
            for name, a in zip(("person", "trial", "z", "contrast"), design)
        },
        "y": array_hash(y),
        "train": array_hash(np.packbits(train)),
        "test": array_hash(np.packbits(test)),
    }


def paired_datasets(rep, cfg, case, base, original_inputs, original_validator=None):
    """Construct one full extension, nested prefixes, and a signal-only jitter."""
    design = formal.design_for_n(base, 500)
    y, truth, _ = formal.generate_outcome(
        cfg, case, design, original_inputs["data_seed"]
    )
    namespace = exact.d5_namespace(cfg, case)
    label = f"main|D5|R{rep:06d}"
    test = formal.outcome_blind_holdout(
        design[0], design[1], 13, f"{namespace}|{label}"
    )
    train = formal.mcar_training_mask(~test, namespace, label, 0.0)
    for key, value in (
        ("data_hash", array_hash(y)),
        ("holdout_hash", array_hash(np.packbits(test))),
        ("training_mask_hash", array_hash(np.packbits(train))),
    ):
        if value != original_inputs[key]:
            raise ValueError(f"Original N500 reconstruction differs: {key}")
    if train.sum() != 25500 or test.sum() != 6500:
        raise ValueError("Original mask counts differ")
    np.testing.assert_array_equal(truth, BASE_TRUTH)
    original_rng = np.random.default_rng(original_inputs["data_seed"])
    original_intercepts = original_rng.normal(0, 0.4, 500)
    original_errors = original_rng.normal(0, 0.4, len(y))
    replay = (
        cfg["design"]["beta0"]
        + cfg["design"]["beta1"] * design[2]
        + original_intercepts[design[0]]
        + design[3] * truth[design[1] - 1]
    )
    replay += original_errors
    np.testing.assert_array_equal(replay, y)
    original_item = dict(
        design=design,
        y=y,
        train=train,
        test=test,
        offsets=np.zeros(500, int),
        G=population_design(np.zeros(500, int)),
        truth=truth,
    )
    if original_validator is not None:
        original_validator(original_item)
    labels = (
        "extension_intercepts",
        "extension_errors",
        "extension_holdout",
        "jitter_permutation",
    )
    seeds = {label: labelled_seed(rep, label) for label in labels}
    if len(set(seeds.values()) | {original_inputs["data_seed"]}) != 5:
        raise ValueError("Seed domains collide")
    full_design = formal.design_for_n(base, 2000)
    for original, full in zip(design, full_design):
        np.testing.assert_array_equal(original, full[: 500 * 64])
    extension = tuple(a[500 * 64 :].copy() for a in full_design)
    u = np.random.default_rng(seeds["extension_intercepts"]).normal(0, 0.4, 1500)
    noise = np.random.default_rng(seeds["extension_errors"]).normal(0, 0.4, 1500 * 64)
    extension_y = (
        cfg["design"]["beta0"]
        + cfg["design"]["beta1"] * extension[2]
        + u[extension[0] - 500]
        + extension[3] * truth[extension[1] - 1]
    )
    extension_y += noise
    extension_test = formal.outcome_blind_holdout(
        extension[0],
        extension[1],
        13,
        f"{NAMESPACE}|R{rep:06d}|extension_holdout|{seeds['extension_holdout']}",
    )
    full_y = np.r_[y, extension_y]
    full_test = np.r_[test, extension_test]
    full_train = np.r_[train, ~extension_test]
    data = {"N500_shared": original_item}
    for n in (1000, 2000):
        end = n * 64
        d = tuple(a[:end] for a in full_design)
        data[f"N{n}_shared"] = dict(
            design=d,
            y=full_y[:end],
            train=full_train[:end],
            test=full_test[:end],
            offsets=np.zeros(n, int),
            G=population_design(np.zeros(n, int)),
            truth=truth,
        )
        if full_train[:end].sum() != n * 51 or full_test[:end].sum() != n * 13:
            raise ValueError("Extension mask counts differ")
        for a, b in zip(design, d):
            np.testing.assert_array_equal(a, b[: 500 * 64])
        np.testing.assert_array_equal(y, full_y[: 500 * 64])
        np.testing.assert_array_equal(test, full_test[: 500 * 64])
        np.testing.assert_array_equal(train, full_train[: 500 * 64])
    offsets = np.random.default_rng(seeds["jitter_permutation"]).permutation(
        np.repeat(np.arange(-2, 3), 100)
    )
    if not np.array_equal(np.bincount(offsets + 2), np.full(5, 100)):
        raise ValueError("Jitter is not balanced")
    jitter_y = jitter_signal(y, design, offsets)
    G = population_design(offsets)
    population_truth = G @ LEVELS
    shifted = np.array(
        [formal.curve_from(CUTS + j, LEVELS) for j in range(-2, 3)]
    ).mean(0)
    np.testing.assert_allclose(population_truth, shifted, rtol=0, atol=3e-17)
    data["N500_jitter2"] = dict(
        design=design,
        y=jitter_y,
        train=train,
        test=test,
        offsets=offsets,
        G=G,
        truth=population_truth,
    )
    prefix1 = design_hashes(
        data["N1000_shared"]["design"],
        full_y[:64000],
        full_train[:64000],
        full_test[:64000],
    )
    prefix2 = design_hashes(
        tuple(a[:64000] for a in data["N2000_shared"]["design"]),
        full_y[:64000],
        full_train[:64000],
        full_test[:64000],
    )
    if prefix1 != prefix2:
        raise ValueError("Nested N1000 prefix differs")
    proof = dict(
        seeds=seeds,
        original_prefix=design_hashes(design, y, train, test),
        nested_N1000=prefix1,
        nested_N2000_prefix1000=prefix2,
        original_intercepts_hash=array_hash(original_intercepts),
        original_errors_hash=array_hash(original_errors),
        extension_intercepts_hash=array_hash(u),
        extension_errors_hash=array_hash(noise),
        extension_first500_intercepts_hash=array_hash(u[:500]),
        extension_first500_errors_hash=array_hash(noise[: 500 * 64]),
        extension_first500_y_hash=array_hash(extension_y[: 500 * 64]),
        extension_all_y_hash=array_hash(extension_y),
        offset_assignment_hash=array_hash(offsets),
        offset_counts=np.bincount(offsets + 2),
        conditions={
            name: dict(
                array_hashes=design_hashes(
                    item["design"], item["y"], item["train"], item["test"]
                ),
                truth_hash=array_hash(item["truth"]),
                target_matrix_hash=array_hash(item["G"]),
                participants=len(item["offsets"]),
                n_train=int(item["train"].sum()),
                n_test=int(item["test"].sum()),
            )
            for name, item in data.items()
        },
    )
    return data, proof


def posterior_case(item):
    train = formal.pack(item["design"], item["y"], item["train"])
    H, h, yqy = formal.collapsed(*train, len(item["offsets"]))
    return train, dict(H=H, h=h, yqy=yqy, n=len(train[-1]))


def validate_baseline(rep, identity, base_case, base_item, states):
    root = Path(identity["roots"]["baseline"])
    old_identity = exact.read_json(root / "run_identity.json")
    saved = diagnostic.validate_result(root, rep, old_identity)
    if saved is None:
        raise ValueError("Missing immutable baseline")
    if saved["inputs"] != identity["inputs"][str(rep)]:
        raise ValueError("Baseline data identity differs")
    for path in diagnostic.paths(root, rep):
        if exact.core.sha(path) != identity["baseline_sha256"][str(path.resolve())]:
            raise ValueError("Baseline result changed")
    cached = exact.case_inputs(rep)[1]
    statistic_errors = {
        key: float(np.max(abs(base_case[key] - cached[key])))
        for key in ("H", "h", "yqy")
    }
    if max(statistic_errors.values()) > 1e-8:
        raise ValueError(f"Original statistics differ {statistic_errors}")
    mixture = shared_posterior(base_case, states)
    train = formal.pack(base_item["design"], base_item["y"], base_item["train"])
    oracle = oracle_posterior(train, 500, base_item["offsets"], base_item["G"])
    oracle_q, _ = oracle_quantiles(oracle)
    with np.load(diagnostic.paths(root, rep)[1], allow_pickle=False) as z:
        errors = dict(
            unknown_weights=float(
                np.max(abs(mixture["weights"] - z["original__unknown__weights"]))
            ),
            unknown_mean=float(
                np.max(
                    abs(
                        mixture["weights"] @ mixture["locations"]
                        - z["original__unknown__mean"]
                    )
                )
            ),
            oracle_quantiles=float(
                np.max(abs(oracle_q - z["original__oracle__quantiles"]))
            ),
            oracle_mean=float(
                np.max(abs(oracle["locations"][0] - z["original__oracle__mean"]))
            ),
        )
    if (
        errors["unknown_weights"] > 1e-10
        or max(errors[k] for k in errors if k != "unknown_weights") > 1e-8
    ):
        raise ValueError(f"Original posterior parity failed {errors}")
    return dict(
        statistic_max_errors=statistic_errors,
        posterior_max_errors=errors,
        baseline_record_sha256=exact.core.sha(diagnostic.paths(root, rep)[0]),
    )


def source_hashes(cfg, protocol):
    paths = [
        Path(__file__),
        PROTOCOL if protocol is None else protocol,
        ROOT / "scripts/coverage_diagnostic.py",
        ROOT / "scripts/exact_d5_500.py",
        ROOT / "scripts/precision.py",
        ROOT / "scripts/simulation.py",
        geometry_path(cfg),
        exact.core.SOURCE / "resolved_config.json",
    ]
    return {str(p.resolve()): exact.core.sha(p) for p in paths}


def make_identity(args, reps):
    roots = {
        key: str(getattr(args, name).resolve())
        for key, name in (
            ("simulation", "simulation_results"),
            ("statistics", "statistics_root"),
            ("records", "records_root"),
            ("baseline", "baseline_diagnostic_root"),
        )
    }
    configure(dict(roots=roots))
    cfg = exact.read_json(exact.core.SOURCE / "resolved_config.json")["config"]
    if exact.d5_namespace(cfg, {"dgp": "D5"}) != "R61-D5-BALANCED-v1":
        raise ValueError("Original namespace differs")
    for key, value in (
        ("participant_sd", 0.4),
        ("residual_sd", 0.4),
        ("beta0", 0.0),
        ("beta1", 0.05),
        ("holdout_trials_per_participant", 13),
    ):
        if cfg["design"][key] != value:
            raise ValueError(f"Original DGP changed: {key}")
    protocol = exact.read_json(args.protocol)
    if (
        protocol["conditions"] != list(CONDITIONS)
        or protocol["namespace"] != NAMESPACE
        or protocol["replications"] != 500
        or protocol["numerical"]["numerical_cdf_pad"] != CDF_PAD
    ):
        raise ValueError("Protocol differs from this runner")
    required = {
        "partitions": 27931,
        "active_omitted_mass_max": OMITTED_LIMIT,
        "active_weights_renormalized": False,
        "location_tolerance": TAU,
        "strict_cdf_margin": MARGIN,
        "cdf_residual_max": 1e-8,
        "brent_xtol": 1e-12,
        "brent_rtol": 1e-14,
        "independent_full_CDF_replications": [1, 250, 500],
        "coarse_bracket_mass": COARSE_BRACKET_MASS,
    }
    if (
        any(protocol["numerical"][key] != value for key, value in required.items())
        or protocol["cuts"] != CUTS.tolist()
        or protocol["levels"] != LEVELS.tolist()
    ):
        raise ValueError("Numerical/truth protocol differs")
    if protocol["priors"] != {
        "nuisance_covariance_sigma_units": "diag(4,1,I_N)",
        "phase_precision": 4,
        "inverse_gamma_shape": 3,
        "inverse_gamma_rate": 2,
        "count_prior": "1/K!",
        "conditional_partition_prior": "uniform legal",
        "Kmax": 4,
        "minimum_segment": 8,
    }:
        raise ValueError("Prior protocol differs")
    archive = exact.read_json(Path(roots["baseline"]) / "run_identity.json")
    if (
        exact.digest({k: v for k, v in archive.items() if k != "identity_sha256"})
        != archive["identity_sha256"]
    ):
        raise ValueError("Baseline identity digest invalid")
    # Historical hashes identify the code used for the completed archive; they
    # are not a current-source lock after public maintenance. Bind the actual
    # current dependencies separately, retain every historical discrepancy,
    # hash the immutable baseline files, and recompute original-data posterior
    # parity before constructing a new-data result.
    historical_source_status = {
        path: dict(
            historical_sha256=expected,
            current_sha256=exact.core.sha(Path(path)) if Path(path).is_file() else None,
        )
        for path, expected in archive["source_sha256"].items()
    }
    for item in historical_source_status.values():
        item["current_matches_historical"] = (
            item["current_sha256"] == item["historical_sha256"]
        )
    inputs = {}
    baseline_hashes = {
        str((Path(roots["baseline"]) / "run_identity.json").resolve()): exact.core.sha(
            Path(roots["baseline"]) / "run_identity.json"
        )
    }
    for rep in reps:
        inputs[str(rep)] = exact.case_inputs(rep)[0]
        record = diagnostic.validate_result(Path(roots["baseline"]), rep, archive)
        if record is None or record["inputs"] != inputs[str(rep)]:
            raise ValueError("Baseline pairing is unavailable")
        for path in diagnostic.paths(Path(roots["baseline"]), rep):
            baseline_hashes[str(path.resolve())] = exact.core.sha(path)
    identity = dict(
        schema=1,
        mode="pilot" if args.pilot else "formal500",
        roots=roots,
        replications=reps,
        inputs=inputs,
        protocol=protocol,
        protocol_path=str(args.protocol.resolve()),
        source_sha256=source_hashes(cfg, args.protocol),
        baseline_identity_sha256=archive["identity_sha256"],
        baseline_historical_source_status=historical_source_status,
        baseline_sha256=baseline_hashes,
        versions=dict(
            python=sys.version, numpy=np.__version__, scipy=exact.scipy.__version__
        ),
    )
    identity["identity_sha256"] = exact.digest(identity)
    return identity


def result_paths(output, rep):
    stem = Path(output) / "replicates" / f"rep_{rep:06d}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def validate_certificate(cert, support, rep=None):
    if not cert.get("certified") or cert["full_support_count"] != support:
        raise ValueError("Uncertified support")
    if not all(
        math.isfinite(cert[k])
        for k in (
            "max_cdf_residual_bound",
            "minimum_lower_margin",
            "minimum_upper_margin",
            "location_tolerance",
            "strict_cdf_margin",
            "numerical_cdf_pad",
        )
    ):
        raise ValueError("Nonfinite numerical certificate")
    if (
        cert["max_cdf_residual_bound"] >= 1e-8
        or cert["minimum_lower_margin"] <= MARGIN
        or cert["minimum_upper_margin"] <= MARGIN
    ):
        raise ValueError("Invalid CDF certificate")
    if (
        cert["location_tolerance"] != TAU
        or cert["strict_cdf_margin"] != MARGIN
        or cert["numerical_cdf_pad"] != CDF_PAD
    ):
        raise ValueError("Certificate tolerances differ")
    if support > 1:
        if (
            cert["active_count"] + cert["omitted_count"] != support
            or not 1 <= cert["active_count"] <= support
        ):
            raise ValueError("Active and omitted support counts differ")
        if (
            cert["omitted_mass_upper"] > OMITTED_LIMIT
            or cert["active_weights_renormalized"]
        ):
            raise ValueError("Invalid active bound")
        if rep in (1, 250, 500) and not (cert.get("independent_full_check") or {}).get(
            "passed"
        ):
            raise ValueError("Missing mandatory independent full-CDF check")


def validate_result(output, rep, identity):
    jp, npz = result_paths(output, rep)
    if not jp.exists():
        return None
    record = exact.read_json(jp)
    if (
        record.get("status") != "complete"
        or record["identity_sha256"] != identity["identity_sha256"]
        or record["inputs"] != identity["inputs"][str(rep)]
    ):
        raise ValueError("Resume identity/status mismatch")
    if not npz.exists() or exact.core.sha(npz) != record["npz_sha256"]:
        raise ValueError("Result arrays missing/corrupt")
    with np.load(npz, allow_pickle=False) as z:
        if z["identity_sha256"].item() != identity["identity_sha256"]:
            raise ValueError("NPZ identity differs")
        for condition in CONDITIONS:
            truth = z[condition + "__truth"]
            if truth.shape != (64,) or not np.isfinite(truth).all():
                raise ValueError("Invalid truth")
            for method in METHODS:
                key = condition + "__" + method
                support = 27931 if method == "working" else 1
                validate_certificate(record["certificates"][key], support, rep)
                q = z[key + "__quantiles"]
                mean = z[key + "__mean"]
                if (
                    q.shape != (2, 64)
                    or mean.shape != (64,)
                    or not np.isfinite(q).all()
                    or not np.isfinite(mean).all()
                    or np.any(q[0] > q[1])
                ):
                    raise ValueError("Invalid intervals/means")
                weights = z[key + "__weights"]
                if (
                    weights.shape != (support,)
                    or np.any(weights < 0)
                    or not np.isfinite(weights).all()
                    or abs(math.fsum(weights) - 1) > 1e-12
                ):
                    raise ValueError("Invalid saved full weights")
                if method == "working" and (
                    z[key + "__logweights"].shape != (support,)
                    or not np.isfinite(z[key + "__logweights"]).all()
                ):
                    raise ValueError("Missing full log targets")
                decisions = [
                    z[key + "__" + name]
                    for name in ("coverage", "miscoverage_below", "miscoverage_above")
                ]
                if any(
                    v.shape != (64,) or v.dtype != bool for v in decisions
                ) or not np.array_equal(decisions[0], ~(decisions[1] | decisions[2])):
                    raise ValueError("Invalid coverage classification")
    return record


def compute_case(rep, output, identity):
    configure(identity)
    started = time.perf_counter()
    exact.verify_hashes(identity["source_sha256"])
    inputs, _cached, _, _ = exact.case_inputs(rep)
    if inputs != identity["inputs"][str(rep)]:
        raise ValueError("Original inputs changed")
    cfg = exact.read_json(exact.core.SOURCE / "resolved_config.json")["config"]
    old = exact.read_json(exact.ORIGINAL / f"rep_{rep:06d}" / "summary.json")
    base, _ = formal.load_frozen_geometry(geometry_path(cfg))
    states = exact.precision.legal_partitions()
    counts = np.array(list(map(len, states)))
    if len(states) != 27931 or np.bincount(counts).tolist() != [
        1,
        49,
        861,
        6545,
        20475,
    ]:
        raise ValueError("Partition support changed")
    baseline = {}

    def baseline_guard(original_item):
        _, base_case = posterior_case(original_item)
        baseline.update(
            validate_baseline(rep, identity, base_case, original_item, states)
        )

    datasets, proof = paired_datasets(
        rep, cfg, old["case"], base, inputs, baseline_guard
    )
    arrays = dict(
        identity_sha256=np.array(identity["identity_sha256"]),
        near=STRATA["near"],
        distance=DISTANCE,
        offsets=datasets["N500_jitter2"]["offsets"],
    )
    certificates = {}
    coverage_checks = {}
    timings = {}
    posterior = {}
    for condition in CONDITIONS:
        item = datasets[condition]
        train, case = posterior_case(item)
        arrays[condition + "__truth"] = item["truth"]
        arrays[condition + "__G"] = item["G"]
        for method in METHODS:
            key = condition + "__" + method
            before = time.perf_counter()
            try:
                if method == "working":
                    mixture = shared_posterior(case, states)
                    q, cert = bounded_quantiles(mixture, rep in (1, 250, 500))
                else:
                    mixture = oracle_posterior(
                        train, len(item["offsets"]), item["offsets"], item["G"]
                    )
                    q, cert = oracle_quantiles(mixture)
                decisions, coverage_check = coverage_decisions(
                    q, mixture, item["truth"]
                )
            except exact.EndpointCertificationError as error:
                exact.atomic_json(
                    Path(output) / "unresolved" / f"rep_{rep:06d}_{key}.json",
                    dict(
                        status="unresolved",
                        replication=rep,
                        condition=condition,
                        method=method,
                        identity_sha256=identity["identity_sha256"],
                        details=error.details,
                        excluded_from_planned_set=False,
                    ),
                )
                raise
            validate_certificate(cert, len(mixture["weights"]), rep)
            certificates[key] = cert
            coverage_checks[key] = coverage_check
            timings[key] = time.perf_counter() - before
            arrays[key + "__quantiles"] = q
            arrays[key + "__mean"] = mixture["weights"] @ mixture["locations"]
            arrays[key + "__weights"] = mixture["weights"]
            for name, value in decisions.items():
                arrays[key + "__" + name] = value
            if method == "working":
                arrays[key + "__logweights"] = mixture["logweights"]
                arrays[key + "__p_k"] = np.bincount(
                    counts, weights=mixture["weights"], minlength=5
                )
                arrays[key + "__nominal_center_pips"] = np.array(
                    [
                        math.fsum(mixture["weights"][[int(t) in s for s in states]])
                        for t in CUTS
                    ]
                )
                posterior[key] = dict(
                    mean_K=float(np.arange(5) @ arrays[key + "__p_k"]),
                    nominal_center_PIP_is_descriptive_only=condition == "N500_jitter2",
                )
            else:
                for name in (
                    "oracle_mu",
                    "oracle_inverse",
                    "oracle_rate",
                    "oracle_shape",
                ):
                    arrays[key + "__" + name] = mixture[name]
    exact.verify_hashes(identity["source_sha256"])
    exact.verify_hashes(inputs["input_sha256"])
    jp, npz = result_paths(output, rep)
    exact.atomic_npz(npz, **arrays)
    record = dict(
        status="complete",
        replication=rep,
        identity_sha256=identity["identity_sha256"],
        inputs=inputs,
        npz_sha256=exact.core.sha(npz),
        generated_provenance=proof,
        baseline_validation=baseline,
        certificates=certificates,
        coverage_classification=coverage_checks,
        posterior=posterior,
        seconds=dict(methods=timings, total=time.perf_counter() - started),
        peak_working_set_bytes=formal.process_peak_working_set(),
        new_MCMC=0,
    )
    exact.atomic_json(jp, record)
    return record


def replication_metrics(q, means, truths, decisions, mask):
    errors = means[:, mask] - truths[:, mask]
    return dict(
        coverage=decisions["coverage"][:, mask].mean(1),
        width=(q[:, 1, mask] - q[:, 0, mask]).mean(1),
        signed_bias=errors.mean(1),
        RMSE=np.square(errors).mean(1),
        miscoverage_below=decisions["miscoverage_below"][:, mask].mean(1),
        miscoverage_above=decisions["miscoverage_above"][:, mask].mean(1),
    )


def estimate(values, metric):
    value = float(np.mean(values))
    if metric == "RMSE":
        root = math.sqrt(value)
        return root, (values - value) / (2 * root) if root else np.zeros_like(values)
    return value, values - value


def summarize_samples(samples, reps):
    """Dataset-unit MCSE, with paired differences and pooled-RMSE influences."""
    stats = {}
    rows = []
    paired = []
    per_dataset = []
    for (condition, method), item in samples.items():
        for stratum, mask in STRATA.items():
            values = replication_metrics(
                item["quantiles"], item["mean"], item["truth"], item["decisions"], mask
            )
            for metric, vector in values.items():
                value, influence = estimate(vector, metric)
                stats[condition, method, stratum, metric] = (value, influence)
                rows.append(
                    dict(
                        condition=condition,
                        method=method,
                        stratum=stratum,
                        metric=metric,
                        estimate=value,
                        mcse=exact.mcse(influence),
                        replications=len(reps),
                        positions=int(mask.sum()),
                        distance_reference="nominal transition centers"
                        if condition == "N500_jitter2"
                        else "shared true boundaries",
                    )
                )
                per_dataset.extend(
                    dict(
                        replication=rep,
                        condition=condition,
                        method=method,
                        stratum=stratum,
                        metric="MSE" if metric == "RMSE" else metric,
                        value=float(v),
                    )
                    for rep, v in zip(reps, vector)
                )

    def contrast(left, right, stratum, metric, right_stratum=None):
        right_stratum = stratum if right_stratum is None else right_stratum
        a, ia = stats[(*left, stratum, metric)]
        b, ib = stats[(*right, right_stratum, metric)]
        paired.append(
            dict(
                left="__".join(left),
                right="__".join(right),
                stratum=stratum
                if stratum == right_stratum
                else stratum + "_minus_" + right_stratum,
                metric=metric,
                difference=a - b,
                paired_mcse=exact.mcse(ia - ib),
                pairs=len(reps),
            )
        )

    for method in METHODS:
        for left, right in (
            ("N1000_shared", "N500_shared"),
            ("N2000_shared", "N500_shared"),
            ("N2000_shared", "N1000_shared"),
            ("N500_jitter2", "N500_shared"),
        ):
            for stratum in STRATA:
                for metric in (
                    "coverage",
                    "width",
                    "signed_bias",
                    "RMSE",
                    "miscoverage_below",
                    "miscoverage_above",
                ):
                    contrast((left, method), (right, method), stratum, metric)
    for condition in ("N500_shared", *CONDITIONS):
        for stratum in STRATA:
            for metric in (
                "coverage",
                "width",
                "signed_bias",
                "RMSE",
                "miscoverage_below",
                "miscoverage_above",
            ):
                contrast((condition, "oracle"), (condition, "working"), stratum, metric)
        for method in METHODS:
            for metric in (
                "coverage",
                "width",
                "signed_bias",
                "RMSE",
                "miscoverage_below",
                "miscoverage_above",
            ):
                contrast(
                    (condition, method), (condition, method), "near", metric, "far"
                )
    return rows, paired, per_dataset


def aggregate(output, identity):
    reps = identity["replications"]
    if identity["mode"] == "formal500" and reps != list(range(1, 501)):
        raise ValueError("Formal aggregation requires all 500 replications")
    keys = [
        (condition, method)
        for condition in ("N500_shared", *CONDITIONS)
        for method in METHODS
    ]
    samples = {
        key: dict(
            quantiles=[],
            mean=[],
            truth=[],
            decisions={
                name: []
                for name in ("coverage", "miscoverage_below", "miscoverage_above")
            },
        )
        for key in keys
    }
    records = []
    root = Path(identity["roots"]["baseline"])
    baseline_identity = exact.read_json(root / "run_identity.json")
    for rep in reps:
        record = validate_result(output, rep, identity)
        if record is None:
            raise ValueError(f"Incomplete replication {rep}; no exclusion is permitted")
        records.append(record)
        if diagnostic.validate_result(root, rep, baseline_identity) is None:
            raise ValueError("Baseline is incomplete")
        with (
            np.load(result_paths(output, rep)[1], allow_pickle=False) as z,
            np.load(diagnostic.paths(root, rep)[1], allow_pickle=False) as baseline,
        ):
            for condition, method in keys:
                item = samples[condition, method]
                source = baseline if condition == "N500_shared" else z
                prefix = (
                    "original__" + ("unknown" if method == "working" else "oracle")
                    if condition == "N500_shared"
                    else condition + "__" + method
                )
                item["quantiles"].append(source[prefix + "__quantiles"])
                item["mean"].append(source[prefix + "__mean"])
                item["truth"].append(
                    BASE_TRUTH
                    if condition == "N500_shared"
                    else source[condition + "__truth"]
                )
                for name in item["decisions"]:
                    item["decisions"][name].append(source[prefix + "__" + name])
    for item in samples.values():
        for name in ("quantiles", "mean", "truth"):
            item[name] = np.array(item[name])
        item["decisions"] = {k: np.array(v) for k, v in item["decisions"].items()}
    rows, paired, per_dataset = summarize_samples(samples, reps)
    for filename, data in (
        ("strata.csv", rows),
        (
            "distance_strata.csv",
            [r for r in rows if r["stratum"].startswith("distance")],
        ),
        ("paired_contrasts.csv", paired),
        (
            "near_far_contrasts.csv",
            [r for r in paired if r["stratum"] == "near_minus_far"],
        ),
        ("per_dataset.csv", per_dataset),
    ):
        exact.write_csv(Path(output) / filename, data)
    summary = dict(
        identity_sha256=identity["identity_sha256"],
        mode=identity["mode"],
        replications=len(reps),
        strata=rows,
        paired_contrasts=paired,
        all_planned_cases_complete=True,
        all_endpoint_certificates_passed=True,
        mean_replication_seconds=float(
            np.mean([r["seconds"]["total"] for r in records])
        ),
        max_peak_working_set_bytes=max(
            r["peak_working_set_bytes"] or 0 for r in records
        ),
        new_MCMC=0,
        numerical_work=dict(
            full_cdf_calls=sum(
                c.get("full_cdf_calls", 0)
                for r in records
                for c in r["certificates"].values()
            ),
            active_cdf_calls=sum(
                c.get("partial_cdf_calls", 0)
                for r in records
                for c in r["certificates"].values()
            ),
            full_fallback_endpoints=sum(
                c.get("full_fallback_endpoints", 0)
                for r in records
                for c in r["certificates"].values()
            ),
            coverage_full_cdf_resolutions=sum(
                c["full_cdf_resolutions"]
                for r in records
                for c in r["coverage_classification"].values()
            ),
        ),
        interpretation=identity["protocol"]["interpretation"],
    )
    exact.atomic_json(Path(output) / "summary.json", summary)
    return summary


def verify_identity(identity):
    if (
        exact.digest({k: v for k, v in identity.items() if k != "identity_sha256"})
        != identity["identity_sha256"]
    ):
        raise ValueError("Run identity digest invalid")
    exact.verify_hashes(identity["source_sha256"])
    exact.verify_hashes(identity["baseline_sha256"])
    for item in identity["inputs"].values():
        exact.verify_hashes(item["input_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("pilot", "full500", "summarize"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument(
        "--replications",
        default="1",
        help="Pilot only; formal execution is always 1..500",
    )
    for name in (
        "simulation-results",
        "statistics-root",
        "records-root",
        "baseline-diagnostic-root",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8, one numerical thread each")
    if args.summarize:
        identity = exact.read_json(output / "run_identity.json")
        configure(identity)
        verify_identity(identity)
        aggregate(output, identity)
        return
    for name in (
        "simulation_results",
        "statistics_root",
        "records_root",
        "baseline_diagnostic_root",
    ):
        path = getattr(args, name)
        if path is None:
            parser.error("--" + name.replace("_", "-") + " is required for execution")
        if output == path.resolve() or output.is_relative_to(path.resolve()):
            parser.error("Output must be outside immutable input roots")
    reps = (
        sorted({int(v) for v in args.replications.split(",")})
        if args.pilot
        else list(range(1, 501))
    )
    if not reps or any(rep not in range(1, 501) for rep in reps):
        parser.error("Pilot replications must lie in 1..500")
    identity = make_identity(args, reps)
    output.mkdir(parents=True, exist_ok=True)
    exact.bind_identity(output, identity)
    frozen = output / "frozen_protocol.json"
    if frozen.exists() and exact.read_json(frozen) != identity["protocol"]:
        raise ValueError("Frozen protocol differs")
    exact.atomic_json(frozen, identity["protocol"])
    states = exact.precision.legal_partitions()
    cuts = np.zeros((len(states), 4), np.int16)
    for i, state in enumerate(states):
        cuts[i, : len(state)] = state
    order = output / "partition_order.npz"
    if order.exists():
        with np.load(order, allow_pickle=False) as z:
            if not np.array_equal(z["cuts"], cuts) or not np.array_equal(
                z["k"], list(map(len, states))
            ):
                raise ValueError("Partition order differs")
    else:
        exact.atomic_npz(order, cuts=cuts, k=np.array(list(map(len, states))))
    todo = [rep for rep in reps if validate_result(output, rep, identity) is None]
    if args.workers == 1:
        for rep in todo:
            item = compute_case(rep, output, identity)
            print(
                exact.precision.native(
                    dict(
                        replication=rep,
                        seconds=item["seconds"],
                        baseline=item["baseline_validation"],
                    )
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(compute_case, rep, str(output), identity) for rep in todo
            ]
            for future in as_completed(futures):
                item = future.result()
                print(
                    exact.precision.native(
                        dict(replication=item["replication"], seconds=item["seconds"])
                    ),
                    flush=True,
                )
    verify_identity(identity)
    aggregate(output, identity)


if __name__ == "__main__":
    main()

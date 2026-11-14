import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import stdtrit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "calibration_tested", ROOT / "scripts/calibration_sensitivity.py"
)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def mixture(weights=(1.0,), locations=(0.0,), scales=(1.0,), df=20.0):
    return dict(
        weights=np.array(weights),
        locations=np.tile(np.array(locations)[:, None], (1, 64)),
        scales=np.tile(np.array(scales)[:, None], (1, 64)),
        df=df,
    )


def test_oracle_matches_independent_dense_joint_nig_and_full_covariance():
    rng = np.random.default_rng(131)
    N = 5
    base = rng.choice([-0.5, 0.5], size=(49, 64))
    design = cal.formal.design_for_n(base, N)
    offsets = np.arange(-2, 3)
    person, trial, z, c = design
    y = rng.normal(size=len(person))
    mask = rng.random(len(y)) > 0.2
    train = cal.formal.pack(design, y, mask)
    person, trial, z, c, y = train
    G = cal.population_design(offsets)
    oracle = cal.oracle_posterior(train, N, offsets, G)
    S = np.zeros((len(y), 5))
    phase = np.array(
        [
            np.searchsorted(cal.CUTS + offsets[i], t, side="left")
            for i, t in zip(person, trial)
        ]
    )
    S[np.arange(len(y)), phase] = c
    Z = np.column_stack((np.ones(len(y)), z, np.eye(N)[person]))
    D = np.column_stack((Z, S))
    prior = np.diag(np.r_[0.25, np.ones(N + 1), np.full(5, 4.0)])
    A = D.T @ D + prior
    inverse = np.linalg.inv(A)
    mu = inverse @ (D.T @ y)
    shape = 3 + len(y) / 2
    rate = 2 + 0.5 * (y @ y - (D.T @ y) @ mu)
    expected_mean = G @ mu[-5:]
    covariance = inverse[-5:, -5:]
    expected_scale = np.sqrt(rate / shape * np.einsum("ij,jk,ik->i", G, covariance, G))
    np.testing.assert_allclose(
        oracle["locations"][0], expected_mean, atol=3e-13, rtol=0
    )
    np.testing.assert_allclose(oracle["scales"][0], expected_scale, atol=3e-13, rtol=0)
    q, _ = cal.oracle_quantiles(oracle)
    np.testing.assert_allclose(
        q,
        expected_mean
        + stdtrit(2 * shape, np.array([0.025, 0.975]))[:, None] * expected_scale,
        atol=5e-13,
        rtol=0,
    )
    diagonal_only = np.sqrt(rate / shape * (G * G) @ np.diag(covariance))
    assert np.max(abs(diagonal_only - expected_scale)) > 1e-6


def test_shared_oracle_reduces_to_existing_known_partition_reference():
    rng = np.random.default_rng(91)
    N = 6
    design = cal.formal.design_for_n(rng.choice([-0.5, 0.5], size=(49, 64)), N)
    train = cal.formal.pack(design, rng.normal(size=N * 64), rng.random(N * 64) > 0.15)
    offsets = np.zeros(N, int)
    G = cal.population_design(offsets)
    oracle = cal.oracle_posterior(train, N, offsets, G)
    H, h, yqy = cal.formal.collapsed(*train, N)
    _, mu, diagonal, rate, shape = cal.exact.precision.partition_posterior_parameters(
        H, h, yqy, len(train[-1]), tuple(cal.CUTS)
    )
    phase = np.searchsorted(cal.CUTS, cal.GRID, side="left")
    np.testing.assert_allclose(oracle["locations"][0], mu[phase], atol=3e-13, rtol=0)
    np.testing.assert_allclose(
        oracle["scales"][0], np.sqrt(rate / shape * diagonal)[phase], atol=3e-13, rtol=0
    )


def test_full_shared_batched_posterior_matches_frozen_scalar_and_batched():
    rng = np.random.default_rng(78)
    X = rng.normal(size=(100, 64))
    y = rng.normal(size=100)
    case = dict(H=X.T @ X, h=X.T @ y, yqy=float(y @ y), n=100)
    states = [(), (13,), (13, 25), (13, 25, 39), (13, 25, 39, 51)]
    actual = cal.shared_posterior(case, states)
    expected = cal.diagnostic.batched_parameters(case, states, precisions=(4.0,))[4.0]
    for key in ("weights", "locations", "scales"):
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=0)
    scalar = np.array(
        [
            cal.exact.precision.partition_posterior_parameters(
                case["H"], case["h"], case["yqy"], 100, state
            )[0]
            for state in states
        ]
    )
    np.testing.assert_allclose(actual["logweights"], scalar, atol=2e-12, rtol=0)
    assert actual["logweights"].shape == (len(states),)


def toy_original_inputs(rep=1):
    cfg = dict(
        namespace="calibration-test-original",
        design=dict(participant_sd=0.4, residual_sd=0.4, beta0=0.0, beta1=0.05),
        dgps={"D5": dict(cuts=cal.CUTS.tolist(), levels=cal.LEVELS.tolist())},
    )
    case = dict(dgp="D5", participants=500, signal_multiplier=1.0)
    base = np.random.default_rng(19).choice([-0.5, 0.5], size=(49, 64))
    design = cal.formal.design_for_n(base, 500)
    seed = 87123
    y, _, _ = cal.formal.generate_outcome(cfg, case, design, seed)
    test = cal.formal.outcome_blind_holdout(
        design[0], design[1], 13, f"{cfg['namespace']}|main|D5|R{rep:06d}"
    )
    inputs = dict(
        data_seed=seed,
        data_hash=cal.array_hash(y),
        holdout_hash=cal.array_hash(np.packbits(test)),
        training_mask_hash=cal.array_hash(np.packbits(~test)),
    )
    return cfg, case, base, inputs


def test_nested_data_preserve_original_prefix_global_geometry_masks_and_replay():
    cfg, case, base, inputs = toy_original_inputs()
    data, proof = cal.paired_datasets(1, cfg, case, base, inputs)
    replay, proof2 = cal.paired_datasets(1, cfg, case, base, inputs)
    assert cal.exact.digest(proof) == cal.exact.digest(proof2)
    baseline = data["N500_shared"]
    small = data["N1000_shared"]
    large = data["N2000_shared"]
    for name in ("y", "train", "test"):
        assert baseline[name].tobytes() == small[name][:32000].tobytes()
        assert small[name].tobytes() == large[name][:64000].tobytes()
        assert large[name].tobytes() == replay["N2000_shared"][name].tobytes()
    for a, b in zip(small["design"], large["design"]):
        assert a.tobytes() == b[:64000].tobytes()
    np.testing.assert_array_equal(
        large["design"][3][500 * 64 : 501 * 64], base[500 % 49]
    )
    assert not np.array_equal(base[500 % 49], base[0])
    np.testing.assert_array_equal(
        large["test"].reshape(2000, 64).sum(1), np.full(2000, 13)
    )
    assert (small["train"].sum(), large["train"].sum()) == (51000, 102000)
    assert len(set(proof["seeds"].values()) | {inputs["data_seed"]}) == 5
    assert proof["nested_N1000"] == proof["nested_N2000_prefix1000"]


def test_balanced_jitter_is_signal_only_legal_and_has_population_average_truth():
    cfg, case, base, inputs = toy_original_inputs()
    data, proof = cal.paired_datasets(1, cfg, case, base, inputs)
    original, jitter = data["N500_shared"], data["N500_jitter2"]
    offsets = jitter["offsets"]
    np.testing.assert_array_equal(np.bincount(offsets + 2), np.full(5, 100))
    shifted = cal.CUTS[None, :] + offsets[:, None]
    lengths = np.diff(
        np.column_stack((np.zeros(500), shifted, np.full(500, 64))), axis=1
    )
    assert lengths.min() >= 8
    np.testing.assert_array_equal(
        cal.jitter_signal(original["y"], original["design"], np.zeros(500, int)),
        original["y"],
    )
    person, trial, _, c = original["design"]
    direct_phase = np.array(
        [
            np.searchsorted(cal.CUTS + offsets[i], t, side="left")
            for i, t in zip(person, trial)
        ]
    )
    expected = original["y"] + c * (
        cal.LEVELS[direct_phase] - cal.BASE_TRUTH[trial - 1]
    )
    np.testing.assert_array_equal(jitter["y"], expected)
    np.testing.assert_array_equal(jitter["train"], original["train"])
    np.testing.assert_array_equal(jitter["test"], original["test"])
    manual = np.array(
        [cal.formal.curve_from(cal.CUTS + j, cal.LEVELS) for j in offsets]
    ).mean(0)
    np.testing.assert_allclose(jitter["truth"], manual, atol=2e-15, rtol=0)
    np.testing.assert_allclose(
        jitter["truth"], jitter["G"] @ cal.LEVELS, atol=0, rtol=0
    )
    assert proof["offset_assignment_hash"] == cal.array_hash(offsets)


def test_original_posterior_guard_runs_before_any_extension_seed(monkeypatch):
    cfg, case, base, inputs = toy_original_inputs()

    def forbidden(*args):
        raise AssertionError("Extension accessed before original posterior validation")

    monkeypatch.setattr(cal, "labelled_seed", forbidden)

    class GuardStop(Exception):
        pass

    def guard(item):
        assert cal.array_hash(item["y"]) == inputs["data_hash"]
        raise GuardStop

    with pytest.raises(GuardStop):
        cal.paired_datasets(1, cfg, case, base, inputs, guard)


def test_tiny_omitted_tail_component_has_conservative_unnormalized_cdf_bounds():
    ex = mixture((1 - 1e-16, 1e-16), (0.0, 50.0), (1.0, 2.0))
    active, omitted, epsilon = cal.active_set(ex["weights"])
    assert active.tolist() == [0] and omitted.tolist() == [1]
    assert ex["weights"][active].sum() < 1
    assert 1e-16 <= epsilon <= 1e-15
    for x in (-100.0, -2.0, 0.0, 2.0, 100.0):
        a = cal.cdf_sum(
            x,
            ex["weights"][active],
            ex["locations"][active, 0],
            ex["scales"][active, 0],
            ex["df"],
        )
        f = cal.cdf_sum(
            x, ex["weights"], ex["locations"][:, 0], ex["scales"][:, 0], ex["df"]
        )
        assert a - cal.CDF_PAD <= f <= a + epsilon + cal.CDF_PAD
    q, cert = cal.bounded_quantiles(ex, independent_full_check=True)
    assert cert["certified"] and not cert["active_weights_renormalized"]
    assert (
        cert["active_count"] == 1 and cert["independent_full_check"]["endpoints"] == 128
    )
    assert q.shape == (2, 64)


def test_underflowed_omitted_components_never_get_zero_mass_upper_bound():
    w = np.array([1.0, 0.0, 0.0])
    active, omitted, epsilon = cal.active_set(w)
    assert active.tolist() == [0] and len(omitted) == 2
    assert epsilon >= 2 * np.nextafter(0.0, 1.0)
    assert epsilon > 0


def test_failed_active_certificate_forces_complete_cdf_fallback(monkeypatch):
    ex = mixture((1 - 1e-16, 1e-16), (0.0, 30.0), (1.0, 1.0))
    original = cal.crossing_certificate

    def reject_active(q, p, cdf, epsilon=0.0, pad=cal.CDF_PAD):
        result = original(q, p, cdf, epsilon, pad)
        if epsilon > 0:
            result["accepted"] = False
        return result

    monkeypatch.setattr(cal, "crossing_certificate", reject_active)
    _, cert = cal.bounded_quantiles(ex)
    assert cert["full_fallback_endpoints"] == 2
    assert cert["full_cdf_calls"] > 0 and cert["certified"]


def test_flat_cdf_cannot_be_certified_only_from_a_small_probability_residual():
    with pytest.raises(cal.exact.EndpointCertificationError):
        cal.bounded_quantiles(mixture(scales=(1e12,)))


def test_truth_near_endpoint_is_classified_by_full_cdf_not_rounded_quantile():
    ex = mixture(scales=(0.05,))
    q, _ = cal.bounded_quantiles(ex)
    truth = q[0] + np.r_[np.full(32, 0.5 * cal.TAU), np.full(32, -0.5 * cal.TAU)]
    decisions, checks = cal.coverage_decisions(q, ex, truth)
    assert checks["near_endpoint_count"] == 64 and checks["full_cdf_resolutions"] == 64
    assert (
        decisions["coverage"][:32].all() and decisions["miscoverage_above"][32:].all()
    )


def test_unresolved_truth_tie_is_not_silently_counted_as_covered():
    ex = mixture(df=20.0)
    q = np.tile(stdtrit(20.0, [0.025, 0.975])[:, None], (1, 64))
    with pytest.raises(cal.exact.EndpointCertificationError):
        cal.coverage_decisions(q, ex, q[0])


def test_paired_mcse_and_distance_strata_use_datasets_as_the_units():
    n = 5
    samples = {}
    for index, condition in enumerate(("N500_shared", *cal.CONDITIONS)):
        for j, method in enumerate(cal.METHODS):
            half = np.arange(1, n + 1) + index * 0.1 + j * 0.05
            q = np.broadcast_to(
                np.stack((-half, half), axis=1)[:, :, None], (n, 2, 64)
            ).copy()
            samples[condition, method] = dict(
                quantiles=q,
                mean=np.broadcast_to(
                    (np.arange(n) * 0.03 + index * 0.01 + j * 0.005)[:, None], (n, 64)
                ),
                truth=np.zeros((n, 64)),
                decisions=dict(
                    coverage=np.ones((n, 64), bool),
                    miscoverage_below=np.zeros((n, 64), bool),
                    miscoverage_above=np.zeros((n, 64), bool),
                ),
            )
    rows, paired, _ = cal.summarize_samples(samples, list(range(1, n + 1)))
    contrast = next(
        r
        for r in paired
        if r["left"] == "N2000_shared__working"
        and r["right"] == "N1000_shared__working"
        and r["stratum"] == "all"
        and r["metric"] == "width"
    )
    assert contrast["paired_mcse"] < 1e-14
    assert (
        next(
            r
            for r in rows
            if r["condition"] == "N2000_shared"
            and r["method"] == "working"
            and r["stratum"] == "all"
            and r["metric"] == "width"
        )["mcse"]
        > 0
    )
    assert [
        cal.STRATA[s].sum()
        for s in ("near", "far", "distance0", "distance1", "distance2", "distance_ge3")
    ] == [20, 44, 4, 8, 8, 44]
    assert any(r["stratum"] == "near_minus_far" for r in paired)
    assert all(
        r["distance_reference"] == "nominal transition centers"
        for r in rows
        if r["condition"] == "N500_jitter2"
    )


def test_resume_refuses_changed_identity_and_formal_aggregate_refuses_subset(tmp_path):
    first = dict(identity_sha256="first", mode="pilot")
    cal.exact.bind_identity(tmp_path, first)
    cal.exact.bind_identity(tmp_path, first)
    with pytest.raises(RuntimeError):
        cal.exact.bind_identity(tmp_path, dict(first, identity_sha256="changed"))
    with pytest.raises(ValueError):
        cal.aggregate(tmp_path, dict(mode="formal500", replications=[1]))
    path, _ = cal.result_paths(tmp_path, 1)
    cal.exact.atomic_json(
        path, dict(status="complete", identity_sha256="wrong", inputs={})
    )
    with pytest.raises(ValueError):
        cal.validate_result(
            tmp_path, 1, dict(identity_sha256="right", inputs={"1": {}})
        )

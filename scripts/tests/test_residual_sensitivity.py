import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import stdtr

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "residual_tested", ROOT / "scripts/residual_sensitivity.py"
)
residual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(residual)


@pytest.mark.parametrize("choice,expected", [
    ("all", ((4, 0.0), (8, 0.0), (4, 0.2))),
    ("t4-iid", ((4, 0.0),)),
    ("t8-iid", ((8, 0.0),)),
    ("t4-ar02", ((4, 0.2),)),
])
def test_student_condition_cli_filter_mapping(choice, expected):
    args = residual.parse_args(["--mode", "student", "--group", "High",
                               "--iterations", "60000", "--student-condition", choice])
    assert residual.STUDENT_CONDITIONS[args.student_condition] == expected
    assert args.mode == "student" and args.group == "High" and args.iterations == 60000


def test_student_condition_cli_default_and_invalid_choice():
    assert residual.parse_args([]).student_condition == "all"
    with pytest.raises(SystemExit):
        residual.parse_args(["--student-condition", "other"])


def toy():
    rng = np.random.default_rng(12)
    n = 300
    N = 5
    train = (
        rng.integers(N, size=n),
        rng.integers(1, 65, size=n),
        rng.normal(size=n),
        rng.choice([-0.5, 0.5], size=n),
        rng.normal(size=n),
    )
    return rng, N, train


def test_sparse_weighted_stats_equal_dense_schur_and_original():
    rng, N, train = toy()
    X, Z, y = residual.designs(train, N)
    w = rng.gamma(2, 0.5, len(y))
    sparse = residual.weighted_stats(*train, N, w)
    dense = residual.dense_stats(X, Z, y, w)
    for a, b in zip(sparse, dense):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)
    original = residual.precision.formal.collapsed(*train, N)
    for a, b in zip(residual.weighted_stats(*train, N, np.ones(len(y))), original):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)


def test_nuisance_draws_match_full_precision():
    rng, N, train = toy()
    X, Z, y = residual.designs(train, N)
    w = rng.gamma(2, 0.5, len(y))
    delta = rng.normal(size=64)
    s2 = 0.3
    cache = residual.weighted_stats(*train, N, w)[3]
    A = Z.T @ (w[:, None] * Z) + np.diag(np.r_[0.25, np.ones(N + 1)])
    expected_mean = np.linalg.solve(A, Z.T @ (w * (y - X @ delta)))
    draws = np.array(
        [residual.draw_nuisance(cache, delta, s2, rng) for _ in range(12000)]
    )
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.012, rtol=0)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), s2 * np.linalg.inv(A), atol=0.003, rtol=0
    )


def test_ar_whitening_matches_gapped_stationary_covariance_and_resets():
    person = np.array([0, 0, 0, 0, 1, 1])
    block = np.array([2, 2, 2, 3, 2, 2])
    trial = np.array([2, 5, 6, 2, 2, 8])
    rho = 0.4
    covariance = np.eye(6)
    for i in range(6):
        for j in range(6):
            if person[i] == person[j] and block[i] == block[j]:
                covariance[i, j] = rho ** abs(trial[i] - trial[j])
    W = residual.ar_whiten(np.eye(6), person, block, trial, rho)
    np.testing.assert_allclose(W @ covariance @ W.T, np.eye(6), atol=1e-14)
    np.testing.assert_array_equal(
        residual.ar_whiten(np.eye(6), person, block, trial, 0), np.eye(6)
    )


def test_exact_phase_uses_full_covariance_and_mixture_quantiles():
    H = np.eye(64) * 2 + np.ones((64, 64)) * 0.1
    h = np.linspace(-1, 1, 64)
    result = residual.exact_gaussian(H, h, 40, 100, states=[(32,)])
    A = np.array(
        [
            [H[:32, :32].sum() + 4, H[:32, 32:].sum()],
            [H[32:, :32].sum(), H[32:, 32:].sum() + 4],
        ]
    )
    inv = np.linalg.inv(A)
    g = np.array([h[:32].sum(), h[32:].sum()])
    mu = inv @ g
    rate = 2 + 0.5 * (40 - g @ mu)
    expected_scale = np.sqrt(
        rate / 53 * (np.array([-1.0, 1.0]) @ inv @ np.array([-1.0, 1.0]))
    )
    np.testing.assert_allclose(result["locations"][0, 2], mu[1] - mu[0])
    np.testing.assert_allclose(result["scales"][0, 2], expected_scale)
    assert (
        abs(
            float(stdtr(106, (result["q025"][2] - (mu[1] - mu[0])) / expected_scale))
            - 0.025
        )
        < 1e-10
    )


def test_point_mass_phase_is_exact_zero_and_not_positive():
    result = residual.exact_gaussian(np.eye(64), np.zeros(64), 10, 80, states=[()])
    np.testing.assert_array_equal(result["q025"][2:], 0)
    np.testing.assert_array_equal(result["q975"][2:], 0)
    assert result["probability_positive"][2] == 0


def test_protocol_partition_count_and_seed():
    assert len(residual.precision.legal_partitions()) == 27931
    assert residual.seed_for("High", 4, 0) == residual.seed_for("High", 4, 0)
    assert residual.seed_for("High", 4, 0) != residual.seed_for("High", 4, 1)


def test_sparse_innovation_stats_equal_dense_weighted_reference():
    rng, N, train = toy()
    person = np.sort(train[0])
    train = (person, *train[1:])
    block = np.ones(len(person), int)
    native = np.arange(len(person)) + 1
    B = residual.innovation_design(train, N, block, native, 0.2)
    w = rng.gamma(2, 0.5, len(person))
    sparse = residual.weighted_sparse_stats(B, N, w)
    dense = B.toarray()
    q = N + 2
    reference = residual.dense_stats(
        dense[:, q : q + 64], dense[:, :q], dense[:, -1], w
    )
    for a, b in zip(sparse, reference):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=0)
    # Frozen augmentation w=c yields the Gaussian W=I baseline, for both dfs.
    for df in (4, 8):
        c = (df - 2) / df
        for a, b in zip(
            residual.weighted_stats(*train, N, np.full(len(person), c) / c),
            residual.precision.formal.collapsed(*train, N),
        ):
            np.testing.assert_allclose(a, b, atol=3e-12, rtol=0)


def test_python_partition_parameters_match_unchanged_kernel_reference_formula():
    rng, N, train = toy()
    H, h, yqy, _ = residual.weighted_stats(*train, N, rng.gamma(2, 0.5, len(train[-1])))
    prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ph = np.r_[0, h.cumsum()]
    for cuts in ((), (32,), (8, 24, 40, 56)):
        lw, mu, L, rate, shape = residual.PythonStepper.parameters(
            prefix, ph, yqy, len(train[-1]), cuts
        )
        expected = residual.precision.partition_posterior_parameters(
            H, h, yqy, len(train[-1]), cuts
        )
        np.testing.assert_allclose(lw, expected[0], atol=3e-12, rtol=0)
        np.testing.assert_allclose(mu, expected[1], atol=3e-12, rtol=0)
        np.testing.assert_allclose(
            np.diag(np.linalg.inv(L @ L.T)), expected[2], atol=3e-12, rtol=0
        )
        np.testing.assert_allclose((rate, shape), expected[3:], atol=3e-12, rtol=0)


def test_atom_quantile_certifies_jump_instead_of_zero_cdf_residual():
    weights = np.array([0.1, 0.9])
    loc = np.array([[0.0], [1.0]])
    scale = np.array([[0.0], [0.1]])
    values, cert = residual.exact_mixture_quantiles(
        weights, loc, scale, 20, probabilities=(0.025, 0.975), return_certificates=True
    )
    assert values[0, 0] == 0
    assert cert[0]["left_cdf"][0] <= 0.025 <= cert[0]["cdf"][0]
    assert cert[0]["is_atom"][0]
    assert abs(cert[1]["cdf"][0] - 0.975) < 1e-10


@pytest.mark.parametrize("rho", [0.0, 0.2])
def test_fixed_partition_student_partially_collapsed_gibbs_matches_full_gibbs(rho):
    """Independent full coefficient/variance Gibbs target, with fixed partition.

    This catches missing nuisance uncertainty, wrong IG shape and wrong latent
    scale factors; the reference never uses Schur/partition helper functions.
    """
    rng = np.random.default_rng(818)
    n = 100
    N = 3
    train = (
        np.sort(rng.integers(N, size=n)),
        rng.integers(1, 65, size=n),
        rng.normal(size=n),
        rng.choice([-0.5, 0.5], size=n),
        rng.standard_t(4, size=n) * 0.2,
    )
    innovations = None
    X, Z, y = residual.designs(train, N)
    if rho:
        innovations = residual.innovation_design(
            train, N, np.ones(n, int), np.arange(n), rho
        )
        dense = innovations.toarray()
        q = N + 2
        X, Z, y = dense[:, q : q + 64], dense[:, :q], dense[:, -1]
    design = np.column_stack((Z, X[:, :32].sum(1), X[:, 32:].sum(1)))
    prior = np.r_[0.25, np.ones(N + 1), 4.0, 4.0]
    d = design.shape[1]
    df = 4.0
    factor = 0.5

    class Fixed(residual.PythonStepper):
        def step(self, H, h, yqy, n, state):
            _, r, _ = state
            P = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            ph = np.r_[0, h.cumsum()]
            _, mu, L, rate, shape = self.parameters(P, ph, yqy, n, (32,))
            s2 = rate / r.gamma(shape)
            delta = mu + np.sqrt(s2) * np.linalg.solve(L.T, r.normal(size=2))
            return s2, np.repeat(delta, 32)

    pc, _ = residual.run_student_chain(
        train, N, 4, 7000, 1000, 114, Fixed(), 1, innovations
    )
    r = np.random.default_rng(115)
    w = np.ones(n)
    s2 = 0.1
    full = []
    for it in range(7000):
        W = w / factor
        A = design.T @ (W[:, None] * design) + np.diag(prior)
        L = np.linalg.cholesky(A)
        mean = np.linalg.solve(A, design.T @ (W * y))
        beta = mean + np.sqrt(s2) * np.linalg.solve(L.T, r.normal(size=d))
        e = y - design @ beta
        s2 = (2 + 0.5 * (W @ (e * e) + prior @ (beta * beta))) / r.gamma(
            3 + (n + d) / 2
        )
        w = r.gamma((df + 1) / 2, 2 / (df + e * e / (s2 * factor)))
        if it >= 1000:
            full.append([beta[-2], beta[-1], s2])
    observed = pc[:, [21, 53, 70]]
    reference = np.array(full)
    np.testing.assert_allclose(observed.mean(0), reference.mean(0), atol=0.018, rtol=0)
    np.testing.assert_allclose(
        np.quantile(observed, [0.025, 0.975], axis=0),
        np.quantile(reference, [0.025, 0.975], axis=0),
        atol=0.035,
        rtol=0,
    )


@pytest.mark.parametrize(
    "stepper_class", [residual.PythonStepper, residual.InformedPythonStepper]
)
def test_persistent_python_partition_chain_recovers_exact_prior_without_data(
    stepper_class,
):
    stepper = stepper_class()
    state = stepper.initialize(4, 883)
    counts = np.zeros(5)
    H = np.zeros((64, 64))
    h = np.zeros(64)
    for it in range(14000):
        stepper.step(H, h, 0.0, 0, state)
        if it >= 1000:
            counts[state[0][0]] += 1
    target = np.array([1.0, 1.0, 0.5, 1 / 6, 1 / 24])
    target /= target.sum()
    np.testing.assert_allclose(counts / counts.sum(), target, atol=0.025, rtol=0)


def test_chain_summary_keeps_constant_indicator_diagnostics_unavailable():
    rng = np.random.default_rng(15)
    draws = np.zeros((3, 300, residual.DRAW_WIDTH))
    draws[:, :, 0] = 2
    draws[:, :, 1] = 12
    draws[:, :, 2] = 32
    draws[:, :, 6:70] = rng.normal(size=(3, 300, 64))
    draws[:, :, 70] = 0.1
    draws[:, :, 71] = np.exp(rng.normal(size=(3, 300)))
    draws[:, :, 72] = rng.normal(size=(3, 300))
    draws[:, :, 73] = rng.uniform(size=(3, 300))
    summary = residual.chain_summary(draws)
    assert summary["diagnostics"]["PIP32"]["constant_chain"]
    assert summary["diagnostics"]["PIP32"]["ess_bulk"] is None
    assert summary["diagnostics"]["PIP32"]["rhat"] is None
    assert summary["phase_contrast"]["q025"] < summary["phase_contrast"]["q975"]
    assert summary["needs_escalation"]  # deliberately fewer than 1500 draws
    for name in (
        "sigma2",
        "mean_effective_precision",
        "mean_log_latent_weight",
        "latent_weight_lower_tail_fraction",
    ):
        assert name in summary["diagnostics"]


def test_student_chain_uses_distinct_reproducible_spawned_streams(monkeypatch):
    _, N, train = toy()
    seed = 82923
    structural_seed, augmented = residual.chain_streams(seed)
    expected = augmented.random()
    assert expected != np.random.default_rng(structural_seed).random()
    recorded = {}

    class Recording(residual.PythonStepper):
        def initialize(self, initk, actual_seed):
            recorded["structural_seed"] = actual_seed
            return super().initialize(initk, actual_seed)

    original = residual.draw_nuisance

    def spy(cache, delta, sigma2, rng):
        if "first_augmented_random" not in recorded:
            recorded["first_augmented_random"] = rng.random()
        return original(cache, delta, sigma2, rng)

    monkeypatch.setattr(residual, "draw_nuisance", spy)
    draws, _ = residual.run_student_chain(train, N, 4, 3, 1, seed, Recording(), 0)
    assert recorded == {
        "structural_seed": structural_seed,
        "first_augmented_random": expected,
    }
    assert draws.shape == (2, residual.DRAW_WIDTH)
    np.testing.assert_array_equal(
        residual.chain_streams(seed)[1].random(20),
        residual.chain_streams(seed)[1].random(20),
    )


def test_archive_checks_are_optional_and_public_protocol_is_available(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("archive accessed when not requested")

    monkeypatch.setattr(residual.np, "load", forbidden)
    assert residual.archived_statistic_check(None, "High", None) == {"requested": False}
    assert residual.archived_exact_check(None, "High", None) == {"requested": False}
    assert (
        residual.PROTOCOL == ROOT / "scripts/config/residual_sensitivity_protocol.json"
    )
    assert residual.PROTOCOL.exists()
    assert residual.OUTPUT == ROOT / "results/reproduction/residual_sensitivity"


def test_informed_batched_weights_equal_independent_scalar_reference():
    rng, N, train = toy()
    H, h, yqy, _ = residual.weighted_stats(*train, N, rng.gamma(2, 0.5, len(train[-1])))
    prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ph = np.r_[0, h.cumsum()]
    stepper = residual.InformedPythonStepper()
    for cuts, kind in (((), 0), ((16, 40), 0), ((16, 32, 48), 1), ((16, 40), 2)):
        states = stepper.neighbors(cuts, kind)
        actual = stepper.logweights(prefix, ph, yqy, len(train[-1]), states)
        expected = [
            residual.precision.partition_posterior_parameters(
                H, h, yqy, len(train[-1]), s
            )[0]
            for s in states
        ]
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)
        q = stepper.probabilities(actual)
        assert np.all(q >= 0.05 / len(q))
        np.testing.assert_allclose(q.sum(), 1, atol=1e-14)


def test_informed_forward_reverse_probability_satisfies_detailed_balance():
    rng, N, train = toy()
    H, h, yqy, _ = residual.weighted_stats(*train, N, np.ones(len(train[-1])))
    prefix = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ph = np.r_[0, h.cumsum()]
    stepper = residual.InformedPythonStepper()
    for cuts, proposal, kind in (
        ((), (32,), 0),
        ((16, 40), (16, 32, 40), 0),
        ((16, 40), (24, 40), 2),
    ):
        states = stepper.neighbors(cuts, kind)
        reverse = 1 - kind if kind < 2 else 2
        backward = stepper.neighbors(proposal, reverse)
        q = stepper.probabilities(
            stepper.logweights(prefix, ph, yqy, len(train[-1]), states)
        )[states.index(proposal)] * stepper.direction(len(cuts), kind)
        qb = stepper.probabilities(
            stepper.logweights(prefix, ph, yqy, len(train[-1]), backward)
        )[backward.index(cuts)] * stepper.direction(len(proposal), reverse)
        delta = (
            stepper.parameters(prefix, ph, yqy, len(train[-1]), proposal)[0]
            - stepper.parameters(prefix, ph, yqy, len(train[-1]), cuts)[0]
        )
        alpha = delta + np.log(qb / q)
        forward = q * np.exp(min(0, alpha))
        back = np.exp(delta) * qb * np.exp(min(0, -alpha))
        np.testing.assert_allclose(forward, back, rtol=2e-13, atol=1e-14)

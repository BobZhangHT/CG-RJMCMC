"""Independent numerical and public C/Python interface checks (synthetic only)."""
import ctypes as ct
import importlib.util
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import invgamma, norm, t

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("precision_interfaces", ROOT / "scripts/precision.py")
precision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precision)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return precision.CWeightProbe(tmp_path_factory.mktemp("public_probe"))


def statistics():
    rng = np.random.default_rng(20261112)
    X = rng.normal(size=(80, 64)) / 4
    y = rng.normal(size=80)
    return X.T @ X, X.T @ y, y @ y, len(y)


def independent_weight(H, h, yqy, n, cuts):
    # Dense indicator-matrix reduction, independent of C prefix sums and
    # precision.partition_posterior_parameters block aggregation.
    B = np.eye(len(cuts) + 1)[np.searchsorted(cuts, np.arange(1, 65), side="left")]
    A = 4 * np.eye(B.shape[1]) + B.T @ H @ B
    g = B.T @ h
    rate = 2 + (yqy - g @ np.linalg.solve(A, g)) / 2
    k = len(cuts)
    return (-math.lgamma(k + 1) - math.log(math.comb(64 - 8 * (k + 1) + k, k))
            + .5 * B.shape[1] * math.log(4) - .5 * np.linalg.slogdet(A)[1]
            - (3 + n / 2) * math.log(rate))


def test_public_probe_matches_independent_logweight_differences(probe):
    args = statistics()
    states = [(), (8,), (56,), (8, 56), (16, 32, 48), (8, 16, 24, 32), (8, 16, 24, 32, 40)]
    got = np.array([probe.weight(*args, cuts) for cuts in states])
    expected = np.array([independent_weight(*args, cuts) for cuts in states])
    np.testing.assert_allclose(got - got[0], expected - expected[0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(got, expected, atol=2e-10, rtol=0)


def test_public_probe_converts_dtype_and_contiguity(probe):
    H, h, yqy, n = statistics()
    H, h = H.astype(np.float32), h.astype(np.float32)
    # Transposed symmetric H and reversed-stride h exercise both memory layouts.
    got = probe.weight(H.T, h[::-1], yqy, n, (16, 32))
    expected = independent_weight(H.T.astype(float), h[::-1].astype(float), yqy, n, (16, 32))
    assert abs(got - expected) < 2e-10


@pytest.mark.parametrize("cuts", [(0,), (7,), (57,), (65,), (32, 16), (16, 16), (8, 15), (8.5,), (True,), tuple(range(8, 57, 8))])
def test_python_rejects_illegal_partition(cuts):
    probe = precision.CWeightProbe.__new__(precision.CWeightProbe)
    with pytest.raises(ValueError):
        probe.weight(*statistics(), cuts)


@pytest.mark.parametrize("field,value", [(0, np.eye(63)), (1, np.zeros((64, 1))), (2, np.nan), (3, 2.5), (3, -1), (3, 2**31)])
def test_python_rejects_unsafe_arguments(field, value):
    probe = precision.CWeightProbe.__new__(precision.CWeightProbe)
    args = list(statistics())
    args[field] = value
    with pytest.raises(ValueError):
        probe.weight(*args, ())


def test_c_abi_rejects_illegal_partition_before_indexing(probe):
    H, h, yqy, n = statistics()
    dp = ct.POINTER(ct.c_double)
    ip = ct.POINTER(ct.c_int)
    for cuts, k, minseg in [([0], 1, 8), ([65], 1, 8), ([32, 16], 2, 8), ([], -1, 8), ([], 6, 8), ([], 0, 0), ([], 0, 65)]:
        a = np.array(cuts or [0], dtype=np.int32)
        got = probe.lib.probe_weight(H.ctypes.data_as(dp), h.ctypes.data_as(dp), yqy, n,
                                     a.ctypes.data_as(ip), k, minseg, 1.)
        assert got == -math.inf


def test_probe_cache_depends_on_wrapper_and_kernel(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    scripts = source_root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("probe.c", "kernel.c"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    monkeypatch.setattr(precision, "ROOT", source_root)
    # Exercise real GCC compilation/cache selection independently of host DLL
    # execution policy; the public-probe tests above require the real loader.
    class Function:
        pass

    class Library:
        probe_weight = Function()

    monkeypatch.setattr(precision.ct, "CDLL", lambda path: Library())
    output = tmp_path / "build"
    first = precision.CWeightProbe(output)
    cached = precision.CWeightProbe(output)
    assert first.library_path == cached.library_path
    for name in ("probe.c", "kernel.c"):
        with (scripts / name).open("a", encoding="utf8") as stream:
            stream.write("\n/* cache invalidation test */\n")
        changed = precision.CWeightProbe(output)
        assert changed.library_path != first.library_path
        first = changed


def test_student_t_conditional_matches_normal_inverse_gamma_integral():
    rate, shape, inverse_variance, mu, x = 4.2, 3.7, .3, .4, 1.1
    scale = precision.student_t_scale(rate, shape, inverse_variance)
    marginal_cdf = t.cdf((x - mu) / scale, 2 * shape)
    integrated_cdf = quad(lambda s2: norm.cdf((x - mu) / np.sqrt(s2 * inverse_variance))
                         * invgamma.pdf(s2, a=shape, scale=rate), 0, np.inf, epsabs=1e-11)[0]
    assert abs(marginal_cdf - integrated_cdf) < 2e-10


def test_mixture_cdf_inversion_matches_independent_root_solver():
    weights = np.array([.2, .8])
    locations = np.array([[-1.], [2.]])
    scales = np.array([[.4], [1.2]])
    probabilities = [.001, .025, .5, .975, .999]
    got = precision.mixture_quantiles(weights, locations, scales, .8, probabilities).ravel()
    expected = [brentq(lambda x: sum(weights * t.cdf((x - locations[:, 0]) / scales[:, 0], .8)) - q,
                       -1e6, 1e6, xtol=1e-10) for q in probabilities]
    np.testing.assert_allclose(got, expected, atol=2e-8, rtol=1e-10)


@pytest.mark.parametrize("weights,df,probabilities", [([-1., 2.], 5., [.5]), ([.5, .5], 0., [.5]), ([.5, .5], 5., [0.]), ([.5, .5], 5., [np.nan])])
def test_mixture_rejects_invalid_distributions(weights, df, probabilities):
    with pytest.raises(ValueError):
        precision.mixture_quantiles(np.array(weights), np.array([0., 1.]), np.ones(2), df, probabilities)


def test_synthetic_precision_smoke_uses_real_c_weights(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(precision, "CWeightProbe", lambda output: probe)
    result = precision.run_smoke(tmp_path)
    assert result["status"] == "passed"
    assert result["manuscript_result"] is False
    assert result["partitions"] == 50
    assert result["max_c_python_logweight_error"] < 2e-8
    assert result["max_mixture_cdf_error"] < 1e-10
    assert (tmp_path / "precision_smoke.json").is_file()


def test_partition_conditionals_match_independent_dense_reduction():
    H, h, yqy, n = statistics()
    for cuts in ((), (8, 56), (16, 32, 48)):
        logw, mu, inverse_diag, rate, shape = precision.partition_posterior_parameters(H, h, yqy, n, cuts)
        B = np.eye(len(cuts) + 1)[np.searchsorted(cuts, np.arange(1, 65), side="left")]
        A = 4 * np.eye(B.shape[1]) + B.T @ H @ B
        expected_mu = np.linalg.solve(A, B.T @ h)
        np.testing.assert_allclose(mu, expected_mu, atol=1e-12, rtol=0)
        np.testing.assert_allclose(inverse_diag, np.diag(np.linalg.inv(A)), atol=1e-12, rtol=0)
        assert abs(rate - (2 + (yqy - (B.T @ h) @ expected_mu) / 2)) < 1e-12
        assert shape == 3 + n / 2
        assert abs(logw - independent_weight(H, h, yqy, n, cuts)) < 2e-10

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("checks", ROOT / "scripts" / "checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_public_defaults_do_not_depend_on_ignored_revision_trees():
    assert checks.DEFAULT_INPUT == ROOT / "data" / "external" / "RawData_Experiment1"
    assert checks.DEFAULT_OUTPUT == ROOT / "results" / "reproduction" / "checks"
    assert checks.DEFAULT_ASSETS == ROOT / "manuscript" / "reproduced"
    assert (
        checks.APPLICATION_PROTOCOL
        == ROOT / "scripts" / "config" / "application_protocol.md"
    )
    assert checks.APPLICATION_PROTOCOL.is_file()


def _small_frame() -> pd.DataFrame:
    person = np.repeat(np.arange(4), 12)
    bins = np.tile(np.arange(1, 13), 4)
    c = np.where((person + bins) % 2, 0.5, -0.5)
    z = (bins - 6.5) / np.std(np.arange(1, 13))
    rng = np.random.default_rng(810)
    y = (
        0.15
        + 0.03 * z
        + np.array([-0.2, 0.1, 0.05, 0.08])[person]
        + c * np.where(bins <= 6, 0.2, 0.05)
    )
    y += rng.normal(0, 0.12, len(y))
    return pd.DataFrame(
        {
            "person": person,
            "bin": bins,
            "z": z,
            "c": c,
            "y": y,
            "block_rank": bins % 10,
            "blockNum": bins % 10 + 2,
            "trialNum": bins,
            "phase": np.where(bins <= 6, "before", "after"),
            "congruency_label": np.where(c > 0, "Incongruent", "Congruent"),
            "uid": person.astype(str),
        }
    )


def test_generalized_collapsed_matches_dense_q_small_case():
    frame = _small_frame()
    W = checks.contrast_design(frame)
    # Exercise a deliberately overcomplete proper-prior block design.
    Z, D_nu, _ = checks.nuisance_design(frame, 4, checks.case_spec("block10"))
    fast = checks.generalized_collapsed(W, Z, D_nu, frame["y"].to_numpy())
    dense = checks.dense_collapsed(W, Z, D_nu, frame["y"].to_numpy())
    assert np.allclose(fast[0], dense[0], atol=2e-12, rtol=2e-12)
    assert np.allclose(fast[1], dense[1], atol=2e-12, rtol=2e-12)
    assert np.isclose(fast[2], dense[2], atol=2e-12, rtol=2e-12)


def test_nominal_generalized_summaries_regenerate_formal_collapsed():
    frame = _small_frame()
    W = checks.contrast_design(frame)
    Z, D_nu, _ = checks.nuisance_design(frame, 4, checks.case_spec("nominal"))
    generalized = checks.generalized_collapsed(W, Z, D_nu, frame["y"].to_numpy())
    original = checks.formal.collapsed(
        frame["person"].to_numpy(),
        frame["bin"].to_numpy(),
        frame["z"].to_numpy(),
        frame["c"].to_numpy(),
        frame["y"].to_numpy(),
        4,
    )
    assert np.allclose(generalized[0], original[0], atol=2e-12, rtol=2e-12)
    assert np.allclose(generalized[1], original[1], atol=2e-12, rtol=2e-12)
    assert np.isclose(generalized[2], original[2], atol=2e-12, rtol=2e-12)


def test_joint_effect_nuisance_and_predictive_draws_have_correct_covariance():
    frame = _small_frame().copy().reset_index(drop=True)
    Z, D_nu, _ = checks.nuisance_design(frame, 4, checks.case_spec("slope_var_0p1"))
    system = checks.posterior_system(frame, Z, D_nu, (6,))
    sigma2 = 0.37
    rng = np.random.default_rng(2901)
    count = 30000
    coefficients = np.stack(
        [
            checks.draw_coefficients_given_sigma(system, sigma2, rng)
            for _ in range(count)
        ]
    )
    expected_cov = sigma2 * np.linalg.inv(system.lambda_post)
    # This checks nuisance/effect marginal variances and their nonzero cross-covariance.
    assert np.allclose(coefficients.mean(axis=0), system.mean, atol=0.012)
    assert np.allclose(
        np.cov(coefficients, rowvar=False), expected_cov, atol=0.006, rtol=0.10
    )

    # A joint conditional posterior-predictive draw adds sigma^2 I, rather than
    # reusing or independently bootstrapping a constant residual vector.
    subset = np.array([0, 1, 3])
    noise = rng.normal(0, np.sqrt(sigma2), size=(count, len(subset)))
    predictive = coefficients @ system.X_full[subset].T + noise
    expected_predictive_cov = system.X_full[subset] @ expected_cov @ system.X_full[
        subset
    ].T + sigma2 * np.eye(len(subset))
    assert np.allclose(
        np.cov(predictive, rowvar=False), expected_predictive_cov, atol=0.012, rtol=0.08
    )


def test_ppc_contrast_is_participant_by_phase_sample_sd():
    frame = _small_frame().reset_index(drop=True)
    # Distinct phase-specific contrast values; pooling by participant would
    # give a different SD and cannot satisfy this check.
    phase_shift = np.where(frame["phase"].eq("before"), -2.0, 3.0)
    contrasts = frame["person"].to_numpy() + phase_shift
    residual = frame["c"].to_numpy() * contrasts
    expected = [i + shift for i in range(4) for shift in (-2.0, 3.0)]
    stats = checks.check_statistics(frame, residual, 2.0, checks.lag_pairs(frame))
    assert np.isclose(stats["participant_contrast_sd"], np.std(expected, ddof=1) / 2.0)


def test_presentation_only_does_not_rewrite_analysis(tmp_path, monkeypatch):
    output, assets = tmp_path / "results", tmp_path / "assets"
    output.mkdir()
    assets.mkdir()
    source = output / "application_results.json"
    original = json.dumps({"groups": [], "participant_level_analysis": {}}).encode()
    source.write_bytes(original)
    monkeypatch.setattr(checks, "supplement_tex", lambda result: "presentation text")
    def forbidden_recalculation(*_):
        raise AssertionError("completed participant analysis was recalculated")
    monkeypatch.setattr(checks, "participant_level_change_summary", forbidden_recalculation)

    def fake_plot(groups, paths):
        for path in paths:
            path.write_bytes(b"plot fixture")

    monkeypatch.setattr(checks, "plot_sensitivity", fake_plot)
    monkeypatch.setattr(checks, "plot_ppc", fake_plot)
    manifest = checks.refresh_presentation(output, assets)
    assert source.read_bytes() == original
    assert len(manifest["asset_sha256"]) == 5
    assert not (output / "run_manifest.json").exists()
    assert (assets / "application_presentation_manifest.json").is_file()

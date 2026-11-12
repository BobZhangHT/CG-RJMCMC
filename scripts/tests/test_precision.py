import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "precision_tested", ROOT / "scripts" / "precision.py"
)
precision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precision)


def test_single_component_mixture_quantile_matches_student_t():
    weights = np.array([1.0])
    location = np.array([[0.7]])
    scale = np.array([[0.2]])
    got = precision.mixture_quantiles(weights, location, scale, df=11.0).ravel()
    expected = 0.7 + 0.2 * t.ppf([0.025, 0.975], 11.0)
    np.testing.assert_allclose(got, expected, atol=2e-12, rtol=0)


def test_symmetric_two_component_mixture_quantiles_are_symmetric():
    got = precision.mixture_quantiles(
        np.array([0.5, 0.5]),
        np.array([[-1.0], [1.0]]),
        np.array([[0.3], [0.3]]),
        df=20.0,
    ).ravel()
    assert got[0] < -1 and got[1] > 1
    np.testing.assert_allclose(got[0], -got[1], atol=2e-12, rtol=0)


def test_student_t_scale_uses_rate_over_shape_and_inverse_diagonal():
    got = precision.student_t_scale(8.0, 2.0, np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(got, [2.0, 4.0, 6.0])


def test_coverage_mcse_uses_replications_not_point_count():
    cover = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool)
    width = np.ones((2, 4))
    near = np.array([1, 1, 0, 0], dtype=bool)
    points, summary = precision.summarize_coverage_matrix(cover, width, near)
    assert all(row["replications"] == 2 for row in points)
    # Per-replication all-position coverages are 0.5 and 1.0: sd/sqrt(2)=0.25.
    assert summary["all"]["coverage_mcse"] == 0.25
    assert summary["near_le_2"]["positions"] == 2
    assert summary["far_gt_2"]["positions"] == 2


def test_partition_count_and_source_hash_contract():
    states = precision.legal_partitions()
    assert len(states) == 27931
    assert len(set(states)) == len(states)
    observed = precision.validate_protocol_sources()
    source_lock = json.loads(precision.SOURCE_LOCK.read_text(encoding="utf-8"))
    assert observed == source_lock["source_sha256"]


def test_original_protocol_is_archival_not_the_current_source_lock():
    protocol = json.loads(precision.PROTOCOL.read_text(encoding="utf-8"))
    source_lock = json.loads(precision.SOURCE_LOCK.read_text(encoding="utf-8"))
    assert protocol["created_before_precision_runs"] is True
    assert protocol["fixed_source_sha256"] != source_lock["source_sha256"]
    assert precision.PROTOCOL.name == "original_precision_protocol.json"


def test_current_source_lock_rejects_any_mismatch(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("released\n", encoding="utf-8")
    digest = precision.sha256_file(source)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"source_sha256": {"source.py": digest}}), encoding="utf-8"
    )
    assert precision.validate_protocol_sources(lock, tmp_path) == {"source.py": digest}
    source.write_text("modified\n", encoding="utf-8")
    try:
        precision.validate_protocol_sources(lock, tmp_path)
    except RuntimeError as error:
        assert "current release source-hash mismatch" in str(error)
    else:
        raise AssertionError("a mismatched source lock must never be accepted")


def test_configure_paths_binds_public_cli_roots(tmp_path):
    simulation = tmp_path / "simulation"
    application = tmp_path / "application"
    output = tmp_path / "output"
    assets = tmp_path / "assets"
    precision.configure_paths(simulation, application, output, assets)
    assert precision.SIMULATION_RESULTS == simulation.resolve()
    assert precision.APPLICATION_RESULTS == application.resolve()
    assert precision.OUTPUT == output.resolve()
    assert precision.ASSETS == assets.resolve()
    assert precision.CWeightProbe.__init__.__defaults__ == (None,)
    assert precision.provenance_path(output).endswith(str(output.resolve()))


def test_fresh_run_completes_conditional_checks_before_manifest(tmp_path, monkeypatch):
    output = tmp_path / "precision"
    assets = tmp_path / "assets"
    simulation = tmp_path / "simulation"
    application = tmp_path / "application"
    simulation.mkdir()
    (simulation / "resolved_config.json").write_text('{"config": {}}', encoding="utf-8")
    precision.configure_paths(simulation, application, output, assets)

    calls = []
    empty = precision.pd.DataFrame()
    monkeypatch.setattr(
        precision,
        "validate_protocol_sources",
        lambda: {"scripts/kernel.c": "kernel-hash"},
    )
    monkeypatch.setattr(
        precision, "summarize_frozen_simulations", lambda: (empty, empty, empty)
    )
    monkeypatch.setattr(
        precision.formal, "compile_kernel", lambda path: (object(), "kernel-hash")
    )
    monkeypatch.setattr(
        precision,
        "CWeightProbe",
        lambda: type("Probe", (), {"library_path": output / "probe.dll"})(),
    )
    monkeypatch.setattr(precision, "prepare_cases", lambda cfg, application_input: [])
    monkeypatch.setattr(precision, "make_assets", lambda *args: calls.append("assets"))
    monkeypatch.setattr(
        precision, "make_results_md", lambda *args: calls.append("results")
    )
    conditional = [
        {
            "q025_absolute_error": 0.001,
            "q975_absolute_error": 0.002,
            "definition": "conditional test fixture",
        }
    ]
    monkeypatch.setattr(
        precision,
        "application_conditional_interval_check",
        lambda application_input: calls.append("conditional") or conditional,
    )
    monkeypatch.setattr(
        precision,
        "add_conditional_check_to_results",
        lambda rows: calls.append("add-to-results"),
    )
    monkeypatch.setattr(
        precision,
        "refresh_coverage_table_from_machine_summary",
        lambda: calls.append("refresh-coverage"),
    )

    precision.run_all(tmp_path / "raw")

    manifest = json.loads(
        (output / "precision_manifest.json").read_text(encoding="utf-8")
    )
    assert calls == [
        "assets",
        "results",
        "conditional",
        "add-to-results",
        "refresh-coverage",
    ]
    assert manifest["application_conditional_interval_check"] == {
        "segments": 1,
        "maximum_endpoint_absolute_error": 0.002,
        "definition": "conditional test fixture",
    }


def test_structural_boundary_support_not_float_rounding_defines_constants():
    states = precision.legal_partitions()
    support = np.asarray(
        [any(trial in state for state in states) for trial in range(1, 65)]
    )
    assert not support[:7].any()
    assert support[7:56].all()
    assert not support[56:].any()


def test_numeric_diagnostic_smoke_includes_explicit_tail_probability():
    values = np.random.default_rng(9).normal(size=(3, 100))
    got = precision.numeric_diagnostic(values, quantiles=True)
    assert got["status"] == "estimated"
    assert got["tail_ess"] > 0
    assert got["mcse_q025"] > 0
    assert got["mcse_q975"] > 0



def test_precision_output_guard_precedes_any_execution(tmp_path, monkeypatch):
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "historical.json"
    sentinel.write_text("historical", encoding="utf8")
    monkeypatch.setattr(precision, "OUTPUT", output)
    def unexpected():
        raise AssertionError("execution must not start")
    monkeypatch.setattr(precision, "validate_protocol_sources", unexpected)
    import pytest
    with pytest.raises(FileExistsError, match="fresh --output"):
        precision.run_all(iterations=100000, warmup=2000, case_ids=["d5_rep_000003"])
    assert sentinel.read_text(encoding="utf8") == "historical"
    assert list(output.iterdir()) == [sentinel]


def test_precision_invocation_validates_public_arguments_before_writes(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(precision, "OUTPUT", tmp_path / "unused")
    for iterations, warmup, case_ids in [(10, 10, None), (True, 0, None), (2.5, 0, None),
                                          (15, -1, None), (2**31, 0, None),
                                          (15, 2, []), (15, 2, ["unknown"]),
                                          (15, 2, ["d5_rep_000003"] * 2)]:
        with pytest.raises(ValueError):
            precision.run_all(iterations=iterations, warmup=warmup, case_ids=case_ids)
    assert not precision.OUTPUT.exists()


def test_selected_longer_precision_run_records_actual_scope_interface_only(tmp_path, monkeypatch):
    """Planning/forwarding test with mocked executors; no native-pass claim."""
    output = tmp_path / "output"
    simulation = tmp_path / "simulation"
    simulation.mkdir()
    (simulation / "resolved_config.json").write_text('{"config": {}}', encoding="utf8")
    monkeypatch.setattr(precision, "OUTPUT", output)
    monkeypatch.setattr(precision, "SIMULATION_RESULTS", simulation)
    monkeypatch.setattr(precision, "validate_protocol_sources", lambda: {"scripts/kernel.c": "kernel"})
    monkeypatch.setattr(precision.formal, "compile_kernel", lambda path: (object(), "kernel"))
    def probe():
        output.mkdir()
        return type("Probe", (), {"library_path": output / "probe.dll"})()
    monkeypatch.setattr(precision, "CWeightProbe", probe)
    selected = ["d5_rep_000003"]
    calls = []
    def prepare(cfg, application_input, case_ids):
        calls.append(tuple(case_ids))
        return [{"case_id": case_ids[0]}]
    monkeypatch.setattr(precision, "prepare_cases", prepare)
    def execute(case, states, probe, library, iterations, warmup):
        calls.append((iterations, warmup))
        return {"case_id": case["case_id"], "probability_threshold_pass": True, "endpoint_threshold_pass": True}
    monkeypatch.setattr(precision, "run_precision_case", execute)
    def forbidden(*args):
        raise AssertionError("selected scope must not require whole-study outputs")
    monkeypatch.setattr(precision, "summarize_frozen_simulations", forbidden)
    monkeypatch.setattr(precision, "application_conditional_interval_check", forbidden)
    precision.run_all(iterations=100000, warmup=2000, case_ids=selected)
    manifest = json.loads((output / "precision_manifest.json").read_text(encoding="utf8"))
    assert calls == [("d5_rep_000003",), (100000, 2000)]
    assert manifest["invocation"] == {"selected_cases": selected, "full_five_case_scope": False,
                                      "iterations_per_chain": 100000, "warmup_per_chain": 2000,
                                      "original_budget": False, "frozen_simulation_summary_rerun": False}
    assert manifest["conditional_extension"]["invoked"] is True
    assert manifest["application_conditional_interval_check"]["status"] == "not_executed_selected_case_scope"



def test_precision_cli_forwards_explicit_budget_and_case_interface_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(precision, "configure_paths", lambda *args: calls.append(("paths", args)))
    monkeypatch.setattr(precision, "run_all", lambda *args: calls.append(("run", args)))
    precision.main(["--run", "--iterations", "100000", "--warmup", "2000", "--case", "d5_rep_000003", "--output", str(tmp_path / "fresh")])
    assert calls[-1][0] == "run"
    assert calls[-1][1][1:] == (100000, 2000, ["d5_rep_000003"])
    calls.clear()
    precision.main(["--run"])
    assert calls[-1][1][1:] == (15000, 2000, None)


def test_precision_cli_rejects_invalid_plans_without_executor(monkeypatch):
    import pytest
    monkeypatch.setattr(precision, "configure_paths", lambda *args: (_ for _ in ()).throw(AssertionError("invalid plan must fail first")))
    for argv in (["--run", "--iterations", "100", "--warmup", "100"],
                 ["--run", "--case", "unknown"], ["--smoke", "--iterations", "100000"]):
        with pytest.raises(SystemExit) as error:
            precision.main(argv)
        assert error.value.code == 2

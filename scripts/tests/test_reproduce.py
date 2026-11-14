from scripts import reproduce
import pytest


def test_protected_outputs_rejected_before_dry_run():
    with pytest.raises(ValueError, match="protected output"):
        reproduce.main(
            [
                "smoke",
                "--dry-run",
                "--output-root",
                str(reproduce.ROOT / "results/formal"),
            ]
        )


def test_safe_defaults_and_dry_run_paths(tmp_path, capsys):
    out, assets, raw = tmp_path / "out", tmp_path / "assets", tmp_path / "raw"
    assert (
        reproduce.main(
            [
                "smoke",
                "--dry-run",
                "--output-root",
                str(out),
                "--assets",
                str(assets),
                "--application-input",
                str(raw),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert str(out / "smoke") in text
    assert "--smoke" in text
    assert not out.exists() and not assets.exists()


def test_application_and_checks_have_explicit_fresh_roots(tmp_path):
    args = type(
        "Args",
        (),
        {
            "stage": "checks",
            "output_root": tmp_path / "out",
            "assets": tmp_path / "assets",
            "application_input": tmp_path / "raw",
            "workers": 1,
        },
    )()
    text = "\n".join(" ".join(x) for x in reproduce.plan(args))
    assert (
        str(args.output_root / "simulation") in text
        and str(args.output_root / "application") in text
    )
    assert "--simulation-results" in text and "--application-results" in text


import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


from scripts.simulation import validate_resume_target


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_IMPORT_MODULES = (
    "scripts.package",
    "scripts.figures",
    "scripts.simulation_figures",
    "scripts.reproduce",
    "scripts.checks",
    "scripts.precision",
    "scripts.simulation",
    "scripts.application",
    "scripts.small_sample_validation",
    "scripts.constrained_dp_validation",
    "scripts.exact_d5_500",
    "scripts.revision_tables",
    "scripts.coverage_diagnostic",
    "scripts.calibration_sensitivity",
    "scripts.residual_sensitivity",
)


def test_validate_resume_target_accepts_absent_or_empty_target(tmp_path: Path) -> None:
    validate_resume_target(tmp_path / "absent", "release-a")
    empty = tmp_path / "empty"
    empty.mkdir()
    validate_resume_target(empty, "release-a")


def test_validate_resume_target_accepts_matching_manifest(tmp_path: Path) -> None:
    output = tmp_path / "matching"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"run_fingerprint": "release-a"}), encoding="utf-8"
    )
    validate_resume_target(output, "release-a")


def test_validate_resume_target_rejects_mismatching_manifest(tmp_path: Path) -> None:
    output = tmp_path / "mismatching"
    output.mkdir()
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({"run_fingerprint": "release-a"}), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="different run fingerprint"):
        validate_resume_target(output, "release-b")
    assert manifest.read_bytes() == before


def test_validate_resume_target_rejects_nonempty_directory_without_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unidentified"
    output.mkdir()
    existing = output / "existing-result.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonempty output has no run manifest"):
        validate_resume_target(output, "release-a")
    assert existing.read_text(encoding="utf-8") == "{}\n"
    assert not (output / "run_manifest.json").exists()


def test_public_modules_import_from_source_only_checkout(tmp_path: Path) -> None:
    """Imports must not require ignored manuscript, results, reviews, or data."""
    source = tmp_path / "source-only"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    for module in PUBLIC_IMPORT_MODULES:
        filename = module.rsplit(".", 1)[1] + ".py"
        shutil.copy2(ROOT / "scripts" / filename, scripts / filename)
    for filename in ("kernel.c", "probe.c"):
        shutil.copy2(ROOT / "scripts" / filename, scripts / filename)
    assert not any(
        (source / name).exists()
        for name in ("data", "manuscript", "results", "reviews")
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    command = "; ".join(f"import {module}" for module in PUBLIC_IMPORT_MODULES)
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_revision_plan_has_real_stages_and_no_implicit_campaign(tmp_path,capsys):
    assert reproduce.main(["revision", "--dry-run", "--output-root",str(tmp_path/"new"),"--assets",str(tmp_path/"assets")]) == 0
    commands=capsys.readouterr().out
    for name in ("small_sample_validation.py","constrained_dp_validation.py","exact_d5_500.py","revision_tables.py"):
        assert name in commands
    assert "simulation.py --config" not in commands
    assert "--simulation-root" in commands


def test_revision_smoke_is_self_contained_and_dry(tmp_path,capsys):
    assert reproduce.main(["revision-smoke","--dry-run","--output-root",str(tmp_path/"new")])==0
    commands=capsys.readouterr().out
    assert commands.count("--smoke")==4
    assert "--application-input" not in commands
    assert not (tmp_path/"new").exists()


def test_original_all_explicitly_excludes_revision_without_optin(tmp_path,capsys,monkeypatch):
    monkeypatch.setattr(reproduce,"ROOT",tmp_path/"source")
    reproduce.main(["all","--dry-run","--output-root",str(reproduce.ROOT/"results/new")])
    text=capsys.readouterr().out
    assert "--include-revision" in text and "exact_d5_500.py" not in text
    reproduce.main(["all","--dry-run","--include-revision","--output-root",str(reproduce.ROOT/"results/new")])
    assert "exact_d5_500.py" in capsys.readouterr().out


def test_validated_n35_cache_is_available_to_fresh_benchmark(tmp_path,capsys):
    cache=tmp_path/"n35_cache"
    reproduce.main(["benchmark","--dry-run","--small-sample-cache",str(cache),"--output-root",str(tmp_path/"new")])
    text=capsys.readouterr().out
    assert "--small-sample-root "+str(cache) in text


def test_table_residual_results_forwarded_without_fitting(tmp_path, capsys):
    residual = tmp_path / "residual"
    reproduce.main(["tables", "--dry-run", "--residual-results", str(residual),
                    "--output-root", str(tmp_path / "out")])
    text = capsys.readouterr().out
    assert "--residual-results " + str(residual) in text
    assert "residual_sensitivity.py" not in text
    assert not residual.exists()

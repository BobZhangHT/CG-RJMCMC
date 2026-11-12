import csv
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "supporting_data_builder", ROOT / "scripts/package.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def coverage_fixture():
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(
        [
            "dgp",
            "trial",
            "replications",
            "near_boundary",
            "coverage",
            "coverage_mcse",
            "mean_width",
            "width_mcse",
        ]
    )
    for d in range(6):
        for trial in range(1, 65):
            writer.writerow([f"D{d}", trial, 500, "False", 0.95, 0.01, 0.02, 0.001])
    return stream.getvalue().encode("utf-8")


def create_fixture(root):
    # Keep unrelated evidence tiny; the pointwise contract still uses all 384 coordinates.
    for relative in builder.selected_paths():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == builder.POINTWISE:
            content = coverage_fixture()
        elif relative.endswith(".csv"):
            content = b"field_a,field_b\r\n1,2\r\n"
        elif relative.endswith(".json"):
            content = b'{"fixture": true}\n'
        else:
            content = b"fixture source bytes\r\n"
        destination.write_bytes(content)


def test_fixture_build_preserves_bytes_and_is_deterministic(tmp_path):
    root, output, archive = (
        tmp_path / "repo",
        tmp_path / "package",
        tmp_path / "package.zip",
    )
    create_fixture(root)
    first = builder.build(root, output, archive)
    first_zip = archive.read_bytes()
    manifest = json.loads((output / "PACKAGE_MANIFEST.json").read_bytes())
    for record in manifest["copied_files"]:
        assert (output / record["path"]).read_bytes() == (
            root / record["source_path"]
        ).read_bytes()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == sorted(bundle.namelist())
        assert all(entry.date_time == builder.ZIP_DATE for entry in bundle.infolist())
        assert all(
            entry.compress_type == zipfile.ZIP_STORED for entry in bundle.infolist()
        )
    # Modification times cannot affect the deterministic ZIP.
    (root / builder.POINTWISE).touch()
    second = builder.build(root, output, archive)
    assert archive.read_bytes() == first_zip
    assert first == second
    assert builder.verify(output)["pointwise_coverage_rows"] == 384


def test_missing_or_duplicate_coverage_is_rejected_before_writes(tmp_path):
    root, output, archive = (
        tmp_path / "repo",
        tmp_path / "package",
        tmp_path / "package.zip",
    )
    create_fixture(root)
    coverage = root / builder.POINTWISE
    lines = coverage.read_bytes().splitlines(keepends=True)
    coverage.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(ValueError, match="384 unique rows"):
        builder.build(root, output, archive)
    assert not output.exists() and not archive.exists()
    # A duplicated position is invalid even when total row count remains 384.
    coverage.write_bytes(b"".join(lines[:-1] + [lines[1]]))
    with pytest.raises(ValueError, match="384 unique rows"):
        builder.build(root, output, archive)
    assert not output.exists() and not archive.exists()


def test_verification_uses_packaged_pointwise_path(tmp_path, monkeypatch):
    root, output = tmp_path / "repo", tmp_path / "package"
    create_fixture(root)
    builder.build(root, output, tmp_path / "package.zip")
    monkeypatch.setattr(builder, "POINTWISE", "unrelated/default.csv")
    assert builder.verify(output)["pointwise_coverage_rows"] == 384


def test_integrity_check_detects_tampering_and_preserves_unknown_files(tmp_path):
    root, output, archive = (
        tmp_path / "repo",
        tmp_path / "package",
        tmp_path / "package.zip",
    )
    create_fixture(root)
    builder.build(root, output, archive)
    (output / builder.POINTWISE).write_bytes(coverage_fixture() + b"\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        builder.verify(output)
    extra = output / "author_notes.txt"
    extra.write_text("keep this", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        builder.build(root, output, archive)
    assert extra.read_text(encoding="utf-8") == "keep this"


@pytest.mark.integration
def test_actual_evidence_package_has_all_reported_rows_and_no_raw_draws(tmp_path):
    missing = [
        relative
        for relative in builder.selected_paths()
        if not (ROOT / relative).is_file()
    ]
    if missing:
        pytest.skip(
            "Full-study generated evidence not installed; fixture-based package tests still run."
        )
    output, archive = tmp_path / "actual_package", tmp_path / "actual_package.zip"
    result = builder.build(ROOT, output, archive)
    manifest = json.loads((output / "PACKAGE_MANIFEST.json").read_bytes())
    files = {row["path"]: row for row in manifest["copied_files"]}
    assert result["pointwise_coverage_rows"] == 384
    assert files[builder.POINTWISE]["rows"] == 384
    for case in builder.CASES:
        base = f"{builder.PRECISION}/cases/{case}"
        assert files[f"{base}/trajectory_diagnostics.csv"]["rows"] == 64
        assert files[f"{base}/discrete_diagnostics.csv"]["rows"] == 70
        assert files[f"{base}/top_partition_stability.csv"]["rows"] >= 1
        assert {"mcse_q025", "mcse_q975", "tail_ess"}.issubset(
            files[f"{base}/trajectory_diagnostics.csv"]["fields"]
        )
    assert files[f"{builder.PRECISION}/paired_rmse_differences.csv"]["rows"] == 60
    assert files[f"{builder.APPLICATION}/case_summary.csv"]["rows"] == 12
    assert files[f"{builder.APPLICATION}/curve_summary.csv"]["rows"] == 768
    assert files[f"{builder.APPLICATION}/ppc_check_summary.csv"]["rows"] == 18
    assert files[f"{builder.APPLICATION}/ppc_draws.csv"]["rows"] == 18000
    assert all(
        Path(path).suffix not in {".npz", ".npy", ".dll", ".so", ".exe"}
        for path in files
    )
    assert not any(
        Path(path).name
        in {"participant_contrasts.csv", "residual_summary.csv", "lag1_summary.csv"}
        for path in files
    )
    # The unmodified probe include resolves to the supplied original kernel.
    probe = output / "scripts/probe.c"
    assert (probe.parent / "kernel.c").resolve().is_file()
    for record in manifest["copied_files"]:
        assert (output / record["path"]).read_bytes() == (
            ROOT / record["source_path"]
        ).read_bytes()

"""Package existing review evidence byte-for-byte; never rerun or alter analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "review_supporting_data"
PRECISION = "results/reproduction/precision"
APPLICATION = "results/reproduction/checks"
SIMULATION_RESULTS = "results/reproduction/simulation"
APPLICATION_RESULTS = "results/reproduction/application"
REPORTED_ASSETS = "manuscript/reproduced"
CASES = (
    "application_high_first",
    "application_low_first",
    "d5_rep_000001",
    "d5_rep_000002",
    "d5_rep_000003",
)
CASE_FILES = (
    "trajectory_diagnostics.csv",
    "discrete_diagnostics.csv",
    "top_partition_stability.csv",
    "summary.json",
)
POINTWISE = f"{PRECISION}/frozen_pointwise_coverage.csv"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def selected_paths() -> tuple[str, ...]:
    paths = [
        f"{PRECISION}/{name}"
        for name in (
            "frozen_pointwise_coverage.csv",
            "frozen_coverage_summary.csv",
            "paired_rmse_differences.csv",
            "application_conditional_interval_check.csv",
            "precision_manifest.json",
        )
    ]
    paths += [
        f"{PRECISION}/cases/{case}/{name}" for case in CASES for name in CASE_FILES
    ]
    paths += [
        f"{APPLICATION}/{name}"
        for name in (
            "case_summary.csv",
            "curve_summary.csv",
            "residual_summary.csv",
            "participant_contrasts.csv",
            "participant_level_summary.csv",
            "lag1_summary.csv",
            "ppc_check_summary.csv",
            "ppc_draws.csv",
            "application_report.md",
            "run_manifest.json",
        )
    ]
    paths += [
        "scripts/config/application_protocol.md",
        "scripts/config/original_precision_protocol.json",
        "scripts/config/reproduction_sources.json",
        *(
            f"{SIMULATION_RESULTS}/{name}"
            for name in (
                "per_method.csv",
                "a3_diagnostics.csv",
                "summary.csv",
                "resolved_config.json",
                "geometry_manifest.json",
                "run_manifest.json",
                "run_status.json",
            )
        ),
        *(
            f"{APPLICATION_RESULTS}/{name}"
            for name in ("results.json", "heldout_rmse.csv", "run_manifest.json")
        ),
        *(
            f"{REPORTED_ASSETS}/tables/{name}"
            for name in (
                "table1_design.csv",
                "table2_rmse.csv",
                "table3_structural_recovery.csv",
                "table4_computation_sensitivity.csv",
                "table5_real_data.csv",
                "publication_manifest.json",
            )
        ),
        f"{REPORTED_ASSETS}/figures/fig2_dgp_trajectories.csv",
        f"{REPORTED_ASSETS}/figures/fig3_structural_recovery.csv",
        "scripts/config/r61_formal.yaml",
        "scripts/config/r61_congruency_geometry.csv",
        "pyproject.toml",
    ]
    paths += [
        f"scripts/{name}"
        for name in (
            "package.py",
            "precision.py",
            "checks.py",
            "simulation.py",
            "application.py",
            "kernel.c",
        )
    ]
    return tuple(sorted(paths))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_data(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("CSV requires unique, nonempty field names")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width differs from its header")
    return fields, rows


def check_pointwise(data: bytes) -> None:
    fields, rows = csv_data(data)
    required = {
        "dgp",
        "trial",
        "replications",
        "coverage",
        "coverage_mcse",
        "mean_width",
        "width_mcse",
        "near_boundary",
    }
    if not required.issubset(fields):
        raise ValueError("pointwise coverage schema is incomplete")
    expected = {(f"D{d}", t) for d in range(6) for t in range(1, 65)}
    keys = [(row["dgp"], int(row["trial"])) for row in rows]
    if len(rows) != 384 or len(set(keys)) != 384 or set(keys) != expected:
        raise ValueError(
            "pointwise coverage requires 384 unique rows: six scenarios by 64 positions"
        )
    for row in rows:
        if int(row["replications"]) != 500:
            raise ValueError("each coverage position must contain 500 replications")
        if not 0 <= float(row["coverage"]) <= 1:
            raise ValueError("coverage must be between zero and one")


def describe_file(path: str, data: bytes) -> dict:
    record = {
        "path": path,
        "source_path": path,
        "bytes": len(data),
        "sha256": sha256(data),
    }
    if path.endswith(".csv"):
        fields, rows = csv_data(data)
        record.update(format="csv", rows=len(rows), fields=fields)
    elif path.endswith(".json"):
        content = json.loads(data)
        record.update(
            format="json",
            top_level_keys=sorted(content) if isinstance(content, dict) else [],
        )
    else:
        record["format"] = Path(path).suffix.lstrip(".")
    return record


def index_text(records: list[dict]) -> str:
    """Generate a compact evidence index from the actual packaged schemas."""
    lines = [
        "# Supporting data",
        "",
        "This package copies saved evidence, reported source data, and source bytes without rerunning analyses. "
        "PACKAGE_MANIFEST.json records each path, SHA-256, byte count and CSV schema. "
        "Raw participant trials, retained chains and binaries are not included.",
        "",
        "## Contents and interpretation",
        "",
        f"- `{PRECISION}/`: pointwise coverage, replication-level coverage summaries, "
        "paired trajectory/prediction RMSE comparisons and five exact/MCMC precision cases.",
        "- Pointwise coverage has 384 rows (six scenarios by 64 positions), each based on "
        "500 independently generated datasets. Near boundaries means distance at most two bins. "
        "Stratum MCSEs use dataset-level averages, not independent positions. "
        "The simultaneous_all64 row counts datasets where all pointwise intervals cover; "
        "it does not make those intervals simultaneous credible bands.",
        "- Paired RMSE differences are method_a minus method_b; negative values favor "
        "method_a. MCSE describes uncertainty in a simulation mean. B0, B1, B2 and A3 "
        "denote NoEffect, ConstEffect, FixedSplit and CG-RJMCMC.",
        "- The five precision cases are both application groups and D5 replicates 1-3. "
        "Exact values enumerate 27,931 partitions. Diagnostic status distinguishes "
        "estimable, constant and unobserved indicators; blank cells are not perfect mixing. "
        "Conditional segment intervals refer to the empirical modal partition, not BMA intervals.",
        f"- `{APPLICATION}/`: 12 fixed sensitivity fits, trajectory summaries and "
        "posterior-predictive discrepancy records for the two condition-order groups. "
       "Participant-level phase changes resample complete participants rather than response rows.",
        f"- `{SIMULATION_RESULTS}/per_method.csv`: all replication-level method metrics "
        "used for distributional and Monte Carlo summaries.",
        f"- `{REPORTED_ASSETS}/`: source values for the reported tables and for the "
        "trajectory and structural-recovery figure geometries.",
        "- PPC observed discrepancies depend on each posterior draw; observed_median "
        "summarizes those discrepancies, not a fixed plug-in residual statistic. "
        "participant_contrast_sd uses participant-by-phase cells. Lag pairs stay within "
        "participants and blocks. Folded tail areas are descriptive, not calibrated tests.",
        "- Sampling precision, model adequacy and interval calibration are separate checks. "
        "The D5 coverage shortfall, early-boundary sensitivity and application residual "
        "mismatch remain limitations.",
        "",
        "## Verification and reproduction",
        "",
        "From the extracted package root: `python scripts/package.py --verify .`.",
        "Full numerical reproduction requires the public code checkout and the "
        "RawData_Experiment1 files from "
        "https://osf.io/6n597/. Run `python scripts/reproduce.py all` in that checkout. "
        "Retained Markov-chain arrays and simulated response matrices are not needed to "
        "inspect the reported source values, but rerunning the stochastic analyses "
        "regenerates them from the supplied configurations and deterministic seed rules.",
        "The archival protocols remain historical records, not new preregistration. "
        "Use the package manifest for the copied inventory and the current release "
        "source lock for reruns. Older analysis manifests may name unbundled paths.",
        "",
        "## CSV schemas",
        "",
    ]
    for record in records:
        if record["format"] == "csv":
            fields = ", ".join(f"`{field}`" for field in record["fields"])
            lines.append(f"- `{record['path']}`: {record['rows']} rows; {fields}.")
    return "\n".join(lines) + "\n"


def encode_json(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build(root: Path, output: Path, archive: Path) -> dict:
    root, output, archive = root.resolve(), output.resolve(), archive.resolve()
    if output == root or output in root.parents or archive.is_relative_to(output):
        raise ValueError(
            "output must be separate from the repository root and archive must be outside the package"
        )
    payload = {}
    for relative in selected_paths():
        source = (root / relative).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"source escapes repository: {relative}")
        payload[relative] = source.read_bytes()
    check_pointwise(payload[POINTWISE])
    records = [describe_file(path, data) for path, data in sorted(payload.items())]
    payload["README.md"] = index_text(records).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "package": PACKAGE_NAME,
        "scope": "curated saved evidence; no participant trial data or retained parameter draws",
        "copied_files": records,
        "generated_files": [
            {
                "path": "README.md",
                "bytes": len(payload["README.md"]),
                "sha256": sha256(payload["README.md"]),
            }
        ],
        "pointwise_coverage_rows": 384,
        "precision_cases": list(CASES),
        "csv_rows_total": sum(r.get("rows", 0) for r in records),
        "archive": {
            "timestamp": list(ZIP_DATE),
            "compression": "stored",
            "root": PACKAGE_NAME,
        },
    }
    payload["PACKAGE_MANIFEST.json"] = encode_json(manifest)
    if output.exists():
        unexpected = {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        } - set(payload)
        if unexpected:
            raise ValueError(
                f"output contains unexpected files; preserve/move them before rebuilding: {sorted(unexpected)}"
            )
        if any(p.is_symlink() for p in output.rglob("*")) or output.is_symlink():
            raise ValueError("package output must not contain symbolic links")
    # All source and schema checks finish before any generated file is written.
    for relative, data in sorted(payload.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for relative, data in sorted(payload.items()):
            info = zipfile.ZipInfo(f"{PACKAGE_NAME}/{relative}", ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            bundle.writestr(info, data)
    result = verify(output)
    with zipfile.ZipFile(archive) as bundle:
        expected_names = [f"{PACKAGE_NAME}/{p}" for p in sorted(payload)]
        if bundle.namelist() != expected_names or any(
            bundle.read(f"{PACKAGE_NAME}/{p}") != data for p, data in payload.items()
        ):
            raise ValueError("archive does not match the prepared package")
    result.update(
        archive=str(archive),
        archive_bytes=archive.stat().st_size,
        archive_sha256=sha256(archive.read_bytes()),
        index=str(output / "README.md"),
    )
    return result


def verify(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "PACKAGE_MANIFEST.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("package") != PACKAGE_NAME:
        raise ValueError("unsupported package manifest")
    records = manifest["copied_files"] + manifest["generated_files"]
    paths = [row["path"] for row in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate manifest paths")
    expected = set(paths) | {"PACKAGE_MANIFEST.json"}
    actual = {
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    }
    if actual != expected:
        raise ValueError("package inventory differs from manifest")
    for record in records:
        path = (output / record["path"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("manifest path escapes package")
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
            raise ValueError(f"hash/size mismatch: {record['path']}")
        if record.get("format") == "csv":
            fields, rows = csv_data(data)
            if fields != record["fields"] or len(rows) != record["rows"]:
                raise ValueError(f"CSV schema mismatch: {record['path']}")
    # Locate the evidence from the verified inventory, not the builder's local
    # default. This also supports standalone verification of relocated bundles.
    pointwise = [p for p in paths if Path(p).name == "frozen_pointwise_coverage.csv"]
    if len(pointwise) != 1:
        raise ValueError("package must identify exactly one pointwise coverage file")
    check_pointwise((output / pointwise[0]).read_bytes())
    return {
        "files": len(expected),
        "copied_files": len(manifest["copied_files"]),
        "csv_rows_total": manifest["csv_rows_total"],
        "pointwise_coverage_rows": 384,
        "package_bytes": sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        ),
    }


def main() -> None:
    global PRECISION, APPLICATION, SIMULATION_RESULTS, APPLICATION_RESULTS
    global REPORTED_ASSETS, POINTWISE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", type=Path, metavar="PACKAGE_DIR")
    parser.add_argument("--precision-results", type=Path)
    parser.add_argument("--application-checks", type=Path)
    parser.add_argument("--simulation-results", type=Path)
    parser.add_argument("--application-results", type=Path)
    parser.add_argument("--reported-assets", type=Path)
    args = parser.parse_args()
    # Public reruns can use fresh result directories without touching the paper's
    # frozen records. Keeping relative paths makes the package independently inspectable.
    for argument, variable in (
        (args.precision_results, "PRECISION"),
        (args.application_checks, "APPLICATION"),
        (args.simulation_results, "SIMULATION_RESULTS"),
        (args.application_results, "APPLICATION_RESULTS"),
        (args.reported_assets, "REPORTED_ASSETS"),
    ):
        if argument is not None:
            globals()[variable] = (
                argument.resolve().relative_to(args.root.resolve()).as_posix()
            )
    POINTWISE = f"{PRECISION}/frozen_pointwise_coverage.csv"
    if args.verify is not None:
        result = verify(args.verify)
    else:
        output = (
            args.output_dir
            or args.root
            / "manuscript"
            / "reproduced"
            / "supporting_information"
            / PACKAGE_NAME
        )
        result = build(args.root, output, args.archive or output.with_suffix(".zip"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

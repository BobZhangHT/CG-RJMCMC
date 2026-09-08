#!/usr/bin/env python
"""One entry point for tests, analysis, figures, packaging and source export."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("smoke", "simulation", "application", "checks", "figures", "supporting")
STAGES = ("validate", "test", *ANALYSES, "export", "all")
PUBLIC_FILES = {
    ".gitignore",
    ".gitattributes",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
}


def plan(args: argparse.Namespace) -> list[list[str]]:
    """Build explicit commands; planning never writes or starts an analysis."""
    out, assets, raw = args.output_root, args.assets, args.application_input
    sim, app = out / "simulation", out / "application"
    precision, checks = out / "precision", out / "checks"
    package = assets / "supporting_information" / "review_supporting_data"

    def task(script: str, *options: object) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, options)]

    simulation = (
        "--config",
        ROOT / "scripts/config/r61_formal.yaml",
        "--stage",
        "all",
        "--workers",
        args.workers,
    )
    commands = {
        "test": [[sys.executable, "-m", "pytest", "-q"]],
        "smoke": [
            task("simulation.py", *simulation, "--smoke", "--output", out / "smoke")
        ],
        "simulation": [task("simulation.py", *simulation, "--output", sim)],
        "application": [task("application.py", "--input", raw, "--output", app)],
        "checks": [
            task(
                "precision.py",
                "--run",
                "--simulation-results",
                sim,
                "--application-results",
                app,
                "--application-input",
                raw,
                "--output",
                precision,
                "--assets",
                assets,
            ),
            task("checks.py", "--input", raw, "--output", checks, "--assets", assets),
        ],
        "figures": [
            task(
                "figures.py",
                "--simulation",
                sim,
                "--application",
                app,
                "--output-root",
                assets,
            )
        ],
        "supporting": [
            task(
                "package.py",
                "--root",
                ROOT,
                "--output-dir",
                package,
                "--archive",
                package.with_suffix(".zip"),
                "--precision-results",
                precision,
                "--application-checks",
                checks,
                "--simulation-results",
                sim,
                "--application-results",
                app,
            )
        ],
    }
    wanted = ANALYSES if args.stage == "all" else (args.stage,)
    return [cmd for stage in wanted for cmd in commands.get(stage, [])]


def validate(args: argparse.Namespace) -> None:
    """Verify the current source lock and compiler without running experiments."""
    lock = ROOT / "scripts/config/reproduction_sources.json"
    expected = json.loads(lock.read_text(encoding="utf-8"))["source_sha256"]
    if not expected:
        raise RuntimeError("empty source lock")
    for relative, digest in expected.items():
        path = ROOT / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(f"source lock mismatch: {relative}")
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc was not found on PATH")
    print("Source lock and GCC: OK")


def validate_output_paths(args: argparse.Namespace) -> None:
    """Protect the manuscript sources and historical results from a new run."""
    protected = [
        ROOT / name
        for name in (
            "scripts",
            ".github",
            "results/formal",
            "results/revision",
            "results/optimization",
            "manuscript/figures",
            "manuscript/tables",
            "manuscript/reviews",
            "manuscript/supporting_information",
            "manuscript/submission_figures",
        )
    ]
    for path in (args.output_root, args.assets):
        if path in (ROOT, ROOT / "results", ROOT / "manuscript") or any(
            path == p or path.is_relative_to(p) or p.is_relative_to(path)
            for p in protected
        ):
            raise ValueError(
                f"protected output location: {path}; choose a fresh subdirectory"
            )
    if args.stage in {"supporting", "all"} and not args.output_root.is_relative_to(
        ROOT
    ):
        raise ValueError(
            "supporting-data packaging requires an output root inside the repository"
        )


def export_sources(destination: Path | None = None) -> list[str]:
    """List/copy Git-visible source files. Never stage, commit or push."""
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = sorted(set(filter(None, raw.decode("utf-8").split("\0"))))
    for relative in paths:
        source = ROOT / relative
        allowed = relative in PUBLIC_FILES or Path(relative).parts[0] in {
            "scripts",
            ".github",
        }
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "--no-index", "-q", "--", relative], cwd=ROOT
            ).returncode
            == 0
        )
        if (
            not allowed
            or ignored
            or source.is_symlink()
            or not source.resolve().is_relative_to(ROOT)
        ):
            raise ValueError(f"non-public upload candidate: {relative}")
    if destination is not None:
        destination = destination.resolve()
        if (
            destination == ROOT
            or ROOT.is_relative_to(destination)
            or destination.is_relative_to(ROOT / "scripts")
            or (destination.exists() and any(destination.iterdir()))
        ):
            raise ValueError(
                "export destination must be new/empty and outside source directories"
            )
        for relative in paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    print("\n".join(paths))
    print(f"{len(paths)} public files; no Git changes made.")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=STAGES, default="validate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--application-input",
        type=Path,
        default=ROOT / "data/external/RawData_Experiment1",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results/reproduction"
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "manuscript/reproduced")
    parser.add_argument(
        "--export-dir", type=Path, help="new empty source-export directory"
    )
    args = parser.parse_args(argv)
    if args.stage == "test" and not (ROOT / "scripts/tests").is_dir():
        parser.error("tests are local-only and are not included in the public release")
    for name in ("output_root", "assets", "application_input"):
        setattr(args, name, getattr(args, name).resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "export":
        export_sources(None if args.dry_run else args.export_dir)
        return 0
    validate_output_paths(args)
    if args.stage in {"validate", "all"} and not args.dry_run:
        validate(args)
    commands = plan(args)
    if args.dry_run:
        for item in commands:
            print(subprocess.list2cmdline(item))
        return 0
    if (
        args.stage in {"all", "application", "checks"}
        and not args.application_input.is_dir()
    ):
        parser.error("obtain external RawData_Experiment1 before running this stage")
    for item in commands:
        subprocess.run(item, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

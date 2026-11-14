#!/usr/bin/env python
"""Explicit primary and revision-stage reproduction workflows; no Git publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import tomllib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("smoke", "simulation", "application", "checks", "figures", "supporting")
REVISION_STAGES = ("small-sample", "benchmark", "exact-d5", "tables", "revision", "revision-smoke")
STAGES = ("validate", "test", *ANALYSES, *REVISION_STAGES, "export", "all")
PUBLIC_FILES = {
    ".gitignore",
    ".gitattributes",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "reports/GITHUB_REPRODUCIBILITY_REPAIR.md",
    "reports/REPRODUCTION_MATRIX.md",
    "reports/TEST_RESULTS.md",
    "reports/PUBLIC_RELEASE_CHECKLIST.md",
}


def plan(args: argparse.Namespace) -> list[list[str]]:
    """Build real commands without writing outputs; revision work is opt-in."""
    out, assets, raw = args.output_root, args.assets, args.application_input
    sim = getattr(args, "simulation_results", None) or out / "simulation"
    app = getattr(args, "application_results", None) or out / "application"
    rev = out / "revision"
    small, benchmark, exact = rev / "small_sample", rev / "benchmark", rev / "exact_d5"
    precision, checks = out / "precision", out / "checks"
    package = assets / "supporting_information" / "review_supporting_data"
    def task(script: str, *options: object) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, options)]
    def add_option(options: list, flag: str, value: object) -> None:
        if value is not None:
            options.extend((flag, value))
    simulation = ("--config", ROOT / "scripts/config/r61_formal.yaml", "--stage", "all", "--workers", args.workers)
    small_options = ["--full", "--output-root", small, "--workers", args.workers]
    design = getattr(args, "design_input", None)
    if design is not None:
        small_options.extend(("--design-input", design))
    else:
        small_options.extend(("--application-input", raw))
    add_option(small_options, "--cache-root", getattr(args, "small_sample_cache", None))
    add_option(small_options, "--statistic-cache-root", getattr(args, "benchmark_cache", None))
    benchmark_options = ["--full", "--output-root", benchmark, "--workers", args.workers]
    benchmark_cache = getattr(args, "benchmark_cache", None)
    balanced = getattr(args, "balanced_d5_root", None)
    if benchmark_cache is not None:
        benchmark_options.extend(("--cache-root", benchmark_cache, "--input-root", ROOT))
    else:
        if balanced is not None:
            benchmark_options.extend(("--primary-root", sim, "--balanced-d5-root", balanced))
        else:
            benchmark_options.extend(("--simulation-root", sim))
        benchmark_options.extend(("--small-sample-root", getattr(args, "small_sample_cache", None) or small, "--application-input", raw))
        add_option(benchmark_options, "--design-input", design)
    exact_cache = getattr(args, "exact_cache", None)
    if exact_cache is not None:
        exact_options = ["--reaggregate-cache", "--cached-exact-root", exact_cache, "--output", exact]
    else:
        exact_options = ["--full", "--simulation-results", balanced / "main" if balanced else sim,
                         "--statistics-root", benchmark / "statistics/D5", "--records-root", benchmark / "replicates/D5",
                         "--output", exact, "--workers", min(args.workers, 12)]
    table_options = ["--simulation", sim, "--application", app / "results.json", "--benchmark", benchmark,
                     "--small-sample", small, "--output-root", assets / "revision_tables"]
    if balanced is not None:
        table_options.extend(("--simulation", balanced / "main"))
    add_option(table_options, "--residual-results", getattr(args, "residual_results", None))
    repeats = getattr(args, "runtime_repeats", None)
    if repeats is not None:
        table_options.extend(("--runtime-repeats", repeats))
    else:
        table_options.extend(("--timing", "--application-input", raw))
    commands = {
        "test": [[sys.executable, "-m", "pytest", "-q"]],
        "smoke": [task("simulation.py", *simulation, "--smoke", "--output", out / "smoke")],
        "simulation": [task("simulation.py", *simulation, "--output", out / "simulation")],
        "application": [task("application.py", "--input", raw, "--output", out / "application")],
        "checks": [task("precision.py", "--run", "--simulation-results", sim, "--application-results", app,
                        "--application-input", raw, "--output", precision, "--assets", assets),
                   task("checks.py", "--input", raw, "--output", checks, "--assets", assets)],
        "small-sample": [task("small_sample_validation.py", *small_options)],
        "benchmark": [task("constrained_dp_validation.py", *benchmark_options)],
        "exact-d5": [task("exact_d5_500.py", *exact_options)],
        "tables": [task("revision_tables.py", *table_options)],
        "figures": [task("figures.py", "--simulation", sim, "--application", app, "--output-root", assets)],
        "supporting": [task("package.py", "--root", ROOT, "--output-dir", package,
                            "--archive", package.with_suffix(".zip"), "--precision-results", precision,
                            "--application-checks", checks, "--simulation-results", sim,
                            "--application-results", app, "--reported-assets", assets)],
        "revision-smoke": [task("precision.py", "--smoke", "--output", out / "revision_smoke/precision"),
                           task("small_sample_validation.py", "--smoke", "--output-root", out / "revision_smoke/small_sample", "--workers", args.workers),
                           task("constrained_dp_validation.py", "--smoke", "--output-root", out / "revision_smoke/benchmark"),
                           task("exact_d5_500.py", "--smoke", "--output", out / "revision_smoke/exact_d5")],
    }
    if args.stage == "revision":
        wanted = ("small-sample", "benchmark", "exact-d5", "tables")
    elif args.stage == "all":
        wanted = ("test", "smoke", "simulation", "application", "checks")
        if getattr(args, "include_revision", False):
            wanted += ("small-sample", "benchmark", "exact-d5", "figures", "tables", "supporting")
        else:
            wanted += ("figures", "supporting")
    elif args.stage == "figures" and getattr(args, "include_revision", False):
        wanted = ("figures", "tables")
    else:
        wanted = (args.stage,)
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
    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12 or newer is required")
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    for requirement in requirements:
        name, version = requirement.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"missing dependency {requirement}; install -e .") from exc
        if actual != version:
            raise RuntimeError(f"dependency mismatch: {name}=={actual}; expected {version}")
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc was not found on PATH")
    compiler = subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]
    print(f"Source lock and pinned dependencies: OK; {compiler}")


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
            "PLOS_ONE_FINAL_SUBMISSION",
            "revision_plosone",
            "final_audit_20261112",
            "production_final_20261112",
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
    parser.add_argument("--include-revision", action="store_true", help="include expensive revision validations in all, or final revision panels in figures")
    parser.add_argument("--simulation-results", type=Path, help="read-only existing simulation root for downstream stages")
    parser.add_argument("--application-results", type=Path, help="read-only existing application result directory")
    parser.add_argument("--balanced-d5-root", type=Path, help="optional legacy split D5 root with main/sensitivity/kmax subdirectories")
    parser.add_argument("--small-sample-cache", type=Path, help="read-only original N35 caches; validate and aggregate without new MCMC")
    parser.add_argument("--benchmark-cache", type=Path, help="read-only stored benchmark inputs; objectives are recalculated")
    parser.add_argument("--exact-cache", type=Path, help="read-only all500 exact cache; verify and reaggregate, not new CDF inversion")
    parser.add_argument("--design-input", type=Path, help="optional frozen application-derived layout NPZ for N35")
    parser.add_argument("--runtime-repeats", type=Path, help="saved 31-repeat-per-group PELT timing CSV; otherwise tables measures new timings")
    parser.add_argument("--residual-results", type=Path, help="completed residual-sensitivity result root for Table 5 Panel B; no new fits")
    args = parser.parse_args(argv)
    if args.stage == "test" and not (ROOT / "scripts/tests").is_dir():
        parser.error("public tests are missing; obtain a complete source checkout")
    for name in ("output_root", "assets", "application_input", "simulation_results", "application_results", "balanced_d5_root", "small_sample_cache", "benchmark_cache", "exact_cache", "design_input", "runtime_repeats", "residual_results"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.stage in {"all", "simulation", "application"} and (args.simulation_results or args.application_results):
        parser.error("existing-result roots are read-only downstream inputs; use individual stages or revision instead of all")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "export":
        export_sources(None if args.dry_run else args.export_dir)
        return 0
    validate_output_paths(args)
    if args.stage in {"validate", "all"} and not args.dry_run:
        validate(args)
    if args.stage == "all" and not args.include_revision:
        print("Scope: original primary/sensitivity/application workflow. Revision validations and final added table panels require --include-revision or the revision stage.")
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

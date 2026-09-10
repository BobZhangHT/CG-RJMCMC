#!/usr/bin/env python
"""Rebuild simulation assets with archived D0--D4 and revised D5 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from scripts import simulation_figures as figures
except ModuleNotFoundError:
    import simulation_figures as figures


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root: Path) -> tuple[list[dict], dict, dict]:
    ok, reason, status = figures.completion(root)
    if not ok or status is None:
        raise RuntimeError(reason)
    fingerprint = status["run_fingerprint"]
    records, paths = figures.collect_replicates(root, fingerprint)
    return records, paths, figures.read_json(root / "resolved_config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, default=ROOT / "results/formal/r61")
    parser.add_argument(
        "--revised-root",
        type=Path,
        default=ROOT / "results/revision/d5_balanced_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/revision/d5_balanced_v1/publication_assets",
    )
    args = parser.parse_args()
    archived = args.archived.resolve()
    revised = args.revised_root.resolve()
    output = args.output.resolve()
    figure_dir, table_dir = output / "figures", output / "tables"
    reporting_dir = output / "combined_reporting"
    for directory in (figure_dir, table_dir, reporting_dir):
        directory.mkdir(parents=True, exist_ok=True)

    old_records, old_paths, old_config = load_run(archived)
    selected_old_records = [row for row in old_records if row["dgp"] != "D5"]
    selected_old_paths = {
        key: value for key, value in old_paths.items() if "D5" not in key[1]
    }

    new_records: list[dict] = []
    new_paths: dict = {}
    revised_configs = []
    for stage in ("main", "sensitivity", "kmax"):
        records, paths, config = load_run(revised / stage)
        if any(row["dgp"] != "D5" for row in records):
            raise RuntimeError(f"{stage} contains a non-D5 revised record")
        new_records.extend(records)
        new_paths.update(paths)
        revised_configs.append(config)

    records = selected_old_records + new_records
    paths = {**selected_old_paths, **new_paths}
    config = json.loads(json.dumps(old_config))
    config["config"]["namespace"] = "R61-D5-BALANCED-v1-COMBINED-REPORTING"
    config["config"]["dgps"]["D5"] = revised_configs[0]["config"]["dgps"]["D5"]

    observed = Counter((row["stage"], row["cell"], row["method"]) for row in records)
    for dgp in (f"D{i}" for i in range(6)):
        for method in figures.METHODS:
            if observed[("main", dgp, method)] != 500:
                raise RuntimeError(f"incomplete main cell: {dgp}/{method}")

    old_diag = pd.read_csv(archived / "a3_diagnostics.csv")
    old_diag = old_diag[old_diag["dgp"].ne("D5")]
    new_diag = pd.concat(
        [pd.read_csv(revised / stage / "a3_diagnostics.csv") for stage in ("main", "sensitivity", "kmax")],
        ignore_index=True,
    )
    pd.concat([old_diag, new_diag], ignore_index=True).to_csv(
        reporting_dir / "a3_diagnostics.csv", index=False
    )

    figures.style()
    figures.figure2(config, paths, figure_dir / "fig2_dgp_trajectories.pdf", False)
    figures.figure3(config, records, paths, figure_dir / "fig3_structural_recovery.pdf", False)
    figures.figure5(records, figure_dir / "fig5_rmse_distributions.pdf", False)
    figures.figure6(
        reporting_dir, records, figure_dir / "fig6_sensitivity_profiles.pdf", False
    )
    figures.table1(config, table_dir, False)
    figures.table2(records, table_dir, False)
    figures.table3(records, table_dir, False)
    figures.table4(records, paths, table_dir, False)

    provenance = {
        "schema_version": 1,
        "design": "archived D0--D4 plus revised balanced-signal D5",
        "archived_source": str(archived),
        "revised_sources": [str(revised / stage) for stage in ("main", "sensitivity", "kmax")],
        "source_manifest_sha256": {
            str(path): digest(path)
            for path in [
                archived / "run_manifest.json",
                revised / "main/run_manifest.json",
                revised / "sensitivity/run_manifest.json",
                revised / "kmax/run_manifest.json",
            ]
        },
        "main_records_per_method": 6 * 500,
        "revised_d5_levels": config["config"]["dgps"]["D5"]["levels"],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rebuilt revised-D5 simulation assets in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

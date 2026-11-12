#!/usr/bin/env python
"""Rebuild validated simulation and PC-Flanker publication reporting assets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from types import SimpleNamespace
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

try:
    from scripts import simulation_figures as simulation
except ModuleNotFoundError:  # Direct invocation from scripts/.
    import simulation_figures as simulation

ROOT = Path(__file__).resolve().parents[1]
SIMULATION_RESULTS = ROOT / "results" / "reproduction" / "simulation"
APPLICATION_RESULTS = ROOT / "results" / "reproduction" / "application"
OUTPUT_ROOT = ROOT / "manuscript" / "reproduced"
GRID_SIZE, PHASE_BOUNDARY, PNG_DPI, TIFF_DPI = 64, 32, 600, 600
BLUE, ORANGE, GREEN, PURPLE, GREY, LIGHT_GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#666666",
    "#D9D9D9",
)

CURRENT = (
    (
        "Fig1.tif",
        "fig5_rmse_distributions",
        "Replication-level trajectory-RMSE distributions",
    ),
    ("Fig2.tif", "fig2_dgp_trajectories", "Simulation trajectories"),
    ("Fig3.tif", "fig3_structural_recovery", "Simulation structural recovery"),
    ("Fig4.tif", "fig6_sensitivity_profiles", "Sensitivity analysis"),
    ("Fig5.tif", "fig4_real_data", "PC-Flanker two-group trajectories"),
    (
        "Fig6.tif",
        "fig7_real_diagnostics",
        "PC-Flanker two-group K posterior and diagnostics",
    ),
)


# PC-Flanker figures deliberately retain their distinct application rc context.
def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def group_label(group: dict[str, Any]) -> str:
    return f"{group['data']['first_pc']}-first"


def load_inputs(result_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    result_path = result_dir / "results.json"
    rmse_path = result_dir / "heldout_rmse.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rmse = pd.read_csv(rmse_path)
    groups = result.get("groups", [])
    if len(groups) != 2:
        raise ValueError(f"Expected two presentation-order groups, found {len(groups)}")
    required_methods = {
        "NoEffect",
        "ConstEffect",
        "FixedSplit",
        "PELT-BIC",
        "CG-RJMCMC",
    }
    for group in groups:
        label = group_label(group)
        a3 = group["CG_RJMCMC"]
        for key in ("bma_curve", "bma_q025", "bma_q975", "boundary_pip"):
            if len(a3[key]) != GRID_SIZE:
                raise ValueError(
                    f"{label}: {key} has length {len(a3[key])}, expected {GRID_SIZE}"
                )
        observed = set(rmse.loc[rmse["group"].eq(label), "method"])
        if observed != required_methods:
            raise ValueError(f"{label}: incomplete method rows in {rmse_path}")
        csv_rows = rmse.loc[rmse["group"].eq(label)].set_index("method")
        for method_row in group["methods"]:
            method = method_row["method"]
            csv_row = csv_rows.loc[method]
            csv_cuts = ast.literal_eval(str(csv_row["cuts"]))
            if (
                int(csv_row["K"]) != int(method_row["K"])
                or csv_cuts != method_row["cuts"]
                or not np.isclose(
                    float(csv_row["test_RMSE"]),
                    float(method_row["test_RMSE"]),
                    rtol=0,
                    atol=1e-14,
                )
                or not np.isclose(
                    float(csv_row["runtime_seconds"]),
                    float(method_row["runtime_seconds"]),
                    rtol=0,
                    atol=1e-14,
                )
            ):
                raise ValueError(
                    f"{label}/{method}: CSV row does not match results.json"
                )
    return result, rmse


def panel_label(ax: plt.Axes, letter: str, label: str) -> None:
    text = f"({letter})" + (f"  {label}" if label else "")
    ax.text(
        -0.12,
        1.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8.3,
    )


def add_phase_boundary(ax: plt.Axes) -> None:
    ax.axvline(
        PHASE_BOUNDARY,
        color=GREY,
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        zorder=1,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=PNG_DPI)
    fig.savefig(
        path.with_suffix(".tif"),
        format="tiff",
        dpi=TIFF_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def make_figure4(result: dict[str, Any], path: Path) -> None:
    x = np.arange(1, GRID_SIZE + 1)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.35),
        sharex=True,
        sharey="col",
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    letters = iter("ABCD")
    for row, group in enumerate(result["groups"]):
        label = group_label(group)
        a3 = group["CG_RJMCMC"]
        mean = np.asarray(a3["bma_curve"], dtype=float)
        lower = np.asarray(a3["bma_q025"], dtype=float)
        upper = np.asarray(a3["bma_q975"], dtype=float)
        joint = np.asarray(a3["joint_map_curve"], dtype=float)
        pip = np.asarray(a3["boundary_pip"], dtype=float)

        ax = axes[row, 0]
        ax.fill_between(x, lower, upper, color=BLUE, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(x, mean, color=BLUE, zorder=3)
        ax.plot(x, joint, color=ORANGE, linestyle=(0, (5, 2)), linewidth=1.7, zorder=4)
        ax.axhline(0, color=LIGHT_GREY, linewidth=0.7, zorder=0)
        add_phase_boundary(ax)
        ax.set_ylabel("Congruency effect (log RT)")
        panel_label(ax, next(letters), label)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)

        ax = axes[row, 1]
        ax.bar(x, pip, width=0.82, color=BLUE, edgecolor="none", zorder=2)
        add_phase_boundary(ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Boundary inclusion probability")
        panel_label(ax, next(letters), label)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)

    for ax in axes[-1, :]:
        ax.set_xlabel("Progression bin (1–64)")
    for ax in axes.flat:
        ax.set_xlim(0.5, GRID_SIZE + 0.5)
        ax.set_xticks([1, 16, 32, 48, 64])

    legend = [
        Line2D([], [], color=BLUE, label="BMA posterior mean"),
        Line2D(
            [], [], color=BLUE, linewidth=6, alpha=0.18, label="Pointwise 95% interval"
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            linestyle=(0, (2.5, 1.5)),
            label="Joint-MAP trajectory",
        ),
        Line2D(
            [],
            [],
            color=GREY,
            linestyle=(0, (4, 2)),
            label="Phase boundary (after bin 32)",
        ),
    ]
    fig.legend(
        legend,
        [item.get_label() for item in legend],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.25,
        handlelength=2.4,
    )
    fig.subplots_adjust(
        left=0.105, right=0.99, bottom=0.105, top=0.91, hspace=0.32, wspace=0.25
    )
    save_figure(fig, path)


def make_figure7(result: dict[str, Any], path: Path) -> None:
    groups = result["groups"]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25))
    group_colors = [BLUE, ORANGE]

    for col_i, (group, color, letter) in enumerate(zip(groups, group_colors, "AB")):
        ax = axes[0, col_i]
        probabilities = np.asarray(group["CG_RJMCMC"]["p_k"], dtype=float)
        kk = np.arange(len(probabilities))
        ax.bar(kk, probabilities, width=0.68, color=color, alpha=0.88)
        ax.set_xticks(kk)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Number of change points, $K$")
        if col_i == 0:
            ax.set_ylabel("Posterior probability")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
        panel_label(ax, letter, "")

    ax = axes[1, 0]
    diagnostics = [
        [
            f"{float(group['CG_RJMCMC']['max_split_Rhat']):.6f}"
            for group in groups
        ],
        [
            f"{float(group['CG_RJMCMC']['min_bulk_ESS']):,.0f}"
            for group in groups
        ],
    ]
    table = ax.table(
        cellText=diagnostics,
        rowLabels=[r"Max $\widehat R$", "Min bulk ESS"],
        colLabels=[group_label(group) for group in groups],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.02, 0.22, 0.96, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")
    ax.set_axis_off()
    panel_label(ax, "C", "")

    ax = axes[1, 1]
    proposal_names = ["Birth", "Death", "Move"]
    width = 0.34
    xx = np.arange(len(proposal_names))
    for group_i, (group, color) in enumerate(zip(groups, group_colors)):
        acceptance = group["CG_RJMCMC"]["acceptance"]
        values = [acceptance["birth"], acceptance["death"], acceptance["move"]]
        offset = (group_i - 0.5) * width
        ax.bar(
            xx + offset,
            values,
            width=width,
            color=color,
            alpha=0.88,
            label=group_label(group),
        )
    ax.set_xticks(xx, proposal_names)
    ax.set_ylim(0, 0.20)
    ax.set_ylabel("Acceptance probability")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    panel_label(ax, "D", "")

    fig.subplots_adjust(
        left=0.10, right=0.91, bottom=0.11, top=0.94, hspace=0.46, wspace=0.34
    )
    save_figure(fig, path)


def format_runtime(value: float) -> str:
    if value < 0.001:
        return f"{value:.6f}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def parse_cuts(value: Any) -> str:
    cuts = ast.literal_eval(str(value)) if not isinstance(value, list) else value
    return ", ".join(str(int(cut)) for cut in cuts) if cuts else "--"


def make_table5(rmse: pd.DataFrame, path: Path) -> None:
    method_order = ["NoEffect", "ConstEffect", "FixedSplit", "PELT-BIC", "CG-RJMCMC"]
    group_order = ["High-first", "Low-first"]
    frame = rmse.copy()
    frame["group"] = pd.Categorical(frame["group"], group_order, ordered=True)
    frame["method"] = pd.Categorical(frame["method"], method_order, ordered=True)
    frame = frame.sort_values(["group", "method"])
    if len(frame) != 10 or frame[["group", "method"]].duplicated().any():
        raise ValueError("Table 5 requires ten unique group-by-method rows")

    csv_frame = frame[
        ["group", "method", "test_RMSE", "K", "cuts", "runtime_seconds"]
    ].copy()
    csv_frame["group"] = csv_frame["group"].astype(str)
    csv_frame["method"] = csv_frame["method"].astype(str)
    csv_frame["cuts"] = csv_frame["cuts"].map(parse_cuts)
    csv_frame.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")

    lines = [
        "% Generated by scripts/figures.py from the sealed PC-Flanker result bundle.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Held-out prediction and selected structure in the proportion-congruency application. High-first and low-first comprise 24 and 21 participants, with 1675 and 1486 held-out responses, respectively. Test RMSE is on the individual log-RT scale. $\widehat K$ is the selected number of cuts; CG-RJMCMC uses the empirical joint MAP partition. Cuts follow progression bins formed from native ordered positions in retained blocks; bin 32 separates the experimental phases. Runtime is recorded fitting time in seconds, with CG-RJMCMC covering three chains and excluding preprocessing, diagnostic aggregation, and file operations.}",
        r"\label{tab:r61-real}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Group & Method & Test RMSE & $\widehat K$ & Selected cuts & Runtime (s) \\",
        r"\midrule",
    ]
    for group_i, group in enumerate(group_order):
        subset = frame[frame["group"].astype(str).eq(group)]
        for row_i, (_, row) in enumerate(subset.iterrows()):
            group_cell = group if row_i == 0 else ""
            lines.append(
                f"{group_cell} & {row['method']} & {float(row['test_RMSE']):.5f} & "
                f"{int(row['K'])} & {parse_cuts(row['cuts'])} & "
                f"{format_runtime(float(row['runtime_seconds']))} " + r"\\"
            )
        if group_i == 0:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Submission TIFF export and recoverable stale-file preservation.
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export(figures: Path, submission: Path, archive: Path) -> Path:
    """Create only Fig1--Fig6 and archive any superseded files recoverably."""
    submission.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    for retired in sorted(submission.glob("Fig*.tif")):
        # Never erase historical upload candidates: isolate them by version.
        if retired.name == "Fig7.tif" or retired.name in {
            name for name, _, _ in CURRENT
        }:
            archived = archive / retired.name
            if not archived.exists():
                shutil.copy2(retired, archived)
            elif digest(archived) != digest(retired):
                # Preserve the original archive across repeated current exports.
                versioned = archive / f"{retired.stem}_{digest(retired)[:12]}.tif"
                if not versioned.exists():
                    shutil.copy2(retired, versioned)
    outputs = []
    for upload_name, stem, description in CURRENT:
        pdf, png = figures / f"{stem}.pdf", figures / f"{stem}.png"
        if not pdf.exists() or not png.exists():
            raise FileNotFoundError(
                f"Current source missing for {upload_name}: {pdf} / {png}"
            )
        target = submission / upload_name
        with Image.open(png) as image:
            image.convert("RGB").save(
                target, format="TIFF", compression="tiff_lzw", dpi=(600, 600)
            )
        with Image.open(target) as image:
            if image.format != "TIFF" or image.width == 0 or image.height == 0:
                raise RuntimeError(f"Invalid TIFF: {target}")
        outputs.append(
            {
                "upload": upload_name,
                "description": description,
                "pdf_source": display_path(pdf),
                "png_render_source": display_path(png),
                "pdf_sha256": digest(pdf),
                "png_sha256": digest(png),
                "tiff_sha256": digest(target),
            }
        )
    # Fig7 is explicitly excluded from the final six-file upload set.
    stale = submission / "Fig7.tif"
    if stale.exists():
        # Its byte-preserving archival copy was made above; remove only the
        # submission-directory duplicate so the final set is exactly six.
        stale.unlink()
    manifest = {
        "schema_version": 1,
        "application": "PC-Flanker",
        "upload_count": 6,
        "retired_upload_set": "Lost-in-Migration / seven-figure mapping; archived, not deleted",
        "outputs": outputs,
    }
    path = submission / "PLOS_FIGURE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def write_publication_manifest(
    simulation_results: Path,
    application_results: Path,
    figures: Path,
    tables: Path,
    submission: Path,
    figure_manifest: Path,
) -> Path:
    """Lock the current PLOS artifacts without retaining historical paths."""
    figure_stems = [stem for _, stem, _ in CURRENT]
    outputs = [
        *(
            figures / f"{stem}.{suffix}"
            for stem in figure_stems
            for suffix in ("pdf", "png")
        ),
        figures / "fig2_dgp_trajectories.csv",
        figures / "fig3_structural_recovery.csv",
        *(
            tables / f"table{number}_{name}.{suffix}"
            for number, name in (
                (1, "design"),
                (2, "rmse"),
                (3, "structural_recovery"),
                (4, "computation_sensitivity"),
                (5, "real_data"),
            )
            for suffix in ("csv", "tex")
        ),
        *(submission / name for name, _, _ in CURRENT),
        figure_manifest,
    ]
    missing = [display_path(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"publication outputs missing: {missing}")

    table5 = pd.read_csv(tables / "table5_real_data.csv")
    if (
        len(table5) != 10
        or table5[["group", "method"]].duplicated().any()
        or set(table5["group"]) != {"High-first", "Low-first"}
    ):
        raise ValueError("current Table 5 CSV failed its group-by-method schema check")

    source_files = [
        simulation_results / "run_manifest.json",
        simulation_results / "resolved_config.json",
        application_results / "results.json",
        application_results / "heldout_rmse.csv",
        application_results / "run_manifest.json",
    ]
    if any(not path.is_file() for path in source_files):
        raise FileNotFoundError("a publication source manifest/result is missing")
    manifest = {
        "schema_version": 2,
        "artifact_status": "current PLOS submission set",
        "generator": "scripts/figures.py",
        "simulation_results": display_path(simulation_results),
        "application_results": display_path(application_results),
        "source_sha256": {
            display_path(path): digest(path) for path in source_files
        },
        "outputs": [display_path(path) for path in outputs],
        "output_sha256": {display_path(path): digest(path) for path in outputs},
        "table5_check": {
            "rows": 10,
            "groups": ["High-first", "Low-first"],
            "methods_per_group": 5,
            "csv_and_tex_generated_together": True,
        },
    }
    path = tables / "publication_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def normalize_table(path, caption):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\caption\{[^\n]*\}", lambda _: "\\caption{" + caption + "}", text)
    text = re.sub(r"\\begin\{minipage\}.*?\\end\{minipage\}", "", text, flags=re.S)
    if path.stem == "table4_computation_sensitivity":
        text = text.replace("Valid &", "\\shortstack{Pass\\\\rate} &")
        text = text.replace("\\resizebox{\\textwidth}{!}{%\n", "")
        text = text.replace("\\end{tabular}\n}", "\\end{tabular}")
        text = text.replace(
            "\\begin{tabular}", "\\setlength{\\tabcolsep}{8pt}\n\\begin{tabular}", 1
        )
    path.write_text(text, encoding="utf-8")


def rebuild_simulation(sim: Path, figures: Path, tables: Path) -> None:
    ok, reason, status = simulation.completion(sim)
    if not ok:
        raise RuntimeError(reason)
    config = simulation.read_json(sim / "resolved_config.json")
    fingerprint = status.get("run_fingerprint") or simulation.read_json(
        sim / "run_manifest.json"
    ).get("run_fingerprint")
    records, paths = simulation.collect_replicates(sim, fingerprint)
    observed = Counter((stage, cell) for stage, cell, _ in paths)
    if any(
        observed[key] != number
        for key, number in simulation.expected_replicate_counts(config).items()
    ):
        raise RuntimeError("Incomplete frozen simulation inventory")
    simulation.style()
    simulation.figure2(config, paths, figures / "fig2_dgp_trajectories.pdf", False)
    simulation.figure3(
        config, records, paths, figures / "fig3_structural_recovery.pdf", False
    )
    simulation.figure5(records, figures / "fig5_rmse_distributions.pdf", False)
    simulation.figure6(sim, records, figures / "fig6_sensitivity_profiles.pdf", False)
    simulation.table1(config, tables, False)
    simulation.table2(records, tables, False)
    simulation.table3(records, tables, False)
    simulation.table4(records, paths, tables, False)
    captions = json.loads(
        (ROOT / "scripts" / "config" / "publication_captions.json").read_text(
            encoding="utf-8"
        )
    )
    for name, caption in captions.items():
        normalize_table(tables / f"{name}.tex", caption)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, default=SIMULATION_RESULTS)
    parser.add_argument("--application", type=Path, default=APPLICATION_RESULTS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--revision-results", type=Path, help="revision root with benchmark/ and small_sample/ for final table panels")
    parser.add_argument("--runtime-repeats", type=Path, help="saved 31-repeat PELT timing CSV for final Table 5")
    parser.add_argument("--application-input", type=Path, help="raw OSF directory, required to measure timings if no saved repeats supplied")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    figures, tables = (
        (args.figures or output_root / "figures").resolve(),
        (args.tables or output_root / "tables").resolve(),
    )
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    # Each family starts from a clean style, as when the original exporters
    # ran in separate processes. A later import cannot leak plot settings.
    with plt.rc_context(matplotlib.rcParamsDefault):
        rebuild_simulation(args.simulation.resolve(), figures, tables)
    with plt.rc_context(matplotlib.rcParamsDefault):
        result, rmse = load_inputs(args.application.resolve())
        set_style()
        make_figure4(result, figures / "fig4_real_data.pdf")
        make_figure7(result, figures / "fig7_real_diagnostics.pdf")
        make_table5(rmse, tables / "table5_real_data.tex")
    if args.revision_results is not None:
        try:
            from scripts import revision_tables
        except ModuleNotFoundError:
            import revision_tables
        revision_tables.generate(SimpleNamespace(
            simulation=[args.simulation.resolve()], application=args.application.resolve()/"results.json",
            benchmark=args.revision_results.resolve()/"benchmark", small_sample=args.revision_results.resolve()/"small_sample",
            output_root=output_root/"revision_tables", runtime_repeats=args.runtime_repeats,
            timing=args.runtime_repeats is None, application_input=args.application_input, allow_partial=False))
    figure_manifest = export(
        figures,
        output_root / "submission_figures",
        output_root / "submission_figures" / "archive_lost_in_migration",
    )
    write_publication_manifest(
        args.simulation.resolve(),
        args.application.resolve(),
        figures,
        tables,
        output_root / "submission_figures",
        figure_manifest,
    )
    print(
        "Rebuilt six figures and original table layouts. Final revised Tables 2/3/5 are in revision_tables/ when --revision-results is supplied; otherwise run reproduce.py tables after revision validation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

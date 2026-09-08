"""Validated simulation reporting, extracted without numerical changes.

Use reproduce.py figures as the command-line entry point.
The retired real-application and seven-figure export functions are excluded.
"""

from __future__ import annotations


import csv


import json

import math


from collections import Counter, defaultdict

from pathlib import Path

from typing import Any, Iterable


import matplotlib.pyplot as plt


import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SUBMISSION_PNG_DPI = 600

METHODS = ("B0", "B1", "B2", "PELT-BIC", "A3")

DISPLAY_METHOD = {
    "B0": "NoEffect",
    "B1": "ConstEffect",
    "B2": "FixedSplit",
    "PELT-BIC": "PELT-BIC",
    "A3": "CG-RJMCMC",
}

DGPS = tuple(f"D{i}" for i in range(6))

BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#666666",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completion(root: Path) -> tuple[bool, str, dict[str, Any] | None]:
    status_path = root / "run_status.json"
    if not root.exists():
        return False, f"missing directory: {root}", None
    if not status_path.exists():
        return False, f"missing completion marker: {status_path}", None
    try:
        status = read_json(status_path)
    except (OSError, ValueError) as exc:
        return False, f"unreadable completion marker: {exc}", None
    if str(status.get("status", "")).lower() != "complete":
        return False, f"run_status is {status.get('status', 'unspecified')!r}", status
    return True, "complete", status


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def mark_provisional(fig: plt.Figure, provisional: bool) -> None:
    if provisional:
        fig.text(
            0.995,
            0.002,
            "PROVISIONAL — incomplete inputs",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=RED,
        )


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        f"({label.upper()})",
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
    )


def missing_panel(ax: plt.Axes, message: str = "Input incomplete") -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=RED,
        fontsize=8,
    )


def save_pdf(fig: plt.Figure, path: Path) -> None:
    """Save a review PDF plus 600-dpi PNG/TIFF submission copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=SUBMISSION_PNG_DPI)
    fig.savefig(
        path.with_suffix(".tif"),
        format="tiff",
        dpi=SUBMISSION_PNG_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def step_curve(cuts: Iterable[int], levels: Iterable[float], trials: int) -> np.ndarray:
    cuts = tuple(int(x) for x in cuts)
    values = np.asarray(list(levels), dtype=float)
    return np.asarray(
        [values[np.searchsorted(cuts, t, side="left")] for t in range(1, trials + 1)]
    )


def collect_replicates(
    sim_root: Path, fingerprint: str | None
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], Path]]:
    records: list[dict[str, Any]] = []
    paths: dict[tuple[str, str, int], Path] = {}
    for path in sorted(sim_root.glob("replicates/*/*/rep_*/summary.json")):
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if fingerprint and value.get("run_fingerprint") != fingerprint:
            continue
        case = value.get("case", {})
        if not {"stage", "cell", "dgp"} <= case.keys():
            continue
        rep = int(value.get("replicate", -1))
        paths[(str(case["stage"]), str(case["cell"]), rep)] = path.parent
        for method in value.get("methods", []):
            row = {
                "stage": case["stage"],
                "cell": case["cell"],
                "dgp": case["dgp"],
                "participants": case.get("participants"),
                "replicate": rep,
                **method,
            }
            records.append(row)
    return records, paths


def expected_replicate_counts(config: dict[str, Any]) -> dict[tuple[str, str], int]:
    cfg, expected = config["config"], {}
    stages = set(config.get("stages", ["main", "sensitivity", "kmax"]))
    if "main" in stages:
        for dgp in cfg["dgps"]:
            expected[("main", dgp)] = int(cfg["replications"]["main_reps"])
    if "sensitivity" in stages:
        reps = int(cfg["replications"]["sensitivity_reps"])
        for family, entries in cfg["sensitivity"].items():
            for item in entries:
                value = item.get("value", item.get("participants"))
                label = item.get("label", str(value).replace(".", "p"))
                expected[("sensitivity", f"{family}_{item['dgp']}_{label}")] = reps
    if "kmax" in stages:
        reps = int(cfg["replications"]["kmax_reps"])
        for item in cfg["kmax_sensitivity"]:
            expected[("kmax", f"Kmax{item['value']}_{item['dgp']}")] = reps
    return expected


def qsummary(values: Iterable[Any]) -> tuple[float, float, float, int]:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    if not len(x):
        return math.nan, math.nan, math.nan, 0
    return (
        float(np.median(x)),
        float(np.quantile(x, 0.25)),
        float(np.quantile(x, 0.75)),
        len(x),
    )


def fmt_num(x: Any, digits: int = 3) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(value):
        return "--"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_iqr(values: Iterable[Any], digits: int = 3) -> tuple[str, int]:
    med, q25, q75, n = qsummary(values)
    return (
        f"{fmt_num(med, digits)} [{fmt_num(q25, digits)}, {fmt_num(q75, digits)}]"
        if n
        else "--",
        n,
    )


class TexCell(str):
    """A table cell containing deliberately authored LaTeX markup."""


def latex_escape(value: Any) -> str:
    if isinstance(value, TexCell):
        return str(value)
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def write_csv(path: Path, rows: list[dict[str, Any]], provisional: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "provisional" if provisional else "final"
    materialized = [{"artifact_status": status, **row} for row in rows]
    fields = list(materialized[0]) if materialized else ["artifact_status", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            materialized or [{"artifact_status": status, "status": "input incomplete"}]
        )


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[Any]],
    align: str,
    provisional: bool,
    notes: str = "",
    add_space_after: set[int] | None = None,
) -> None:
    status = (
        "% PROVISIONAL: generated from incomplete inputs\n"
        if provisional
        else "% FINAL: all required run_status markers were complete\n"
    )
    lines = [
        status.rstrip(),
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    if rows:
        for row_i, row in enumerate(rows):
            lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
            if add_space_after and row_i in add_space_after:
                lines.append(r"\addlinespace")
    else:
        lines.append(
            r"\multicolumn{" + str(len(headers)) + r"}{c}{Input incomplete} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.append(
            r"\begin{minipage}{0.98\linewidth}\footnotesize "
            + notes
            + r"\end{minipage}"
        )
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure2(
    config: dict[str, Any],
    rep_paths: dict[tuple[str, str, int], Path],
    out: Path,
    provisional: bool,
) -> None:
    cfg = config["config"]
    trials = int(cfg["design"]["trials"])
    source_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.15), sharex=True, sharey=True)
    for i, (dgp, ax) in enumerate(zip(DGPS, axes.flat)):
        arrays = []
        for (stage, cell, _), directory in rep_paths.items():
            if stage != "main" or cell != dgp:
                continue
            path = directory / "a3_bma_curve.csv"
            try:
                frame = pd.read_csv(path)
                if len(frame) == trials and "estimate" in frame:
                    arrays.append(frame["estimate"].to_numpy(float))
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        if not arrays or dgp not in cfg["dgps"]:
            missing_panel(ax)
        else:
            estimates = np.stack(arrays)
            truth = step_curve(
                cfg["dgps"][dgp]["cuts"], cfg["dgps"][dgp]["levels"], trials
            )
            x = np.arange(1, trials + 1)
            q25 = np.quantile(estimates, 0.25, axis=0)
            median = np.median(estimates, axis=0)
            q75 = np.quantile(estimates, 0.75, axis=0)
            source_rows.extend(
                {
                    "dgp": dgp,
                    "trial": int(trial),
                    "replications": len(estimates),
                    "truth": float(true),
                    "estimate_q25": float(lo),
                    "estimate_median": float(mid),
                    "estimate_q75": float(hi),
                }
                for trial, true, lo, mid, hi in zip(x, truth, q25, median, q75)
            )
            ax.fill_between(
                x,
                q25,
                q75,
                color=BLUE,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                x, median, color=BLUE, label="CG-RJMCMC median"
            )
            ax.plot(x, truth, color=ORANGE, linestyle="--", label="Truth")
            ax.axhline(0, color="#BBBBBB", linewidth=0.55, zorder=0)
            ax.set_xlim(1, trials)
        ax.text(
            -0.15,
            1.04,
            f"({chr(65 + i)})  {dgp}",
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            fontsize=8.5,
        )
        if i >= 3:
            ax.set_xlabel("Trial position")
        if i % 3 == 0:
            ax.set_ylabel("Congruency effect")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.005),
        )
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.105, top=0.91, hspace=0.22, wspace=0.15
    )
    mark_provisional(fig, provisional)
    save_pdf(fig, out)
    pd.DataFrame(source_rows).to_csv(
        out.with_suffix(".csv"), index=False, lineterminator="\n"
    )


def figure3(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    rep_paths: dict[tuple[str, str, int], Path],
    out: Path,
    provisional: bool,
) -> None:
    frame = pd.DataFrame(records)
    cfg = config["config"]
    source_rows = []
    # Model-dimension uncertainty and boundary selection are the two quantities
    # that are not already reported numerically in the structural-recovery table.
    # Hausdorff summaries remain in Table 3 to avoid repeating the same result
    # in both a figure and a table.
    fig, axes = plt.subplots(6, 2, figsize=(7.15, 7.55), sharex="col")
    for row_i, dgp in enumerate(DGPS):
        ax_pk, ax_pip = axes[row_i]
        dirs = [
            directory
            for (stage, cell, _), directory in rep_paths.items()
            if stage == "main" and cell == dgp
        ]
        pks, pips = [], []
        for directory in dirs:
            try:
                pk = pd.read_csv(directory / "a3_p_k.csv")
                pks.append(pk.set_index("K")["probability"])
                curve = pd.read_csv(directory / "a3_bma_curve.csv")
                pips.append(curve["boundary_PIP"].to_numpy(float))
            except (OSError, KeyError, ValueError, pd.errors.ParserError):
                pass
        if pks:
            pkf = pd.concat(pks, axis=1).fillna(0)
            mean_pk = pkf.mean(axis=1)
            ax_pk.bar(pkf.index.astype(int), mean_pk, color=BLUE, width=0.75)
            source_rows.extend(
                {
                    "dgp": dgp,
                    "metric": "mean_posterior_count_probability",
                    "index": int(k),
                    "value": float(value),
                    "replications": len(pks),
                    "true_boundary": False,
                }
                for k, value in mean_pk.items()
            )
            ax_pk.set_ylim(0, 1)
            ax_pk.set_xticks(pkf.index.astype(int))
        else:
            missing_panel(ax_pk)
        if pips:
            mean_pip = np.mean(np.stack(pips), axis=0)
            x = np.arange(1, len(mean_pip) + 1)
            ax_pip.plot(x, mean_pip, color=BLUE, label="CG-RJMCMC mean PIP")
            subset = frame[
                (frame.get("stage") == "main")
                & (frame.get("dgp") == dgp)
                & (frame.get("method") == "PELT-BIC")
            ]
            freq = Counter()
            for cuts in subset.get("estimated_cuts", []):
                for cut in cuts if isinstance(cuts, list) else str(cuts).split(","):
                    if str(cut).strip():
                        freq[int(cut)] += 1
            denom = max(len(subset), 1)
            xx = np.arange(1, len(mean_pip) + 1)
            pelt_frequency = np.asarray([freq[int(v)] / denom for v in xx])
            if freq:
                ax_pip.plot(
                    xx,
                    pelt_frequency,
                    color=ORANGE,
                    linestyle="--",
                    label="PELT frequency",
                )
            truth = set(cfg["dgps"][dgp]["cuts"])
            for trial, pip, pelt in zip(xx, mean_pip, pelt_frequency):
                source_rows.extend(
                    [
                        {
                            "dgp": dgp,
                            "metric": "mean_boundary_inclusion_probability",
                            "index": int(trial),
                            "value": float(pip),
                            "replications": len(pips),
                            "true_boundary": int(trial) in truth,
                        },
                        {
                            "dgp": dgp,
                            "metric": "pelt_selection_frequency",
                            "index": int(trial),
                            "value": float(pelt),
                            "replications": len(subset),
                            "true_boundary": int(trial) in truth,
                        },
                    ]
                )
            for cut in cfg["dgps"][dgp]["cuts"]:
                ax_pip.axvline(cut, color=GREY, linestyle=":", linewidth=1.0, alpha=0.7)
            ax_pip.set_xlim(1, len(mean_pip))
            ax_pip.set_ylim(bottom=0)
        else:
            missing_panel(ax_pip)
        ax_pk.set_ylabel(f"{dgp}")
        if row_i == 0:
            ax_pk.text(
                0.5,
                1.13,
                "Posterior $P(K)$",
                transform=ax_pk.transAxes,
                ha="center",
                fontsize=9,
            )
            ax_pip.text(
                0.5,
                1.13,
                "Boundary selection",
                transform=ax_pip.transAxes,
                ha="center",
                fontsize=9,
            )
        if row_i == 5:
            ax_pk.set_xlabel("Number of change points, $K$")
            ax_pip.set_xlabel("Boundary after trial")
    handles = [
        plt.Line2D([], [], color=BLUE, label="CG-RJMCMC"),
        plt.Line2D([], [], color=ORANGE, linestyle="--", label="PELT-BIC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.075, top=0.94, hspace=0.36, wspace=0.25
    )
    mark_provisional(fig, provisional)
    save_pdf(fig, out)
    pd.DataFrame(source_rows).to_csv(
        out.with_suffix(".csv"), index=False, lineterminator="\n"
    )


def col(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    return next((lookup[n.lower()] for n in names if n.lower() in lookup), None)


def table1(config: dict[str, Any], out_dir: Path, provisional: bool) -> None:
    cfg = config["config"]
    design, reps, nominal = cfg["design"], cfg["replications"], cfg["nominal_model"]
    rows = []
    for dgp in DGPS:
        spec = cfg["dgps"].get(dgp)
        if not spec:
            continue
        rows.append(
            {
                "DGP": dgp,
                "participants": design["participants"],
                "trials": design["trials"],
                "true_K": len(spec["cuts"]),
                "true_cuts": ";".join(map(str, spec["cuts"])) or "none",
                "true_levels": ";".join(f"{x:.3f}" for x in spec["levels"]),
                "replications": reps["main_reps"],
                "Kmax": nominal["kmax"],
                "minimum_segment": nominal["min_segment"],
            }
        )
    write_csv(out_dir / "table1_design.csv", rows, provisional)
    latex_rows = [
        [
            (
                TexCell(str(r[k]).replace(";", r";\,"))
                if k in {"true_cuts", "true_levels"}
                else r[k]
            )
            for k in (
                "DGP",
                "participants",
                "trials",
                "true_K",
                "true_cuts",
                "true_levels",
                "replications",
            )
        ]
        for r in rows
    ]
    notes = (
        f"Common CG-RJMCMC settings: $K_{{\\max}}={nominal['kmax']}$, minimum segment length {nominal['min_segment']}, "
        f"{cfg['sampler']['chains']} chains, {cfg['sampler']['iterations']} iterations per chain, and {cfg['sampler']['warmup']} warm-up iterations."
    )
    write_latex_table(
        out_dir / "table1_design.tex",
        "Simulation design and reproducibility settings.",
        "tab:r61-design",
        ["DGP", "$N$", "$T$", "True $K$", "True cuts", "True levels", "Replications"],
        latex_rows,
        "lrrrrlr",
        provisional,
        notes,
    )


def table2(records: list[dict[str, Any]], out_dir: Path, provisional: bool) -> None:
    frame = pd.DataFrame(records)
    rows = []
    trajectory: dict[tuple[str, str], str] = {}
    prediction: dict[tuple[str, str], str] = {}
    for dgp in DGPS:
        for method in METHODS:
            q = (
                frame[
                    (frame["stage"] == "main")
                    & (frame["dgp"] == dgp)
                    & (frame["method"] == method)
                ]
                if len(frame)
                else pd.DataFrame()
            )
            traj, tn = fmt_iqr(q.get("trajectory_RMSE", []), 4)
            test, pn = fmt_iqr(q.get("test_RMSE", []), 4)
            rows.append(
                {
                    "DGP": dgp,
                    "method": method,
                    "trajectory_RMSE_median_IQR": traj,
                    "trajectory_RMSE_n": tn,
                    "test_RMSE_median_IQR": test,
                    "test_RMSE_n": pn,
                }
            )
            trajectory[(dgp, method)] = traj.split(" [", 1)[0]
            prediction[(dgp, method)] = test.split(" [", 1)[0]
    write_csv(out_dir / "table2_rmse.csv", rows, provisional)

    counts_by_dgp: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        counts_by_dgp[str(row["DGP"])].update(
            (int(row["trajectory_RMSE_n"]), int(row["test_RMSE_n"]))
        )
    all_counts = {n for values in counts_by_dgp.values() for n in values}
    if all_counts == {500}:
        n_note = "Each DGP--method cell contains 500 Monte Carlo replications."
    elif all(len(values) == 1 for values in counts_by_dgp.values()):
        dgp_counts = ", ".join(
            f"{dgp}: $n={next(iter(counts_by_dgp[dgp]))}$" for dgp in DGPS
        )
        n_note = f"Available replications per method and metric are {dgp_counts}."
    else:
        nonzero = sorted(n for n in all_counts if n > 0)
        range_note = f"$n={nonzero[0]}$--{nonzero[-1]}" if nonzero else "$n=0$"
        n_note = (
            f"Available cell sizes vary ({range_note}); method- and metric-specific values "
            "are retained in the companion CSV."
        )

    status = (
        "% PROVISIONAL: generated from incomplete inputs"
        if provisional
        else "% FINAL: all required run_status markers were complete"
    )
    headers = ["DGP", *(DISPLAY_METHOD[m] for m in METHODS)]
    lines = [
        status,
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Trajectory estimation and held-out prediction error.}",
        r"\label{tab:r61-rmse}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
    ]
    for panel_title, values in (
        ("Panel A: Trajectory RMSE", trajectory),
        ("Panel B: Held-out test RMSE", prediction),
    ):
        lines.extend(
            [
                r"\multicolumn{6}{l}{\textit{" + panel_title + r"}} \\",
                " & ".join(headers) + r" \\",
                r"\midrule",
            ]
        )
        lines.extend(
            " & ".join([dgp, *(values[(dgp, method)] for method in METHODS)]) + r" \\"
            for dgp in DGPS
        )
        if panel_title.startswith("Panel A"):
            lines.append(r"\midrule")
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Entries are medians across Monte Carlo replications; IQRs are provided in S2 Table and the companion CSV. "
            + n_note
            + r" RMSE denotes root mean squared error.\end{minipage}",
            r"\end{table}",
        ]
    )
    (out_dir / "table2_rmse.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table3(records: list[dict[str, Any]], out_dir: Path, provisional: bool) -> None:
    frame = pd.DataFrame(records)
    rows, latex_rows = [], []
    for dgp in DGPS:
        for method in ("PELT-BIC", "A3"):
            q = (
                frame[
                    (frame["stage"] == "main")
                    & (frame["dgp"] == dgp)
                    & (frame["method"] == method)
                ]
                if len(frame)
                else pd.DataFrame()
            )
            ck = pd.to_numeric(
                q.get("correct_K", pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            rate = fmt_num(ck.mean(), 3) if len(ck) else "--"
            haus, hn = fmt_iqr(q.get("Hausdorff", []), 1)
            level, ln = fmt_iqr(q.get("segment_level_RMSE", []), 4)
            rows.append(
                {
                    "DGP": dgp,
                    "method": method,
                    "correct_K_rate": rate,
                    "correct_K_n": len(ck),
                    "Hausdorff_median_IQR": haus,
                    "Hausdorff_n": hn,
                    "conditional_segment_level_RMSE_median_IQR": level,
                    "conditional_n": ln,
                }
            )
            latex_rows.append(
                [dgp, DISPLAY_METHOD[method], rate, len(ck), haus, level, ln]
            )
    write_csv(out_dir / "table3_structural_recovery.csv", rows, provisional)
    write_latex_table(
        out_dir / "table3_structural_recovery.tex",
        "Structural recovery by change-point method.",
        "tab:r61-structure",
        ["DGP", "Method", "Correct $K$", "$n$", "Hausdorff", "Level RMSE", "$n_{K}$"],
        latex_rows,
        "llrrrrr",
        provisional,
        "Hausdorff and segment-level RMSE are median [IQR]. Level RMSE is conditional on recovering the correct number of segments; $n_K$ is its denominator.",
        add_space_after=set(range(1, len(latex_rows) - 1, 2)),
    )


def table4(
    records: list[dict[str, Any]],
    rep_paths: dict[tuple[str, str, int], Path],
    out_dir: Path,
    provisional: bool,
) -> None:
    diag_rows = []
    for (stage, cell, rep), directory in rep_paths.items():
        try:
            summary = read_json(directory / "summary.json")
            a3 = summary["A3"]
            diag_rows.append(
                {
                    "stage": stage,
                    "cell": cell,
                    "dgp": summary["case"]["dgp"],
                    "replicate": rep,
                    "diagnostics_valid": a3.get("diagnostics_valid"),
                    "max_split_Rhat": a3.get("max_split_Rhat"),
                    "min_bulk_ESS": a3.get("min_bulk_ESS"),
                    "min_ESS_per_second": a3.get("min_ESS_per_second"),
                    "birth_acceptance": a3.get("acceptance", {}).get("birth"),
                    "death_acceptance": a3.get("acceptance", {}).get("death"),
                    "move_acceptance": a3.get("acceptance", {}).get("move"),
                    "runtime_seconds": a3.get("three_chain_walltime_seconds"),
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    diagnostics = pd.DataFrame(diag_rows)
    rows, latex_rows = [], []
    for dgp in DGPS:
        q = (
            diagnostics[(diagnostics["stage"] == "main") & (diagnostics["dgp"] == dgp)]
            if len(diagnostics)
            else pd.DataFrame()
        )
        valid = pd.Series(q.get("diagnostics_valid", []), dtype="boolean").dropna()
        fields = {
            name: fmt_iqr(q.get(name, []), digits)
            for name, digits in (
                ("max_split_Rhat", 3),
                ("min_bulk_ESS", 0),
                ("min_ESS_per_second", 0),
                ("birth_acceptance", 3),
                ("death_acceptance", 3),
                ("move_acceptance", 3),
                ("runtime_seconds", 2),
            )
        }
        row = {
            "section": "diagnostics",
            "cell": dgp,
            "diagnostic_valid_rate": fmt_num(valid.mean(), 3) if len(valid) else "--",
            "n": len(valid),
            **{k + "_median_IQR": v[0] for k, v in fields.items()},
            "trajectory_RMSE_median_IQR": "",
            "test_RMSE_median_IQR": "",
            "correct_K_rate": "",
        }
        rows.append(row)
        latex_rows.append(
            [
                dgp,
                row["diagnostic_valid_rate"],
                len(valid),
                fields["max_split_Rhat"][0],
                fields["min_bulk_ESS"][0],
                fields["min_ESS_per_second"][0],
                fields["runtime_seconds"][0],
            ]
        )
    frame = pd.DataFrame(records)
    sens = (
        frame[
            (frame["stage"].isin(["sensitivity", "kmax"])) & (frame["method"] == "A3")
        ]
        if len(frame)
        else pd.DataFrame()
    )
    sensitivity_latex_rows = []
    for cell in sorted(sens["cell"].unique()) if len(sens) else []:
        q = sens[sens["cell"] == cell]
        traj, n = fmt_iqr(q.get("trajectory_RMSE", []), 4)
        test, _ = fmt_iqr(q.get("test_RMSE", []), 4)
        ck = pd.to_numeric(q.get("correct_K", []), errors="coerce").dropna()
        rows.append(
            {
                "section": "sensitivity",
                "cell": cell,
                "diagnostic_valid_rate": "",
                "n": n,
                "max_split_Rhat_median_IQR": "",
                "min_bulk_ESS_median_IQR": "",
                "min_ESS_per_second_median_IQR": "",
                "birth_acceptance_median_IQR": "",
                "death_acceptance_median_IQR": "",
                "move_acceptance_median_IQR": "",
                "runtime_seconds_median_IQR": "",
                "trajectory_RMSE_median_IQR": traj,
                "test_RMSE_median_IQR": test,
                "correct_K_rate": fmt_num(ck.mean(), 3) if len(ck) else "--",
            }
        )
        sensitivity_latex_rows.append(
            [cell, traj, test, fmt_num(ck.mean(), 3) if len(ck) else "--", n]
        )
    write_csv(out_dir / "table4_computation_sensitivity.csv", rows, provisional)
    write_latex_table(
        out_dir / "table4_computation_sensitivity.tex",
        "CG-RJMCMC computation diagnostics for the main simulations.",
        "tab:r61-computation",
        [
            "DGP",
            "Valid",
            "$n$",
            "Max $\\widehat R$",
            "Min ESS",
            "Min ESS/s",
            "Runtime (s)",
        ],
        latex_rows,
        "lrrrrrr",
        provisional,
        "Continuous entries are median [IQR] across replications. The companion CSV contains acceptance rates and the complete sensitivity block without duplicating RMSE rows in the typeset main panel.",
    )
    table4_path = out_dir / "table4_computation_sensitivity.tex"
    table4_text = table4_path.read_text(encoding="utf-8")
    table4_text = table4_text.replace(
        r"\begin{tabular}{lrrrrrr}",
        r"\resizebox{\textwidth}{!}{%" + "\n" + r"\begin{tabular}{lrrrrrr}",
        1,
    )
    table4_text = table4_text.replace(
        r"\end{tabular}", r"\end{tabular}" + "\n" + "}", 1
    )
    table4_path.write_text(table4_text, encoding="utf-8")
    write_latex_table(
        out_dir / "table4_sensitivity.tex",
        "CG-RJMCMC sensitivity analyses.",
        "tab:r61-sensitivity",
        ["Sensitivity cell", "Trajectory RMSE", "Test RMSE", "Correct $K$", "$n$"],
        sensitivity_latex_rows,
        "lrrrr",
        provisional,
        "RMSE entries are median [IQR]. Cell names map directly to the frozen sensitivity configuration and the traceable companion CSV.",
    )


def figure5(records: list[dict[str, Any]], out: Path, provisional: bool) -> None:
    """Show the repeated-sample distribution of trajectory error."""
    frame = pd.DataFrame(records)
    frame = frame[frame["stage"].eq("main")].copy()
    colors = [GREY, GREEN, ORANGE, RED, BLUE]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.65), sharex=True)
    for i, dgp in enumerate(DGPS):
        ax = axes.flat[i]
        values = [
            frame.loc[
                (frame["dgp"].eq(dgp)) & (frame["method"].eq(method)), "trajectory_RMSE"
            ]
            .dropna()
            .to_numpy(float)
            for method in METHODS
        ]
        boxes = ax.boxplot(
            values,
            tick_labels=[DISPLAY_METHOD[m] for m in METHODS],
            showfliers=False,
            widths=0.62,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(dgp, fontweight="bold", pad=3)
        ax.tick_params(axis="x", rotation=36)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        if i % 3 == 0:
            ax.set_ylabel("Trajectory RMSE")
        panel(ax, chr(97 + i))
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.18, top=0.94, hspace=0.4, wspace=0.25
    )
    mark_provisional(fig, provisional)
    save_pdf(fig, out)


def figure6(
    sim_root: Path, records: list[dict[str, Any]], out: Path, provisional: bool
) -> None:
    """Grouped sensitivity dot intervals, with scenario-specific main-run references."""
    frame = pd.DataFrame(records)
    frame = frame[frame["method"].eq("A3")].copy()
    diag = pd.read_csv(sim_root / "a3_diagnostics.csv")
    frame = frame.merge(
        diag[["stage", "cell", "dgp", "replicate", "min_bulk_ESS"]],
        on=["stage", "cell", "dgp", "replicate"],
        validate="one_to_one",
    )
    # Categorical conditions have no meaningful connecting order or interpolation.
    settings = [
        ("sample_size_{d}_1000", "$N=1000$"),
        ("signal_multiplier_{d}_0p6", "Signal $\\times 0.6$"),
        ("signal_multiplier_{d}_1p4", "Signal $\\times 1.4$"),
        ("training_mcar_{d}_0p05", "Missing 5%"),
        ("training_mcar_{d}_0p15", "Missing 15%"),
        ("prior_factorial_power_{d}_flat", "Uniform count prior"),
        ("prior_factorial_power_{d}_strong_occam", "Stronger count penalty"),
        ("min_segment_{d}_6", "$m=6$"),
        ("min_segment_{d}_10", "$m=10$"),
        ("Kmax5_{d}", "$K_{\\max}=5$"),
        ("Kmax3_{d}", "$K_{\\max}=3$"),
    ]
    metrics = [
        ("trajectory_RMSE", "Trajectory RMSE", BLUE),
        ("correct_K", "Correct $K$ proportion", GREEN),
        ("min_bulk_ESS", "Minimum bulk ESS", PURPLE),
        ("runtime_seconds", "Runtime (s)", ORANGE),
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(8.2, 7.2), gridspec_kw={"height_ratios": [6, 12]}
    )
    for row, dgp in enumerate(("D1", "D5")):
        available = set(frame.loc[frame["dgp"].eq(dgp), "cell"])
        cells = [(dgp, "Main specification")] + [
            (
                key.format(d=dgp),
                label + (" *" if dgp == "D5" and key == "Kmax3_{d}" else ""),
            )
            for key, label in settings
            if key.format(d=dgp) in available
        ]
        base = frame[frame["stage"].eq("main") & frame["dgp"].eq(dgp)]
        if len(base) != 500 and not provisional:
            raise ValueError(f"Expected 500 main replications for {dgp}")
        for col, (metric, xlabel, color) in enumerate(metrics):
            ax = axes[row, col]
            reference = (
                base[metric].mean() if metric == "correct_K" else base[metric].median()
            )
            ax.axvline(reference, color=GREY, linestyle="--", linewidth=0.9, zorder=1)
            for y, (cell, label) in enumerate(cells):
                q = (
                    base
                    if y == 0
                    else frame[frame["cell"].eq(cell) & frame["dgp"].eq(dgp)]
                )
                values = q[metric].dropna().to_numpy(float)
                if not len(values):
                    raise ValueError(f"No values for {cell}, {metric}")
                if not provisional and len(values) != (500 if y == 0 else 100):
                    raise ValueError(
                        f"Unexpected replication count for {cell}, {metric}: {len(values)}"
                    )
                if metric == "correct_K":
                    n = len(values)
                    p = float(values.mean())
                    z = 1.959963984540054
                    den = 1 + z * z / n
                    center = (p + z * z / (2 * n)) / den
                    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
                    lo, hi, mid = max(0.0, center - half), min(1.0, center + half), p
                    if dgp == "D5" and cell == "Kmax3_D5":
                        # Recovery is impossible under this model support, not an uncertain rate.
                        lo = hi = mid = 0.0
                else:
                    lo, mid, hi = np.quantile(values, [0.25, 0.5, 0.75])
                ax.errorbar(
                    mid,
                    y,
                    xerr=[[max(0.0, mid - lo)], [max(0.0, hi - mid)]],
                    fmt="D" if y == 0 else "o",
                    color=GREY if y == 0 else color,
                    markersize=4,
                    capsize=2,
                    elinewidth=1,
                    zorder=3,
                )
            ax.set_yticks(range(len(cells)))
            ax.set_ylim(len(cells) - 0.45, -0.65)
            if col == 0:
                ax.set_yticklabels([label for _, label in cells], fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(xlabel, fontsize=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
            ax.text(
                0.0,
                1.02,
                f"({chr(65 + row * 4 + col)})",
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
            )
            ax.set_xlim(left=0)
            if metric == "correct_K":
                ax.set_xlim(-0.03, 1.04)
                ax.set_xticks([0, 0.5, 1])
            elif metric == "min_bulk_ESS":
                ax.set_xlim(0, 42000)
                ax.set_xticks([0, 20000, 40000], ["0", "20", "40"])
                ax.set_xlabel("Minimum bulk ESS\n(thousands)", fontsize=9)
            elif metric == "trajectory_RMSE":
                ax.set_xlim(0, 0.032)
                ax.set_xticks([0, 0.015, 0.03])
            else:
                ax.set_xlim(0, 16)
                ax.set_xticks([0, 8, 16])
        axes[row, 0].text(
            -0.02,
            1.13,
            f"{dgp}: " + ("constant effect" if dgp == "D1" else "four changes"),
            transform=axes[row, 0].transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.subplots_adjust(
        left=0.24, right=0.99, bottom=0.09, top=0.93, hspace=0.48, wspace=0.25
    )
    mark_provisional(fig, provisional)
    save_pdf(fig, out)

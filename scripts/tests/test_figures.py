"""Publication-asset regression tests: tables, plots and submission TIFFs."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from scripts.figures import normalize_table
from scripts.figures import CURRENT, export


def test_table_normalization_preserves_iqr_as_text(tmp_path: Path):
    path = tmp_path / "table4_computation_sensitivity.tex"
    path.write_text(
        "\\caption{Old}\n\\begin{tabular}{lr}\nValid & 1.000 [0.9, 1.0] \\\\\n\\end{tabular}\n",
        encoding="utf-8",
    )
    normalize_table(path, "Current caption.")
    text = path.read_text(encoding="utf-8")
    assert "\\caption{Current caption.}" in text
    assert "1.000 [0.9, 1.0]" in text
    assert "Valid" not in text


def test_repeated_export_preserves_original_archive(tmp_path: Path):
    figures, upload, archive = (tmp_path / x for x in ("figures", "upload", "archive"))
    figures.mkdir()
    upload.mkdir()
    for _, stem, _ in CURRENT:
        (figures / (stem + ".pdf")).write_bytes(b"%PDF-test")
        Image.new("RGB", (12, 12), "white").save(figures / (stem + ".png"))
    original = b"original historical candidate"
    (upload / "Fig1.tif").write_bytes(original)
    export(figures, upload, archive)
    export(figures, upload, archive)
    assert (archive / "Fig1.tif").read_bytes() == original


import json

import matplotlib.pyplot as plt

from scripts.figures import TIFF_DPI, save_figure


def test_pc_save_figure_emits_submission_tiff(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.5, 1.0))
    ax.plot([0, 1], [0, 1])
    save_figure(fig, tmp_path / "pc.pdf")
    with Image.open(tmp_path / "pc.tif") as image:
        assert image.format == "TIFF"
        assert abs(image.info["dpi"][0] - TIFF_DPI) < 0.1


def test_export_has_exactly_six_current_tiffs_and_archives_retired_file(
    tmp_path: Path,
) -> None:
    figures, upload, archive = (
        tmp_path / "figures",
        tmp_path / "upload",
        tmp_path / "archive",
    )
    figures.mkdir()
    upload.mkdir()
    for _, stem, _ in CURRENT:
        (figures / f"{stem}.pdf").write_bytes(b"%PDF-current")
        Image.new("RGB", (16, 12), "white").save(
            figures / f"{stem}.png", dpi=(600, 600)
        )
    Image.new("RGB", (8, 8), "black").save(upload / "Fig7.tif")
    manifest_path = export(figures, upload, archive)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["upload_count"] == 6
    assert [x["upload"] for x in manifest["outputs"]] == [
        name for name, _, _ in CURRENT
    ]
    assert (archive / "Fig7.tif").exists()
    assert not (upload / "Fig7.tif").exists()
    for name, _, _ in CURRENT:
        with Image.open(upload / name) as image:
            assert image.format == "TIFF" and image.size == (16, 12)


import pandas as pd

from scripts.simulation_figures import (
    DGPS,
    METHODS,
    SUBMISSION_PNG_DPI,
    save_pdf,
    table2,
)


def test_save_pdf_also_emits_600_dpi_submission_png(tmp_path) -> None:
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.plot([0, 1], [0, 1])
    pdf_path = tmp_path / "figure.pdf"

    save_pdf(fig, pdf_path)

    png_path = pdf_path.with_suffix(".png")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.exists()
    with Image.open(png_path) as image:
        assert image.width > 0 and image.height > 0
        dpi_x, dpi_y = image.info["dpi"]
    assert abs(dpi_x - SUBMISSION_PNG_DPI) < 0.01
    assert abs(dpi_y - SUBMISSION_PNG_DPI) < 0.01


def test_table2_uses_compact_two_panel_layout_and_preserves_detailed_csv(
    tmp_path,
) -> None:
    records = []
    for dgp in DGPS:
        for method in METHODS:
            for replicate in range(2):
                records.append(
                    {
                        "stage": "main",
                        "dgp": dgp,
                        "method": method,
                        "trajectory_RMSE": 0.01 + replicate * 0.001,
                        "test_RMSE": 0.40 + replicate * 0.001,
                    }
                )

    table2(records, tmp_path, provisional=True)

    latex = (tmp_path / "table2_rmse.tex").read_text(encoding="utf-8")
    detail = pd.read_csv(tmp_path / "table2_rmse.csv")
    assert "Panel A: Trajectory RMSE" in latex
    assert "Panel B: Held-out test RMSE" in latex
    assert latex.count(r"\begin{tabular}") == 1
    # Keep type at the manuscript font size; PLOS discourages shrinking table text.
    assert r"\resizebox{\linewidth}{!}" not in latex
    assert len(detail) == len(DGPS) * len(METHODS)
    assert set(detail["trajectory_RMSE_n"]) == {2}
    assert set(detail["test_RMSE_n"]) == {2}

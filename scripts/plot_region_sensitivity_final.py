"""Reproduce final matched-clean Figure 2 only; no training or Figure 1 writes.

Usage: python scripts/plot_region_sensitivity_final.py
Requires matplotlib. The packaged source CSV is hash-locked to final evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("REGION_PLOT_SOURCE_10_REGIONS.csv")
SOURCE_SHA256 = "3f4c546c1760407ddfa990b482c0d6fb55899b65ed46fc7ab9aeefbfe3d726be"
REGIONS = [
    "Final token-prediction projection",
    "Feed-forward compression module",
    "Complete decoder-layer output",
    "Feed-forward expansion module",
    "Post-attention normalization",
    "Attention output transformation",
    "Input-side normalization",
    "Gated feed-forward module",
    "Attention value transformation",
    "Other regions",
]
EXPECTED_ADE = [225, 1, 0, 0, 0, 0, 3, 0, 4, 2]
EXPECTED_STRUCTURAL = [430, 163, 155, 118, 100, 99, 94, 69, 65, 112]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "figures" / UPSTREAM.name)
    parser.add_argument("--out", type=Path, default=ROOT / "generated" / "figure2")
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    assert sha256(source) == SOURCE_SHA256, "Source CSV differs from frozen final evidence"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert [int(row["display_order"]) for row in rows] == list(range(1, 11))
    names = [row["display_region"] for row in rows]
    ade_only = [int(row["ade_only"]) for row in rows]
    structural_parsing = [int(row["structural_parsing"]) for row in rows]
    assert names == REGIONS
    assert ade_only == EXPECTED_ADE
    assert structural_parsing == EXPECTED_STRUCTURAL
    assert sum(ade_only) == 235
    assert sum(structural_parsing) == 1405
    assert names[-1] == "Other regions"
    assert ade_only[-1] == 2 and structural_parsing[-1] == 112
    # geometry_related exists in the source but is deliberately not plotted.
    fig_dir = out / "Fig"
    report_dir = out / "outputs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(9.3, 5.15))
    height = 0.31
    bar_records = []
    labels = []
    for offset, values, color, label in [
        (-height / 2, ade_only, "#2C6FA5", "ADE-only failures"),
        (height / 2, structural_parsing, "#59A16D", "Structural/parsing failures"),
    ]:
        bars = ax.barh([i + offset for i in range(10)], values,
                       height=height, color=color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            assert bar.get_width() == value
            bar_records.append((bar, value))
            if value:
                labels.append(ax.text(value + 5, bar.get_y() + bar.get_height() / 2,
                                      str(value), va="center", fontsize=9.5))
    ax.set_yticks(range(10))
    ax.set_yticklabels(names, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 470)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Number of failure samples", fontsize=12)
    ax.set_ylabel("Model region", fontsize=12, labelpad=9)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#C5CDD4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    legend = ax.legend(loc="lower right", fontsize=10.3, frameon=True,
                       facecolor="white", edgecolor="#B0B0B0")
    legend.get_frame().set_linewidth(0.5)
    fig.subplots_adjust(left=0.385, right=0.985, bottom=0.125, top=0.97)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_objects = labels + ax.get_xticklabels() + ax.get_yticklabels() + [
        ax.xaxis.label, ax.yaxis.label,
    ] + list(legend.get_texts())
    for obj in text_objects:
        if obj.get_text():
            box = obj.get_window_extent(renderer)
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), obj.get_text()
    legend_box = legend.get_window_extent(renderer)
    for bar, value in bar_records:
        if value:
            assert not legend_box.overlaps(bar.get_window_extent(renderer)), "Legend covers a bar"
    for obj in labels:
        assert not legend_box.overlaps(obj.get_window_extent(renderer)), "Legend covers a count"
    assert len(labels) == sum(v > 0 for v in ade_only + structural_parsing) == 15

    pdf = fig_dir / "region_sensitivity_combined.pdf"
    png = fig_dir / "region_sensitivity_combined_preview.png"
    fig.savefig(pdf, metadata={"Title": "Figure 2: Final matched-clean region distribution"})
    fig.savefig(png, dpi=320)
    plt.close(fig)
    report = {
        "status": "PASS", "label_definition": "matched-clean", "delta_ADE_m": 1.0,
        "regions": names, "ADE_only": ade_only, "structural_parsing": structural_parsing,
        "ADE_only_total": sum(ade_only), "structural_parsing_total": sum(structural_parsing),
        "other_ADE_only": 2, "other_structural_parsing": 112,
        "source_csv": str(source), "source_sha256": sha256(source),
        "upstream_csv": str(UPSTREAM), "upstream_sha256": SOURCE_SHA256,
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "outputs": {str(p): sha256(p) for p in (pdf, png)},
        "style": {"figsize_inches": [9.3, 5.15], "bar_height": height,
                  "colors": ["#2C6FA5", "#59A16D"], "PNG_dpi": 320,
                  "matplotlib": matplotlib.__version__, "python": platform.python_version()},
        "geometry_not_plotted": True, "assertions_passed": True,
        "text_within_canvas": True, "legend_does_not_overlap_bars_or_counts": True,
        "full_manuscript_recompiled": False,
    }
    (report_dir / "figure2_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "figure2_validation.txt").write_text(
        "ADE-only total = 235\nStructural/parsing total = 1405\n"
        "Other ADE-only = 2\nOther Structural/parsing = 112\n"
        "Final projection ADE-only = 225\nSource: final matched-clean canonical evidence\n"
        "All script asserts = PASS\nText clipping check = PASS\nLegend overlap check = PASS\n"
        "Geometry-related failures = NOT PLOTTED\nPNG = 320 dpi\n"
        "Full manuscript recompilation = NOT RUN\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

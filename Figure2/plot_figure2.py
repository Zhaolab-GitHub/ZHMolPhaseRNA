#!/usr/bin/env python3
"""Reproduce the six physicochemical boxplots from the accompanying TSV files."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
GROUP_COLORS = {"Reviewed": "#fa7f6f", "Unlabeled": "#82afda"}
GROUP_BY_LABEL = {"1": "Reviewed", "0": "Unlabeled"}
OUTPUT = HERE / "Figure2_physicochemical_boxplots.png"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def configure_style(font_dir):
    if font_dir is not None:
        for path in sorted(font_dir.glob("*.ttf")):
            font_manager.fontManager.addfont(str(path))

    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        family = "DejaVu Sans"
        print(f"Arial unavailable; using {family}. Supply --font-dir for Arial.")
    else:
        family = "Arial"

    plt.rcParams.update({
        "font.family": family,
        "font.size": 22,
        "axes.linewidth": 1.25,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": family,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    if "mathtext.fallback" in plt.rcParams:
        plt.rcParams["mathtext.fallback"] = None


def p_label(raw_p):
    if raw_p >= 0.001:
        return f"$P = {raw_p:.3g}$"
    mantissa, exponent = f"{raw_p:.1e}".split("e")
    return rf"$P = {mantissa} \times 10^{{{int(exponent)}}}$"


def draw_panel(ax, panel, grouped):
    feature = panel["feature"]
    arrays = [[float(row[feature]) for row in rows] for rows in grouped.values()]
    if any(not values or not all(map(math.isfinite, values)) for values in arrays):
        raise ValueError(f"{feature} requires finite values in both groups.")
    low = min(map(min, arrays))
    high = max(map(max, arrays))
    span = high - low
    if span <= 0:
        raise ValueError(f"{feature} has no variation.")

    chart = ax.boxplot(
        arrays, positions=[1, 2], widths=0.5, whis=1.5,
        showfliers=True, patch_artist=True, manage_ticks=False,
        boxprops={"edgecolor": "#444444", "linewidth": 1.3},
        whiskerprops={"color": "#555555", "linewidth": 1.2},
        capprops={"color": "#555555", "linewidth": 1.2},
        medianprops={"color": "#333333", "linewidth": 1.9},
        flierprops={"marker": "o", "markersize": 2.4, "markeredgecolor": "none", "alpha": 0.35},
    )
    for box, flier, color in zip(chart["boxes"], chart["fliers"], GROUP_COLORS.values()):
        box.set_facecolor(color)
        box.set_alpha(0.9)
        flier.set_markerfacecolor(color)

    y = high + 0.10 * span
    h = 0.025 * span
    ax.plot([1, 1, 2, 2], [y, y + h, y + h, y], color="#444444", linewidth=1.1, clip_on=False)
    ax.text(1.5, y + h + 0.025 * span, p_label(float(panel["P_raw"])),
            ha="center", va="bottom", fontsize=20)
    ax.set_ylim(0 if low >= 0 else low - 0.04 * span, high + 0.30 * span)
    ax.set_xlim(0.45, 2.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{group}\n(n={len(rows)})" for group, rows in grouped.items()])
    ax.set_title(panel["title"], fontsize=22, pad=20)
    ax.set_ylabel(panel["y_label"], fontsize=21, labelpad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=18, pad=8, width=1.1, length=5)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="upper"))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.text(-0.16, 1.07, panel["panel_letter"], transform=ax.transAxes,
            fontsize=28, fontweight="bold", ha="left", va="bottom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path,
                        default=HERE.parent.parent / "assets" / "fonts" / "arial",
                        help="Directory containing Arial TTF files.")
    args = parser.parse_args()

    panels = read_tsv(HERE / "panels.tsv")
    if len(panels) != 6:
        raise ValueError("panels.tsv must describe all six panels in display order.")
    grouped = {group: [] for group in GROUP_COLORS}
    for row in read_tsv(HERE / "source_data.tsv"):
        grouped[GROUP_BY_LABEL[row["label"]]].append(row)

    configure_style(args.font_dir)
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.6))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.92, wspace=0.32, hspace=0.57)
    for ax, panel in zip(axes.flat, panels):
        draw_panel(ax, panel, grouped)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()

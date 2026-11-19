"""Recreate Figure 6c from the bundled 4BS2 sequence scores."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter


DATA_DIR = Path(__file__).resolve().parent


def load_histogram():
    with (DATA_DIR / "4BS2_native_score.csv").open(newline="") as handle:
        native, = csv.DictReader(handle)
    with (DATA_DIR / "4BS2_all_34650_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    background = [row for row in rows if row["sequence"] != native["sequence"]]
    native_rows = [row for row in rows if row["sequence"] == native["sequence"]]
    if len(rows) != 34650 or native_rows != [native]:
        raise ValueError("Expected 34,650 score rows and one matching native record.")

    edges = np.linspace(0.30, 1.00, 71)
    scores = [float(row["mean_probability"]) for row in background]
    counts, _ = np.histogram(scores, bins=edges)
    if counts.sum() != len(background):
        raise ValueError("One or more non-native scores fall outside the histogram range.")

    native_label = f"{float(native['mean_probability']):.4f}"
    return edges, counts, native_label, len(rows)


def plot(output):
    edges, counts, native_label, total = load_histogram()
    native_marker = float(native_label)
    coral = "#FA7F6F"
    axis_color = "#66717A"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
    })
    width, height = 7.6, 2.63
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes((0.80 / width, 0.59 / height, 6.16 / width, 1.18 / height))

    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#82AFDA")
    ax.set(xlim=(0.30, 1.01), ylim=(0, 3000),
           xticks=np.linspace(0.30, 1.00, 8), yticks=[0, 1000, 2000, 3000],
           xlabel="ZHMolPhaseRNA score", ylabel="Number of sequences")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_color)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(color=axis_color, width=0.5, length=2.3, pad=4)
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 6

    ax.axvline(native_marker, color=coral, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.77, 2700, f"Native = {native_label}", color=coral, weight="bold",
            ha="center", va="center")
    ax.hlines(2700, 0.86, native_marker - 0.003, colors=coral, linewidth=0.75)

    fig.text(0.26 / width, 0.895, "c", fontsize=11, weight="bold", va="center")
    fig.text(3.88 / width, 0.895, "Sequence permutation analysis",
             fontsize=9, ha="center", va="center")
    fig.text(0.54 / width, 0.782, f"{total:,} unique permutations", va="center")
    fig.savefig(output, dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DATA_DIR / "Figure6c.png",
                        help="Figure output path (.png, .pdf, or .svg)")
    args = parser.parse_args()
    plot(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

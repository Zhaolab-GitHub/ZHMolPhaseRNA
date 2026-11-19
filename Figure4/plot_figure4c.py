"""Draw the CAG47 pairing map and pairing statistics in Figure 4c."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


DATA_DIR = Path(__file__).resolve().parent


def load_data():
    _, *sequence_lines = (DATA_DIR / "CAG47.fasta").read_text().splitlines()
    sequence = "".join(sequence_lines)

    probabilities = np.zeros((len(sequence), len(sequence)))
    with (DATA_DIR / "panel_c_pair_probabilities.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            i, j = int(row["i"]) - 1, int(row["j"]) - 1
            probabilities[i, j] = probabilities[j, i] = float(row["probability"])
    return sequence, probabilities


def draw_probability_map(ax, probabilities):
    n = len(probabilities)
    left, right, top, bottom = 13, 61, 110, 158
    cmap = LinearSegmentedColormap.from_list(
        "pairing", [(0, "#ffffff"), (0.25, "#add3e2"), (1, "#3f7eae")], N=1024
    )
    image_options = dict(cmap=cmap, vmin=0, vmax=0.25, origin="lower",
                         aspect="auto", interpolation="nearest")
    ax.imshow(probabilities, extent=(left, right, bottom, top), **image_options)
    ax.imshow(np.linspace(0, 0.25, 256)[:, None], extent=(63, 65, bottom, top), **image_options)
    ax.hlines(bottom, left, right, colors="black", linewidth=0.6)
    ax.vlines(left, top, bottom, colors="black", linewidth=0.6)
    ax.text(41, 101, "Pairing probability map", ha="center", va="center", weight="bold")
    for nt in (1, 47, 94, n):
        offset = (nt - 0.5) * (right - left) / n
        x, y = left + offset, bottom - offset
        ax.hlines(y, left - 0.8, left, colors="black", linewidth=0.5)
        ax.vlines(x, bottom, bottom + 0.8, colors="black", linewidth=0.5)
        ax.text(11.3, y, str(nt), ha="right", va="center")
        ax.text(x, 161.1, str(nt), ha="center", va="center")
    ax.text(37, 167.1, "Nucleotide j", ha="center", va="center")
    ax.text(3.7, 134, "Nucleotide i", ha="center", va="center", rotation=90)
    ax.text(65, 106, "p(i,j)", ha="center", va="center")
    for value in (0, 0.125, 0.25):
        y = bottom - (bottom - top) * value / 0.25
        ax.text(66, y, f"{value:g}", va="center")


def draw_pairing_features(ax, sequence, probabilities):
    cg = np.array([base in "CG" for base in sequence])
    occupancy = probabilities.sum(axis=1)[cg].mean()
    maximum_pair_probability = np.triu(probabilities, k=1).max()
    ax.text(104, 101, "Pairing features", ha="center", va="center", weight="bold")
    features = [
        (123, "Mean pairing occupancy\nof C/G nucleotides", occupancy, "#8dcec8"),
        (153, "Maximum individual\nbase-pair probability", maximum_pair_probability, "#98c0de"),
    ]
    for y, label, value, color in features:
        ax.text(80, y - 9.5, label, va="center")
        ax.text(130, y - 3, f"{value:.2%}", ha="right", va="center")
        ax.add_patch(Rectangle((81, y), 49, 4, facecolor="#edf1f4", linewidth=0))
        ax.add_patch(Rectangle((81, y), 49 * value, 4, facecolor=color, linewidth=0))
        for fraction, tick_label in ((0, "0"), (0.5, "50"), (1, "100%")):
            x = 81 + 49 * fraction
            ax.vlines(x, y + 4, y + 4.7, colors="black", linewidth=0.45)
            ax.text(x, y + 6.9, tick_label, ha="center", va="center")


def plot(output):
    sequence, probabilities = load_data()
    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(138 / 25.4, 87 / 25.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 138), ylim=(173, 86))
    ax.set_axis_off()
    ax.text(4, 90.5, "c", va="center", size=13, weight="bold")
    ax.text(10, 90.7, "Pairing characteristics", va="center", weight="bold")
    draw_probability_map(ax, probabilities)
    draw_pairing_features(ax, sequence, probabilities)
    fig.savefig(output, dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DATA_DIR / "Figure4c.png")
    plot(parser.parse_args().output)

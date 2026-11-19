"""Draw the four CAG47 representative structures in Figure 4a."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as PlotPath


DATA_DIR = Path(__file__).resolve().parent
PALETTE = {"S001": "#fa7f6f", "S002": "#8dcec8", "S003": "#a7b8ee", "S004": "#add3e2"}


def base_pairs(dot_bracket):
    stack = []
    pairs = []
    for position, symbol in enumerate(dot_bracket, start=1):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")" and stack:
            pairs.append((stack.pop(), position))
        elif symbol != ".":
            raise ValueError(f"Invalid dot-bracket notation at position {position}.")
    if stack:
        raise ValueError("Unclosed base pair in dot-bracket notation.")
    return sorted(pairs, key=lambda pair: pair[1] - pair[0], reverse=True)


def draw_arc(ax, x_left, x_right, baseline, height, color):
    # Match the two cubic Bézier segments used for each arc in v8.
    center = (x_left + x_right) / 2
    k = 0.5522847498307936
    control_dx = k * (x_right - x_left) / 2
    vertices = [
        (x_left, baseline),
        (x_left, baseline - k * height), (center - control_dx, baseline - height),
        (center, baseline - height),
        (center + control_dx, baseline - height), (x_right, baseline - k * height),
        (x_right, baseline),
    ]
    path = PlotPath(vertices, [PlotPath.MOVETO] + [PlotPath.CURVE4] * 6)
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, linewidth=0.4))


def plot(output):
    with (DATA_DIR / "panel_a_structures.csv").open(newline="") as handle:
        structures = list(csv.DictReader(handle))
    seq_length = len(structures[0]["dot_bracket"])
    mfe = min(float(row["energy_kcal_mol"]) for row in structures)

    def x_at(position):
        return 35 + (position - 1) * 49 / (seq_length - 1)

    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(88 / 25.4, 86 / 25.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 88), ylim=(86, 0))
    ax.set_axis_off()
    ax.text(4, 5.5, "a", va="center", size=13, weight="bold")
    ax.text(10, 5.7, "Representative secondary-structure states", va="center", weight="bold")

    for index, row in enumerate(structures):
        sid = row["structure_id"]
        energy = float(row["energy_kcal_mol"])
        baseline = 25 + index * 14
        label = f"{sid} (MFE)" if energy == mfe else sid
        ax.text(5, baseline - 8.65, label, va="center", weight="bold")
        energy_label = f"{energy:.2f}".replace("-", "−")
        ax.text(5, baseline - 4.8, f"ΔG {energy_label} kcal/mol", va="center")
        ax.text(5, baseline - 1.2, f"P(model) {float(row['P_model']):.2%}", va="center")
        ax.hlines(baseline, x_at(1), x_at(seq_length), colors="#aeb9c1", linewidth=0.45)
        for i, j in base_pairs(row["dot_bracket"]):
            draw_arc(ax, x_at(i), x_at(j), baseline, 8.2 * (j - i) / (seq_length - 1), PALETTE[sid])
        for hairpin in row["hairpin_positions_1based"].split(";"):
            start, end = map(int, hairpin.split("-"))
            ax.hlines(baseline, x_at(start) - 0.1176, x_at(end) + 0.1176,
                      colors="#fb9f8a", linewidth=2)
            ax.text(x_at((start + end) / 2), baseline + 2.2, f"{start}–{end}",
                    ha="center", va="center")
        for position in map(int, row["highlighted_AA_positions_1based"].split(";")):
            ax.add_patch(Ellipse((x_at(position), baseline), 1.1, 1.1, facecolor="#fb9f8a",
                                 edgecolor="white", linewidth=0.2))

    for position in (1, 35, 70, 105, seq_length):
        x = x_at(position)
        ax.vlines(x, 72, 72.8, colors="#59656e", linewidth=0.45)
        ax.text(x, 74.65, str(position), ha="center", va="center")
    ax.text(60.088, 82, "Nucleotide position", ha="center", va="center")
    fig.savefig(output, dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DATA_DIR / "Figure4a.png")
    plot(parser.parse_args().output)

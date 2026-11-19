"""Draw the saved CAG47 metric MDS projection in Figure 4b."""

import argparse
import csv
from math import hypot, sqrt
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


DATA_DIR = Path(__file__).resolve().parent
REPRESENTATIVES = {
    "S001": ("#fa7f6f", 112, 52),
    "S002": ("#8dcec8", 151.5, 66),
    "S003": ("#a7b8ee", 151.5, 32.5),
    "S004": ("#add3e2", 111, 36.2),
}


def canvas_position(mds1, mds2):
    """Convert saved MDS coordinates to v8 figure coordinates in millimeters."""
    return 136.8 + 0.36 * mds1, 43.08 - 0.36 * mds2


def plot(output):
    with (DATA_DIR / "panel_b_mds.csv").open(newline="") as handle:
        structures = list(csv.DictReader(handle))
    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(102 / 25.4, 86 / 25.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(88, 190), ylim=(86, 0))
    ax.set_axis_off()
    ax.text(90, 5.5, "b", va="center", size=13, weight="bold")
    ax.text(96, 5.7, "Structural heterogeneity of sampled CAG47 conformations",
            va="center", weight="bold")
    ax.text(103, 11.25, "Point area represents sampling frequency", va="center")

    left, right, top, bottom = 108, 165.6, 15, 70.44
    ax.vlines(left, top, bottom, linewidth=0.65, color="black")
    ax.hlines(bottom, left, right, linewidth=0.65, color="black")
    for tick in (-60, -30, 0, 30, 60):
        x, y = canvas_position(tick, tick)
        ax.vlines(x, bottom, bottom + 0.9, linewidth=0.5, color="black")
        ax.hlines(y, left - 0.9, left, linewidth=0.5, color="black")
        ax.text(x, 73.49, str(tick), ha="center", va="center")
        ax.text(106, y, str(tick), ha="right", va="center")
    ax.text(137, 78.2, "Metric MDS coordinate 1", ha="center", va="center")
    ax.text(98, 43.8, "Metric MDS coordinate 2", ha="center", va="center", rotation=90)

    palette = LinearSegmentedColormap.from_list(
        "energy", ["#82afda", "#a7b8ee", "#e8a9d0", "#fa7f6f"], N=1024
    )
    points = {}
    for row in structures:
        sid = row["structure_id"]
        x, y = canvas_position(float(row["display_MDS1"]), float(row["display_MDS2"]))
        raw_x, raw_y = canvas_position(float(row["MDS1"]), float(row["MDS2"]))
        if hypot(x - raw_x, y - raw_y) > 1e-6:
            ax.plot((raw_x, x), (raw_y, y), color="#bdcbd5", linewidth=0.28, zorder=1)
        # A diameter proportional to sqrt(count) makes the disk area proportional to count.
        radius = 0.95 * sqrt(int(row["sample_count"])) / 2
        points[sid] = (x, y, radius)
        energy = float(row["deltaG_from_MFE_kcal_mol"])
        alpha = 1 if sid in REPRESENTATIVES else 0.88
        rgb = tuple(int(channel * 255) / 255 for channel in palette(energy / 12)[:3])
        ax.add_patch(Circle((x, y), radius, facecolor=rgb, edgecolor="#5e8099",
                            alpha=alpha, linewidth=0.30, zorder=2))
    for sid, (color, label_x, label_y) in REPRESENTATIVES.items():
        x, y, radius = points[sid]
        ax.add_patch(Circle((x, y), radius + 0.275, fill=False, edgecolor=color,
                            linewidth=1.1, zorder=3))
        start_x, start_y = (label_x + 1, label_y + 0.4) if sid == "S002" else (
            label_x + 8, label_y + 3.1
        )
        dx, dy = x - start_x, y - start_y
        cutoff = (radius + 0.65) / hypot(dx, dy)
        ax.plot((start_x, x - dx * cutoff), (start_y, y - dy * cutoff),
                color="black", linewidth=0.5, zorder=3)
        ax.text(label_x, label_y + 1.85, sid, va="center", weight="bold")

    ax.imshow(np.linspace(0, 12, 256)[:, None], cmap=palette, vmin=0, vmax=12,
              extent=(174, 176.4, bottom, top), origin="lower", aspect="auto")
    for tick in (0, 4, 8, 12):
        ax.text(177.4, bottom - (bottom - top) * tick / 12, str(tick), va="center")
    ax.text(185.5, 43.5, "ΔG from MFE (kcal/mol)", ha="center", va="center", rotation=90)
    fig.savefig(output, dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DATA_DIR / "Figure4b.png")
    plot(parser.parse_args().output)

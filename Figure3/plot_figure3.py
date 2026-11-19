#!/usr/bin/env python3
"""Plot the six main-results panels from the tables beside this script.

Run without arguments for the combined PNG, or use --panel a through --panel f
for a single panel. The script uses saved predictions and requires no models.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
MODELS = {
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient boosting",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "extra_trees": "Extra Trees",
    "mlp": "MLP",
    "random_forest": "Random forest",
    "svm_rbf": "SVM (RBF)",
    "lda": "LDA",
    "logistic_regression": "Logistic regression",
    "adaboost": "AdaBoost",
    "knn": "KNN",
    "ridge": "Ridge",
    "gaussian_nb": "Gaussian NB",
}
COLORS = [
    "#fa7f6f", "#fb9f8a", "#ffbe7a", "#f8cc79", "#f0d879",
    "#dfdf82", "#cde58b", "#b4d28a", "#9bbf8a", "#94c7a9",
    "#8dcec8", "#9dcfe5", "#add3e2", "#98c0de", "#82afda", "#94a3e4",
]
ROC_GRID = np.linspace(0, 1, 1001)
RECALL_GRID = np.arange(365) / 364
METRICS = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "MCC"]
EXTERNAL_GROUPS = {
    "HTP-supported": "d_htp_supported_scores.txt",
    "Background": "d_background_scores.txt",
}
ORIGIN_GROUPS = {
    "Natural": "e_natural_scores.txt",
    "Synthetic": "e_synthetic_scores.txt",
    "Unlabeled": "e_unlabeled_scores.txt",
}


def configure_font(directory):
    for path in sorted(directory.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        family = "DejaVu Sans"
        print(f"Arial unavailable; using {family}. Supply --font-dir for Arial.")
    else:
        family = "Arial"

    plt.rcParams.update({
        "font.family": family, "font.size": 22,
        "axes.labelsize": 28, "axes.titlesize": 28, "axes.linewidth": 1.5,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
        "mathtext.cal": family, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.tick_params(labelsize=20, pad=8)
    ax.xaxis.label.set_fontsize(24)
    ax.yaxis.label.set_fontsize(24)


def summarize_oof():
    source = pd.read_csv(HERE / "abc_oof_scores.tsv", sep="\t", float_precision="round_trip")
    datasets = list(source.groupby("dataset_id"))
    if (
        [dataset_id for dataset_id, _ in datasets] != list(range(1, 11))
        or source.duplicated(["dataset_id", "RNA_ID"]).any()
        or any(frame["label"].value_counts().to_dict() != {0: 364, 1: 364} for _, frame in datasets)
    ):
        raise ValueError("Expected ten OOF datasets, each with 364 reviewed and 364 unlabeled RNAs.")

    scores = source[list(MODELS)].to_numpy()
    if not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("OOF scores must be in [0, 1].")

    means = []
    curves = {}
    for model in MODELS:
        rows, roc, pr = [], [], []
        for _, frame in datasets:
            y = frame["label"].to_numpy()
            score = frame[model].to_numpy()
            predicted = score >= 0.5
            fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
            roc.append(np.interp(ROC_GRID, fpr, tpr))
            precision, recall, _ = precision_recall_curve(y, score)
            ap = -np.sum(np.diff(recall) * precision[:-1])
            indices = np.searchsorted(recall[::-1], RECALL_GRID, side="left")
            steps = precision[::-1][indices]
            if not np.isclose(np.sum(np.diff(RECALL_GRID) * steps[1:]), ap, rtol=0, atol=1e-12):
                raise ValueError(f"PR step area differs from average precision for {model}.")
            pr.append(steps)
            rows.append([
                roc_auc_score(y, score), ap, precision_score(y, predicted, zero_division=0),
                recall_score(y, predicted, zero_division=0), f1_score(y, predicted, zero_division=0),
                matthews_corrcoef(y, predicted),
            ])

        metrics = pd.DataFrame(rows, columns=METRICS)
        means.append((model, metrics.mean(), metrics.std(ddof=1)))
        curves[model] = {"roc": np.mean(roc, axis=0), "pr": np.mean(pr, axis=0)}
    means.sort(key=lambda summary: summary[1]["AUROC"], reverse=True)
    return means, curves


def draw_curve(ax, summaries, curves, kind):
    is_roc = kind == "roc"
    grid, metric = (ROC_GRID, "AUROC") if is_roc else (RECALL_GRID, "AUPRC")
    patterns = ["solid", (0, (5, 2)), (0, (2, 1.6))]
    for i, (model, mean, sd) in enumerate(summaries):
        label = f"{MODELS[model]}  {mean[metric]:.3f} ± {sd[metric]:.3f}"
        ax.plot(grid, curves[model][kind], color=COLORS[i], linewidth=3,
                linestyle=patterns[i % 3], drawstyle="default" if is_roc else "steps-pre",
                label=label)
    baseline = dict(color="#888888", linewidth=1.6, linestyle=(0, (4, 3)), zorder=0)
    if is_roc:
        ax.plot([0, 1], [0, 1], **baseline)
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
    else:
        ax.axhline(0.5, **baseline)
        ax.set(xlabel="Recall", ylabel="Precision")
    ax.set(xlim=(0, 1), ylim=(0, 1.015))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("ROC curves" if is_roc else "AUPRC curves", fontsize=30, pad=22)
    legend = ax.legend(loc="lower right" if is_roc else "lower left", frameon=False, fontsize=13.5,
                       handlelength=2.2, handletextpad=0.6, labelspacing=0.42, borderaxespad=0.7,
                       borderpad=0.55)
    for label in legend.get_texts():
        label.set_fontsize(15.5)


def draw_performance(ax, summaries):
    _, mean, sd = next(row for row in summaries if row[0] == "lightgbm")
    x = np.arange(len(METRICS))
    ax.bar(x, mean, 0.67, color=[COLORS[i] for i in (12, 15, 3, 6, 9, 0)],
           edgecolor="none", yerr=sd, zorder=3,
           error_kw=dict(ecolor="#454545", capsize=7, capthick=1.7, elinewidth=1.7))
    for i, (value, error) in enumerate(zip(mean, sd)):
        ax.text(i, value + error + 0.026, f"{value:.3f}", ha="center", va="bottom", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(["AUROC", "AUPRC", "Pre", "Recall", "F1", "MCC"])
    ax.set(ylabel="Score", ylim=(0, 1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title("Performance of ZHMolPhaseRNA", fontsize=30, pad=22)


def draw_scores(ax, files, colors, title, ylabel):
    groups = list(files)
    arrays = [
        pd.read_csv(HERE / filename, sep="\t", float_precision="round_trip")["score"].to_numpy()
        for filename in files.values()
    ]
    if any(not len(values) or not np.all((values >= 0) & (values <= 1)) for values in arrays):
        raise ValueError("Each score file must contain values in [0, 1].")

    positions = np.arange(1, len(groups) + 1)
    violin = ax.violinplot(arrays, positions=positions, widths=0.80, points=256, bw_method="scott",
                           showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set(facecolor=color, edgecolor=color, alpha=0.65, linewidth=1.1)
    ax.boxplot(arrays, positions=positions, widths=0.20, whis=1.5, patch_artist=True,
               showfliers=True, manage_ticks=False, zorder=4,
               boxprops=dict(facecolor="white", edgecolor="#454545", linewidth=1.4),
               whiskerprops=dict(color="#454545", linewidth=1.4),
               capprops=dict(color="#454545", linewidth=1.4),
               medianprops=dict(color="#222222", linewidth=2.2),
               flierprops=dict(marker="o", markersize=2.4, markerfacecolor="#444444",
                               markeredgecolor="none", alpha=0.30))
    ax.axhline(0.5, color="#888888", linewidth=1.15, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(groups) + 0.43, 0.51, "0.5", ha="right", va="bottom",
            fontsize=14, color="#666666")
    ax.set(xlim=(0.45, len(groups) + 0.48), ylim=(-0.025, 1.205))
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{group}\n(n = {len(values):,})" for group, values in zip(groups, arrays)])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_title(title, fontsize=30, pad=22, linespacing=1.2)
    return arrays


def draw_external(ax):
    x, y = draw_scores(ax, EXTERNAL_GROUPS, [COLORS[0], COLORS[14]],
                       "Prediction scores of HTP-supported\nand background RNAs",
                       "Prediction score")
    for i, values in enumerate((x, y), 1):
        ax.text(i, 1.035, f"Median = {np.median(values):.3f}",
                ha="center", va="bottom", fontsize=18)

    # Two-sided Mann–Whitney U with tie and continuity corrections. Evaluate
    # the normal tail in log space because the original P value underflows.
    scores = np.concatenate([x, y])
    n1, n2 = len(x), len(y)
    n = n1 + n2
    u1 = rankdata(scores)[:n1].sum() - n1 * (n1 + 1) / 2
    ties = np.unique(scores, return_counts=True)[1].astype(float)
    variance = n1 * n2 / 12 * (n + 1 - np.sum(ties**3 - ties) / (n * (n - 1)))
    z = max(0., abs(u1 - n1 * n2 / 2) - 0.5) / np.sqrt(variance) if variance > 0 else 0.
    log_p = np.log(2) + norm.logsf(z)
    annotation = (r"Mann–Whitney P < $10^{-300}$" if log_p < -300 * np.log(10)
                  else f"Mann–Whitney P = {np.exp(log_p):.2g}")
    ax.plot([1, 1, 2, 2], [1.10, 1.12, 1.12, 1.10], color="#454545", linewidth=1.4)
    ax.text(1.5, 1.135, annotation, ha="center", va="bottom", fontsize=18)


def draw_ablation(ax):
    data = pd.read_csv(HERE / "f_ablation_plot_data.tsv", sep="\t")
    x = np.arange(len(data))
    for offset, metric, color in [(-0.19, "AUROC", COLORS[0]), (0.19, "AUPRC", COLORS[14])]:
        mean, sd = data[f"{metric}_mean"], data[f"{metric}_SD"]
        ax.bar(x + offset, mean, 0.34, color=color, edgecolor="none", yerr=sd,
               error_kw=dict(ecolor="#454545", capsize=5, capthick=1.3, elinewidth=1.3),
               label=metric, zorder=3)
        for i, (value, error) in enumerate(zip(mean, sd)):
            ax.text(i + offset, value + error + 0.012, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace(" + ", " +\n").replace(" only", "\nonly")
                       for name in data["feature_set"]])
    for label, name in zip(ax.get_xticklabels(), data["feature_set"]):
        if name == "ERNIE-RNA + Handcrafted":
            label.set_fontweight("bold")
    ax.set(ylabel="Score", ylim=(0, 1.02))
    ax.set_title("Feature ablation", fontsize=30, pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=2, frameon=False, fontsize=18)


def layout(fig):
    """Use the original upper-panel sizes and the added third row of scores."""
    grid = fig.add_gridspec(3, 2, height_ratios=[1.12, 0.72, 0.90],
                           left=0.075, right=0.975, bottom=0.055, top=0.965,
                           wspace=0.25, hspace=0.36)
    height = fig.get_figheight()
    positions = {}
    for letter, spec in zip("abcf", [grid[0, 0], grid[0, 1], grid[1, :], grid[2, :]]):
        box = spec.get_position(fig)
        positions[letter] = [box.x0, (box.y0 * 25 + 10) / height, box.width, box.height * 25 / height]

    ablation = positions["f"]
    score_letter_y = ablation[1] + ablation[3] * 1.08 + 0.45 / height
    score_y = score_letter_y - 8.45 / height
    ablation[1] = 1.6 / height
    positions["d"] = [0.075, score_y, 0.40, 7.2 / height]
    positions["e"] = [0.575, score_y, 0.40, 7.2 / height]
    return positions, score_letter_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=list("abcdef"), help="Save just the specified panel.")
    parser.add_argument("--font-dir", type=Path,
                        default=HERE.parent.parent / "assets" / "fonts" / "arial",
                        help="Directory containing Arial TTF files.")
    args = parser.parse_args()
    configure_font(args.font_dir)

    fig = plt.figure(figsize=(20, 35))
    positions, score_letter_y = layout(fig)
    axes = {letter: fig.add_axes(position) for letter, position in positions.items()
            if args.panel is None or args.panel == letter}
    if any(letter in axes for letter in "abc"):
        summaries, curves = summarize_oof()
        for letter, kind in [("a", "roc"), ("b", "pr")]:
            if letter in axes:
                draw_curve(axes[letter], summaries, curves, kind)
        if "c" in axes:
            draw_performance(axes["c"], summaries)
    if "d" in axes:
        draw_external(axes["d"])
    if "e" in axes:
        draw_scores(axes["e"], ORIGIN_GROUPS, [COLORS[i] for i in (0, 3, 14)],
                    "OOF prediction score distributions\nacross RNA groups",
                    "Mean OOF prediction score")
    if "f" in axes:
        draw_ablation(axes["f"])

    letter_style = dict(fontsize=32, fontweight="bold", ha="left", va="bottom")
    for letter, ax in axes.items():
        style_axis(ax)
        x = 0.5255 if letter in "be" else 0.0255
        if letter in "de":
            fig.text(x, score_letter_y, letter, **letter_style)
        else:
            transform = blended_transform_factory(fig.transFigure, ax.transAxes)
            ax.text(x, 1.04 if letter in "ab" else 1.08, letter, transform=transform, **letter_style)

    suffix = args.panel or "combined"
    path = HERE / f"Figure3_main_results_{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

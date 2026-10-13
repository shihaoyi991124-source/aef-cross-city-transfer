"""Stage14C-P3: compact, vector five-city domain-signature display."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure3_source.csv"
FROZEN = ROOT / "outputs/tables/stage07_domain_classifier.csv"
OUT = ROOT / "outputs/paper/figures/Figure3_domain_signature_v2"
ORDER = ("SZ", "KM", "WH", "XA", "BJ")
NAMES = {"Shenzhen": "SZ", "Kunming": "KM", "Wuhan": "WH",
         "Xian": "XA", "Beijing": "BJ"}
INK = "#26343D"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read_rows(SOURCE)
    frozen = read_rows(FROZEN)
    matrix_rows = [r for r in source if r["panel"] == "confusion_matrix"]
    metric_rows = {r["metric"]: r for r in source if r["panel"] == "summary_metrics"}
    assert len(matrix_rows) == 25 and set(metric_rows) == {"accuracy", "macro_f1"}
    assert len(frozen) == 5 and {r["model"] for r in frozen} == {"Logistic Regression"}
    matrix: dict[tuple[str, str], float] = {}
    for row in matrix_rows:
        actual, predicted = NAMES[row["actual_city"]], NAMES[row["predicted_city"]]
        count = sum(int(f[f"cm_{actual}_{predicted}"]) for f in frozen)
        assert abs(float(row["count_across_five_seeds"]) - count) < 1e-9
        row_total = sum(int(f[f"cm_{actual}_{p}"]) for f in frozen for p in ORDER)
        assert abs(float(row["row_normalized_proportion"]) - count / row_total) < 5e-11
        matrix[actual, predicted] = float(row["row_normalized_proportion"])
        assert row["n_seeds"] == "5"
    assert len(matrix) == 25
    for metric, row in metric_rows.items():
        values = [float(f[metric]) for f in frozen]
        assert abs(float(row["mean"]) - statistics.mean(values)) < 5e-11
        assert abs(float(row["std"]) - statistics.stdev(values)) < 5e-11

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.8,
        "axes.titlesize": 9.3, "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(6.65, 3.55), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 0.60],
                           left=0.11, right=0.96, top=0.84, bottom=0.15,
                           wspace=0.20)
    ax = fig.add_subplot(grid[0, 0])
    bar = fig.add_subplot(grid[0, 1])
    info = fig.add_subplot(grid[0, 2])
    cmap = mpl.colormaps["cividis"]
    norm = Normalize(0, 1)
    for i, actual in enumerate(ORDER):
        for j, predicted in enumerate(ORDER):
            value = matrix[actual, predicted]
            rgba = cmap(norm(value))
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=rgba, edgecolor="white",
                                   linewidth=.55))
            label = f"{value * 100:.3f}%" if value else "0"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.0,
                    color=INK if value > .55 else "white")
    ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(5), ORDER)
    ax.set_yticks(range(5), ORDER)
    ax.set_xlabel("Predicted city", labelpad=5, color=INK)
    ax.set_ylabel("True city", labelpad=5, color=INK)
    ax.set_title("Five-city domain classification", loc="left", pad=9,
                 fontweight="bold", color=INK)
    ax.tick_params(length=0, pad=4, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for step in range(100):
        lo, hi = step / 100, (step + 1) / 100
        bar.add_patch(Rectangle((0, lo), 1, hi - lo,
                                facecolor=cmap((lo + hi) / 2), edgecolor="none"))
    bar.set(xlim=(0, 1), ylim=(0, 1))
    bar.set_xticks([])
    bar.yaxis.tick_right()
    bar.set_yticks([0, .25, .50, .75, 1.0])
    bar.tick_params(axis="y", labelsize=8.0, colors=INK, length=2, width=.65)
    for spine in bar.spines.values():
        spine.set_color("#66767E")
        spine.set_linewidth(.65)

    info.axis("off")
    info.text(0.05, .80, "Five-seed mean ± SD", transform=info.transAxes,
              fontsize=8.8, fontweight="bold", color=INK)
    info.text(0.05, .64,
              f"Accuracy\n{float(metric_rows['accuracy']['mean']):.6f} ± "
              f"{float(metric_rows['accuracy']['std']):.6f}",
              transform=info.transAxes, fontsize=8.6, color=INK, linespacing=1.6)
    info.text(0.05, .39,
              f"Macro-F1\n{float(metric_rows['macro_f1']['mean']):.6f} ± "
              f"{float(metric_rows['macro_f1']['std']):.6f}",
              transform=info.transAxes, fontsize=8.6, color=INK, linespacing=1.6)
    info.text(0.05, .12, "Spatially held-out\ntest blocks", transform=info.transAxes,
              fontsize=8.0, color="#52616A", linespacing=1.35)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure3 v2: 25 pooled confusion cells and two frozen metrics verified")


if __name__ == "__main__":
    main()

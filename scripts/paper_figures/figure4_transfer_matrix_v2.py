"""Stage14C-P2: render the frozen directed-transfer matrices as vector art."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure4_source.csv"
FROZEN = {
    "Macro-F1": ROOT / "outputs/tables/stage07_transfer_matrix_macro_f1.csv",
    "Expansion-F1": ROOT / "outputs/tables/stage07_transfer_matrix_expansion_f1.csv",
}
OUT = ROOT / "outputs/paper/figures/Figure4_transfer_matrix_v2"
ORDER = ["SZ", "KM", "WH", "XA", "BJ"]
INK = "#26343D"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read_rows(SOURCE)
    assert len(source) == 50, "Expected 2 metrics × 25 source–target cells"
    matrices: dict[str, dict[tuple[str, str], float]] = {}
    for metric in FROZEN:
        data = [r for r in source if r["metric"] == metric]
        assert len(data) == 25
        matrix = {(r["source_abbreviation"], r["target_abbreviation"]): float(r["value"]) for r in data}
        assert len(matrix) == 25
        frozen = read_rows(FROZEN[metric])
        assert [r["source_abbreviation"] for r in frozen] == ORDER
        for row in frozen:
            for target in ORDER:
                assert abs(matrix[row["source_abbreviation"], target] - float(row[target])) < 1e-10
        for row in data:
            diagonal = row["source_abbreviation"] == row["target_abbreviation"]
            assert row["evaluation"].startswith("same-city") == diagonal
            assert row["seeds"] == "5"
        matrices[metric] = matrix
    assert abs(matrices["Macro-F1"]["SZ", "KM"] - 0.8922977732) < 1e-10
    assert abs(matrices["Expansion-F1"]["SZ", "KM"] - 0.8646454491) < 1e-10
    assert abs(matrices["Macro-F1"]["KM", "SZ"] - 0.9862516189) < 1e-10
    assert abs(matrices["Expansion-F1"]["KM", "SZ"] - 0.9823961172) < 1e-10

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.8,
        "axes.titlesize": 9.4, "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    cmap = mpl.colormaps["viridis"]
    norm = Normalize(vmin=0.70, vmax=1.00)
    fig = plt.figure(figsize=(7.6, 3.75), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055],
                           left=0.09, right=0.945, top=0.85, bottom=0.17, wspace=0.27)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for panel, metric, ax in zip("ab", FROZEN, axes[:2]):
        matrix = matrices[metric]
        for i, source_city in enumerate(ORDER):
            for j, target_city in enumerate(ORDER):
                value = matrix[source_city, target_city]
                rgba = cmap(norm(value))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=rgba, edgecolor="none", zorder=1))
                # The diagonal keeps its performance fill; a dark dashed border
                # identifies same-city references without recolouring values.
                if i == j:
                    ax.add_patch(Rectangle((j - 0.488, i - 0.488), 0.976, 0.976,
                                           fill=False, edgecolor="#25333B", linewidth=1.15,
                                           linestyle=(0, (3.2, 1.7)), zorder=3))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=8.2, color="white" if luminance < 0.48 else "#17242A",
                        zorder=4)
        ax.set(xlim=(-0.5, 4.5), ylim=(4.5, -0.5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(range(5), ORDER)
        ax.set_yticks(range(5), ORDER)
        ax.set_xlabel("Target city", labelpad=4, color=INK)
        ax.set_ylabel("Source city", labelpad=4, color=INK)
        ax.set_title(f"({panel})  {metric}", loc="left", pad=8,
                     fontweight="bold", color=INK)
        ax.tick_params(length=0, pad=5, colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = axes[2]
    steps = 100
    for step in range(steps):
        lo = 0.70 + 0.30 * step / steps
        hi = 0.70 + 0.30 * (step + 1) / steps
        bar.add_patch(Rectangle((0, lo), 1, hi - lo, facecolor=cmap(norm((lo + hi) / 2)),
                                edgecolor="none", zorder=1))
    bar.set(xlim=(0, 1), ylim=(0.70, 1.00))
    bar.set_xticks([])
    bar.yaxis.tick_right()
    bar.yaxis.set_label_position("right")
    bar.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    bar.set_ylabel("F1 score", fontsize=8.8, color=INK, labelpad=5)
    bar.tick_params(axis="y", colors=INK, labelsize=8.0, length=2.5, width=0.7)
    for spine in bar.spines.values():
        spine.set_color("#66767E")
        spine.set_linewidth(0.7)
    fig.text(0.465, 0.065, "Dashed cell outline: same-city reference",
             ha="center", color="#52616A", fontsize=8.0)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 v2: 50 frozen values checked; vector rectangles and colorbar rendered")


if __name__ == "__main__":
    main()

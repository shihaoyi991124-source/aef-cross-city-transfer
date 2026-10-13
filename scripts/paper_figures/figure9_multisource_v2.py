"""Stage14C-P1 point-pair Figure 9 from the frozen Stage11 source table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure9_source.csv"
TABLE5 = ROOT / "outputs/paper/tables/Table5_multisource_results.csv"
OUT = ROOT / "outputs/paper/figures/Figure9_multisource_v2"
DEV = "#276D84"
EXT = "#A86228"
INK = "#26343D"
GRAY = "#66767E"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows(SOURCE)
    old = {r["City"]: r for r in read_rows(TABLE5) if r["Role"] != "Summary"}
    order = ["Beijing", "Kunming", "Shenzhen", "Wuhan", "Xian", "Chengdu", "Hangzhou"]
    assert [r["target_city"] for r in rows] == order
    keys = [
        ("mean_single_source_macro_f1", "Mean Single-source Macro-F1"),
        ("pooled_macro_f1", "Pooled Macro-F1"),
        ("uniform_ensemble_macro_f1", "Uniform Ensemble Macro-F1"),
        ("oracle_best_single_macro_f1", "Best Single-source Oracle"),
    ]
    for r in rows:
        assert all(abs(float(r[a]) - float(old[r["target_city"]][b])) < 1e-10 for a, b in keys)
        assert float(r["pooled_macro_f1"]) > float(r["mean_single_source_macro_f1"])
    assert float(rows[1]["uniform_ensemble_macro_f1"]) > float(rows[1]["pooled_macro_f1"])

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, (devax, extax) = plt.subplots(
        2, 1, figsize=(7.5, 5.3), sharex=True,
        gridspec_kw={"height_ratios": [5, 2], "hspace": 0.34},
    )
    fig.patch.set_facecolor("white")
    for idx, (ax, group, role, color, shape) in enumerate(
        [
            (devax, rows[:5], "Development targets", DEV, "o"),
            (extax, rows[5:], "External targets", EXT, "D"),
        ]
    ):
        for j, r in enumerate(group):
            y = len(group) - 1 - j
            mean = float(r["mean_single_source_macro_f1"])
            pooled = float(r["pooled_macro_f1"])
            ensemble = float(r["uniform_ensemble_macro_f1"])
            oracle = float(r["oracle_best_single_macro_f1"])
            ax.plot([mean, pooled], [y, y], color=color, linewidth=1.55, zorder=2)
            ax.scatter(mean, y, s=46, marker=shape, facecolors="white",
                       edgecolors=color, linewidths=1.35, zorder=4)
            ax.scatter(pooled, y, s=48, marker=shape, facecolors=color,
                       edgecolors=color, linewidths=1.2, zorder=5)
            ax.scatter(ensemble, y + 0.17, s=24, marker="^", facecolors="white",
                       edgecolors=GRAY, linewidths=0.9, zorder=3)
            ax.scatter(oracle, y - 0.17, s=29, marker="x", color=GRAY,
                       linewidths=0.85, zorder=3)
        names = ["Xi'an" if r["target_city"] == "Xian" else r["target_city"] for r in group]
        ax.set_yticks(list(reversed(range(len(group)))), names)
        ax.set_ylim(-0.52, len(group) - 0.48)
        ax.set_xlim(0.84, 1.005)
        ax.grid(axis="x", color="#E7ECEE", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#86939A")
        ax.tick_params(axis="y", length=0, colors=INK, pad=8)
        ax.tick_params(axis="x", length=2.5, width=0.65, colors=INK)
        ax.set_title(f"({'ab'[idx]})  {role} · pooled > mean single: {len(group)}/{len(group)}",
                     loc="left", pad=6, fontweight="bold", color=INK)

    extax.set_xticks([0.84, 0.88, 0.92, 0.96, 1.00])
    extax.set_xlabel("Macro-F1 on held-out target test blocks", labelpad=7, color=INK)
    handles = [
        Line2D([], [], marker="o", markersize=5.5, linestyle="None",
               markerfacecolor="white", markeredgecolor=DEV, label="Mean single source"),
        Line2D([], [], marker="o", markersize=5.5, linestyle="None",
               markerfacecolor=DEV, markeredgecolor=DEV, label="Pooled"),
        Line2D([], [], marker="^", markersize=5.5, linestyle="None",
               markerfacecolor="white", markeredgecolor=GRAY, label="Uniform ensemble"),
        Line2D([], [], marker="x", markersize=5.5, linestyle="None",
               color=GRAY, label="Retrospective oracle"),
        Line2D([], [], marker="D", markersize=5.2, linestyle="None",
               markerfacecolor=EXT, markeredgecolor=EXT, label="External target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, -0.006), columnspacing=1.5,
               handletextpad=0.48, fontsize=7.8)
    fig.subplots_adjust(left=0.175, right=0.975, top=0.92, bottom=0.21)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 9 v2 rendered from 7 frozen target rows; all four strategies cross-checked to original Table 5")


if __name__ == "__main__":
    main()

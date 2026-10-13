"""Stage14C-P3: frozen LOCO and fixed-source/target rank diagnostics."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure6_source.csv"
LOCO = ROOT / "outputs/tables/stage07b_leave_one_city_out.csv"
FIXED_SOURCE = ROOT / "outputs/tables/stage07b_within_source_target_difficulty.csv"
FIXED_TARGET = ROOT / "outputs/tables/stage07b_within_target_ranking.csv"
OUT = ROOT / "outputs/paper/figures/Figure6_robustness_v2"
ORDER = ("BJ", "KM", "SZ", "WH", "XA")
BLUE = "#276D84"
ORANGE = "#AF6B2D"
INK = "#26343D"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read_rows(SOURCE)
    loco = {r["removed_abbreviation"]: r for r in read_rows(LOCO)}
    fixed_source = read_rows(FIXED_SOURCE)
    fixed_target = read_rows(FIXED_TARGET)
    assert len(loco) == len(fixed_source) == len(fixed_target) == 5
    assert set(loco) == set(ORDER)
    old_loco = {r["category"]: r for r in source if r["panel"] == "leave-one-city-out"}
    for city in ORDER:
        old = old_loco[f"Remove {city}"]
        assert abs(float(old["spearman_rho"]) - float(loco[city]["macro_spearman_rho"])) < 1e-10
        assert old["n_pairs"] == loco[city]["directed_pairs"] == "12"
    specs = [
        ("Fixed source · Macro-F1", fixed_source, "macro_spearman_rho",
         "Within source: target difficulty", "Macro-F1 gap", BLUE, "o"),
        ("Fixed source · Expansion-F1", fixed_source, "expansion_spearman_rho",
         "Within source: target difficulty", "Expansion-F1 gap", ORANGE, "D"),
        ("Fixed target · Macro-F1", fixed_target, "macro_spearman_rho",
         "Within target: source selection", "Macro-F1 gap", BLUE, "o"),
        ("Fixed target · Expansion-F1", fixed_target, "expansion_spearman_rho",
         "Within target: source selection", "Expansion-F1 gap", ORANGE, "D"),
    ]
    old_medians = {(r["category"], r["outcome"]): float(r["spearman_rho"])
                   for r in source if r["panel"] == "within-target-vs-within-source"}
    for _, frozen, column, category, outcome, _, _ in specs:
        median = statistics.median(float(row[column]) for row in frozen)
        assert abs(median - old_medians[category, outcome]) < 1e-10

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.4, "axes.labelsize": 8.5,
        "axes.titlesize": 9.2, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.75),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    for i, city in enumerate(ORDER):
        y = 4 - i
        row = loco[city]
        ax.scatter(float(row["macro_spearman_rho"]), y + .12, s=30,
                   facecolor=BLUE, edgecolor="#164558", linewidth=.55,
                   marker="o", zorder=4)
        ax.scatter(float(row["expansion_spearman_rho"]), y - .12, s=27,
                   facecolor="white", edgecolor=ORANGE, linewidth=1.2,
                   marker="D", zorder=4)
    ax.set_yticks(range(5), [f"Remove {c}" for c in reversed(ORDER)])
    ax.set_xlim(0, .9)
    ax.set_ylim(-.45, 4.55)
    ax.set_xlabel("Spearman ρ", labelpad=5, color=INK)
    ax.set_title("(a)  Leave-one-city-out", loc="left", pad=9,
                 fontweight="bold", color=INK)
    ax.grid(axis="x", color="#E7ECEE", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.7, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#86939A")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE,
               markeredgecolor="#164558", label="Macro-F1", markersize=5.4),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="white",
               markeredgecolor=ORANGE, label="Expansion-F1", markersize=5.1),
    ], frameon=False, loc="lower left", fontsize=7.6, handletextpad=.4,
              labelspacing=.35, borderaxespad=.25)

    ax = axes[1]
    offsets = (-.14, -.07, 0, .07, .14)
    labels = []
    for idx, (label, frozen, column, category, outcome, color, marker) in enumerate(specs):
        y = 3 - idx
        role = "Fixed source" if label.startswith("Fixed source") else "Fixed target"
        outcome_label = "Macro-F1" if outcome == "Macro-F1 gap" else "Expansion-F1"
        labels.append(f"{role}\n{outcome_label}")
        values = [float(row[column]) for row in frozen]
        for value, offset in zip(values, offsets):
            ax.scatter(value, y + offset, s=25, marker=marker,
                       facecolor="white", edgecolor=color, linewidth=1.0,
                       zorder=3)
        median = old_medians[category, outcome]
        ax.plot([median, median], [y - .20, y + .20], color=INK,
                linewidth=2.4, solid_capstyle="round", zorder=5)
    ax.axvline(0, color="#7E8B90", linewidth=.75, linestyle="--", zorder=1)
    ax.set_yticks(range(4), list(reversed(labels)))
    ax.set_xlim(-.28, 1.06)
    ax.set_ylim(-.47, 3.47)
    ax.set_xlabel("Within-city Spearman ρ", labelpad=5, color=INK)
    ax.set_title("(b)  Target difficulty / source ranking", loc="left", pad=9,
                 fontweight="bold", color=INK)
    ax.grid(axis="x", color="#E7ECEE", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.7, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#86939A")
    ax.text(.985, .975, "Five city values; dark bar = median",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            color="#52616A", bbox={"facecolor": "white", "edgecolor": "none",
                                   "alpha": .85, "pad": 2.0})
    fig.subplots_adjust(left=.105, right=.985, top=.87, bottom=.19, wspace=.45)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Figure6 v2: five frozen LOCO subsets for both outcomes and 20 city-specific ranks verified")


if __name__ == "__main__":
    main()

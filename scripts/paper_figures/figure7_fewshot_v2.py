"""Stage14C-P3: categorical-budget rendering of frozen few-shot evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure7_source.csv"
FROZEN = ROOT / "outputs/tables/fewshot_summary.csv"
OUT = ROOT / "outputs/paper/figures/Figure7_fewshot_v2"
PERCENTS = (0, 1, 5, 10, 25)
INK = "#26343D"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read_rows(SOURCE)
    frozen = read_rows(FROZEN)
    recovery = sorted((r for r in source if r["point_type"] == "recovery"),
                      key=lambda r: int(r["target_label_percent"]))
    ceiling = [r for r in source if r["point_type"] == "full-target ceiling"]
    assert [int(r["target_label_percent"]) for r in recovery] == list(PERCENTS)
    assert len(ceiling) == 1 and len(source) == 6
    for row in source:
        assert (row["direction"], row["model"], row["feature_set"]) == (
            "Shenzhen -> Kunming", "XGBoost", "AEF-192")
        match = [f for f in frozen if
                 f["direction"] == row["direction"] and
                 f["model"] == row["model"] and
                 f["feature_set"] == row["feature_set"] and
                 f["strategy"] == row["strategy"] and
                 f["target_label_percent"] == row["target_label_percent"]]
        assert len(match) == 1
        for field in ("macro_f1_mean", "macro_f1_std",
                      "expansion_f1_mean", "expansion_f1_std"):
            assert abs(float(row[field]) - float(match[0][field])) < 1e-10
    for pct, macro, expansion in ((0, .8922977732, .8646454491),
                                  (5, .9412467966, .9261665537),
                                  (25, .9623898967, .9539356288)):
        row = next(r for r in recovery if int(r["target_label_percent"]) == pct)
        assert abs(float(row["macro_f1_mean"]) - macro) < 1e-10
        assert abs(float(row["expansion_f1_mean"]) - expansion) < 1e-10

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.8,
        "axes.titlesize": 9.3, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.45))
    fig.patch.set_facecolor("white")
    panels = (
        ("(a)  Macro-F1", "macro_f1_mean", "macro_f1_std", "#276D84", "o", .86),
        ("(b)  Expansion-F1", "expansion_f1_mean", "expansion_f1_std", "#AF6B2D", "D", .82),
    )
    for ax, (title, mean_key, std_key, color, marker, ymin) in zip(axes, panels):
        x = list(range(len(recovery)))  # categorical positions, not metric spacing
        y = [float(r[mean_key]) for r in recovery]
        sd = [float(r[std_key]) for r in recovery]
        ax.errorbar(x, y, yerr=sd, fmt=marker + "-", color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.25, markersize=5.3, linewidth=1.25,
                    ecolor=color, elinewidth=.9, capsize=2.2, zorder=4)
        full = float(ceiling[0][mean_key])
        ax.axhline(full, color="#626E73", linestyle=(0, (4, 2.5)),
                   linewidth=.9, zorder=2)
        ax.text(4.15, full + .0012, "Full-target", color="#52616A",
                fontsize=7.4, ha="right", va="bottom")
        ax.set_xlim(-.24, 4.25)
        ax.set_ylim(ymin, .985)
        ax.set_xticks(x, [str(p) for p in PERCENTS])
        ax.set_xlabel("Nominal target-label fraction (%)", labelpad=5, color=INK)
        ax.set_ylabel(title.split("  ")[1], labelpad=5, color=INK)
        ax.set_title(title, loc="left", pad=8, fontweight="bold", color=INK)
        ax.grid(axis="y", color="#E7ECEE", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.7, colors=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#86939A")
    fig.subplots_adjust(left=.09, right=.985, top=.86, bottom=.20, wspace=.30)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Figure7 v2: five categorical budgets and full-target ceiling verified")


if __name__ == "__main__":
    main()

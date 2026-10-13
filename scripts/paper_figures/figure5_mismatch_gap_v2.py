"""Stage14C-P2: re-render frozen support-distance/gap associations."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure5_source.csv"
PAIR = ROOT / "outputs/tables/stage07_pair_mechanism.csv"
QAP = ROOT / "outputs/tables/stage07b_qap_results.csv"
OUT = ROOT / "outputs/paper/figures/Figure5_mismatch_gap_v2"
BLUE = "#276D84"
INK = "#26343D"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = read_rows(SOURCE)
    pairs = {(r["source_city"], r["target_city"]): r for r in read_rows(PAIR)}
    qap = {r["test"]: r for r in read_rows(QAP)}
    assert len(source) == 40 and len(pairs) == 20
    panel_spec = [
        ("Macro mismatch-gap", "support_distance_mean", "support_distance_seed_std",
         "macro_f1_gap_mean", "macro_f1_gap_std", "Overall distance vs Macro gap",
         "Macro-F1 gap", "Same-class support distance (cosine)", (0.14, 0.53), (-0.012, 0.21)),
        ("Class2 mismatch-expansion gap", "class2_support_distance_mean",
         "class2_support_distance_seed_std", "expansion_f1_gap_mean",
         "expansion_f1_gap_std", "Class2 distance vs Expansion gap",
         "Expansion-F1 gap", "Class 2 support distance (cosine)", (0.14, 0.57), (-0.018, 0.33)),
    ]
    for panel, xkey, xsdkey, ykey, ysdkey, qkey, _, _, _, _ in panel_spec:
        data = [r for r in source if r["panel"] == panel]
        assert len(data) == 20
        for row in data:
            frozen = pairs[row["source_city"], row["target_city"]]
            for src_key, frozen_key in [("x", xkey), ("x_std", xsdkey),
                                        ("y", ykey), ("y_std", ysdkey)]:
                assert abs(float(row[src_key]) - float(frozen[frozen_key])) < 1e-10
            assert abs(float(row["spearman_rho"]) - float(qap[qkey]["observed_statistic"])) < 5e-7
            assert abs(float(row["qap_one_sided_p"]) - float(qap[qkey]["exact_qap_one_sided_p"])) < 1e-10

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.8,
        "axes.titlesize": 9.3, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.70))
    fig.patch.set_facecolor("white")
    offsets = {
        "Macro mismatch-gap": {"WH>KM": (7, -12), "SZ>KM": (8, 8), "KM>SZ": (7, 8)},
        "Class2 mismatch-expansion gap": {"WH>KM": (7, -9), "SZ>KM": (8, -11), "KM>SZ": (25, 13)},
    }
    for idx, (spec, ax) in enumerate(zip(panel_spec, axes)):
        panel, _, _, _, _, _, outcome, xlabel, xlim, ylim = spec
        data = [r for r in source if r["panel"] == panel]
        x = [float(r["x"]) for r in data]
        y = [float(r["y"]) for r in data]
        xerr = [float(r["x_std"]) for r in data]
        yerr = [float(r["y_std"]) for r in data]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=4.7,
                    markeredgecolor="#1C5265", markeredgewidth=0.55,
                    color=BLUE, ecolor="#8C989E", elinewidth=0.85,
                    capsize=1.6, capthick=0.85, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, labelpad=5, color=INK)
        ax.set_ylabel(f"Target-normalized {outcome}", labelpad=5, color=INK)
        ax.set_title(f"({'ab'[idx]})  {outcome}", loc="left", pad=8,
                     fontweight="bold", color=INK)
        ax.grid(color="#E7ECEE", linewidth=0.58)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#86939A")
        ax.tick_params(length=2.5, width=0.7, colors=INK)
        first = data[0]
        ax.text(0.035, 0.975,
                f"Spearman ρ = {float(first['spearman_rho']):.3f}\n"
                f"Exact QAP p = {float(first['qap_one_sided_p']):.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.0,
                color=INK, linespacing=1.25,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.0})
        for row in data:
            label = row["pair_label"]
            if label not in offsets[panel]:
                continue
            dx, dy = offsets[panel][label]
            ax.annotate(label.replace(">", "→"),
                        xy=(float(row["x"]), float(row["y"])),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.3, color=INK,
                        arrowprops={"arrowstyle": "-", "color": "#6D7A81", "lw": 0.55,
                                    "shrinkA": 1.0, "shrinkB": 3.5}, zorder=4)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.21, wspace=0.30)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure 5 v2: 40 frozen pair-panel rows checked; no fitted trend line")


if __name__ == "__main__":
    main()

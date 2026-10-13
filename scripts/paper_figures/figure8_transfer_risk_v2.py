"""Stage14C-P1 publication rendering of frozen Figure 8 values.

No risk model is fitted here. The Stage11 figure source and frozen Table 4
are read and checked before plotting; printed metrics come from Table 4.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper/source_data/Figure8_source.csv"
TABLE4 = ROOT / "outputs/paper/tables/Table4_label_free_transfer_risk.csv"
OUT = ROOT / "outputs/paper/figures/Figure8_transfer_risk_v2"
DEV = "#276D84"
EXT = "#A86228"
INK = "#26343D"
GRAY = "#69777E"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics() -> dict[str, dict[str, tuple[float, float]]]:
    rows = read_rows(TABLE4)
    assert len(rows) == 4, "Frozen Table 4 must contain four risk-role rows"
    result: dict[str, dict[str, tuple[float, float]]] = {"Development": {}, "External": {}}
    for row in rows:
        name = row["Risk model and role"]
        role = "Development" if "Development" in name else "External"
        risk = "R1" if name.startswith("R1") else "R4"
        result[role][risk] = (float(row["Spearman"]), float(row["MAE"]))
    assert abs(result["External"]["R4"][0] - 0.7939393939) < 1e-10
    assert abs(result["External"]["R4"][1] - 0.0137564222) < 1e-10
    return result


def main() -> None:
    rows = read_rows(SOURCE)
    assert len(rows) == 30
    groups = {
        "Development OOF": [r for r in rows if r["evaluation_role"] == "Development OOF"],
        "External frozen validation": [
            r for r in rows if r["evaluation_role"] == "External frozen validation"
        ],
    }
    assert [len(v) for v in groups.values()] == [20, 10]
    assert all(r["R4_status"].startswith("POST-HOC DEVELOPMENT") for r in rows)
    stats = metrics()

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))
    fig.patch.set_facecolor("white")
    choices = [
        (axes[0], groups["Development OOF"], "Development", DEV, "o", 0.18, 0.03),
        (axes[1], groups["External frozen validation"], "External", EXT, "D", 0.09, 0.015),
    ]
    for idx, (ax, data, role, color, marker, upper, step) in enumerate(choices):
        ax.plot([0, upper], [0, upper], color="#8C989E", linewidth=0.8,
                linestyle=(0, (4, 3)), zorder=0)
        x = [float(r["actual_macro_gap"]) for r in data]
        for risk, field, size, zorder in [
            ("R1", "predicted_macro_gap_R1", 32, 2),
            ("R4", "predicted_macro_gap_R4", 29, 3),
        ]:
            y = [float(r[field]) for r in data]
            ax.scatter(
                x, y, s=size, marker=marker, linewidths=0.9,
                edgecolors=color, facecolors="white" if risk == "R1" else color,
                zorder=zorder,
            )
        ticks = [round(i * step, 3) for i in range(round(upper / step) + 1)]
        ax.set(xlim=(-0.003, upper + 0.003), ylim=(-0.003, upper + 0.003))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E9EDEF", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#86939A")
        ax.tick_params(colors=INK, length=2.4, width=0.65)
        ax.set_title(
            f"({'ab'[idx]})  {role}  ·  n = {len(data)} directed pairs",
            loc="left", color=INK, fontweight="bold", pad=8,
        )
        r1_rho, r1_mae = stats[role]["R1"]
        r4_rho, r4_mae = stats[role]["R4"]
        annotation = (
            f"R1   ρ = {r1_rho:.3f}   MAE = {r1_mae:.3f}\n"
            f"R4*  ρ = {r4_rho:.3f}   MAE = {r4_mae:.3f}"
        )
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top",
                ha="left", fontsize=7.7, linespacing=1.3, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0))

    fig.supxlabel("Observed target-normalized Macro-F1 gap", y=0.20,
                  fontsize=9, color=INK)
    fig.supylabel("Predicted Macro-F1 gap", x=0.035, fontsize=9, color=INK)
    handles = [
        Line2D([], [], marker="o", markersize=5, linestyle="None",
               markerfacecolor="white", markeredgecolor=DEV, label="Development"),
        Line2D([], [], marker="D", markersize=5, linestyle="None",
               markerfacecolor="white", markeredgecolor=EXT, label="External"),
        Line2D([], [], marker="o", markersize=5, linestyle="None",
               markerfacecolor="white", markeredgecolor=GRAY,
               label="R1 · predefined PCD-5NN"),
        Line2D([], [], marker="o", markersize=5, linestyle="None",
               markerfacecolor=GRAY, markeredgecolor=GRAY,
               label="R4* · predictive entropy"),
        Line2D([], [], color="#8C989E", linewidth=0.8,
               linestyle=(0, (4, 3)), label="y = x"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.005), columnspacing=1.25,
               handletextpad=0.45, fontsize=7.7)
    fig.text(0.52, 0.145, "Different axis ranges: (a) 0–0.18; (b) 0–0.09",
             ha="center", va="center", color=GRAY, fontsize=7.5)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.84, bottom=0.30, wspace=0.24)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 8 v2 rendered from {len(rows)} frozen pair rows; metrics read from frozen Table 4")


if __name__ == "__main__":
    main()

"""Journal-style Figure 1: frozen AEF cross-city transfer framework.

Conceptual artwork only: no outcome tables, model training, or measured results.
The accepted four-panel scientific structure from v2 is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper" / "figures"
STEM = "Figure1_framework_v3"
# Artwork is drawn at 2x the intended 7.6-inch journal width.
DESIGN_IN = (15.2, 7.5)
FINAL_WIDTH_IN = 7.6

INK = "#26343D"
MUTED = "#59666E"
RULE = "#D6DCDF"
DEV = "#276D84"
DEV_PALE = "#E8F2F4"
EXT = "#A86228"
EXT_PALE = "#FAEFE4"
CLASS_0 = "#93A68A"
CLASS_1 = "#668BA0"
CLASS_2 = "#BD8C62"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "path",
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
})


def txt(ax, x, y, label, *, size=14.5, color=INK, weight="normal",
        ha="left", va="center", rotation=0):
    return ax.text(x, y, label, fontsize=size, color=color,
                   fontweight=weight, ha=ha, va=va, rotation=rotation,
                   linespacing=1.08)


def rule(ax, x1, y1, x2, y2, *, color=RULE, width=0.9, style="solid", z=1):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
            linestyle=style, solid_capstyle="round", zorder=z)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, width=1.1, head=9, z=2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head,
        linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=z))


def raster(ax, cx, cy, palette):
    """Compact vector glyph, not observational imagery."""
    s = 8
    x0, y0 = cx - 1.5 * s, cy - 1.5 * s
    for row in range(3):
        for col in range(3):
            ax.add_patch(Rectangle(
                (x0 + col * s, y0 + row * s), s - 0.6, s - 0.6,
                facecolor=palette[(row * 3 + col) % len(palette)],
                edgecolor="white", linewidth=0.6))
    ax.add_patch(Rectangle((x0 - 1, y0 - 1), 3 * s + 1, 3 * s + 1,
                           fill=False, edgecolor=MUTED, linewidth=0.8))


def annual_vector(ax, cx, cy, seed):
    heights = [(11, 22, 16, 28, 19), (17, 12, 26, 20, 15),
               (22, 17, 13, 27, 21)][seed]
    for j, height in enumerate(heights):
        ax.add_patch(Rectangle((cx - 28 + j * 11, cy - 13), 7, height,
                               facecolor=DEV if j % 2 == 0 else "#8AB1BF",
                               edgecolor="none"))
    rule(ax, cx - 31, cy - 15, cx + 28, cy - 15,
         color=MUTED, width=0.8)


def city(ax, cx, cy, label, *, external=False):
    if external:
        r = 17
        ax.add_patch(Polygon(
            [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
            closed=True, facecolor=EXT_PALE, edgecolor=EXT, linewidth=1.55,
            linestyle=(0, (4, 2)), joinstyle="miter", zorder=3))
    else:
        ax.add_patch(Circle((cx, cy), 17, facecolor="white", edgecolor=DEV,
                            linewidth=1.55, linestyle="solid", zorder=3))
    txt(ax, cx, cy - 31, label, size=14.5, ha="center")


def panel_header(ax, x, letter, title):
    # v2: letter 19 pt / title 18 pt; v3: 17 pt / 16 pt.
    txt(ax, x, 720, letter, size=17, weight="bold", color=DEV)
    txt(ax, x + 46, 720, title, size=16, weight="bold")


def benchmark(ax):
    panel_header(ax, 45, "(a)", "Benchmark")
    raster(ax, 75, 632, [CLASS_0, CLASS_1, "#C5D5C3"])
    raster(ax, 181, 632, [CLASS_0, CLASS_1, CLASS_2])
    raster(ax, 294, 632, ["#D2D8D7", CLASS_1, "#E6E8E5"])
    txt(ax, 75, 593, "Dynamic\nWorld", size=14, ha="center", color=MUTED)
    txt(ax, 181, 593, "ESRI land\ncover", size=14, ha="center", color=MUTED)
    txt(ax, 294, 593, "GHSL\ncontext", size=14, ha="center", color=MUTED)

    # Only Dynamic World + ESRI define consensus labels; GHSL joins sampling.
    rule(ax, 75, 566, 75, 558, color=MUTED)
    rule(ax, 181, 566, 181, 558, color=MUTED)
    rule(ax, 75, 558, 181, 558, color=MUTED)
    arrow(ax, 128, 558, 128, 542)
    rule(ax, 294, 566, 294, 382, color=MUTED)
    txt(ax, 58, 525, "Consensus labels", size=15, weight="bold")
    for y, color, label in [
        (490, CLASS_0, "Stable Non-built"),
        (455, CLASS_1, "Stable Built"),
        (420, CLASS_2, "Strict Expansion"),
    ]:
        ax.add_patch(Rectangle((63, y - 8), 17, 17, facecolor=color,
                               edgecolor=MUTED, linewidth=0.7))
        txt(ax, 90, y, label, size=14.5)
    rule(ax, 151, 409, 151, 382, color=MUTED)
    rule(ax, 151, 382, 294, 382, color=MUTED)
    arrow(ax, 217, 382, 217, 363)
    txt(ax, 217, 340, "Sample locations", size=15, ha="center",
        weight="bold")
    arrow(ax, 217, 320, 217, 301)
    txt(ax, 195, 280, "Annual AEF", size=15, weight="bold", ha="center")
    for idx, (x, year) in enumerate([(88, "2020"), (186, "2022"),
                                     (284, "2024")]):
        annual_vector(ax, x, 235, idx)
        txt(ax, x, 191, year, size=14.5, ha="center")
    rule(ax, 88, 171, 284, 171, color=DEV, width=1.1)
    arrow(ax, 186, 171, 186, 147, color=DEV)
    txt(ax, 186, 122, "AEF-192", size=17, ha="center", weight="bold",
        color=DEV)


def directed_transfer(ax):
    panel_header(ax, 377, "(b)", "Directed transfer")
    for x, y, label in [
        (398, 628, "Shenzhen"), (545, 628, "Kunming"),
        (398, 581, "Wuhan"), (545, 581, "Xi'an"),
        (398, 534, "Beijing"),
    ]:
        ax.add_patch(Circle((x, y), 11, facecolor="white", edgecolor=DEV,
                            linewidth=1.4, linestyle="solid"))
        txt(ax, x + 21, y, label, size=14)
    txt(ax, 530, 500, "Target", size=15, ha="center", weight="bold")
    txt(ax, 380, 360, "Source", size=14.5, ha="center", rotation=90)
    abbreviations = ["SZ", "KM", "WH", "XA", "BJ"]
    x0, y0, step = 438, 270, 38
    for j, code in enumerate(abbreviations):
        txt(ax, x0 + step * j + 17, 476, code, size=14, ha="center")
    for i, code in enumerate(abbreviations):
        y = y0 + step * (4 - i)
        txt(ax, 418, y + 17, code, size=14, ha="center")
        for j in range(5):
            x = x0 + step * j
            diagonal = i == j
            ax.add_patch(Rectangle(
                (x, y), 34, 34,
                facecolor="#DDE4E6" if diagonal else (
                    DEV_PALE if (i + j) % 2 else "white"),
                edgecolor="#87949C", linewidth=0.8))
            if diagonal:
                ax.add_patch(Circle((x + 17, y + 17), 3.5, color=MUTED))
            else:
                arrow(ax, x + 9, y + 17, x + 25, y + 17,
                      color=DEV, width=0.9, head=5)
    txt(ax, 452, 245, "Diagonal: same-city", size=14, color=MUTED)
    txt(ax, 452, 215, "Off-diagonal: transfer", size=14, color=MUTED)
    rule(ax, 527, 197, 527, 185, color=MUTED)
    arrow(ax, 527, 185, 527, 173)
    txt(ax, 527, 154, "Target-normalized gap", size=15,
        weight="bold", ha="center")
    txt(ax, 527, 125, "same-city − transfer", size=14, ha="center",
        color=MUTED)


def diagnosis_and_mitigation(ax):
    panel_header(ax, 704, "(c)", "Diagnosis / risk / mitigation")
    txt(ax, 711, 655, "Representation diagnosis", size=15.5,
        weight="bold")
    for x, y in [(740, 589), (775, 611), (783, 567), (818, 592),
                 (850, 610)]:
        ax.add_patch(Circle((x, y), 5.5, facecolor=DEV, edgecolor=DEV,
                            linewidth=0.8))
    for x, y in [(819, 626), (872, 571)]:
        ax.add_patch(Circle((x, y), 6.5, facecolor="white", edgecolor=INK,
                            linewidth=1.3))
    rule(ax, 819, 626, 850, 610, color=MUTED, width=0.9,
         style=(0, (3, 2)))
    rule(ax, 872, 571, 818, 592, color=MUTED, width=0.9,
         style=(0, (3, 2)))
    arrow(ax, 881, 589, 982, 589)
    txt(ax, 991, 612, "Support", size=14.5, weight="bold")
    txt(ax, 991, 587, "mismatch", size=14.5, weight="bold")
    txt(ax, 991, 550, "Gap association", size=14)
    txt(ax, 711, 544, "True-class diagnostic", size=14, color=MUTED)
    rule(ax, 706, 528, 1126, 528)

    txt(ax, 711, 501, "Transfer-risk estimation", size=15.5,
        weight="bold")
    for x, y in [(735, 438), (750, 452), (758, 425)]:
        ax.add_patch(Circle((x, y), 5, facecolor="white", edgecolor=INK,
                            linewidth=1.0))
    txt(ax, 745, 390, "Target vectors", size=14, ha="center")
    arrow(ax, 770, 440, 793, 440)
    ax.add_patch(Rectangle((793, 417), 91, 46, facecolor="white",
                           edgecolor=INK, linewidth=1.0))
    txt(ax, 838, 441, "Source\nclassifier", size=14, ha="center")
    arrow(ax, 884, 440, 911, 440)
    for j, height in enumerate((22, 37, 15)):
        ax.add_patch(Rectangle((916 + j * 14, 424), 9, height,
                               facecolor=DEV if j == 1 else "#9FB9C1",
                               edgecolor="none"))
    txt(ax, 935, 390, "Probabilities", size=14, ha="center")
    arrow(ax, 961, 440, 984, 440)
    txt(ax, 1053, 451, "Predictive", size=14.5, weight="bold", ha="center")
    txt(ax, 1053, 425, "uncertainty*", size=14.5, weight="bold", ha="center")
    arrow(ax, 1053, 410, 1053, 392)
    txt(ax, 1053, 371, "Risk estimate", size=14.5, ha="center")
    txt(ax, 826, 355, "No target labels at inference", size=14,
        color=MUTED, ha="center")
    rule(ax, 706, 337, 1126, 337)

    txt(ax, 711, 311, "Pooled-source mitigation", size=15.5,
        weight="bold")
    for x in (746, 781, 816):
        ax.add_patch(Circle((x, 245), 13, facecolor="white", edgecolor=DEV,
                            linewidth=1.4))
        rule(ax, x, 232, x, 218, color=DEV, width=0.9)
    rule(ax, 746, 218, 816, 218, color=DEV, width=0.9)
    txt(ax, 781, 189, "Labeled sources", size=14, ha="center")
    arrow(ax, 829, 245, 875, 245, color=DEV)
    ax.add_patch(Rectangle((875, 217), 112, 56, facecolor="white",
                           edgecolor=DEV, linewidth=1.1))
    txt(ax, 931, 246, "Pooled\ntraining", size=14.5, ha="center",
        weight="bold")
    arrow(ax, 987, 245, 1005, 245, color=DEV)
    ax.add_patch(Rectangle((1005, 221), 120, 48, facecolor="white",
                           edgecolor=INK, linewidth=1.0))
    txt(ax, 1065, 245, "Target\nprediction", size=14, ha="center")


def external_evaluation(ax):
    panel_header(ax, 1170, "(d)", "Frozen external test")
    city(ax, 1240, 626, "Chengdu", external=True)
    city(ax, 1394, 626, "Hangzhou", external=True)
    rule(ax, 1240, 577, 1240, 558, color=EXT)
    rule(ax, 1394, 577, 1394, 558, color=EXT)
    rule(ax, 1240, 558, 1394, 558, color=EXT)
    arrow(ax, 1317, 558, 1317, 532, color=EXT)
    # Three scientific stages, in temporal order, without audit-log phrasing.
    items = [
        (487, "Risk model +\npredictions fixed"),
        (352, "Outcome reveal"),
        (217, "External evaluation"),
    ]
    for idx, (y, label) in enumerate(items):
        ax.add_patch(Circle((1209, y), 7, facecolor=EXT_PALE,
                            edgecolor=EXT, linewidth=1.35))
        txt(ax, 1233, y, label, size=15, weight="bold" if idx == 0
            else "normal")
        if idx < len(items) - 1:
            arrow(ax, 1209, y - 11, 1209, items[idx + 1][0] + 12,
                  color=EXT)


def legend(ax):
    rule(ax, 45, 96, 1480, 96, color=RULE, width=0.8)
    ax.add_patch(Circle((264, 59), 10, facecolor="white", edgecolor=DEV,
                        linewidth=1.5))
    txt(ax, 285, 59, "Development city", size=14.5)
    ax.add_patch(Polygon([(512, 70), (523, 59), (512, 48), (501, 59)],
                         closed=True, facecolor=EXT_PALE, edgecolor=EXT,
                         linewidth=1.5, linestyle=(0, (4, 2))))
    txt(ax, 534, 59, "External target", size=14.5)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=DESIGN_IN, dpi=300, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1520)
    ax.set_ylim(20, 770)
    ax.set_aspect("equal")
    ax.axis("off")
    for x in (356, 684, 1148):
        rule(ax, x, 108, x, 668, color="#E0E5E7", width=0.8, z=0)
    for x1, x2 in ((45, 344), (369, 672), (696, 1136), (1160, 1480)):
        rule(ax, x1, 686, x2, 686, color=DEV if x1 < 1140 else EXT,
             width=1.3)
    for x1, x2 in ((344, 369), (672, 696), (1136, 1160)):
        arrow(ax, x1, 686, x2, 686, width=1.0, head=8)
    benchmark(ax)
    directed_transfer(ax)
    diagnosis_and_mitigation(ax)
    external_evaluation(ax)
    legend(ax)
    fig.savefig(OUT / f"{STEM}.svg", format="svg", facecolor="white")
    fig.savefig(OUT / f"{STEM}.pdf", format="pdf", facecolor="white")
    fig.savefig(OUT / f"{STEM}.png", format="png", dpi=300,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()

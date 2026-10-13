"""Final Figure 2: seven-city locator and selected benchmark spatial windows.

This is a contextual illustration of existing labels/coordinates. Window selection
uses class counts and location alone, never prediction or performance information.
Run from any working directory with the project's Python environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
SOURCE_POINTS = ROOT / "outputs/paper/source_data/Figure2_study_area_v2_source.csv"
BOUNDARY_ARCHIVE = ROOT / "data/reference/ne_10m_admin_1_states_provinces_v5.1.1.zip"
BOUNDARY_SHA256 = "EFC59726337323058F9446210ADC96673179CD344E053666EE3D28CB58BA2B05"
OUTPUT_DIR = ROOT / "outputs/figures/final"
WINDOW_CSV = ROOT / "outputs/manuscript/figure2_spatial_window_selection.csv"
CITY_FILES = {
    "Shenzhen": PROCESSED / "Shenzhen_AEF_195D.parquet",
    "Kunming": PROCESSED / "Kunming_AEF_195D.parquet",
    "Wuhan": PROCESSED / "Wuhan_AEF_195D.parquet",
    "Xian": PROCESSED / "Xian_AEF_195D.parquet",
    "Beijing": PROCESSED / "Beijing_AEF_195D.parquet",
    "Chengdu": PROCESSED / "external/Chengdu_External_AEF_195D.parquet",
    "Hangzhou": PROCESSED / "external/Hangzhou_External_AEF_195D.parquet",
}
LOCAL_CRS = {"Shenzhen": "EPSG:32650", "Kunming": "EPSG:32648", "Chengdu": "EPSG:32648"}
LOCAL_CITIES = ("Shenzhen", "Kunming", "Chengdu")
CITY_ORDER = tuple(CITY_FILES)
GRID_M = 2000
WINDOW_M = 6000
RULE = "2-km-aligned 6x6-km cells; require all 3 classes; maximize minimum class count, then total count; tie: lowest xmin then ymin"

INK = "#26343D"
MUTED = "#59666E"
DEV = "#276D84"
EXT = "#A86228"
BOUNDARY = "#B3BDC2"
GRID = "#E4E9EB"
CLASS_COLORS = {0: "#93A68A", 1: "#668BA0", 2: "#BD8C62"}  # Figure 1 palette
CLASS_MARKERS = {0: "X", 1: "s", 2: "^"}
CLASS_NAMES = {0: "Stable Non-built", 1: "Stable Built", 2: "Strict Expansion"}
VIEW = (102.00, 120.90, 22.20, 41.20)
LABEL_OFFSETS_PT = {
    "Shenzhen": (10, 9), "Kunming": (9, -10), "Wuhan": (10, 8),
    "Xian": (-28, 9), "Beijing": (-43, -7),
    "Chengdu": (8, -10), "Hangzhou": (-53, -10),
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
})


def load_city(city: str) -> pd.DataFrame:
    cols = ["city", "class", "longitude", "latitude", "utm_x", "utm_y", "block_id"]
    data = pd.read_parquet(CITY_FILES[city], columns=cols)
    if len(data) != 18000 or set(data["class"]) != {0, 1, 2}:
        raise ValueError(f"Unexpected sample/class inventory: {city}")
    if data[cols].isna().any().any() or set(data["city"]) != {city}:
        raise ValueError(f"Invalid city, coordinates, or labels: {city}")
    if data["class"].value_counts().to_dict() != {0: 6000, 1: 6000, 2: 6000}:
        raise ValueError(f"Class balance changed: {city}")
    return data


def load_map_sources() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    digest = hashlib.sha256(BOUNDARY_ARCHIVE.read_bytes()).hexdigest().upper()
    if digest != BOUNDARY_SHA256:
        raise ValueError("Natural Earth ADM1 archive hash changed")
    with zipfile.ZipFile(BOUNDARY_ARCHIVE) as z:
        version = z.read("ne_10m_admin_1_states_provinces.VERSION.txt").decode().strip()
    if version != "5.1.1":
        raise ValueError("Natural Earth ADM1 version changed")
    boundaries = gpd.read_file("zip://" + str(BOUNDARY_ARCHIVE))
    if boundaries.crs is None or boundaries.crs.to_epsg() != 4326:
        raise ValueError("Natural Earth ADM1 CRS changed")
    china = boundaries[boundaries["adm0_a3"].eq("CHN")]
    if len(china) != 32 or china.geometry.is_empty.any():
        raise ValueError("Natural Earth China ADM1 geometry changed")
    points = pd.read_csv(SOURCE_POINTS)
    if tuple(points["City"]) != CITY_ORDER:
        raise ValueError("Frozen Figure 2 city order changed")
    for row in points.itertuples(index=False):
        data = load_city(row.City)
        if not np.isclose(data.longitude.median(), row.Longitude, atol=1e-9):
            raise ValueError(f"Figure 2 longitude median changed: {row.City}")
        if not np.isclose(data.latitude.median(), row.Latitude, atol=1e-9):
            raise ValueError(f"Figure 2 latitude median changed: {row.City}")
    return china, points


def choose_window(data: pd.DataFrame) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    """Search all occupied 2-km-grid 3x3-cell windows without looking at models.

    Score each eligible candidate lexicographically by its least represented
    class, then its total point count. Resolve exact ties by smallest origin.
    """
    bx = np.floor(data.utm_x.to_numpy() / GRID_M).astype(int)
    by = np.floor(data.utm_y.to_numpy() / GRID_M).astype(int)
    cls = data["class"].to_numpy(dtype=int)
    cell = Counter(zip(bx, by, cls))
    occupied = {(int(x), int(y)) for x, y in zip(bx, by)}
    candidates = {(x - dx, y - dy) for x, y in occupied
                  for dx in range(3) for dy in range(3)}
    best = None
    for left, bottom in sorted(candidates):
        counts = {k: sum(cell[(left + dx, bottom + dy, k)]
                         for dx in range(3) for dy in range(3)) for k in (0, 1, 2)}
        if min(counts.values()) == 0:
            continue
        score = (min(counts.values()), sum(counts.values()), -left, -bottom)
        if best is None or score > best[0]:
            best = (score, left, bottom, counts)
    if best is None:
        raise ValueError("No common-scale window contains all classes")
    _, left, bottom, counts = best
    xmin, ymin = left * GRID_M, bottom * GRID_M
    xmax, ymax = xmin + WINDOW_M, ymin + WINDOW_M
    mask = data.utm_x.ge(xmin) & data.utm_x.lt(xmax) & data.utm_y.ge(ymin) & data.utm_y.lt(ymax)
    independent_counts = data.loc[mask, "class"].value_counts().to_dict()
    if counts != {k: independent_counts.get(k, 0) for k in (0, 1, 2)}:
        raise AssertionError("Window count differs from direct point filter")
    return (xmin, ymin, xmax, ymax), counts


def draw_locator(ax, china: gpd.GeoDataFrame, points: pd.DataFrame) -> None:
    xmin, xmax, ymin, ymax = VIEW
    china.cx[xmin:xmax, ymin:ymax].boundary.plot(ax=ax, color=BOUNDARY,
                                                  linewidth=0.54, zorder=1)
    for row in points.itertuples(index=False):
        is_ext = row.Role == "External validation"
        color = EXT if is_ext else DEV
        ax.scatter(row.Longitude, row.Latitude, s=43, marker="D" if is_ext else "o",
                   facecolor="#FAEFE4" if is_ext else "white", edgecolor=color,
                   linewidth=1.12, zorder=5)
        off = LABEL_OFFSETS_PT[row.City]
        ax.annotate("Xi'an" if row.City == "Xian" else row.City,
                    xy=(row.Longitude, row.Latitude), xytext=off,
                    textcoords="offset points", ha="left",
                    va="bottom" if off[1] > 0 else "top", color=color,
                    fontsize=8.0, fontweight="medium",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": .35},
                    arrowprops={"arrowstyle": "-", "color": color,
                                "linestyle": "--" if is_ext else "-", "linewidth": .64,
                                "shrinkA": 2, "shrinkB": 3}, zorder=6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([104, 108, 112, 116, 120])
    ax.set_yticks([24, 28, 32, 36, 40])
    ax.set_xticklabels([f"{v}\N{DEGREE SIGN}E" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v}\N{DEGREE SIGN}N" for v in ax.get_yticks()])
    ax.set_xlabel("Longitude", fontsize=8.3, color=INK)
    ax.set_ylabel("Latitude", fontsize=8.3, color=INK)
    ax.grid(color=GRID, linewidth=.45, zorder=0)
    handles = [
        Line2D([], [], color=DEV, marker="o", markerfacecolor="white",
               linestyle="-", linewidth=.8, markersize=5, label="Development (5)"),
        Line2D([], [], color=EXT, marker="D", markerfacecolor="#FAEFE4",
               linestyle="--", linewidth=.8, markersize=5, label="External (2)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", fontsize=8.0, ncol=1, borderpad=.24,
              handlelength=1.2, labelspacing=.2)


def draw_local(ax, city: str, data: pd.DataFrame,
               bounds: tuple[int, int, int, int], counts: dict[int, int]) -> None:
    xmin, ymin, xmax, ymax = bounds
    selected = data.loc[data.utm_x.ge(xmin) & data.utm_x.lt(xmax) &
                        data.utm_y.ge(ymin) & data.utm_y.lt(ymax)]
    for k in (0, 1, 2):
        part = selected[selected["class"].eq(k)]
        ax.scatter((part.utm_x.to_numpy() - xmin) / 1000,
                   (part.utm_y.to_numpy() - ymin) / 1000,
                   s=14 if k == 0 else 10, marker=CLASS_MARKERS[k], color=CLASS_COLORS[k],
                   edgecolors=INK, linewidths=.15, alpha=.72, rasterized=False,
                   zorder=k+2)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("Easting (km)", fontsize=8.2, color=INK)
    ax.set_ylabel("Northing (km)", fontsize=8.2, color=INK)
    ax.grid(color=GRID, linewidth=.55, zorder=0)
    ax.annotate("N", xy=(5.58, 5.75), xytext=(5.58, 5.0),
                ha="center", va="bottom", fontsize=8.3, color=INK,
                arrowprops={"arrowstyle": "-|>", "lw": .85, "color": INK},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": .25},
                zorder=9)
    # A projected metric scale, with white backing to remain legible over points.
    ax.plot([.38, 2.38], [.36, .36], color="white", linewidth=5, zorder=9)
    ax.plot([.38, 2.38], [.36, .36], color=INK, linewidth=1.3, zorder=10)
    ax.plot([.38, .38], [.25, .47], color=INK, linewidth=1, zorder=10)
    ax.plot([2.38, 2.38], [.25, .47], color=INK, linewidth=1, zorder=10)
    ax.text(1.38, .50, "2 km", ha="center", va="bottom", fontsize=8.0,
            color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .15}, zorder=10)
    role = "external target" if city == "Chengdu" else "development"
    ax.text(.03, .98, f"{role} · {LOCAL_CRS[city]}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.0, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .91, "pad": 1.5}, zorder=10)
    assert len(selected) == sum(counts.values())


def main() -> None:
    china, points = load_map_sources()
    local_data = {city: load_city(city) for city in LOCAL_CITIES}
    windows = {city: choose_window(local_data[city]) for city in LOCAL_CITIES}
    WINDOW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with WINDOW_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["city", "xmin", "ymin", "xmax", "ymax", "window_size",
                         "n_stable_nonbuilt", "n_stable_built", "n_strict_expansion", "selection_rule"])
        for city in LOCAL_CITIES:
            bounds, counts = windows[city]
            writer.writerow([city, *bounds, "6 km × 6 km", counts[0], counts[1], counts[2], RULE])

    fig = plt.figure(figsize=(7.6, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=.085, right=.975, bottom=.18, top=.94,
                            wspace=.24, hspace=.25)
    axes = [fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    draw_locator(axes[0], china, points)
    titles = ["(a) Seven-city geography", "(b) Shenzhen — selected window",
              "(c) Kunming — selected window", "(d) Chengdu — selected window"]
    for city, ax in zip(LOCAL_CITIES, axes[1:]):
        bounds, counts = windows[city]
        draw_local(ax, city, local_data[city], bounds, counts)
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontsize=9.0, color=INK, fontweight="semibold", pad=7)
        ax.tick_params(axis="both", labelsize=8.1, length=2.5, width=.7, colors=INK)
        ax.spines["left"].set_color("#98A6AD")
        ax.spines["bottom"].set_color("#98A6AD")

    class_handles = [Line2D([], [], marker=CLASS_MARKERS[k], linestyle="none",
                            color=CLASS_COLORS[k], markeredgecolor=INK,
                            markeredgewidth=.4, markersize=6, label=CLASS_NAMES[k])
                     for k in (0, 1, 2)]
    fig.legend(handles=class_handles, loc="lower center", bbox_to_anchor=(.53, .025),
               ncol=3, frameon=False, fontsize=8.4, columnspacing=1.8,
               handletextpad=.45)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "Figure2_spatial_context.pdf"
    png = OUTPUT_DIR / "Figure2_spatial_context.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.07)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)
    print(json.dumps({"pdf": str(pdf), "png": str(png), "windows": {
        c: {"bounds": windows[c][0], "counts": windows[c][1]} for c in LOCAL_CITIES}}, indent=2))


if __name__ == "__main__":
    main()

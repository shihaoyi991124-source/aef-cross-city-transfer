"""Render supplementary external-target comparisons from frozen aggregate tables.

No model is fitted and no score is recalculated. Source bars use the reported
five-seed single-source means; pooled, ensemble, and same-city values come
from the target summary. The latter is a supervised reference, not a transfer
strategy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isclose
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
PAIR_TABLE = ROOT / "results" / "supplementary_tables" / "TableS9.csv"
TARGET_TABLE = ROOT / "results" / "canonical_tables" / "stage09b_external_summary.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper" / "figures"

SOURCES = ("Beijing", "Kunming", "Shenzhen", "Wuhan", "Xian")
ABBREVIATIONS = ("BJ", "KM", "SZ", "WH", "XA")
TARGETS = (("Chengdu", "FigureS16_clean"), ("Hangzhou", "FigureS17_clean"))
COLORS = ("#5B83BE",) * 5 + ("#C87323", "#1C837B")


@dataclass(frozen=True)
class Comparison:
    target: str
    categories: tuple[str, ...]
    values: tuple[float, ...]
    same_city_reference: float


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_comparisons() -> tuple[Comparison, ...]:
    """Read only the packaged final aggregate values and validate their joins."""
    pairs = _read(PAIR_TABLE)
    summaries = _read(TARGET_TABLE)
    if len(pairs) != 10 or len(summaries) != 2:
        raise ValueError("Expected ten external directions and two target summaries")
    by_target = {row["target_city"]: row for row in summaries}
    if set(by_target) != {name for name, _ in TARGETS}:
        raise ValueError("External target summary mismatch")

    comparisons = []
    for target, _ in TARGETS:
        target_pairs = {row["source_city"]: row for row in pairs if row["target_city"] == target}
        if set(target_pairs) != set(SOURCES):
            raise ValueError(f"Missing or duplicate source direction for {target}")
        summary = by_target[target]
        reference = float(summary["same_city_macro_f1"])
        source_values = tuple(float(target_pairs[source]["actual_cross_macro_f1"]) for source in SOURCES)
        for source, value in zip(SOURCES, source_values):
            gap = float(target_pairs[source]["actual_macro_gap"])
            if not isclose(reference - value, gap, abs_tol=2e-10):
                raise ValueError(f"Target-normalized gap mismatch: {source} to {target}")
        if not isclose(sum(source_values) / 5, float(summary["mean_single_source_macro_f1"]), abs_tol=2e-10):
            raise ValueError(f"Mean single-source mismatch for {target}")
        values = source_values + (float(summary["pooled_macro_f1"]), float(summary["ensemble_macro_f1"]))
        comparisons.append(Comparison(target, ABBREVIATIONS + ("POOLED5", "ENSEMBLE5"), values, reference))
    return tuple(comparisons)


def render(comparison: Comparison, stem: str) -> None:
    """Draw the seven reported strategies and the supervised reference."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(comparison.categories, comparison.values, color=COLORS, width=0.68)
    ax.axhline(comparison.same_city_reference, color="#9C3D3D", linestyle="--", linewidth=1.4)
    ax.annotate(
        "same-city reference",
        xy=(0.96, comparison.same_city_reference),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right", va="bottom", color="#9C3D3D", fontsize=9,
    )
    ax.set(title=f"{comparison.target}: single-source and multi-source comparison", ylabel="Macro-F1")
    ax.set_ylim(0.85, 1.008)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    for comparison, (_, stem) in zip(load_comparisons(), TARGETS):
        render(comparison, stem)
        print(f"{stem}: 7 categories, supervised reference {comparison.same_city_reference:.10f}")


if __name__ == "__main__":
    main()

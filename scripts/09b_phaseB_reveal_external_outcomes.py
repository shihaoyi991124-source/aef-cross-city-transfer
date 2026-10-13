"""Stage 09B Phase B: hash-gated external label reveal and one-time evaluation.

This process verifies every Phase A hash before reading target labels.  It never
regenerates a Phase A prediction and never fits or changes a risk calibrator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "FiveCity_AEF_195D.parquet"
EXTERNAL = {
    "Chengdu": ROOT / "data" / "processed" / "external" / "Chengdu_External_AEF_195D.parquet",
    "Hangzhou": ROOT / "data" / "processed" / "external" / "Hangzhou_External_AEF_195D.parquet",
}
CONFIG = ROOT / "config.json"
BASELINE = ROOT / "scripts" / "02_baseline_experiment.py"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
LOGS = ROOT / "outputs" / "logs"
FROZEN = ROOT / "outputs" / "frozen_external"

MANIFEST_PATH = FROZEN / "stage09b_phaseA_manifest.txt"
SPLITS_PATH = TABLES / "stage09b_external_block_splits.csv"
CALIBRATORS_PATH = TABLES / "stage09b_frozen_risk_calibrators.csv"
SAMPLE_PREDICTIONS_PATH = FROZEN / "stage09b_external_sample_predictions.parquet"
PAIR_PREDICTIONS_PATH = FROZEN / "stage09b_external_pair_predictions.csv"
STAGE08_PAIRS = TABLES / "stage08_label_free_pair_features.csv"

SPLIT_AUDIT_PATH = TABLES / "stage09b_external_split_audit.csv"
SAME_PATH = TABLES / "stage09b_external_same_city.csv"
SINGLE_PATH = TABLES / "stage09b_external_single_source.csv"
PAIR_RESULTS_PATH = TABLES / "stage09b_external_pair_results.csv"
RISK_PATH = TABLES / "stage09b_external_risk_validation.csv"
HIGH_RISK_PATH = TABLES / "stage09b_external_high_risk.csv"
MULTISOURCE_PATH = TABLES / "stage09b_external_multisource.csv"
ENSEMBLE_PATH = TABLES / "stage09b_external_ensemble.csv"
SUMMARY_PATH = TABLES / "stage09b_external_summary.csv"
REPORT_PATH = LOGS / "stage09b_external_validation_report.txt"

FIGURE_PATHS = {
    "risk_scatter": FIGURES / "40_external_predicted_vs_actual_gap.png",
    "single": FIGURES / "41_external_single_source_transfer.png",
    "pooled": FIGURES / "42_external_pooled_vs_single.png",
    "chengdu": FIGURES / "43_external_chengdu_comparison.png",
    "hangzhou": FIGURES / "44_external_hangzhou_comparison.png",
    "high_risk": FIGURES / "45_external_risk_classification.png",
}

CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
TARGETS = ["Chengdu", "Hangzhou"]
ABBR = {
    "Shenzhen": "SZ", "Kunming": "KM", "Wuhan": "WH", "Xian": "XA",
    "Beijing": "BJ", "Chengdu": "CD", "Hangzhou": "HZ",
}
SEEDS = [20261008, 20261009, 20261010, 20261011, 20261012]
FEATURES = [f"A{i:02d}_{year}" for year in (2020, 2022, 2024) for i in range(64)]
METRICS = [
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "class0_f1", "class1_f1", "expansion_f1",
]
HIGH_RISK_THRESHOLD = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def load_stage03() -> Any:
    spec = importlib.util.spec_from_file_location("stage03_baseline_for_stage09b_b", BASELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Stage 03 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_id_hash(values: pd.Series | np.ndarray | list[str]) -> str:
    items = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest().upper()


def verify_phase_a_gate() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Phase A manifest is absent")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required_false = [
        "target_labels_read_for_prediction", "target_f1_computed", "target_gap_computed",
        "threshold_changed", "model_changed",
    ]
    if manifest.get("status") != "FROZEN_LABEL_BLIND" or any(manifest.get(key) is not False for key in required_false):
        raise RuntimeError("Phase A manifest flags fail the hard reveal gate")
    for name, expected in manifest.get("files", {}).items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Phase A file missing: {name}")
        actual = sha256(path)
        if actual != str(expected).upper():
            raise RuntimeError(f"Frozen Phase A hash mismatch: {name}: {actual} != {expected}")
    required = {
        "outputs/tables/stage09b_external_block_splits.csv",
        "outputs/tables/stage09b_frozen_risk_calibrators.csv",
        "outputs/frozen_external/stage09b_external_sample_predictions.parquet",
        "outputs/frozen_external/stage09b_external_pair_predictions.csv",
    }
    if not required.issubset(manifest.get("files", {})):
        raise RuntimeError("Phase A manifest does not cover every frozen prediction input")
    return manifest


def safe_correlations(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if len(np.unique(predicted)) < 2 or len(np.unique(actual)) < 2:
        return np.nan, np.nan
    return float(spearmanr(actual, predicted).statistic), float(pearsonr(actual, predicted).statistic)


def evaluate_prediction(stage03: Any, labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return stage03.land_cover_metrics(labels.astype(np.int64), prediction.astype(np.int64))


def external_risk_judgment(spearman: float, improvement: float, auroc: float) -> str:
    if spearman >= 0.50 and improvement >= 0.15 and auroc >= 0.75:
        return "EXTERNALLY SUPPORTED"
    if spearman > 0 and (improvement >= 0.15 or auroc >= 0.75):
        return "PARTIALLY SUPPORTED"
    return "NOT SUPPORTED"


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(pair: pd.DataFrame, summary: pd.DataFrame, single: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    colors = {"R1": "#1565C0", "R4": "#D84315"}

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    low = float(min(pair.actual_macro_gap.min(), pair.R1_predicted_macro_gap.min(), pair.R4_predicted_macro_gap.min())) - 0.01
    high = float(max(pair.actual_macro_gap.max(), pair.R1_predicted_macro_gap.max(), pair.R4_predicted_macro_gap.max())) + 0.01
    ax.plot([low, high], [low, high], color="#666666", linestyle="--", linewidth=1, label="Perfect calibration")
    for name, column in [("R1", "R1_predicted_macro_gap"), ("R4", "R4_predicted_macro_gap")]:
        ax.scatter(pair.actual_macro_gap, pair[column], s=52, alpha=0.85, color=colors[name], label=name)
    for row in pair.itertuples(index=False):
        ax.annotate(f"{row.source_abbreviation}→{row.target_abbreviation}", (row.actual_macro_gap, row.R4_predicted_macro_gap), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set(xlabel="Actual target-normalized Macro-F1 gap", ylabel="Frozen predicted Macro-F1 gap", title="External frozen risk predictions vs actual gaps", xlim=(low, high), ylim=(low, high))
    ax.legend()
    save_figure(fig, FIGURE_PATHS["risk_scatter"])

    pair_order = pair.sort_values(["target_city", "source_city"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(pair_order))
    ax.bar(positions, pair_order.actual_cross_macro_f1, color=["#5B8FF9" if c == "Chengdu" else "#61DDAA" for c in pair_order.target_city])
    ax.set_xticks(positions, pair_order.direction.str.replace(" -> ", "→"), rotation=45, ha="right")
    ax.set(ylabel="Macro-F1", title="Frozen single-source external transfer")
    ax.set_ylim(max(0.0, pair_order.actual_cross_macro_f1.min() - 0.08), 1.0)
    save_figure(fig, FIGURE_PATHS["single"])

    labels = [ABBR[city] for city in TARGETS]
    x = np.arange(len(TARGETS))
    width = 0.19
    methods = [
        ("Mean single", "mean_single_source_macro_f1", "#9E9E9E"),
        ("Oracle single", "oracle_best_single_macro_f1", "#7E57C2"),
        ("POOLED5", "pooled_macro_f1", "#EF6C00"),
        ("ENSEMBLE5", "ensemble_macro_f1", "#00897B"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ordered_summary = summary.set_index("target_city").loc[TARGETS]
    for index, (name, column, color) in enumerate(methods):
        ax.bar(x + (index - 1.5) * width, ordered_summary[column], width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Macro-F1", title="External multi-source strategies vs single-source baselines")
    ax.set_ylim(max(0.0, ordered_summary[[column for _, column, _ in methods]].min().min() - 0.08), 1.0)
    ax.legend(ncol=2)
    save_figure(fig, FIGURE_PATHS["pooled"])

    for target, key in [("Chengdu", "chengdu"), ("Hangzhou", "hangzhou")]:
        source_means = single.loc[single.target_city == target].groupby("source_city")["macro_f1"].mean().reindex(CITIES)
        target_row = summary.loc[summary.target_city == target].iloc[0]
        names = [ABBR[city] for city in CITIES] + ["POOLED5", "ENSEMBLE5"]
        values = source_means.tolist() + [target_row.pooled_macro_f1, target_row.ensemble_macro_f1]
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        ax.bar(names, values, color=["#5B8FF9"] * 5 + ["#EF6C00", "#00897B"])
        ax.axhline(target_row.same_city_macro_f1, color="#B71C1C", linestyle="--", linewidth=1.2, label="Same-city")
        ax.set(ylabel="Macro-F1", title=f"{target} external comparison")
        ax.set_ylim(max(0.0, min(values) - 0.08), 1.0)
        ax.legend()
        save_figure(fig, FIGURE_PATHS[key])

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)
    positions = np.arange(len(pair_order))
    for ax, name, column in [(axes[0], "R1", "R1_predicted_macro_gap"), (axes[1], "R4", "R4_predicted_macro_gap")]:
        actual_high = pair_order.actual_macro_gap >= HIGH_RISK_THRESHOLD
        ax.scatter(positions, pair_order[column], c=np.where(actual_high, "#D32F2F", "#388E3C"), s=48)
        ax.axhline(HIGH_RISK_THRESHOLD, color="#222222", linestyle="--", linewidth=1)
        ax.set_ylabel(f"{name} predicted gap")
        ax.set_title(f"{name}: red = actual high risk")
    axes[1].set_xticks(positions, pair_order.direction.str.replace(" -> ", "→"), rotation=45, ha="right")
    save_figure(fig, FIGURE_PATHS["high_risk"])


def report_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "NaN")
    return view.to_string(index=False)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Hard gate: this entire block completes before any target label column is read.
    manifest = verify_phase_a_gate()
    reveal_timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    REPORT_PATH.write_text(
        f"Stage 09B Frozen External Validation\nLABEL_REVEAL_STARTED {reveal_timestamp}\n",
        encoding="utf-8",
    )
    print(f"LABEL_REVEAL_STARTED {reveal_timestamp}", flush=True)

    # Labels are first read after the manifest and frozen file hashes have passed.
    external_frames: dict[str, pd.DataFrame] = {}
    for city, path in EXTERNAL.items():
        frame = pd.read_parquet(path, columns=["sample_id", "city", "block_id", "class"] + FEATURES, engine="pyarrow")
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["block_id"] = frame["block_id"].astype(str)
        if len(frame) != 18000 or set(frame["class"].astype(int).unique()) != {0, 1, 2}:
            raise AssertionError(f"{city}: invalid external labelled dataset")
        external_frames[city] = frame

    split_table = pd.read_csv(SPLITS_PATH, dtype={"block_id": str})
    sample_predictions = pd.read_parquet(SAMPLE_PREDICTIONS_PATH, engine="pyarrow")
    sample_predictions["sample_id"] = sample_predictions["sample_id"].astype(str)
    pair_predictions = pd.read_csv(PAIR_PREDICTIONS_PATH)
    calibrators = pd.read_csv(CALIBRATORS_PATH)
    if "class" in sample_predictions.columns or any(column.startswith("actual") for column in pair_predictions.columns):
        raise RuntimeError("Phase A frozen files contain prohibited outcome fields")
    if len(pair_predictions) != 10 or set(calibrators.risk_model) != {"R1", "R4"}:
        raise AssertionError("Frozen risk artifacts are incomplete")
    if not np.allclose(sample_predictions[["prob_class0", "prob_class1", "prob_class2"]].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Frozen probabilities do not sum to one")

    warnings: list[str] = []
    split_audit_rows: list[dict[str, Any]] = []
    split_indices: dict[tuple[str, int, str], np.ndarray] = {}
    for city in TARGETS:
        frame = external_frames[city]
        for seed in SEEDS:
            part = split_table.loc[(split_table.city == city) & (split_table.seed == seed)]
            blocks_by_split = {name: set(part.loc[part.split == name, "block_id"]) for name in ("train", "validation", "test")}
            if blocks_by_split["train"] & blocks_by_split["validation"] or blocks_by_split["train"] & blocks_by_split["test"] or blocks_by_split["validation"] & blocks_by_split["test"]:
                raise AssertionError(f"{city} seed={seed}: frozen block overlap")
            if set().union(*blocks_by_split.values()) != set(frame.block_id.unique()):
                raise AssertionError(f"{city} seed={seed}: frozen block coverage failure")
            for name in ("train", "validation", "test"):
                indices = np.flatnonzero(frame.block_id.isin(blocks_by_split[name]).to_numpy())
                split_indices[(city, seed, name)] = indices
                counts = np.bincount(frame.iloc[indices]["class"].to_numpy(np.int64), minlength=3)
                split_audit_rows.append({
                    "city": city, "city_abbreviation": ABBR[city], "seed": seed, "split": name,
                    "blocks": len(blocks_by_split[name]), "samples": len(indices),
                    "class0_samples": int(counts[0]), "class1_samples": int(counts[1]), "class2_samples": int(counts[2]),
                })
                if name == "test" and np.any(counts == 0):
                    warnings.append(f"{city} seed={seed}: a class is absent from the frozen test fold; no resplit performed")
            expected_test_ids = set(frame.iloc[split_indices[(city, seed, "test")]]["sample_id"])
            for model_name in CITIES + ["POOLED5", "ENSEMBLE5"]:
                frozen = sample_predictions.loc[
                    (sample_predictions.target_city == city) & (sample_predictions.seed == seed) &
                    (sample_predictions.source_city_or_model == model_name)
                ]
                if len(frozen) != len(expected_test_ids) or set(frozen.sample_id) != expected_test_ids:
                    raise AssertionError(f"{city} seed={seed} {model_name}: frozen test sample mismatch")
    split_audit = pd.DataFrame(split_audit_rows)
    save_csv(split_audit, SPLIT_AUDIT_PATH)

    stage03 = load_stage03()
    stage_config = json.loads(CONFIG.read_text(encoding="utf-8"))["stage_03"]
    same_rows: list[dict[str, Any]] = []
    for city in TARGETS:
        frame = external_frames[city]
        matrix = frame[FEATURES].to_numpy(np.float32, copy=False)
        labels = frame["class"].to_numpy(np.int64, copy=False)
        for seed in SEEDS:
            train = split_indices[(city, seed, "train")]
            validation = split_indices[(city, seed, "validation")]
            test = split_indices[(city, seed, "test")]
            model = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
            model, seconds, iterations = stage03.fit_model(
                model, "XGBoost", matrix[train], labels[train], matrix[validation], labels[validation],
                warnings, f"Stage 09B Phase B same-city={city} seed={seed}",
            )
            metrics = evaluate_prediction(stage03, labels[test], model.predict(matrix[test]).astype(int))
            same_rows.append({
                "target_city": city, "target_abbreviation": ABBR[city], "seed": seed,
                "model": "XGBoost", "feature_set": "AEF-192", "experiment_type": "external same-city spatial",
                "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
                "train_blocks": int(split_audit.loc[(split_audit.city == city) & (split_audit.seed == seed) & (split_audit.split == "train"), "blocks"].iloc[0]),
                "validation_blocks": int(split_audit.loc[(split_audit.city == city) & (split_audit.seed == seed) & (split_audit.split == "validation"), "blocks"].iloc[0]),
                "test_blocks": int(split_audit.loc[(split_audit.city == city) & (split_audit.seed == seed) & (split_audit.split == "test"), "blocks"].iloc[0]),
                "target_test_sample_id_sha256": sample_id_hash(frame.iloc[test]["sample_id"]),
                "training_seconds": seconds, "fitted_iterations": iterations, **metrics,
            })
            print(f"SAME_CITY city={city} seed={seed} macro={metrics['macro_f1']:.6f} expansion={metrics['expansion_f1']:.6f}", flush=True)
    same_city = pd.DataFrame(same_rows).sort_values(["target_city", "seed"]).reset_index(drop=True)
    save_csv(same_city, SAME_PATH)

    single_rows: list[dict[str, Any]] = []
    for target_city in TARGETS:
        labels_by_id = external_frames[target_city].set_index("sample_id")["class"]
        for seed in SEEDS:
            same = same_city.loc[(same_city.target_city == target_city) & (same_city.seed == seed)].iloc[0]
            expected_hash = same.target_test_sample_id_sha256
            for source in CITIES:
                frozen = sample_predictions.loc[
                    (sample_predictions.target_city == target_city) & (sample_predictions.seed == seed) &
                    (sample_predictions.source_city_or_model == source)
                ].copy()
                if sample_id_hash(frozen.sample_id) != expected_hash:
                    raise AssertionError(f"{source}->{target_city} seed={seed}: same target test set assertion failed")
                y_true = labels_by_id.loc[frozen.sample_id].to_numpy(np.int64)
                metrics = evaluate_prediction(stage03, y_true, frozen.predicted_class.to_numpy(np.int64))
                single_rows.append({
                    "source_city": source, "source_abbreviation": ABBR[source],
                    "target_city": target_city, "target_abbreviation": ABBR[target_city],
                    "direction": f"{source} -> {target_city}", "seed": seed,
                    "model": "XGBoost", "feature_set": "AEF-192", "experiment_type": "frozen external zero-shot",
                    "target_test_samples": len(frozen), "target_test_sample_id_sha256": expected_hash,
                    **metrics,
                    "target_same_city_macro_f1": same.macro_f1,
                    "target_same_city_expansion_f1": same.expansion_f1,
                    "macro_gap": float(same.macro_f1 - metrics["macro_f1"]),
                    "expansion_gap": float(same.expansion_f1 - metrics["expansion_f1"]),
                    "shared_target_test_set_assertion": True,
                })
    single = pd.DataFrame(single_rows).sort_values(["target_city", "source_city", "seed"]).reset_index(drop=True)
    if len(single) != 50:
        raise AssertionError("All 50 frozen single-source runs were not evaluated")
    save_csv(single, SINGLE_PATH)

    aggregate_rows: list[dict[str, Any]] = []
    for (source, target), group in single.groupby(["source_city", "target_city"], sort=False):
        aggregate_rows.append({
            "source_city": source, "target_city": target,
            "actual_cross_macro_f1": float(group.macro_f1.mean()),
            "actual_cross_macro_f1_seed_std": float(group.macro_f1.std(ddof=1)),
            "actual_cross_expansion_f1": float(group.expansion_f1.mean()),
            "actual_cross_expansion_f1_seed_std": float(group.expansion_f1.std(ddof=1)),
            "actual_macro_gap": float(group.macro_gap.mean()),
            "actual_macro_gap_seed_std": float(group.macro_gap.std(ddof=1)),
            "actual_expansion_gap": float(group.expansion_gap.mean()),
            "actual_expansion_gap_seed_std": float(group.expansion_gap.std(ddof=1)),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    pair_results = pair_predictions.merge(aggregate, on=["source_city", "target_city"], how="left", validate="one_to_one")
    if len(pair_results) != 10 or pair_results[["actual_macro_gap", "actual_cross_macro_f1"]].isna().any().any():
        raise AssertionError("External pair outcome join failed")
    pair_results = pair_results.sort_values(["target_city", "source_city"]).reset_index(drop=True)
    save_csv(pair_results, PAIR_RESULTS_PATH)

    development_pairs = pd.read_csv(STAGE08_PAIRS)
    r0_value = float(development_pairs.actual_macro_gap.mean())
    actual = pair_results.actual_macro_gap.to_numpy(float)
    predictions = {
        "R0": np.full(len(pair_results), r0_value),
        "R1": pair_results.R1_predicted_macro_gap.to_numpy(float),
        "R4": pair_results.R4_predicted_macro_gap.to_numpy(float),
    }
    r0_mae = float(mean_absolute_error(actual, predictions["R0"]))
    risk_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    risk_lookup: dict[str, dict[str, Any]] = {}
    for name, predicted in predictions.items():
        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
        rho, pearson = safe_correlations(actual, predicted)
        improvement = 0.0 if name == "R0" else float((r0_mae - mae) / r0_mae)
        if name in {"R1", "R4"}:
            actual_high = (actual >= HIGH_RISK_THRESHOLD).astype(int)
            predicted_high = (predicted >= HIGH_RISK_THRESHOLD).astype(int)
            auroc = float(roc_auc_score(actual_high, predicted)) if len(np.unique(actual_high)) == 2 else np.nan
            high = {
                "risk_model": name, "threshold": HIGH_RISK_THRESHOLD, "n_pairs": len(actual),
                "actual_high_risk_count": int(actual_high.sum()), "predicted_high_risk_count": int(predicted_high.sum()),
                "accuracy": float(accuracy_score(actual_high, predicted_high)),
                "balanced_accuracy": float(balanced_accuracy_score(actual_high, predicted_high)),
                "precision": float(precision_score(actual_high, predicted_high, zero_division=0)),
                "recall": float(recall_score(actual_high, predicted_high, zero_division=0)),
                "f1": float(f1_score(actual_high, predicted_high, zero_division=0)),
                "auroc": auroc,
            }
            high_rows.append(high)
            judgment = external_risk_judgment(rho, improvement, auroc)
        else:
            auroc = np.nan
            judgment = "REFERENCE BASELINE"
        status = (
            "TRAINING-MEAN BASELINE" if name == "R0" else
            "PREDEFINED GEOMETRY CANDIDATE" if name == "R1" else
            "POST-HOC DEVELOPMENT CANDIDATE WITH EXTERNAL VALIDATION"
        )
        row = {
            "risk_model": name, "candidate_status": status, "n_external_pairs": len(actual),
            "mae": mae, "rmse": rmse, "spearman_rho": rho, "pearson_r": pearson,
            "r0_mae": r0_mae, "relative_mae_improvement_vs_R0": improvement,
            "high_risk_auroc": auroc, "external_judgment": judgment,
        }
        risk_rows.append(row)
        risk_lookup[name] = row
    risk_validation = pd.DataFrame(risk_rows)
    high_risk = pd.DataFrame(high_rows)
    save_csv(risk_validation, RISK_PATH)
    save_csv(high_risk, HIGH_RISK_PATH)

    # Evaluate only frozen Phase A POOLED5 and ENSEMBLE5 sample predictions.
    pooled_metrics: dict[tuple[str, int], dict[str, Any]] = {}
    ensemble_rows: list[dict[str, Any]] = []
    for target_city in TARGETS:
        labels_by_id = external_frames[target_city].set_index("sample_id")["class"]
        for seed in SEEDS:
            for model_name in ["POOLED5", "ENSEMBLE5"]:
                frozen = sample_predictions.loc[
                    (sample_predictions.target_city == target_city) & (sample_predictions.seed == seed) &
                    (sample_predictions.source_city_or_model == model_name)
                ]
                same_hash = same_city.loc[(same_city.target_city == target_city) & (same_city.seed == seed), "target_test_sample_id_sha256"].iloc[0]
                if sample_id_hash(frozen.sample_id) != same_hash:
                    raise AssertionError(f"{target_city} seed={seed} {model_name}: frozen test mismatch")
                metrics = evaluate_prediction(stage03, labels_by_id.loc[frozen.sample_id].to_numpy(np.int64), frozen.predicted_class.to_numpy(np.int64))
                if model_name == "POOLED5":
                    pooled_metrics[(target_city, seed)] = metrics
                else:
                    ensemble_rows.append({
                        "target_city": target_city, "target_abbreviation": ABBR[target_city], "seed": seed,
                        "model": model_name, "target_test_samples": len(frozen), "target_test_sample_id_sha256": same_hash,
                        **metrics,
                    })
    ensemble = pd.DataFrame(ensemble_rows).sort_values(["target_city", "seed"]).reset_index(drop=True)
    save_csv(ensemble, ENSEMBLE_PATH)

    selected_source = pair_predictions.sort_values(["target_city", "PCD_5NN"]).groupby("target_city", as_index=False).first().set_index("target_city")["source_city"].to_dict()
    multi_rows: list[dict[str, Any]] = []
    for target_city in TARGETS:
        for seed in SEEDS:
            source_runs = single.loc[(single.target_city == target_city) & (single.seed == seed)].copy()
            same = same_city.loc[(same_city.target_city == target_city) & (same_city.seed == seed)].iloc[0]
            ensemble_row = ensemble.loc[(ensemble.target_city == target_city) & (ensemble.seed == seed)].iloc[0]
            pooled = pooled_metrics[(target_city, seed)]
            oracle_macro = source_runs.loc[source_runs.macro_f1.idxmax()]
            worst_macro = source_runs.loc[source_runs.macro_f1.idxmin()]
            oracle_expansion = source_runs.loc[source_runs.expansion_f1.idxmax()]
            worst_expansion = source_runs.loc[source_runs.expansion_f1.idxmin()]
            selected = source_runs.loc[source_runs.source_city == selected_source[target_city]].iloc[0]
            multi_rows.append({
                "target_city": target_city, "target_abbreviation": ABBR[target_city], "seed": seed,
                "source_city_count": 5, "target_test_samples": int(same.test_samples),
                **{f"pooled_{name}": pooled[name] for name in METRICS},
                "target_same_city_macro_f1": same.macro_f1, "target_same_city_expansion_f1": same.expansion_f1,
                "pooled_macro_gap": float(same.macro_f1 - pooled["macro_f1"]),
                "pooled_expansion_gap": float(same.expansion_f1 - pooled["expansion_f1"]),
                "mean_single_source_macro_f1": float(source_runs.macro_f1.mean()),
                "mean_single_source_expansion_f1": float(source_runs.expansion_f1.mean()),
                "oracle_best_macro_source": oracle_macro.source_city, "oracle_best_macro_f1": oracle_macro.macro_f1,
                "oracle_best_expansion_source": oracle_expansion.source_city, "oracle_best_expansion_f1": oracle_expansion.expansion_f1,
                "worst_macro_source": worst_macro.source_city, "worst_macro_f1": worst_macro.macro_f1,
                "worst_expansion_source": worst_expansion.source_city, "worst_expansion_f1": worst_expansion.expansion_f1,
                "distance_selected_source": selected_source[target_city],
                "distance_selected_macro_f1": selected.macro_f1,
                "distance_selected_expansion_f1": selected.expansion_f1,
                "uniform_ensemble_macro_f1": ensemble_row.macro_f1,
                "uniform_ensemble_expansion_f1": ensemble_row.expansion_f1,
                "pooled_minus_mean_single_macro_f1": float(pooled["macro_f1"] - source_runs.macro_f1.mean()),
                "pooled_minus_mean_single_expansion_f1": float(pooled["expansion_f1"] - source_runs.expansion_f1.mean()),
                "oracle_is_retrospective_not_deployable": True,
                "external_target_used_in_training_or_early_stopping": False,
            })
    multisource = pd.DataFrame(multi_rows).sort_values(["target_city", "seed"]).reset_index(drop=True)
    save_csv(multisource, MULTISOURCE_PATH)

    summary_rows: list[dict[str, Any]] = []
    for target_city in TARGETS:
        same_group = same_city.loc[same_city.target_city == target_city]
        source_group = single.loc[single.target_city == target_city]
        multi_group = multisource.loc[multisource.target_city == target_city]
        ensemble_group = ensemble.loc[ensemble.target_city == target_city]
        summary_rows.append({
            "target_city": target_city, "target_abbreviation": ABBR[target_city], "seed_count": len(multi_group),
            "same_city_macro_f1": float(same_group.macro_f1.mean()),
            "same_city_macro_f1_std": float(same_group.macro_f1.std(ddof=1)),
            "same_city_expansion_f1": float(same_group.expansion_f1.mean()),
            "same_city_expansion_f1_std": float(same_group.expansion_f1.std(ddof=1)),
            "mean_single_source_macro_f1": float(multi_group.mean_single_source_macro_f1.mean()),
            "mean_single_source_expansion_f1": float(multi_group.mean_single_source_expansion_f1.mean()),
            "oracle_best_single_macro_f1": float(multi_group.oracle_best_macro_f1.mean()),
            "oracle_best_single_expansion_f1": float(multi_group.oracle_best_expansion_f1.mean()),
            "worst_single_macro_f1": float(multi_group.worst_macro_f1.mean()),
            "worst_single_expansion_f1": float(multi_group.worst_expansion_f1.mean()),
            "distance_selected_source": selected_source[target_city],
            "distance_selected_macro_f1": float(multi_group.distance_selected_macro_f1.mean()),
            "distance_selected_expansion_f1": float(multi_group.distance_selected_expansion_f1.mean()),
            "pooled_macro_f1": float(multi_group.pooled_macro_f1.mean()),
            "pooled_macro_f1_std": float(multi_group.pooled_macro_f1.std(ddof=1)),
            "pooled_expansion_f1": float(multi_group.pooled_expansion_f1.mean()),
            "pooled_expansion_f1_std": float(multi_group.pooled_expansion_f1.std(ddof=1)),
            "ensemble_macro_f1": float(ensemble_group.macro_f1.mean()),
            "ensemble_macro_f1_std": float(ensemble_group.macro_f1.std(ddof=1)),
            "ensemble_expansion_f1": float(ensemble_group.expansion_f1.mean()),
            "ensemble_expansion_f1_std": float(ensemble_group.expansion_f1.std(ddof=1)),
            "pooled_minus_mean_single_macro_f1": float(multi_group.pooled_minus_mean_single_macro_f1.mean()),
            "pooled_minus_mean_single_expansion_f1": float(multi_group.pooled_minus_mean_single_expansion_f1.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    improved_count = int((summary.pooled_minus_mean_single_macro_f1 > 0).sum())
    mean_improvement = float(summary.pooled_minus_mean_single_macro_f1.mean())
    pooled_judgment = (
        "STRONG EXTERNAL REPLICATION" if improved_count == 2 and mean_improvement > 0 else
        "PARTIAL EXTERNAL REPLICATION" if improved_count == 1 and mean_improvement > 0 else
        "NOT REPLICATED"
    )
    summary["pooled_targets_improved"] = improved_count
    summary["pooled_mean_target_improvement"] = mean_improvement
    summary["external_pooled_judgment"] = pooled_judgment
    save_csv(summary, SUMMARY_PATH)

    create_figures(pair_results, summary, single)

    protocol_warnings = [message for message in warnings if "absent" in message or "protocol" in message.lower()]
    stage08_r1 = "PARTIALLY SUPPORTED during development"
    stage08_r4 = "post-hoc development candidate; strong development association"
    contradictions: list[str] = []
    if risk_lookup["R1"]["external_judgment"] == "NOT SUPPORTED":
        contradictions.append("R1 did not retain support on the untouched external pairs despite partial development support.")
    if risk_lookup["R4"]["external_judgment"] == "NOT SUPPORTED":
        contradictions.append("R4's post-hoc development association did not retain support externally.")
    if pooled_judgment == "NOT REPLICATED":
        contradictions.append("The Stage08 pooled advantage did not replicate on the external targets.")
    elif pooled_judgment == "PARTIAL EXTERNAL REPLICATION":
        contradictions.append("The Stage08 pooled advantage replicated for only one of two external targets.")
    contradiction_text = " ".join(contradictions) if contradictions else "No material contradiction: risk/pooling conclusions retained the required external support."

    calibrator_lines = calibrators[["risk_model", "predictor", "intercept", "coefficient", "training_n", "candidate_status"]].to_string(index=False)
    lines = [
        "Stage 09B — Frozen External Validation",
        "=" * 78,
        "",
        "1. Stage09A hash verification",
        f"- PASS. Chengdu={manifest['files']['data/processed/external/Chengdu_External_AEF_195D.parquet']}",
        f"- PASS. Hangzhou={manifest['files']['data/processed/external/Hangzhou_External_AEF_195D.parquet']}",
        f"- PASS. FiveCity={manifest['files']['data/processed/FiveCity_AEF_195D.parquet']}",
        "",
        "2. Label-blind block split protocol",
        "- Deterministic SHA256 ordering of seed + block_id; approximately 70/15/15 blocks.",
        "- Split function received only city, block IDs, and seed. Frozen block sets are disjoint and exhaustive.",
        "",
        "3. Phase A leakage audit",
        "- PASS: external label column was not read by Phase A; target features were sample_id/city/block_id/AEF-192/split only.",
        "- PASS: risk functions had no target-label argument; no F1, actual gap, or confusion matrix was computed in Phase A.",
        f"- Stage07 reproduction max errors: {manifest['stage07_reproduction_max_absolute_error']}",
        "",
        "4. Phase A frozen file hashes",
        *[f"- {name}: {value}" for name, value in manifest["files"].items()],
        "",
        "5. Frozen R1 coefficients",
        calibrator_lines.splitlines()[0], calibrator_lines.splitlines()[1],
        "",
        "6. Frozen R4 coefficients",
        calibrator_lines.splitlines()[0], calibrator_lines.splitlines()[2],
        "- R4 is a POST-HOC DEVELOPMENT CANDIDATE WITH EXTERNAL VALIDATION; its Stage08 discovery remains disclosed.",
        "",
        "7. External risk predictions BEFORE reveal",
        report_table(pair_predictions, ["direction", "PCD_5NN", "mean_entropy", "R1_predicted_macro_gap", "R4_predicted_macro_gap"]),
        "",
        "8. Label reveal timestamp",
        f"- LABEL_REVEAL_STARTED {reveal_timestamp}",
        "- No risk model fitting or Phase A prediction regeneration occurred after this line.",
        "",
        "9. Post-reveal class distributions",
        report_table(split_audit, ["city", "seed", "split", "samples", "class0_samples", "class1_samples", "class2_samples"]),
        "",
        "10. External same-city baselines",
        report_table(same_city, ["target_city", "seed", "macro_f1", "expansion_f1", "accuracy", "balanced_accuracy"]),
        "",
        "11. Ten single-source results",
        report_table(pair_results, ["direction", "actual_cross_macro_f1", "actual_cross_expansion_f1"]),
        "",
        "12. Actual target-normalized gaps",
        report_table(pair_results, ["direction", "actual_macro_gap", "actual_expansion_gap"]),
        "",
        "13. R1 external validation",
        report_table(risk_validation.loc[risk_validation.risk_model == "R1"], ["candidate_status", "mae", "rmse", "spearman_rho", "pearson_r", "relative_mae_improvement_vs_R0", "high_risk_auroc", "external_judgment"]),
        f"- Development reference: {stage08_r1}.",
        "",
        "14. R4 external validation",
        report_table(risk_validation.loc[risk_validation.risk_model == "R4"], ["candidate_status", "mae", "rmse", "spearman_rho", "pearson_r", "relative_mae_improvement_vs_R0", "high_risk_auroc", "external_judgment"]),
        f"- Development reference: {stage08_r4}.",
        "",
        "15. High-risk classification",
        report_table(high_risk, ["risk_model", "threshold", "actual_high_risk_count", "predicted_high_risk_count", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]),
        "",
        "16. POOLED5 external results",
        report_table(summary, ["target_city", "pooled_macro_f1", "pooled_expansion_f1", "mean_single_source_macro_f1", "pooled_minus_mean_single_macro_f1"]),
        f"- Judgment: {pooled_judgment}; improved targets={improved_count}/2; mean Macro-F1 improvement={mean_improvement:.6f}.",
        "",
        "17. ENSEMBLE5 external results",
        report_table(summary, ["target_city", "ensemble_macro_f1", "ensemble_expansion_f1", "pooled_macro_f1", "oracle_best_single_macro_f1"]),
        "",
        "18. Chengdu summary",
        report_table(summary.loc[summary.target_city == "Chengdu"], ["same_city_macro_f1", "same_city_expansion_f1", "mean_single_source_macro_f1", "oracle_best_single_macro_f1", "distance_selected_macro_f1", "pooled_macro_f1", "ensemble_macro_f1"]),
        "",
        "19. Hangzhou summary",
        report_table(summary.loc[summary.target_city == "Hangzhou"], ["same_city_macro_f1", "same_city_expansion_f1", "mean_single_source_macro_f1", "oracle_best_single_macro_f1", "distance_selected_macro_f1", "pooled_macro_f1", "ensemble_macro_f1"]),
        "",
        "20. Predefined judgments",
        f"- R1 PREDEFINED GEOMETRY CANDIDATE: {risk_lookup['R1']['external_judgment']}.",
        f"- R4 POST-HOC DEVELOPMENT CANDIDATE WITH EXTERNAL VALIDATION: {risk_lookup['R4']['external_judgment']}.",
        f"- POOLED5: {pooled_judgment}.",
        "",
        "21. Limitations",
        "- Only two external cities and ten directed source-target pairs are available; correlation and AUROC uncertainty is substantial.",
        "- Oracle best-source results are retrospective and not deployable. Same-city baselines use post-reveal labels solely for target normalization.",
        "- R4 was identified post hoc in Stage08 development and is not reclassified as predefined.",
        "",
        "22. Warnings",
        f"- Protocol warnings: {protocol_warnings if protocol_warnings else 'None.'}",
        f"- Runtime/model warnings: {warnings if warnings else 'None.'}",
        f"- Development/external comparison: {contradiction_text}",
        "",
        "23. Final paper recommendation",
        "- Report all external targets, directions, seeds, and frozen predictions. Preserve the predefined R1 and post-hoc R4 labels.",
        f"- Risk conclusion: R1={risk_lookup['R1']['external_judgment']}; R4={risk_lookup['R4']['external_judgment']}.",
        f"- Multi-source conclusion: {pooled_judgment}. Do not claim that POOLED5 beats the retrospective oracle unless the table shows it.",
        "- Stage 09B final protocol status: SUCCESS. No method changed after label reveal.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    expected_tables = [
        SPLIT_AUDIT_PATH, SAME_PATH, SINGLE_PATH, PAIR_RESULTS_PATH, RISK_PATH,
        HIGH_RISK_PATH, MULTISOURCE_PATH, ENSEMBLE_PATH, SUMMARY_PATH,
    ]
    missing = [str(path) for path in expected_tables + list(FIGURE_PATHS.values()) + [REPORT_PATH] if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Stage 09B outputs missing/empty: {missing}")
    print(f"PHASE_B_SUCCESS report={REPORT_PATH}", flush=True)
    print(risk_validation.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

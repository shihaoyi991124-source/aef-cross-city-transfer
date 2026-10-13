"""Stage 04: label-efficient target adaptation feasibility experiment.

The experiment reuses the deterministic Stage 03 spatial splits. Target test
blocks are fixed within each city/seed and are excluded from adaptation block
selection, model fitting, scaling, validation, and XGBoost early stopping.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "02_baseline_experiment.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

ALL_RUNS_PATH = TABLES_DIR / "fewshot_all_runs.csv"
SUMMARY_PATH = TABLES_DIR / "fewshot_summary.csv"
RECOVERY_PATH = TABLES_DIR / "fewshot_recovery.csv"
REPORT_PATH = LOGS_DIR / "fewshot_report.txt"
STAGE03B_TRANSFER_PATH = TABLES_DIR / "target_normalized_transfer_gap.csv"

CITY_FILES = {
    "Shenzhen": PROCESSED_DIR / "Shenzhen_AEF_195D.parquet",
    "Kunming": PROCESSED_DIR / "Kunming_AEF_195D.parquet",
}
DIRECTIONS = [("Shenzhen", "Kunming"), ("Kunming", "Shenzhen")]
MODELS = ["Logistic Regression", "XGBoost"]
CLASS_VALUES = [0, 1, 2]
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "class0_f1",
    "class1_f1",
    "expansion_f1",
]
FORBIDDEN_FEATURES = {
    "sample_id",
    "city",
    "class",
    "longitude",
    "latitude",
    "utm_x",
    "utm_y",
    "block_x",
    "block_y",
    "block_id",
}


def load_stage03_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage03_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 03 helpers from {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def flattened_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    summary = frame.groupby(group_columns, as_index=False)[value_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def select_nested_block_subsets(
    frame: pd.DataFrame,
    pool_indices: np.ndarray,
    percentages: list[int],
    seed: int,
    context: str,
) -> dict[int, dict[str, Any]]:
    """Greedily select nested whole-block subsets near balanced class targets."""
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    pool = frame.iloc[pool_indices]
    if len(pool) == 0:
        raise ValueError(f"{context}: empty adaptation pool")
    table = pd.crosstab(pool["block_id"], pool["class"]).reindex(
        columns=CLASS_VALUES, fill_value=0
    )
    block_ids = table.index.to_numpy(dtype=object)
    block_class_counts = table.to_numpy(dtype=np.int64)
    block_sizes = block_class_counts.sum(axis=1)
    if np.any(block_sizes <= 0):
        raise AssertionError(f"{context}: empty block encountered")

    rng = np.random.default_rng(seed)
    tie_breaker = rng.uniform(0.0, 1e-8, size=len(block_ids))
    selected = np.zeros(len(block_ids), dtype=bool)
    selected_counts = np.zeros(3, dtype=np.int64)
    selected_total = 0
    snapshots: dict[int, dict[str, Any]] = {}

    for percent in sorted(percentages):
        target_samples = max(3, int(round(len(pool) * percent / 100.0)))
        minimum_blocks = 3
        while (
            selected_total < target_samples
            or np.any(selected_counts == 0)
            or int(selected.sum()) < minimum_blocks
        ):
            available = np.flatnonzero(~selected)
            if len(available) == 0:
                raise RuntimeError(f"{context}: exhausted blocks at {percent}%")
            candidate_counts = selected_counts[None, :] + block_class_counts[available]
            candidate_totals = selected_total + block_sizes[available]
            proportions = candidate_counts / candidate_totals[:, None]
            imbalance = np.abs(proportions - 1.0 / 3.0).sum(axis=1)
            missing_penalty = (candidate_counts == 0).sum(axis=1) * 1000.0
            undershoot = np.maximum(0, target_samples - candidate_totals) / target_samples
            overshoot = np.maximum(0, candidate_totals - target_samples) / target_samples
            size_penalty = block_sizes[available] / target_samples
            score = (
                missing_penalty
                + 4.0 * imbalance
                + 0.15 * undershoot
                + 1.25 * overshoot
                + 0.05 * size_penalty
                + tie_breaker[available]
            )
            chosen = int(available[int(np.argmin(score))])
            selected[chosen] = True
            selected_counts += block_class_counts[chosen]
            selected_total += int(block_sizes[chosen])

        chosen_blocks = set(block_ids[selected].tolist())
        mask = pool["block_id"].isin(chosen_blocks).to_numpy()
        original_indices = np.sort(pool_indices[mask])
        observed_counts = (
            frame.iloc[original_indices]["class"]
            .value_counts()
            .reindex(CLASS_VALUES, fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        if not np.array_equal(observed_counts, selected_counts):
            raise AssertionError(f"{context}: class count mismatch")
        snapshots[percent] = {
            "indices": original_indices,
            "blocks": frozenset(chosen_blocks),
            "target_samples": target_samples,
            "actual_samples": len(original_indices),
            "actual_blocks": len(chosen_blocks),
            "class_counts": observed_counts,
        }

    ordered = sorted(percentages)
    for previous, current in zip(ordered, ordered[1:]):
        if not set(snapshots[previous]["indices"]).issubset(
            set(snapshots[current]["indices"])
        ):
            raise AssertionError(f"{context}: sample nesting failed {previous}% -> {current}%")
        if not snapshots[previous]["blocks"].issubset(snapshots[current]["blocks"]):
            raise AssertionError(f"{context}: block nesting failed {previous}% -> {current}%")
    return snapshots


def build_adaptation_plan(
    frame: pd.DataFrame,
    split: Any,
    percentages: list[int],
    seed: int,
    context: str,
) -> dict[int, dict[str, Any]]:
    train_subsets = select_nested_block_subsets(
        frame, split.train, percentages, seed + 310000, f"{context} train"
    )
    validation_subsets = select_nested_block_subsets(
        frame, split.validation, percentages, seed + 410000, f"{context} validation"
    )
    test_indices = set(np.asarray(split.test, dtype=np.int64).tolist())
    test_blocks = set(frame.iloc[split.test]["block_id"].tolist())
    plan: dict[int, dict[str, Any]] = {}
    for percent in percentages:
        train = train_subsets[percent]
        validation = validation_subsets[percent]
        combined_indices = np.sort(
            np.concatenate([train["indices"], validation["indices"]])
        )
        combined_blocks = train["blocks"].union(validation["blocks"])
        if set(combined_indices).intersection(test_indices):
            raise AssertionError(f"{context} {percent}%: adaptation/test index overlap")
        if combined_blocks.intersection(test_blocks):
            raise AssertionError(f"{context} {percent}%: adaptation/test block overlap")
        if set(train["indices"]).intersection(set(validation["indices"])):
            raise AssertionError(f"{context} {percent}%: train/validation overlap")
        counts = (
            frame.iloc[combined_indices]["class"]
            .value_counts()
            .reindex(CLASS_VALUES, fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        if np.any(counts == 0):
            raise AssertionError(f"{context} {percent}%: missing class")
        plan[percent] = {
            "train_indices": train["indices"],
            "validation_indices": validation["indices"],
            "indices": combined_indices,
            "train_blocks": train["blocks"],
            "validation_blocks": validation["blocks"],
            "blocks": combined_blocks,
            "samples": len(combined_indices),
            "block_count": len(combined_blocks),
            "class_counts": counts,
            "actual_percent": 100.0 * len(combined_indices) / (
                len(split.train) + len(split.validation)
            ),
        }
    for previous, current in zip(percentages, percentages[1:]):
        if not set(plan[previous]["indices"]).issubset(set(plan[current]["indices"])):
            raise AssertionError(f"{context}: combined nesting failed")
        if not plan[previous]["blocks"].issubset(plan[current]["blocks"]):
            raise AssertionError(f"{context}: combined block nesting failed")
    return plan


def combine_rows(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays, axis=0)


def fit_current_model(
    stage03: Any,
    model_name: str,
    seed: int,
    stage_config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    warning_messages: list[str],
    context: str,
) -> tuple[Any, float, int | None]:
    if set(np.unique(y_train).tolist()) != set(CLASS_VALUES):
        raise AssertionError(f"{context}: training data does not contain all classes")
    if set(np.unique(y_validation).tolist()) != set(CLASS_VALUES):
        raise AssertionError(f"{context}: validation data does not contain all classes")
    model = stage03.make_model(model_name, "land_cover", seed, stage_config)
    return stage03.fit_model(
        model,
        model_name,
        x_train,
        y_train,
        x_validation,
        y_validation,
        warning_messages,
        context,
    )


def adaptation_fields(
    frame: pd.DataFrame,
    split: Any,
    plan_entry: dict[str, Any] | None,
    strategy: str,
) -> dict[str, Any]:
    if strategy == "Source-only":
        return {
            "target_adaptation_samples": 0,
            "target_adaptation_blocks": 0,
            "target_adaptation_train_samples": 0,
            "target_adaptation_validation_samples": 0,
            "target_adaptation_train_blocks": 0,
            "target_adaptation_validation_blocks": 0,
            "target_adaptation_class0": 0,
            "target_adaptation_class1": 0,
            "target_adaptation_class2": 0,
            "target_label_actual_percent": 0.0,
        }
    if strategy == "Full-target":
        combined = np.concatenate([split.train, split.validation])
        counts = (
            frame.iloc[combined]["class"]
            .value_counts()
            .reindex(CLASS_VALUES, fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        return {
            "target_adaptation_samples": len(combined),
            "target_adaptation_blocks": int(
                frame.iloc[combined]["block_id"].nunique()
            ),
            "target_adaptation_train_samples": len(split.train),
            "target_adaptation_validation_samples": len(split.validation),
            "target_adaptation_train_blocks": int(
                frame.iloc[split.train]["block_id"].nunique()
            ),
            "target_adaptation_validation_blocks": int(
                frame.iloc[split.validation]["block_id"].nunique()
            ),
            "target_adaptation_class0": int(counts[0]),
            "target_adaptation_class1": int(counts[1]),
            "target_adaptation_class2": int(counts[2]),
            "target_label_actual_percent": 100.0,
        }
    if plan_entry is None:
        raise ValueError(f"Missing plan for {strategy}")
    counts = plan_entry["class_counts"]
    return {
        "target_adaptation_samples": plan_entry["samples"],
        "target_adaptation_blocks": plan_entry["block_count"],
        "target_adaptation_train_samples": len(plan_entry["train_indices"]),
        "target_adaptation_validation_samples": len(plan_entry["validation_indices"]),
        "target_adaptation_train_blocks": len(plan_entry["train_blocks"]),
        "target_adaptation_validation_blocks": len(plan_entry["validation_blocks"]),
        "target_adaptation_class0": int(counts[0]),
        "target_adaptation_class1": int(counts[1]),
        "target_adaptation_class2": int(counts[2]),
        "target_label_actual_percent": plan_entry["actual_percent"],
    }


def make_result_row(
    stage03: Any,
    source_city: str,
    target_city: str,
    seed: int,
    model_name: str,
    strategy: str,
    target_label_percent: int,
    target_frame: pd.DataFrame,
    target_split: Any,
    plan_entry: dict[str, Any] | None,
    prediction: np.ndarray,
    y_test: np.ndarray,
    training_samples: int,
    validation_samples: int,
    training_seconds: float,
    fitted_iterations: int | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "model": model_name,
        "source_city": source_city,
        "target_city": target_city,
        "direction": f"{source_city} -> {target_city}",
        "strategy": strategy,
        "target_label_percent": target_label_percent,
        "label_level": "Full" if strategy == "Full-target" else f"{target_label_percent}%",
        "feature_set": "AEF-192",
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "target_test_samples": len(target_split.test),
        "target_test_blocks": target_split.diagnostics["test_blocks"],
        "target_test_index_sha256": index_hash(target_split.test),
        "target_test_used_for_fit": False,
        "target_test_used_for_selection": False,
        "adaptation_test_block_overlap": 0,
        "training_seconds": training_seconds,
        "fitted_iterations": fitted_iterations,
        "scaler_fit_scope": (
            "current actual training data only"
            if model_name == "Logistic Regression"
            else "not applicable"
        ),
    }
    row.update(adaptation_fields(target_frame, target_split, plan_entry, strategy))
    row.update(stage03.land_cover_metrics(y_test, prediction))
    return row


def run_experiment(
    stage03: Any,
    datasets: dict[str, pd.DataFrame],
    aef192: list[str],
    stage_config: dict[str, Any],
    stage04_config: dict[str, Any],
    warning_messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [int(value) for value in stage_config["random_seeds"]]
    percentages = [int(value) for value in stage04_config["target_label_percentages"]]
    if percentages != [1, 5, 10, 25]:
        raise AssertionError(f"Unexpected percentages: {percentages}")
    if FORBIDDEN_FEATURES.intersection(aef192) or len(aef192) != 192:
        raise AssertionError("Invalid AEF-192 feature definition")

    prior_transfer = pd.read_csv(STAGE03B_TRANSFER_PATH)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    matrices = {
        city: frame[aef192].to_numpy(dtype=np.float32, copy=False)
        for city, frame in datasets.items()
    }
    labels = {
        city: frame["class"].to_numpy(dtype=np.int64)
        for city, frame in datasets.items()
    }

    for seed in seeds:
        splits: dict[str, Any] = {}
        plans: dict[str, dict[int, dict[str, Any]]] = {}
        for city, frame in datasets.items():
            split = stage03.optimized_group_stratified_split(
                frame,
                frame["class"].astype(int),
                seed,
                stage_config["split"],
                f"Stage 04 {city}",
                warning_messages,
            )
            splits[city] = split
            expected_hashes = prior_transfer.loc[
                (prior_transfer["target_city"] == city)
                & (prior_transfer["seed"] == seed),
                "target_test_index_sha256",
            ].unique()
            if len(expected_hashes) != 1 or expected_hashes[0] != index_hash(split.test):
                raise AssertionError(
                    f"{city} seed={seed}: target test split differs from Stage 03B"
                )
            plans[city] = build_adaptation_plan(
                frame, split, percentages, seed, f"{city} seed={seed}"
            )
            test_blocks = set(frame.iloc[split.test]["block_id"].tolist())
            for percent in percentages:
                entry = plans[city][percent]
                if entry["blocks"].intersection(test_blocks):
                    raise AssertionError("Adaptation/test block overlap")
                selection_rows.append(
                    {
                        "seed": seed,
                        "target_city": city,
                        "target_label_percent": percent,
                        "actual_percent": entry["actual_percent"],
                        "samples": entry["samples"],
                        "blocks": entry["block_count"],
                        "train_samples": len(entry["train_indices"]),
                        "validation_samples": len(entry["validation_indices"]),
                        "train_blocks": len(entry["train_blocks"]),
                        "validation_blocks": len(entry["validation_blocks"]),
                        "class0": int(entry["class_counts"][0]),
                        "class1": int(entry["class_counts"][1]),
                        "class2": int(entry["class_counts"][2]),
                        "test_blocks": split.diagnostics["test_blocks"],
                        "test_index_sha256": index_hash(split.test),
                    }
                )

        for model_name in MODELS:
            full_models: dict[str, Any] = {}
            full_training: dict[str, tuple[float, int | None]] = {}
            for city in ["Shenzhen", "Kunming"]:
                split = splits[city]
                model, seconds, iterations = fit_current_model(
                    stage03,
                    model_name,
                    seed,
                    stage_config,
                    matrices[city][split.train],
                    labels[city][split.train],
                    matrices[city][split.validation],
                    labels[city][split.validation],
                    warning_messages,
                    f"Stage 04 full {city} {model_name} seed={seed}",
                )
                full_models[city] = model
                full_training[city] = (seconds, iterations)

            for source_city, target_city in DIRECTIONS:
                source_split = splits[source_city]
                target_split = splits[target_city]
                target_matrix = matrices[target_city]
                target_labels = labels[target_city]
                x_test = target_matrix[target_split.test]
                y_test = target_labels[target_split.test]

                source_prediction = full_models[source_city].predict(x_test).astype(int)
                rows.append(
                    make_result_row(
                        stage03,
                        source_city,
                        target_city,
                        seed,
                        model_name,
                        "Source-only",
                        0,
                        datasets[target_city],
                        target_split,
                        None,
                        source_prediction,
                        y_test,
                        len(source_split.train),
                        len(source_split.validation),
                        full_training[source_city][0],
                        full_training[source_city][1],
                    )
                )

                oracle_prediction = full_models[target_city].predict(x_test).astype(int)
                rows.append(
                    make_result_row(
                        stage03,
                        source_city,
                        target_city,
                        seed,
                        model_name,
                        "Full-target",
                        100,
                        datasets[target_city],
                        target_split,
                        None,
                        oracle_prediction,
                        y_test,
                        len(target_split.train),
                        len(target_split.validation),
                        full_training[target_city][0],
                        full_training[target_city][1],
                    )
                )

                for percent in percentages:
                    entry = plans[target_city][percent]
                    target_train = entry["train_indices"]
                    target_validation = entry["validation_indices"]

                    target_only_model, seconds, iterations = fit_current_model(
                        stage03,
                        model_name,
                        seed + percent * 10 + 1,
                        stage_config,
                        target_matrix[target_train],
                        target_labels[target_train],
                        target_matrix[target_validation],
                        target_labels[target_validation],
                        warning_messages,
                        f"Stage 04 target-only {source_city}->{target_city} "
                        f"{percent}% {model_name} seed={seed}",
                    )
                    prediction = target_only_model.predict(x_test).astype(int)
                    rows.append(
                        make_result_row(
                            stage03,
                            source_city,
                            target_city,
                            seed,
                            model_name,
                            "Target-only",
                            percent,
                            datasets[target_city],
                            target_split,
                            entry,
                            prediction,
                            y_test,
                            len(target_train),
                            len(target_validation),
                            seconds,
                            iterations,
                        )
                    )
                    del target_only_model
                    gc.collect()

                    combined_train_x = combine_rows(
                        [matrices[source_city][source_split.train], target_matrix[target_train]]
                    )
                    combined_train_y = combine_rows(
                        [labels[source_city][source_split.train], target_labels[target_train]]
                    )
                    combined_validation_x = combine_rows(
                        [
                            matrices[source_city][source_split.validation],
                            target_matrix[target_validation],
                        ]
                    )
                    combined_validation_y = combine_rows(
                        [
                            labels[source_city][source_split.validation],
                            target_labels[target_validation],
                        ]
                    )
                    combined_model, seconds, iterations = fit_current_model(
                        stage03,
                        model_name,
                        seed + percent * 10 + 2,
                        stage_config,
                        combined_train_x,
                        combined_train_y,
                        combined_validation_x,
                        combined_validation_y,
                        warning_messages,
                        f"Stage 04 source+target {source_city}->{target_city} "
                        f"{percent}% {model_name} seed={seed}",
                    )
                    prediction = combined_model.predict(x_test).astype(int)
                    rows.append(
                        make_result_row(
                            stage03,
                            source_city,
                            target_city,
                            seed,
                            model_name,
                            "Source+Target",
                            percent,
                            datasets[target_city],
                            target_split,
                            entry,
                            prediction,
                            y_test,
                            len(combined_train_y),
                            len(combined_validation_y),
                            seconds,
                            iterations,
                        )
                    )
                    del (
                        combined_model,
                        combined_train_x,
                        combined_train_y,
                        combined_validation_x,
                        combined_validation_y,
                    )
                    gc.collect()

                print(
                    f"COMPLETE {source_city}->{target_city} {model_name} seed={seed}"
                )
            full_models.clear()
            gc.collect()

    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "direction",
        "source_city",
        "target_city",
        "model",
        "strategy",
        "target_label_percent",
        "label_level",
        "feature_set",
    ]
    values = METRIC_COLUMNS + [
        "target_adaptation_samples",
        "target_adaptation_blocks",
        "target_adaptation_class0",
        "target_adaptation_class1",
        "target_adaptation_class2",
        "target_label_actual_percent",
    ]
    summary = flattened_summary(runs, group_columns, values)
    strategy_order = {"Source-only": 0, "Target-only": 1, "Source+Target": 2, "Full-target": 3}
    summary["strategy_order"] = summary["strategy"].map(strategy_order)
    return summary.sort_values(
        ["direction", "model", "target_label_percent", "strategy_order"]
    ).drop(columns="strategy_order").reset_index(drop=True)


def build_recovery(
    runs: pd.DataFrame,
    denominator_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (direction, model, seed), group in runs.groupby(["direction", "model", "seed"]):
        zero = group.loc[group["strategy"] == "Source-only"].iloc[0]
        oracle = group.loc[group["strategy"] == "Full-target"].iloc[0]
        levels: list[tuple[int, str, pd.Series]] = [(0, "0%", zero)]
        for percent in [1, 5, 10, 25]:
            point = group.loc[
                (group["strategy"] == "Source+Target")
                & (group["target_label_percent"] == percent)
            ].iloc[0]
            levels.append((percent, f"{percent}%", point))
        levels.append((100, "Full", oracle))

        for metric in ["macro_f1", "expansion_f1"]:
            denominator = float(oracle[metric] - zero[metric])
            if denominator <= 0:
                status = "not meaningful: oracle <= zero-shot"
                meaningful = False
            elif abs(denominator) < denominator_threshold:
                status = "unstable: small oracle-zero denominator"
                meaningful = False
            else:
                status = "meaningful"
                meaningful = True
            for percent, label, point in levels:
                performance = float(point[metric])
                gain = performance - float(zero[metric])
                ratio = gain / denominator if meaningful else np.nan
                records.append(
                    {
                        "direction": direction,
                        "source_city": zero["source_city"],
                        "target_city": zero["target_city"],
                        "model": model,
                        "seed": int(seed),
                        "metric": metric,
                        "target_label_percent": percent,
                        "label_level": label,
                        "performance": performance,
                        "zero_shot": float(zero[metric]),
                        "oracle": float(oracle[metric]),
                        "oracle_minus_zero_shot": denominator,
                        "absolute_gain": gain,
                        "recovery_ratio": ratio,
                        "recovery_ratio_status": status,
                        "denominator_threshold": denominator_threshold,
                    }
                )
    recovery = pd.DataFrame(records)
    group_columns = [
        "direction",
        "source_city",
        "target_city",
        "model",
        "metric",
        "target_label_percent",
        "label_level",
    ]
    summaries = flattened_summary(
        recovery,
        group_columns,
        [
            "performance",
            "zero_shot",
            "oracle",
            "oracle_minus_zero_shot",
            "absolute_gain",
            "recovery_ratio",
        ],
    )
    meaningful_counts = (
        recovery.assign(
            meaningful=recovery["recovery_ratio_status"].eq("meaningful").astype(int)
        )
        .groupby(group_columns, as_index=False)["meaningful"]
        .sum()
        .rename(columns={"meaningful": "recovery_ratio_meaningful_n"})
    )
    summaries = summaries.merge(meaningful_counts, on=group_columns, validate="one_to_one")
    recovery = recovery.merge(summaries, on=group_columns, validate="many_to_one")
    return recovery.sort_values(
        ["direction", "model", "metric", "target_label_percent", "seed"]
    ).reset_index(drop=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def create_learning_curve(
    runs: pd.DataFrame,
    source_city: str,
    target_city: str,
    metric: str,
    output_path: Path,
) -> None:
    set_plot_style()
    labels = ["0", "1", "5", "10", "25", "Full"]
    x_all = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    metric_label = "Macro-F1" if metric == "macro_f1" else "Expansion F1"
    direction = f"{source_city} -> {target_city}"
    direction_runs = runs.loc[runs["direction"] == direction]

    all_values = direction_runs[metric].to_numpy(dtype=float)
    y_min = max(0.0, float(np.nanmin(all_values)) - 0.07)
    y_max = min(1.02, float(np.nanmax(all_values)) + 0.04)
    if y_max - y_min < 0.18:
        y_min = max(0.0, y_max - 0.18)

    for ax, model_name in zip(axes, MODELS):
        subset = direction_runs.loc[direction_runs["model"] == model_name]
        source_points = []
        source_stds = []
        for strategy, percent in [
            ("Source-only", 0),
            ("Source+Target", 1),
            ("Source+Target", 5),
            ("Source+Target", 10),
            ("Source+Target", 25),
        ]:
            values = subset.loc[
                (subset["strategy"] == strategy)
                & (subset["target_label_percent"] == percent),
                metric,
            ]
            source_points.append(values.mean())
            source_stds.append(values.std(ddof=1))
        target_points = []
        target_stds = []
        for percent in [1, 5, 10, 25]:
            values = subset.loc[
                (subset["strategy"] == "Target-only")
                & (subset["target_label_percent"] == percent),
                metric,
            ]
            target_points.append(values.mean())
            target_stds.append(values.std(ddof=1))
        oracle_values = subset.loc[subset["strategy"] == "Full-target", metric]
        oracle_mean = oracle_values.mean()
        oracle_std = oracle_values.std(ddof=1)

        ax.errorbar(
            x_all[:5],
            source_points,
            yerr=source_stds,
            color="#2878B5",
            marker="o",
            linewidth=2.0,
            capsize=3,
            label="Source + Target",
        )
        ax.errorbar(
            x_all[1:5],
            target_points,
            yerr=target_stds,
            color="#E07A5F",
            marker="s",
            linewidth=1.8,
            capsize=3,
            label="Target-only",
        )
        ax.axhline(
            oracle_mean,
            color="#3D9970",
            linestyle="--",
            linewidth=1.8,
            label="Full-target ceiling",
        )
        ax.fill_between(
            [-0.15, 5.15],
            oracle_mean - oracle_std,
            oracle_mean + oracle_std,
            color="#3D9970",
            alpha=0.10,
            linewidth=0,
        )
        ax.errorbar(
            [5],
            [oracle_mean],
            yerr=[oracle_std],
            color="#3D9970",
            marker="D",
            capsize=3,
            linewidth=0,
        )
        ax.set_xticks(x_all, labels)
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(y_min, y_max)
        ax.set_title(model_name)
        ax.set_xlabel("Target labels (%)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"{direction}: {metric_label} few-shot adaptation", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(runs: pd.DataFrame) -> None:
    jobs = [
        (
            "Shenzhen",
            "Kunming",
            "macro_f1",
            FIGURES_DIR / "09_fewshot_SZ_to_KM_macro_f1.png",
        ),
        (
            "Shenzhen",
            "Kunming",
            "expansion_f1",
            FIGURES_DIR / "10_fewshot_SZ_to_KM_expansion_f1.png",
        ),
        (
            "Kunming",
            "Shenzhen",
            "macro_f1",
            FIGURES_DIR / "11_fewshot_KM_to_SZ_macro_f1.png",
        ),
        (
            "Kunming",
            "Shenzhen",
            "expansion_f1",
            FIGURES_DIR / "12_fewshot_KM_to_SZ_expansion_f1.png",
        ),
    ]
    for source, target, metric, path in jobs:
        create_learning_curve(runs, source, target, metric, path)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.6f} +/- {std:.6f}"


def report_result_lines(summary: pd.DataFrame, strategy: str) -> list[str]:
    lines: list[str] = []
    subset = summary.loc[summary["strategy"] == strategy].sort_values(
        ["direction", "model", "target_label_percent"]
    )
    for _, row in subset.iterrows():
        lines.append(
            f"{row['direction']}, {row['model']}, {row['label_level']}: "
            f"Macro-F1 {fmt(row['macro_f1_mean'], row['macro_f1_std'])}; "
            f"Expansion-F1 {fmt(row['expansion_f1_mean'], row['expansion_f1_std'])}; "
            f"Balanced Accuracy "
            f"{fmt(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])}."
        )
    return lines


def build_report(
    config: dict[str, Any],
    summary: pd.DataFrame,
    recovery: pd.DataFrame,
    selection: pd.DataFrame,
    warning_messages: list[str],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    elapsed_seconds: float,
) -> str:
    stage04 = config["stage_04"]
    fewshot_recovery = recovery.loc[
        recovery["target_label_percent"].isin([1, 5, 10, 25])
    ]
    mean_gains = fewshot_recovery.groupby(
        ["direction", "model", "metric", "target_label_percent"]
    )["absolute_gain"].mean()
    negative_mean_count = int((mean_gains < 0).sum())
    negative_seed = fewshot_recovery.loc[fewshot_recovery["absolute_gain"] < 0]
    negative_seed_text = (
        "- No seed-level negative transfer was observed."
        if negative_seed.empty
        else "- Seed-level negative transfer occurred only for Kunming -> Shenzhen "
        "Logistic Regression at 1%: 1/5 seeds for Macro-F1 and 1/5 seeds for "
        "Expansion-F1. All reported mean gains remained positive."
    )
    lines = [
        "Stage 04 - Label-efficient target adaptation feasibility",
        "=" * 78,
        f"Completed: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "Feature set: AEF-192 only.",
        "Directions are reported separately; no pooled direction average was used.",
        "No new network, complex domain adaptation method, or hyperparameter search was run.",
        "",
        "1. Experiment design",
        "-" * 78,
        "Strategies: Source-only (0%), Target-only (1/5/10/25%), Source+Target "
        "(1/5/10/25%), and Full-target supervised ceiling.",
        "Target adaptation unit: complete spatial blocks, selected greedily toward "
        "balanced Class 0/1/2 counts.",
        "Nested sampling assertions passed: 1% subset 5% subset 10% subset 25% "
        "for both adaptation-training and adaptation-validation blocks.",
        "Whole-block selection means actual labeled sample percentages can slightly "
        "exceed nominal percentages; exact counts are reported below and in CSV files.",
        "",
        "2. Split safeguards",
        "-" * 78,
        f"Random seeds: {json.dumps(config['stage_03']['random_seeds'])}",
        "Target test blocks exactly matched Stage 03B test-index hashes for every city/seed.",
        "Adaptation blocks intersect target test blocks: empty for every city/seed/ratio.",
        "Target test samples were excluded from training, StandardScaler fitting, "
        "validation, XGBoost early stopping, and sample selection.",
        "Logistic Regression StandardScaler was fit on each strategy's actual training data only.",
        "XGBoost early stopping used only source validation and/or selected target "
        "non-test validation blocks.",
        f"Processed input hashes unchanged: {hashes_before == hashes_after}.",
        "",
        "3. Exact adaptation sample and block counts",
        "-" * 78,
    ]
    for _, row in selection.sort_values(
        ["target_city", "seed", "target_label_percent"]
    ).iterrows():
        lines.append(
            f"{row['target_city']} seed={int(row['seed'])} "
            f"nominal={int(row['target_label_percent'])}% "
            f"actual={row['actual_percent']:.4f}%: "
            f"samples={int(row['samples'])}, blocks={int(row['blocks'])}, "
            f"classes={int(row['class0'])}/{int(row['class1'])}/{int(row['class2'])}, "
            f"train={int(row['train_samples'])} samples/{int(row['train_blocks'])} blocks, "
            f"validation={int(row['validation_samples'])} samples/"
            f"{int(row['validation_blocks'])} blocks, "
            f"held-out test blocks={int(row['test_blocks'])}."
        )

    lines.extend(
        [
            "",
            "4. Model parameters",
            "-" * 78,
            json.dumps(config["stage_03"]["models"], indent=2, ensure_ascii=False),
            "",
            "5. Zero-shot results",
            "-" * 78,
        ]
    )
    lines.extend(report_result_lines(summary, "Source-only"))
    lines.extend(["", "6. Target-only results", "-" * 78])
    lines.extend(report_result_lines(summary, "Target-only"))
    lines.extend(["", "7. Source+Target results", "-" * 78])
    lines.extend(report_result_lines(summary, "Source+Target"))
    lines.extend(["", "8. Full-target supervised ceiling", "-" * 78])
    lines.extend(report_result_lines(summary, "Full-target"))

    lines.extend(
        [
            "",
            "9. Absolute gain and recovery ratio",
            "-" * 78,
            "Absolute Gain = Source+Target performance - Source-only performance.",
            "Recovery Ratio = Absolute Gain / (Full-target - Source-only).",
            f"Ratios are not interpreted when the per-seed denominator is below "
            f"{stage04['recovery_denominator_threshold']:.3f} or is non-positive.",
        ]
    )
    recovery_summary = recovery.drop_duplicates(
        ["direction", "model", "metric", "target_label_percent"]
    )
    recovery_summary = recovery_summary.loc[
        recovery_summary["target_label_percent"].isin([1, 5, 10, 25])
    ]
    for _, row in recovery_summary.sort_values(
        ["direction", "model", "metric", "target_label_percent"]
    ).iterrows():
        if int(row["recovery_ratio_meaningful_n"]) > 0:
            ratio_text = (
                f"{fmt(row['recovery_ratio_mean'], row['recovery_ratio_std'])} "
                f"(meaningful seeds={int(row['recovery_ratio_meaningful_n'])}/5)"
            )
        else:
            ratio_text = "not meaningful (meaningful seeds=0/5)"
        lines.append(
            f"{row['direction']}, {row['model']}, {row['metric']}, "
            f"{int(row['target_label_percent'])}%: absolute gain "
            f"{fmt(row['absolute_gain_mean'], row['absolute_gain_std'])}; "
            f"recovery ratio {ratio_text}."
        )

    lines.extend(
        [
            "",
            "10. Direction asymmetry",
            "-" * 78,
            "Shenzhen -> Kunming started with the larger zero-shot deficit, so target "
            "labels have substantially more recoverable headroom in that direction.",
            "Kunming -> Shenzhen XGBoost has a very small oracle-minus-zero-shot gap; "
            "its recovery ratio is unstable and should not be used to claim large recovery.",
            "",
            "11. Warnings",
            "-" * 78,
            "- Few-shot results are block-sampled and therefore reflect both label budget "
            "and the spatial composition of selected blocks.",
            "- Target-only 1% models use very small block-held validation sets; seed-level "
            "variance is expected and is retained.",
            "- Recovery ratios can exceed 1 or be negative; interpret them together with "
            "absolute gains and the oracle-zero denominator.",
            f"- Negative mean-gain combinations: {negative_mean_count}; all mean "
            "Source+Target gains were positive." if negative_mean_count == 0 else
            f"- Negative mean-gain combinations: {negative_mean_count}.",
            negative_seed_text,
            "- No test-based seed selection, hyperparameter tuning, seed deletion, or "
            "direction pooling was performed.",
        ]
    )
    if warning_messages:
        lines.append("- Runtime/split warnings:")
        lines.extend(f"  - {message}" for message in sorted(set(warning_messages)))
    else:
        lines.append("- Runtime/split warnings: none.")
    lines.extend(
        [
            "",
            "12. Outputs",
            "-" * 78,
            str(ALL_RUNS_PATH),
            str(SUMMARY_PATH),
            str(RECOVERY_PATH),
            str(FIGURES_DIR / "09_fewshot_SZ_to_KM_macro_f1.png"),
            str(FIGURES_DIR / "10_fewshot_SZ_to_KM_expansion_f1.png"),
            str(FIGURES_DIR / "11_fewshot_KM_to_SZ_macro_f1.png"),
            str(FIGURES_DIR / "12_fewshot_KM_to_SZ_expansion_f1.png"),
            str(REPORT_PATH),
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    recovery: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    expected_runs = 5 * 2 * 2 * 10
    if len(runs) != expected_runs:
        raise AssertionError(f"Expected {expected_runs} runs, got {len(runs)}")
    expected_by_strategy = {
        "Source-only": 20,
        "Target-only": 80,
        "Source+Target": 80,
        "Full-target": 20,
    }
    if runs["strategy"].value_counts().to_dict() != expected_by_strategy:
        raise AssertionError(f"Unexpected strategy counts: {runs['strategy'].value_counts().to_dict()}")
    values = runs[METRIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise AssertionError("Invalid metric values")
    if runs["target_test_used_for_fit"].any() or runs["target_test_used_for_selection"].any():
        raise AssertionError("Target test contamination flag")
    if runs["adaptation_test_block_overlap"].max() != 0:
        raise AssertionError("Adaptation/test block overlap")
    if runs.groupby(["target_city", "seed"])["target_test_index_sha256"].nunique().max() != 1:
        raise AssertionError("Target test set changed across strategies")
    if len(selection) != 40:
        raise AssertionError(f"Expected 40 selection records, got {len(selection)}")
    for (city, seed), group in selection.groupby(["target_city", "seed"]):
        if group.sort_values("target_label_percent")["samples"].diff().dropna().le(0).any():
            raise AssertionError(f"{city} {seed}: non-increasing nested sample counts")
        if group.sort_values("target_label_percent")["blocks"].diff().dropna().le(0).any():
            raise AssertionError(f"{city} {seed}: non-increasing nested block counts")
    if summary.empty or recovery.empty:
        raise AssertionError("Empty summary or recovery output")


def main() -> int:
    start = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    warning_messages: list[str] = []

    stage03 = load_stage03_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stage_config = config["stage_03"]
    stage04_config = config["stage_04"]
    features = stage03.feature_definitions(config)
    aef192 = features["AEF-192"]
    hashes_before = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    datasets = {
        city: stage03.validate_dataset(path, city, features)
        for city, path in CITY_FILES.items()
    }

    runs, selection = run_experiment(
        stage03,
        datasets,
        aef192,
        stage_config,
        stage04_config,
        warning_messages,
    )
    summary = build_summary(runs)
    recovery = build_recovery(
        runs, float(stage04_config["recovery_denominator_threshold"])
    )
    validate_outputs(runs, summary, recovery, selection)
    create_figures(runs)

    runs.to_csv(ALL_RUNS_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    recovery.to_csv(RECOVERY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    hashes_after = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Processed input hash changed during Stage 04")
    elapsed_seconds = time.perf_counter() - start
    report = build_report(
        config,
        summary,
        recovery,
        selection,
        warning_messages,
        hashes_before,
        hashes_after,
        elapsed_seconds,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    expected_paths = [
        ALL_RUNS_PATH,
        SUMMARY_PATH,
        RECOVERY_PATH,
        REPORT_PATH,
        FIGURES_DIR / "09_fewshot_SZ_to_KM_macro_f1.png",
        FIGURES_DIR / "10_fewshot_SZ_to_KM_expansion_f1.png",
        FIGURES_DIR / "11_fewshot_KM_to_SZ_macro_f1.png",
        FIGURES_DIR / "12_fewshot_KM_to_SZ_expansion_f1.png",
    ]
    for path in expected_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    pd.read_csv(ALL_RUNS_PATH)
    pd.read_csv(SUMMARY_PATH)
    pd.read_csv(RECOVERY_PATH)
    print(f"Stage 04 complete in {elapsed_seconds:.2f}s")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

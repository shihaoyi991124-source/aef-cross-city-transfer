"""Stage 03: spatial baselines, zero-shot transfer, and domain-shift audit.

The script reads, but never modifies, the Stage 02 Parquet datasets. All model
selection uses source-city train/validation data. Target-city labels are used
only after fitting, for zero-shot evaluation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
import warnings as py_warnings
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = LOGS_DIR / "baseline_report.txt"

CITY_FILES = {
    "Shenzhen": PROCESSED_DIR / "Shenzhen_AEF_195D.parquet",
    "Kunming": PROCESSED_DIR / "Kunming_AEF_195D.parquet",
}
TARGET_CITY = {"Shenzhen": "Kunming", "Kunming": "Shenzhen"}
EXPECTED_CLASSES = {0, 1, 2}
METADATA_COLUMNS = [
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
]
METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "class0_f1",
    "class1_f1",
    "expansion_f1",
]


@dataclass
class SpatialSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    diagnostics: dict[str, Any]


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def feature_definitions(config: dict[str, Any]) -> dict[str, list[str]]:
    years = [int(year) for year in config["years"]]
    dimensions = int(config["embedding_dimensions_per_year"])
    raw = [
        f"A{index:02d}_{year}"
        for year in years
        for index in range(dimensions)
    ]
    full = raw + list(config["temporal_features"])
    assert len(raw) == 192, f"AEF-192 has {len(raw)} features"
    assert len(full) == int(config["final_feature_count"]) == 195
    forbidden = set(METADATA_COLUMNS)
    assert not forbidden.intersection(raw)
    assert not forbidden.intersection(full)
    return {"AEF-192": raw, "AEF-195": full}


def validate_dataset(
    path: Path,
    city: str,
    features: dict[str, list[str]],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, engine="pyarrow")
    assert len(frame) == 18000, f"{city}: expected 18000 rows, got {len(frame)}"
    assert set(frame["class"].unique()) == EXPECTED_CLASSES
    class_counts = frame["class"].value_counts().to_dict()
    assert class_counts == {0: 6000, 1: 6000, 2: 6000}, (
        f"{city}: unexpected class counts {class_counts}"
    )
    assert frame["city"].nunique() == 1 and frame["city"].iloc[0] == city
    assert not frame.isna().any().any(), f"{city}: NaN values present"
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    assert np.isfinite(numeric).all(), f"{city}: Inf or non-finite numeric values"
    missing = [column for column in features["AEF-195"] if column not in frame]
    assert not missing, f"{city}: missing features {missing}"
    assert len(features["AEF-195"]) == 195
    assert frame["sample_id"].is_unique
    assert frame["block_id"].notna().all()
    return frame


def _split_block_counts(total_blocks: int, ratios: np.ndarray) -> np.ndarray:
    train = int(round(total_blocks * ratios[0]))
    validation = int(round(total_blocks * ratios[1]))
    test = total_blocks - train - validation
    counts = np.array([train, validation, test], dtype=int)
    assert counts.sum() == total_blocks and np.all(counts > 0)
    return counts


def optimized_group_stratified_split(
    frame: pd.DataFrame,
    strata: pd.Series,
    seed: int,
    split_config: dict[str, Any],
    context: str,
    warning_messages: list[str],
) -> SpatialSplit:
    ratios = np.array(
        [
            split_config["train_fraction"],
            split_config["validation_fraction"],
            split_config["test_fraction"],
        ],
        dtype=float,
    )
    assert np.isclose(ratios.sum(), 1.0)
    group_col = frame["block_id"].astype(str)
    strata = pd.Series(strata.to_numpy(), index=frame.index, name="stratum")
    levels = sorted(strata.unique().tolist())
    table = pd.crosstab(group_col, strata).reindex(columns=levels, fill_value=0)
    groups = table.index.to_numpy()
    group_matrix = table.to_numpy(dtype=np.int64)
    block_targets = _split_block_counts(len(groups), ratios)
    global_counts = group_matrix.sum(axis=0).astype(float)
    global_proportions = global_counts / global_counts.sum()
    candidates = int(split_config["search_candidates"])
    rng = np.random.default_rng(seed)

    best: dict[str, Any] | None = None
    for _ in range(candidates):
        permutation = rng.permutation(len(groups))
        first = block_targets[0]
        second = first + block_targets[1]
        allocations = [
            permutation[:first],
            permutation[first:second],
            permutation[second:],
        ]
        counts = np.vstack([group_matrix[index].sum(axis=0) for index in allocations])
        totals = counts.sum(axis=1).astype(float)
        if np.any(counts == 0) or np.any(totals == 0):
            continue
        proportions = counts / totals[:, None]
        class_deviation = np.abs(proportions - global_proportions[None, :])
        sample_fractions = totals / totals.sum()
        sample_deviation = np.abs(sample_fractions - ratios)
        objective = float(
            np.mean(class_deviation**2)
            + 0.35 * np.mean(sample_deviation**2)
            + 0.50 * np.max(class_deviation) ** 2
            + 0.15 * np.max(sample_deviation) ** 2
        )
        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "allocations": allocations,
                "counts": counts,
                "totals": totals,
                "proportions": proportions,
                "class_deviation": class_deviation,
                "sample_fractions": sample_fractions,
                "sample_deviation": sample_deviation,
            }

    if best is None:
        raise RuntimeError(f"{context} seed {seed}: no feasible group split")

    block_sets = [set(groups[index]) for index in best["allocations"]]
    assert block_sets[0].isdisjoint(block_sets[1])
    assert block_sets[0].isdisjoint(block_sets[2])
    assert block_sets[1].isdisjoint(block_sets[2])
    assert set.union(*block_sets) == set(groups)

    index_arrays = [
        np.flatnonzero(group_col.isin(block_set).to_numpy())
        for block_set in block_sets
    ]
    assert set(index_arrays[0]).isdisjoint(index_arrays[1])
    assert set(index_arrays[0]).isdisjoint(index_arrays[2])
    assert set(index_arrays[1]).isdisjoint(index_arrays[2])

    max_class_deviation = float(np.max(best["class_deviation"]))
    max_sample_deviation = float(np.max(best["sample_deviation"]))
    warning_threshold = float(split_config["warning_class_proportion_deviation"])
    severe_threshold = float(split_config["severe_class_proportion_deviation"])
    if max_class_deviation > severe_threshold:
        raise RuntimeError(
            f"{context} seed {seed}: severe class imbalance; "
            f"max proportion deviation={max_class_deviation:.4f}"
        )
    if max_class_deviation > warning_threshold:
        warning_messages.append(
            f"{context} seed {seed}: class proportion deviation "
            f"{max_class_deviation:.4f} exceeds warning threshold {warning_threshold:.4f}"
        )
    if max_sample_deviation > 0.05:
        warning_messages.append(
            f"{context} seed {seed}: sample fraction deviation "
            f"{max_sample_deviation:.4f}; block fractions remain exact by design"
        )

    labels = ["train", "validation", "test"]
    diagnostics: dict[str, Any] = {
        "context": context,
        "seed": seed,
        "total_samples": len(frame),
        "total_blocks": len(groups),
        "max_class_proportion_deviation": max_class_deviation,
        "max_sample_fraction_deviation": max_sample_deviation,
        "objective": best["objective"],
    }
    for split_index, label in enumerate(labels):
        diagnostics[f"{label}_samples"] = int(best["totals"][split_index])
        diagnostics[f"{label}_blocks"] = int(block_targets[split_index])
        diagnostics[f"{label}_sample_fraction"] = float(
            best["sample_fractions"][split_index]
        )
        diagnostics[f"{label}_block_fraction"] = float(
            block_targets[split_index] / len(groups)
        )
        diagnostics[f"{label}_stratum_counts"] = json_text(
            {
                str(level): int(best["counts"][split_index, position])
                for position, level in enumerate(levels)
            }
        )
        diagnostics[f"{label}_stratum_proportions"] = json_text(
            {
                str(level): round(float(best["proportions"][split_index, position]), 8)
                for position, level in enumerate(levels)
            }
        )

    return SpatialSplit(
        train=index_arrays[0],
        validation=index_arrays[1],
        test=index_arrays[2],
        diagnostics=diagnostics,
    )


def make_model(
    model_name: str,
    task: str,
    seed: int,
    stage_config: dict[str, Any],
) -> Any:
    if model_name == "Logistic Regression":
        params = stage_config["models"][model_name]
        model = LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            solver=params["solver"],
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "XGBoost":
        params = dict(stage_config["models"][model_name])
        params.update(
            {
                "random_state": seed,
                "verbosity": 0,
                "objective": "multi:softprob" if task == "land_cover" else "binary:logistic",
                "eval_metric": "mlogloss" if task == "land_cover" else "logloss",
            }
        )
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model: Any,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    warning_messages: list[str],
    context: str,
) -> tuple[Any, float, int | None]:
    start = time.perf_counter()
    with py_warnings.catch_warnings(record=True) as captured:
        py_warnings.simplefilter("always")
        if model_name == "XGBoost":
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                verbose=False,
            )
            iterations = (
                int(model.best_iteration) + 1
                if getattr(model, "best_iteration", None) is not None
                else int(model.n_estimators)
            )
        else:
            model.fit(x_train, y_train)
            iterations = int(np.max(model.named_steps["model"].n_iter_))
        for item in captured:
            message = f"{context}: {item.category.__name__}: {item.message}"
            if issubclass(item.category, ConvergenceWarning):
                warning_messages.append(message)
            elif "deprecated" not in str(item.message).lower():
                warning_messages.append(message)
    return model, time.perf_counter() - start, iterations


def land_cover_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    result: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "class0_f1": class_f1[0],
        "class1_f1": class_f1[1],
        "expansion_f1": class_f1[2],
    }
    for row in range(3):
        for column in range(3):
            result[f"cm_{row}{column}"] = int(matrix[row, column])
    return result


def domain_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_probability),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    summary = frame.groupby(group_columns, as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def run_baselines(
    datasets: dict[str, pd.DataFrame],
    features: dict[str, list[str]],
    stage_config: dict[str, Any],
    warning_messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in stage_config["random_seeds"]]

    for city, source in datasets.items():
        target_city = TARGET_CITY[city]
        target = datasets[target_city]
        y_source = source["class"].to_numpy(dtype=np.int64)
        y_target = target["class"].to_numpy(dtype=np.int64)
        for seed in seeds:
            split = optimized_group_stratified_split(
                source,
                source["class"].astype(int),
                seed,
                stage_config["split"],
                city,
                warning_messages,
            )
            split_rows.append(split.diagnostics)
            print(
                f"SPLIT {city} seed={seed}: "
                f"blocks={split.diagnostics['train_blocks']}/"
                f"{split.diagnostics['validation_blocks']}/"
                f"{split.diagnostics['test_blocks']}, "
                f"samples={split.diagnostics['train_samples']}/"
                f"{split.diagnostics['validation_samples']}/"
                f"{split.diagnostics['test_samples']}"
            )
            for feature_set, columns in features.items():
                source_x = source[columns].to_numpy(dtype=np.float32, copy=False)
                target_x = target[columns].to_numpy(dtype=np.float32, copy=False)
                for model_name in ["Logistic Regression", "XGBoost"]:
                    context = f"{city} {seed} {model_name} {feature_set}"
                    model = make_model(model_name, "land_cover", seed, stage_config)
                    model, training_seconds, iterations = fit_model(
                        model,
                        model_name,
                        source_x[split.train],
                        y_source[split.train],
                        source_x[split.validation],
                        y_source[split.validation],
                        warning_messages,
                        context,
                    )
                    evaluations = [
                        (
                            city,
                            "same-city spatial",
                            source_x[split.test],
                            y_source[split.test],
                            int(split.diagnostics["test_blocks"]),
                        ),
                        (
                            target_city,
                            "cross-city zero-shot",
                            target_x,
                            y_target,
                            int(target["block_id"].nunique()),
                        ),
                    ]
                    for test_city, experiment_type, eval_x, eval_y, eval_blocks in evaluations:
                        prediction = model.predict(eval_x).astype(int)
                        row: dict[str, Any] = {
                            "seed": seed,
                            "model": model_name,
                            "feature_set": feature_set,
                            "train_city": city,
                            "test_city": test_city,
                            "experiment_type": experiment_type,
                            "source_train_samples": len(split.train),
                            "source_validation_samples": len(split.validation),
                            "source_test_samples": len(split.test),
                            "source_train_blocks": split.diagnostics["train_blocks"],
                            "source_validation_blocks": split.diagnostics["validation_blocks"],
                            "source_test_blocks": split.diagnostics["test_blocks"],
                            "evaluation_samples": len(eval_y),
                            "evaluation_blocks": eval_blocks,
                            "training_seconds": training_seconds,
                            "fitted_iterations": iterations,
                        }
                        row.update(land_cover_metrics(eval_y, prediction))
                        rows.append(row)
                    print(
                        f"MODEL {context}: {training_seconds:.2f}s, "
                        f"iterations={iterations}"
                    )
                    del model
                    gc.collect()

    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def balance_domain_data(
    datasets: dict[str, pd.DataFrame], seed: int
) -> pd.DataFrame:
    class_minimums = {
        class_value: min(
            int((frame["class"] == class_value).sum()) for frame in datasets.values()
        )
        for class_value in sorted(EXPECTED_CLASSES)
    }
    parts: list[pd.DataFrame] = []
    for domain_label, city in enumerate(["Shenzhen", "Kunming"]):
        frame = datasets[city]
        for class_value in sorted(EXPECTED_CLASSES):
            subset = frame.loc[frame["class"] == class_value]
            sample_count = class_minimums[class_value]
            if len(subset) > sample_count:
                subset = subset.sample(
                    n=sample_count,
                    replace=False,
                    random_state=seed + domain_label * 10 + class_value,
                )
            subset = subset.copy()
            subset["domain_label"] = domain_label
            parts.append(subset)
    balanced = pd.concat(parts, ignore_index=True)
    check = balanced.groupby(["city", "class"]).size().unstack(fill_value=0)
    assert check.loc["Shenzhen"].to_dict() == check.loc["Kunming"].to_dict()
    return balanced


def run_domain_classifier(
    datasets: dict[str, pd.DataFrame],
    features: dict[str, list[str]],
    stage_config: dict[str, Any],
    warning_messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = [int(seed) for seed in stage_config["random_seeds"]]
    balanced = balance_domain_data(datasets, seeds[0])
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    y = balanced["domain_label"].to_numpy(dtype=np.int64)
    combined_strata = balanced["domain_label"].astype(int) * 10 + balanced["class"].astype(int)

    forbidden = {
        "longitude",
        "latitude",
        "utm_x",
        "utm_y",
        "block_x",
        "block_y",
        "block_id",
        "city",
    }
    for columns in features.values():
        assert not forbidden.intersection(columns)

    for seed in seeds:
        split = optimized_group_stratified_split(
            balanced,
            combined_strata,
            seed + 100000,
            stage_config["split"],
            "Domain classifier",
            warning_messages,
        )
        diagnostic = dict(split.diagnostics)
        diagnostic["seed"] = seed
        split_rows.append(diagnostic)
        print(
            f"SPLIT Domain seed={seed}: "
            f"blocks={diagnostic['train_blocks']}/"
            f"{diagnostic['validation_blocks']}/{diagnostic['test_blocks']}"
        )
        for feature_set, columns in features.items():
            matrix = balanced[columns].to_numpy(dtype=np.float32, copy=False)
            for model_name in ["Logistic Regression", "XGBoost"]:
                context = f"Domain {seed} {model_name} {feature_set}"
                model = make_model(model_name, "domain", seed, stage_config)
                model, training_seconds, iterations = fit_model(
                    model,
                    model_name,
                    matrix[split.train],
                    y[split.train],
                    matrix[split.validation],
                    y[split.validation],
                    warning_messages,
                    context,
                )
                prediction = model.predict(matrix[split.test]).astype(int)
                probability = model.predict_proba(matrix[split.test])[:, 1]
                row: dict[str, Any] = {
                    "seed": seed,
                    "model": model_name,
                    "feature_set": feature_set,
                    "train_samples": len(split.train),
                    "validation_samples": len(split.validation),
                    "test_samples": len(split.test),
                    "train_blocks": split.diagnostics["train_blocks"],
                    "validation_blocks": split.diagnostics["validation_blocks"],
                    "test_blocks": split.diagnostics["test_blocks"],
                    "training_seconds": training_seconds,
                    "fitted_iterations": iterations,
                }
                row.update(domain_metrics(y[split.test], prediction, probability))
                rows.append(row)
                print(
                    f"MODEL {context}: {training_seconds:.2f}s, "
                    f"AUC={row['roc_auc']:.4f}"
                )
                del model
                gc.collect()
    balance_table = balanced.groupby(["city", "class"], as_index=False).size()
    return pd.DataFrame(rows), pd.DataFrame(split_rows), balance_table


def cross_city_degradation(baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "model", "feature_set", "train_city"]
    same = baseline.loc[
        baseline["experiment_type"] == "same-city spatial",
        keys + ["test_city", "macro_f1", "expansion_f1"],
    ].rename(
        columns={
            "test_city": "same_city_test_city",
            "macro_f1": "same_city_macro_f1",
            "expansion_f1": "same_city_expansion_f1",
        }
    )
    cross = baseline.loc[
        baseline["experiment_type"] == "cross-city zero-shot",
        keys + ["test_city", "macro_f1", "expansion_f1"],
    ].rename(
        columns={
            "test_city": "cross_city_test_city",
            "macro_f1": "cross_city_macro_f1",
            "expansion_f1": "cross_city_expansion_f1",
        }
    )
    result = same.merge(cross, on=keys, validate="one_to_one")
    result["delta_macro_f1"] = (
        result["same_city_macro_f1"] - result["cross_city_macro_f1"]
    )
    result["delta_expansion_f1"] = (
        result["same_city_expansion_f1"] - result["cross_city_expansion_f1"]
    )
    groups = ["model", "feature_set", "train_city", "cross_city_test_city"]
    for metric in ["delta_macro_f1", "delta_expansion_f1"]:
        result[f"{metric}_mean"] = result.groupby(groups)[metric].transform("mean")
        result[f"{metric}_std"] = result.groupby(groups)[metric].transform("std")
    return result


def temporal_contribution(baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "model", "train_city", "test_city", "experiment_type"]
    columns = keys + ["feature_set", "macro_f1", "expansion_f1"]
    wide = baseline[columns].pivot(
        index=keys,
        columns="feature_set",
        values=["macro_f1", "expansion_f1"],
    )
    wide.columns = [f"{metric}_{feature}" for metric, feature in wide.columns]
    wide = wide.reset_index()
    wide["delta_macro_f1_AEF195_minus_AEF192"] = (
        wide["macro_f1_AEF-195"] - wide["macro_f1_AEF-192"]
    )
    wide["delta_expansion_f1_AEF195_minus_AEF192"] = (
        wide["expansion_f1_AEF-195"] - wide["expansion_f1_AEF-192"]
    )
    groups = ["model", "train_city", "test_city", "experiment_type"]
    for metric in [
        "delta_macro_f1_AEF195_minus_AEF192",
        "delta_expansion_f1_AEF195_minus_AEF192",
    ]:
        wide[f"{metric}_mean"] = wide.groupby(groups)[metric].transform("mean")
        wide[f"{metric}_std"] = wide.groupby(groups)[metric].transform("std")
    return wide


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_same_cross(baseline: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    model_order = ["Logistic Regression", "XGBoost"]
    feature_order = ["AEF-192", "AEF-195"]
    experiment_order = ["same-city spatial", "cross-city zero-shot"]
    colors = ["#4C78A8", "#E07B39"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, city in zip(axes, ["Shenzhen", "Kunming"]):
        subset = baseline.loc[baseline["train_city"] == city]
        positions = np.arange(4)
        labels: list[str] = []
        for model in model_order:
            for feature in feature_order:
                labels.append(("LR" if model.startswith("Logistic") else "XGB") + f"\n{feature}")
        width = 0.36
        for exp_index, experiment in enumerate(experiment_order):
            means: list[float] = []
            stds: list[float] = []
            for model in model_order:
                for feature in feature_order:
                    values = subset.loc[
                        (subset["model"] == model)
                        & (subset["feature_set"] == feature)
                        & (subset["experiment_type"] == experiment),
                        metric,
                    ]
                    means.append(float(values.mean()))
                    stds.append(float(values.std(ddof=1)))
            label = "Same-city" if experiment.startswith("same") else "Cross-city"
            axis.bar(
                positions + (exp_index - 0.5) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=colors[exp_index],
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )
        axis.set_title(f"Models trained in {city}")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_ablation(baseline: pd.DataFrame) -> None:
    model_order = ["Logistic Regression", "XGBoost"]
    experiment_order = ["same-city spatial", "cross-city zero-shot"]
    feature_order = ["AEF-192", "AEF-195"]
    colors = ["#8C8C8C", "#59A14F"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["macro_f1", "expansion_f1"],
        ["Macro-F1", "Expansion F1"],
    ):
        positions = np.arange(4)
        labels = ["LR\nSame", "LR\nCross", "XGB\nSame", "XGB\nCross"]
        width = 0.36
        for feature_index, feature in enumerate(feature_order):
            means: list[float] = []
            stds: list[float] = []
            for model in model_order:
                for experiment in experiment_order:
                    values = baseline.loc[
                        (baseline["model"] == model)
                        & (baseline["feature_set"] == feature)
                        & (baseline["experiment_type"] == experiment),
                        metric,
                    ]
                    means.append(float(values.mean()))
                    stds.append(float(values.std(ddof=1)))
            axis.bar(
                positions + (feature_index - 0.5) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=colors[feature_index],
                label=feature,
                edgecolor="white",
                linewidth=0.5,
            )
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("F1 score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    fig.suptitle("Feature ablation: annual embeddings and temporal cosine distances")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(FIGURES_DIR / "03_feature_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_domain_auc(domain_runs: pd.DataFrame) -> None:
    model_order = ["Logistic Regression", "XGBoost"]
    feature_order = ["AEF-192", "AEF-195"]
    colors = ["#8C8C8C", "#59A14F"]
    positions = np.arange(len(model_order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    for feature_index, feature in enumerate(feature_order):
        means: list[float] = []
        stds: list[float] = []
        for model in model_order:
            values = domain_runs.loc[
                (domain_runs["model"] == model)
                & (domain_runs["feature_set"] == feature),
                "roc_auc",
            ]
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        axis.bar(
            positions + (feature_index - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=colors[feature_index],
            label=feature,
            edgecolor="white",
            linewidth=0.5,
        )
    axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="Chance")
    axis.axhline(0.75, color="#B23A48", linestyle=":", linewidth=1, label="0.75 reference")
    axis.set_xticks(positions, ["Logistic Regression", "XGBoost"])
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0.45, 1.01)
    axis.set_title("City-domain classifier")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIGURES_DIR / "04_domain_classifier_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    baseline: pd.DataFrame,
    train_city: str,
    test_city: str,
    filename: str,
) -> None:
    subset = baseline.loc[
        (baseline["train_city"] == train_city)
        & (baseline["test_city"] == test_city)
        & (baseline["experiment_type"] == "cross-city zero-shot")
    ].copy()
    best = (
        subset.groupby(["model", "feature_set"], as_index=False)["macro_f1"]
        .mean()
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    chosen = subset.loc[
        (subset["model"] == best["model"])
        & (subset["feature_set"] == best["feature_set"])
    ]
    matrix = np.array(
        [
            [chosen[f"cm_{row}{column}"].sum() for column in range(3)]
            for row in range(3)
        ],
        dtype=np.int64,
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(3):
        for column in range(3):
            color = "white" if normalized[row, column] > 0.55 else "#222222"
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.1%}\n({matrix[row, column]:,})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    axis.set_xticks([0, 1, 2], ["Stable non-built", "Stable built", "Expansion"])
    axis.set_yticks([0, 1, 2], ["Stable non-built", "Stable built", "Expansion"])
    axis.tick_params(axis="x", rotation=18)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"{train_city} → {test_city}\n{best['model']}, {best['feature_set']} (5 seeds summed)"
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(baseline: pd.DataFrame, domain_runs: pd.DataFrame) -> None:
    set_plot_style()
    plot_same_cross(
        baseline,
        "macro_f1",
        "01_same_vs_cross_macro_f1.png",
        "Macro-F1",
    )
    plot_same_cross(
        baseline,
        "expansion_f1",
        "02_same_vs_cross_expansion_f1.png",
        "Expansion F1",
    )
    plot_feature_ablation(baseline)
    plot_domain_auc(domain_runs)
    plot_confusion_matrix(
        baseline,
        "Shenzhen",
        "Kunming",
        "05_confusion_matrix_SZ_to_KM.png",
    )
    plot_confusion_matrix(
        baseline,
        "Kunming",
        "Shenzhen",
        "06_confusion_matrix_KM_to_SZ.png",
    )


def format_summary_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}")


def build_report(
    config: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
    parquet_hashes: dict[str, str],
    baseline: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    degradation: pd.DataFrame,
    contribution: pd.DataFrame,
    domain_runs: pd.DataFrame,
    domain_summary: pd.DataFrame,
    city_splits: pd.DataFrame,
    domain_splits: pd.DataFrame,
    domain_balance: pd.DataFrame,
    warning_messages: list[str],
    go_conclusion: str,
    criteria: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("GeoAI-AEF Stage 03 Baseline Report")
    lines.append("=" * 78)
    lines.append(f"Run completed: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(f"Python: {platform.python_version()} ({sys.executable})")
    for package in [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "matplotlib",
        "xgboost",
        "joblib",
        "scipy",
    ]:
        lines.append(f"{package}: {version(package)}")

    lines.extend(["", "1. Dataset information", "-" * 78])
    for city, frame in datasets.items():
        counts = frame["class"].value_counts().sort_index().to_dict()
        lines.append(
            f"{city}: samples={len(frame)}, blocks={frame['block_id'].nunique()}, "
            f"classes={counts}, source_sha256={parquet_hashes[city]}"
        )
    lines.append("NaN=0, Inf=0, feature sets validated at 192 and 195 dimensions.")

    lines.extend(["", "2. Split strategy", "-" * 78])
    lines.append(
        "Randomized group-stratified block search. Each seed evaluates "
        f"{config['stage_03']['split']['search_candidates']} candidate assignments, "
        "uses exact block-count targets near 70%/15%/15%, minimizes sample-fraction "
        "and class-proportion deviations, and asserts pairwise block disjointness."
    )
    lines.append("Random seeds: " + json_text(config["stage_03"]["random_seeds"]))
    for _, row in city_splits.sort_values(["context", "seed"]).iterrows():
        lines.append(
            f"{row['context']} seed {int(row['seed'])}: blocks "
            f"{int(row['train_blocks'])}/{int(row['validation_blocks'])}/"
            f"{int(row['test_blocks'])}; samples "
            f"{int(row['train_samples'])}/{int(row['validation_samples'])}/"
            f"{int(row['test_samples'])}; max class deviation "
            f"{row['max_class_proportion_deviation']:.4f}"
        )
        lines.append(
            f"  class counts train={row['train_stratum_counts']}, "
            f"validation={row['validation_stratum_counts']}, "
            f"test={row['test_stratum_counts']}"
        )
    for _, row in domain_splits.sort_values("seed").iterrows():
        lines.append(
            f"Domain seed {int(row['seed'])}: blocks "
            f"{int(row['train_blocks'])}/{int(row['validation_blocks'])}/"
            f"{int(row['test_blocks'])}; samples "
            f"{int(row['train_samples'])}/{int(row['validation_samples'])}/"
            f"{int(row['test_samples'])}; max stratum deviation "
            f"{row['max_class_proportion_deviation']:.4f}"
        )
    lines.append("Domain balance by city and land-cover class:")
    lines.append(domain_balance.to_string(index=False))

    lines.extend(["", "3. Model parameters", "-" * 78])
    lines.append(json.dumps(config["stage_03"]["models"], indent=2, ensure_ascii=False))
    lines.append(
        "Logistic Regression uses StandardScaler fitted on source training data only. "
        "XGBoost uses source validation data only for early stopping."
    )

    lines.extend(["", "4. Feature definitions", "-" * 78])
    lines.append("AEF-192: 64 annual embedding dimensions for 2020, 2022 and 2024.")
    lines.append("AEF-195: AEF-192 plus D_20_22, D_22_24 and D_20_24.")
    lines.append("Metadata leakage assertion passed for both feature sets.")

    result_columns = [
        "model",
        "feature_set",
        "train_city",
        "test_city",
        "experiment_type",
        "macro_f1_mean",
        "macro_f1_std",
        "expansion_f1_mean",
        "expansion_f1_std",
    ]
    lines.extend(["", "5–9. Baseline, zero-shot and Expansion F1 results", "-" * 78])
    lines.append(format_summary_table(baseline_summary, result_columns))

    lines.extend(["", "10. Cross-city degradation", "-" * 78])
    degradation_summary = (
        degradation.groupby(
            ["model", "feature_set", "train_city", "cross_city_test_city"],
            as_index=False,
        )[["delta_macro_f1", "delta_expansion_f1"]]
        .agg(["mean", "std"])
    )
    degradation_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in degradation_summary.columns
    ]
    lines.append(degradation_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    lines.extend(["", "11. Domain classifier", "-" * 78])
    lines.append(
        domain_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    lines.append(
        "Coordinates, UTM fields, block IDs and city strings were excluded from domain features."
    )

    lines.extend(["", "12. Temporal feature contribution", "-" * 78])
    contribution_summary = contribution.groupby(
        ["model", "train_city", "test_city", "experiment_type"], as_index=False
    )[
        [
            "delta_macro_f1_AEF195_minus_AEF192",
            "delta_expansion_f1_AEF195_minus_AEF192",
        ]
    ].agg(["mean", "std"])
    contribution_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in contribution_summary.columns
    ]
    lines.append(contribution_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    lines.extend(["", "13. Warnings", "-" * 78])
    if warning_messages:
        lines.extend(f"WARNING: {message}" for message in sorted(set(warning_messages)))
    else:
        lines.append("None")

    lines.extend(["", "14. Go / No-Go conclusion", "-" * 78])
    lines.append(
        f"Mean cross-city Macro-F1 degradation: {criteria['mean_macro_degradation']:.6f} "
        f"(reference threshold {criteria['macro_threshold']:.2f})."
    )
    lines.append(
        "Directional mean Macro-F1 degradation: "
        + ", ".join(
            f"{city}→{TARGET_CITY[city]}={value:.6f}"
            for city, value in criteria["directional_macro_degradation"].items()
        )
    )
    lines.append(
        f"Mean Expansion-F1 degradation: {criteria['mean_expansion_degradation']:.6f}."
    )
    lines.append(
        f"Best Logistic Regression domain ROC-AUC mean: {criteria['best_lr_domain_auc']:.6f} "
        f"(reference threshold {criteria['domain_threshold']:.2f})."
    )
    lines.append(f"Conclusion: {go_conclusion}")
    lines.append(
        "No few-shot adaptation was run. Target-city labels were used only for final scoring."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    warning_messages: list[str] = []
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        stage_config = config["stage_03"]
        features = feature_definitions(config)
        hashes_before = {city: sha256(path) for city, path in CITY_FILES.items()}
        datasets = {
            city: validate_dataset(path, city, features)
            for city, path in CITY_FILES.items()
        }
        print("DATA VALIDATION PASSED")
        print("PACKAGE VERSIONS")
        for package in [
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "matplotlib",
            "xgboost",
            "joblib",
            "scipy",
        ]:
            print(f"{package}={version(package)}")

        baseline, city_splits = run_baselines(
            datasets, features, stage_config, warning_messages
        )
        expected_baseline_rows = 2 * 5 * 2 * 2 * 2
        assert len(baseline) == expected_baseline_rows, len(baseline)
        assert baseline[METRIC_COLUMNS].notna().all().all()
        assert np.isfinite(baseline[METRIC_COLUMNS].to_numpy()).all()

        baseline_summary = summarize(
            baseline,
            ["model", "feature_set", "train_city", "test_city", "experiment_type"],
            METRIC_COLUMNS,
        )
        degradation = cross_city_degradation(baseline)
        contribution = temporal_contribution(baseline)
        domain_runs, domain_splits, domain_balance = run_domain_classifier(
            datasets, features, stage_config, warning_messages
        )
        assert len(domain_runs) == 5 * 2 * 2
        assert domain_runs[["roc_auc", "accuracy", "balanced_accuracy", "f1"]].notna().all().all()
        domain_summary = summarize(
            domain_runs,
            ["model", "feature_set"],
            ["roc_auc", "accuracy", "balanced_accuracy", "f1"],
        )

        table_paths = {
            "baseline_all_runs": TABLES_DIR / "baseline_all_runs.csv",
            "baseline_summary": TABLES_DIR / "baseline_summary.csv",
            "cross_city_degradation": TABLES_DIR / "cross_city_degradation.csv",
            "domain_classifier_runs": TABLES_DIR / "domain_classifier_runs.csv",
            "domain_classifier_summary": TABLES_DIR / "domain_classifier_summary.csv",
            "temporal_feature_contribution": TABLES_DIR / "temporal_feature_contribution.csv",
            "spatial_split_summary": TABLES_DIR / "spatial_split_summary.csv",
        }
        baseline.to_csv(table_paths["baseline_all_runs"], index=False, encoding="utf-8-sig")
        baseline_summary.to_csv(
            table_paths["baseline_summary"], index=False, encoding="utf-8-sig"
        )
        degradation.to_csv(
            table_paths["cross_city_degradation"], index=False, encoding="utf-8-sig"
        )
        domain_runs.to_csv(
            table_paths["domain_classifier_runs"], index=False, encoding="utf-8-sig"
        )
        domain_summary.to_csv(
            table_paths["domain_classifier_summary"], index=False, encoding="utf-8-sig"
        )
        contribution.to_csv(
            table_paths["temporal_feature_contribution"], index=False, encoding="utf-8-sig"
        )
        pd.concat([city_splits, domain_splits], ignore_index=True).to_csv(
            table_paths["spatial_split_summary"], index=False, encoding="utf-8-sig"
        )

        create_figures(baseline, domain_runs)
        expected_figures = [
            "01_same_vs_cross_macro_f1.png",
            "02_same_vs_cross_expansion_f1.png",
            "03_feature_ablation.png",
            "04_domain_classifier_auc.png",
            "05_confusion_matrix_SZ_to_KM.png",
            "06_confusion_matrix_KM_to_SZ.png",
        ]
        for name in expected_figures:
            path = FIGURES_DIR / name
            assert path.is_file() and path.stat().st_size > 0, path

        mean_macro_degradation = float(degradation["delta_macro_f1"].mean())
        mean_expansion_degradation = float(degradation["delta_expansion_f1"].mean())
        lr_domain = domain_summary.loc[domain_summary["model"] == "Logistic Regression"]
        best_lr_domain_auc = float(lr_domain["roc_auc_mean"].max())
        macro_threshold = float(
            stage_config["go_no_go"]["cross_city_macro_f1_degradation_threshold"]
        )
        domain_threshold = float(stage_config["go_no_go"]["domain_auc_threshold"])
        directional_macro_degradation = {
            city: float(values.mean())
            for city, values in degradation.groupby("train_city")["delta_macro_f1"]
        }
        directional_values = list(directional_macro_degradation.values())
        asymmetric_gap = max(directional_values) - min(directional_values)
        if asymmetric_gap > macro_threshold:
            warning_messages.append(
                "Cross-city degradation is strongly asymmetric; the pooled mean obscures "
                "the direction-specific generalization gap."
            )
        go = (
            max(directional_values) > macro_threshold
            and best_lr_domain_auc > domain_threshold
        )
        go_conclusion = (
            "GO — the domain classifier shows a strong city signature and at least one "
            "transfer direction has a meaningful Macro-F1 gap. The evidence is asymmetric, "
            "so follow-up adaptation must report both directions separately."
            if go
            else "NO-GO — neither a sufficiently strong direction-specific generalization "
            "gap nor the required domain-signature evidence was jointly established."
        )
        criteria = {
            "mean_macro_degradation": mean_macro_degradation,
            "mean_expansion_degradation": mean_expansion_degradation,
            "best_lr_domain_auc": best_lr_domain_auc,
            "macro_threshold": macro_threshold,
            "domain_threshold": domain_threshold,
            "directional_macro_degradation": directional_macro_degradation,
        }

        overall_temporal_macro = float(
            contribution["delta_macro_f1_AEF195_minus_AEF192"].mean()
        )
        if abs(overall_temporal_macro) < 0.005:
            warning_messages.append(
                "Temporal cosine features changed overall Macro-F1 by less than 0.005; "
                "no meaningful aggregate contribution was detected."
            )

        hashes_after = {city: sha256(path) for city, path in CITY_FILES.items()}
        assert hashes_before == hashes_after, "Processed Parquet files changed during Stage 03"

        report = build_report(
            config,
            datasets,
            hashes_before,
            baseline,
            baseline_summary,
            degradation,
            contribution,
            domain_runs,
            domain_summary,
            city_splits,
            domain_splits,
            domain_balance,
            warning_messages,
            go_conclusion,
            criteria,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")

        for path in table_paths.values():
            assert path.is_file() and path.stat().st_size > 0
            pd.read_csv(path)
        assert REPORT_PATH.is_file() and REPORT_PATH.stat().st_size > 0
        print("STAGE 03 SUCCESS")
        print(f"REPORT={REPORT_PATH}")
        print(f"MEAN_MACRO_DEGRADATION={mean_macro_degradation:.8f}")
        print(f"MEAN_EXPANSION_DEGRADATION={mean_expansion_degradation:.8f}")
        print(f"BEST_LR_DOMAIN_AUC={best_lr_domain_auc:.8f}")
        print(f"CONCLUSION={go_conclusion}")
        print(f"WARNINGS={len(set(warning_messages))}")
        return 0
    except Exception as exc:
        failure = (
            "GeoAI-AEF Stage 03 Baseline Report\n"
            + "=" * 78
            + f"\nStatus: FAILED\nError: {type(exc).__name__}: {exc}\n"
        )
        REPORT_PATH.write_text(failure, encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())


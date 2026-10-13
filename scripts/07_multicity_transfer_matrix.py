"""Stage 07: five-city transfer matrix and representation-distance audit.

The experiment reuses the exact Stage 03 spatial split and model helpers.  It
does not tune any model on a target city.  Every target-normalized gap compares
the cross-city prediction with a same-city model on the identical held-out
target blocks.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "02_baseline_experiment.py"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "FiveCity_AEF_195D.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = LOGS_DIR / "stage07_multicity_transfer_report.txt"

SPATIAL_SPLITS_PATH = TABLES_DIR / "stage07_spatial_splits.csv"
SAME_CITY_PATH = TABLES_DIR / "stage07_same_city_runs.csv"
CROSS_CITY_PATH = TABLES_DIR / "stage07_cross_city_runs.csv"
MACRO_MATRIX_PATH = TABLES_DIR / "stage07_transfer_matrix_macro_f1.csv"
EXPANSION_MATRIX_PATH = TABLES_DIR / "stage07_transfer_matrix_expansion_f1.csv"
GAP_PATH = TABLES_DIR / "stage07_target_normalized_gap.csv"
SUPPORT_PATH = TABLES_DIR / "stage07_directed_support_distance.csv"
MECHANISM_PATH = TABLES_DIR / "stage07_pair_mechanism.csv"
SOURCE_PATH = TABLES_DIR / "stage07_source_transferability.csv"
TARGET_PATH = TABLES_DIR / "stage07_target_difficulty.csv"
DOMAIN_PATH = TABLES_DIR / "stage07_domain_classifier.csv"

FIGURE_PATHS = {
    "macro": FIGURES_DIR / "19_multicity_macro_f1_matrix.png",
    "expansion": FIGURES_DIR / "20_multicity_expansion_f1_matrix.png",
    "h1": FIGURES_DIR / "21_support_distance_vs_macro_gap.png",
    "h2": FIGURES_DIR / "22_class2_distance_vs_expansion_gap.png",
    "source": FIGURES_DIR / "23_source_transferability.png",
    "target": FIGURES_DIR / "24_target_difficulty.png",
    "domain": FIGURES_DIR / "25_five_city_domain_confusion_matrix.png",
}

CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
ABBREVIATIONS = {
    "Shenzhen": "SZ",
    "Kunming": "KM",
    "Wuhan": "WH",
    "Xian": "XA",
    "Beijing": "BJ",
}
CLASS_VALUES = [0, 1, 2]
CLASS_NAMES = {
    0: "Stable Non-built",
    1: "Stable Built",
    2: "Strict Expansion",
}
MODEL_NAMES = ["XGBoost", "Logistic Regression"]
PRIMARY_MODEL = "XGBoost"
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
    "D_20_22",
    "D_22_24",
    "D_20_24",
}
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261008
K_NEIGHBORS = 5


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def load_stage03_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage03_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 03 helpers from {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def sample_hash(frame: pd.DataFrame, indices: np.ndarray) -> str:
    values = sorted(frame.iloc[np.asarray(indices, dtype=np.int64)]["sample_id"].astype(str))
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def feature_columns(config: dict[str, Any]) -> list[str]:
    columns = [
        f"A{dimension:02d}_{year}"
        for year in [int(value) for value in config["years"]]
        for dimension in range(int(config["embedding_dimensions_per_year"]))
    ]
    if len(columns) != 192 or len(set(columns)) != 192:
        raise AssertionError("AEF-192 feature definition is invalid")
    if FORBIDDEN_FEATURES.intersection(columns):
        raise AssertionError("Forbidden metadata/temporal columns entered AEF-192")
    return columns


def validate_input(config: dict[str, Any], features: list[str]) -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    frame = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if frame.shape != (90000, 205):
        raise AssertionError(f"Expected (90000, 205), got {frame.shape}")
    if frame["city"].value_counts().to_dict() != {city: 18000 for city in CITIES}:
        raise AssertionError("City counts are not 18,000 each")
    expected_counts = {(city, cls): 6000 for city in CITIES for cls in CLASS_VALUES}
    if frame.groupby(["city", "class"]).size().to_dict() != expected_counts:
        raise AssertionError("City-class counts are not 6,000 each")
    if frame["sample_id"].duplicated().any():
        raise AssertionError("sample_id is not globally unique")
    if frame["block_id"].isna().any():
        raise AssertionError("block_id contains missing values")
    missing = [column for column in features if column not in frame]
    if missing:
        raise AssertionError(f"Missing AEF features: {missing}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or frame.isna().any().any():
        raise AssertionError("Input contains NaN or Inf")
    configured = config.get("selected_multicity_cities")
    if configured != CITIES:
        raise AssertionError(f"Configured city order differs: {configured}")
    return frame


def build_city_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for city in CITIES:
        city_frame = frame.loc[frame["city"] == city].copy().reset_index(drop=True)
        if len(city_frame) != 18000:
            raise AssertionError(f"{city}: invalid row count")
        outputs[city] = city_frame
    return outputs


def make_spatial_splits(
    stage03: Any,
    city_frames: dict[str, pd.DataFrame],
    stage_config: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[tuple[str, int], Any], pd.DataFrame, list[str]]:
    seeds = [int(value) for value in stage_config["random_seeds"]]
    prior_summary_path = TABLES_DIR / "spatial_split_summary.csv"
    prior_gap_path = TABLES_DIR / "target_normalized_transfer_gap.csv"
    prior_summary = pd.read_csv(prior_summary_path)
    prior_gap = pd.read_csv(prior_gap_path)
    splits: dict[tuple[str, int], Any] = {}
    rows: list[dict[str, Any]] = []
    confirmations: list[str] = []

    for city in CITIES:
        city_frame = city_frames[city]
        labels = city_frame["class"].astype(int)
        for seed in seeds:
            split = stage03.optimized_group_stratified_split(
                city_frame,
                labels,
                seed,
                stage_config["split"],
                f"Stage 07 {city}",
                warnings,
            )
            splits[(city, seed)] = split
            assignment: dict[str, str] = {}
            for split_name, indices in [
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            ]:
                for block_id in city_frame.iloc[indices]["block_id"].astype(str).unique():
                    if block_id in assignment:
                        raise AssertionError(f"{city} seed={seed}: overlapping block {block_id}")
                    assignment[block_id] = split_name
            grouped = (
                city_frame.assign(_class=city_frame["class"].astype(int))
                .groupby("block_id", sort=True)
                .agg(
                    block_samples=("sample_id", "size"),
                    class0_samples=("_class", lambda values: int((values == 0).sum())),
                    class1_samples=("_class", lambda values: int((values == 1).sum())),
                    class2_samples=("_class", lambda values: int((values == 2).sum())),
                )
                .reset_index()
            )
            if len(grouped) != split.diagnostics["total_blocks"]:
                raise AssertionError("Persisted split block count mismatch")
            for item in grouped.itertuples(index=False):
                block_id = str(item.block_id)
                rows.append(
                    {
                        "city": city,
                        "city_abbreviation": ABBREVIATIONS[city],
                        "seed": seed,
                        "block_id": block_id,
                        "split": assignment[block_id],
                        "block_samples": int(item.block_samples),
                        "class0_samples": int(item.class0_samples),
                        "class1_samples": int(item.class1_samples),
                        "class2_samples": int(item.class2_samples),
                        "split_algorithm": "Stage03 optimized group-stratified spatial block",
                        "spatial_block_size_m": 2000,
                    }
                )

            if city in {"Shenzhen", "Kunming"}:
                prior_row = prior_summary.loc[
                    (prior_summary["context"] == city) & (prior_summary["seed"] == seed)
                ]
                if len(prior_row) != 1:
                    raise AssertionError(f"Missing Stage 03 split summary for {city}, {seed}")
                prior_record = prior_row.iloc[0]
                fields = [
                    "train_samples",
                    "validation_samples",
                    "test_samples",
                    "train_blocks",
                    "validation_blocks",
                    "test_blocks",
                    "objective",
                ]
                for field in fields:
                    current = float(split.diagnostics[field])
                    expected = float(prior_record[field])
                    if not np.isclose(current, expected, rtol=0.0, atol=1e-12):
                        raise AssertionError(
                            f"{city} seed={seed}: Stage 03 {field} mismatch "
                            f"({current} vs {expected})"
                        )
                expected_hashes = prior_gap.loc[
                    (prior_gap["target_city"] == city) & (prior_gap["seed"] == seed),
                    "target_test_index_sha256",
                ].drop_duplicates()
                current_hash = index_hash(split.test)
                if len(expected_hashes) != 1 or expected_hashes.iloc[0] != current_hash:
                    raise AssertionError(
                        f"{city} seed={seed}: Stage 03B test-index hash mismatch"
                    )
                confirmations.append(
                    f"{city} seed={seed}: Stage 03 diagnostics and test-index hash reproduced"
                )

    output = pd.DataFrame(rows).sort_values(
        ["city", "seed", "split", "block_id"]
    ).reset_index(drop=True)
    expected_rows = sum(city_frames[city]["block_id"].nunique() for city in CITIES) * len(seeds)
    if len(output) != expected_rows:
        raise AssertionError("Spatial split export has an unexpected row count")
    return splits, output, confirmations


def run_land_cover_models(
    stage03: Any,
    city_frames: dict[str, pd.DataFrame],
    features: list[str],
    splits: dict[tuple[str, int], Any],
    stage_config: dict[str, Any],
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = [int(value) for value in stage_config["random_seeds"]]
    matrices = {
        city: city_frames[city][features].to_numpy(dtype=np.float32, copy=False)
        for city in CITIES
    }
    labels = {
        city: city_frames[city]["class"].to_numpy(dtype=np.int64)
        for city in CITIES
    }
    evaluation_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for model_name in MODEL_NAMES:
            for source_city in CITIES:
                split = splits[(source_city, seed)]
                model = stage03.make_model(model_name, "land_cover", seed, stage_config)
                model, training_seconds, iterations = stage03.fit_model(
                    model,
                    model_name,
                    matrices[source_city][split.train],
                    labels[source_city][split.train],
                    matrices[source_city][split.validation],
                    labels[source_city][split.validation],
                    warnings,
                    f"Stage 07 {source_city} {model_name} seed={seed}",
                )
                for target_city in CITIES:
                    target_split = splits[(target_city, seed)]
                    target_indices = np.asarray(target_split.test, dtype=np.int64)
                    prediction = model.predict(matrices[target_city][target_indices]).astype(int)
                    metrics = stage03.land_cover_metrics(
                        labels[target_city][target_indices], prediction
                    )
                    evaluation_rows.append(
                        {
                            "source_city": source_city,
                            "source_abbreviation": ABBREVIATIONS[source_city],
                            "target_city": target_city,
                            "target_abbreviation": ABBREVIATIONS[target_city],
                            "direction": f"{source_city} -> {target_city}",
                            "seed": seed,
                            "model": model_name,
                            "feature_set": "AEF-192",
                            "experiment_type": (
                                "same-city spatial" if source_city == target_city
                                else "cross-city zero-shot"
                            ),
                            "source_train_samples": len(split.train),
                            "source_validation_samples": len(split.validation),
                            "target_test_samples": len(target_indices),
                            "source_train_blocks": split.diagnostics["train_blocks"],
                            "source_validation_blocks": split.diagnostics["validation_blocks"],
                            "target_test_blocks": target_split.diagnostics["test_blocks"],
                            "target_test_index_sha256": index_hash(target_indices),
                            "target_test_sample_id_sha256": sample_hash(
                                city_frames[target_city], target_indices
                            ),
                            "training_seconds": training_seconds,
                            "fitted_iterations": iterations,
                            **metrics,
                        }
                    )
                print(f"MODEL {model_name} source={source_city} seed={seed} complete", flush=True)
                del model
                gc.collect()

    evaluations = pd.DataFrame(evaluation_rows)
    same = evaluations.loc[evaluations["source_city"] == evaluations["target_city"]].copy()
    cross = evaluations.loc[evaluations["source_city"] != evaluations["target_city"]].copy()
    same = same.sort_values(["model", "source_city", "seed"]).reset_index(drop=True)
    cross = cross.sort_values(["model", "source_city", "target_city", "seed"]).reset_index(drop=True)

    same_lookup = same.rename(
        columns={
            **{metric: f"target_same_city_{metric}" for metric in METRIC_COLUMNS},
            "target_test_index_sha256": "same_target_test_index_sha256",
            "target_test_sample_id_sha256": "same_target_test_sample_id_sha256",
        }
    )
    same_lookup = same_lookup[
        ["target_city", "seed", "model", "same_target_test_index_sha256",
         "same_target_test_sample_id_sha256"]
        + [f"target_same_city_{metric}" for metric in METRIC_COLUMNS]
    ]
    gaps = cross.merge(same_lookup, on=["target_city", "seed", "model"], how="left")
    if gaps[[f"target_same_city_{metric}" for metric in METRIC_COLUMNS]].isna().any().any():
        raise AssertionError("Same-city reference missing during gap construction")
    if not (
        gaps["target_test_index_sha256"] == gaps["same_target_test_index_sha256"]
    ).all():
        raise AssertionError("Cross-city and same-city target test indices differ")
    if not (
        gaps["target_test_sample_id_sha256"] == gaps["same_target_test_sample_id_sha256"]
    ).all():
        raise AssertionError("Cross-city and same-city target sample IDs differ")
    for metric in METRIC_COLUMNS:
        gaps[f"{metric}_gap"] = gaps[f"target_same_city_{metric}"] - gaps[metric]
    gaps["shared_target_test_set_assertion"] = True
    gaps = gaps.sort_values(["model", "source_city", "target_city", "seed"]).reset_index(drop=True)
    return same, cross, gaps


def make_matrix(
    same: pd.DataFrame,
    cross: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in CITIES:
        row: dict[str, Any] = {
            "source_city": source,
            "source_abbreviation": ABBREVIATIONS[source],
            "model": PRIMARY_MODEL,
            "feature_set": "AEF-192",
            "metric": metric,
        }
        for target in CITIES:
            data = same if source == target else cross
            values = data.loc[
                (data["model"] == PRIMARY_MODEL)
                & (data["source_city"] == source)
                & (data["target_city"] == target),
                metric,
            ]
            if len(values) != 5:
                raise AssertionError(f"Matrix cell {source}->{target} has {len(values)} runs")
            row[ABBREVIATIONS[target]] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def l2_normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise AssertionError("Non-finite or zero representation norm")
    output = normalize(matrix, norm="l2", axis=1, copy=True)
    if float(np.max(np.abs(np.linalg.norm(output, axis=1) - 1.0))) > 2e-6:
        raise AssertionError("L2 normalization failed")
    return np.asarray(output, dtype=np.float32)


def run_support_distance(
    city_frames: dict[str, pd.DataFrame],
    features: list[str],
    splits: dict[tuple[str, int], Any],
    seeds: list[int],
) -> pd.DataFrame:
    normalized = {
        city: l2_normalized(
            city_frames[city][features].to_numpy(dtype=np.float32, copy=False)
        )
        for city in CITIES
    }
    labels = {
        city: city_frames[city]["class"].to_numpy(dtype=np.int64)
        for city in CITIES
    }
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for source_city in CITIES:
            source_split = splits[(source_city, seed)]
            by_target: dict[str, list[np.ndarray]] = {
                city: [] for city in CITIES if city != source_city
            }
            for class_value in CLASS_VALUES:
                reference_indices = source_split.train[
                    labels[source_city][source_split.train] == class_value
                ]
                estimator = NearestNeighbors(
                    n_neighbors=K_NEIGHBORS,
                    metric="cosine",
                    algorithm="brute",
                    n_jobs=4,
                )
                estimator.fit(normalized[source_city][reference_indices])
                for target_city in CITIES:
                    if target_city == source_city:
                        continue
                    target_split = splits[(target_city, seed)]
                    target_indices = target_split.test[
                        labels[target_city][target_split.test] == class_value
                    ]
                    distances, _ = estimator.kneighbors(
                        normalized[target_city][target_indices], return_distance=True
                    )
                    mean_distance = distances.mean(axis=1).astype(np.float64)
                    if not np.isfinite(mean_distance).all():
                        raise AssertionError("Non-finite support distance")
                    by_target[target_city].append(mean_distance)
                    rows.append(
                        {
                            "source_city": source_city,
                            "source_abbreviation": ABBREVIATIONS[source_city],
                            "target_city": target_city,
                            "target_abbreviation": ABBREVIATIONS[target_city],
                            "direction": f"{source_city} -> {target_city}",
                            "seed": seed,
                            "scope": f"Class {class_value}",
                            "class": class_value,
                            "class_name": CLASS_NAMES[class_value],
                            "feature_set": "AEF-192",
                            "normalization": "per-sample L2",
                            "metric": "cosine",
                            "k": K_NEIGHBORS,
                            "source_reference_samples": len(reference_indices),
                            "target_query_samples": len(target_indices),
                            "support_distance_mean": float(mean_distance.mean()),
                            "support_distance_std": float(mean_distance.std(ddof=1)),
                            "support_distance_median": float(np.median(mean_distance)),
                            "support_distance_p90": float(np.quantile(mean_distance, 0.90)),
                            "target_test_index_sha256": index_hash(target_split.test),
                        }
                    )
                del estimator
            for target_city, arrays in by_target.items():
                values = np.concatenate(arrays)
                target_split = splits[(target_city, seed)]
                rows.append(
                    {
                        "source_city": source_city,
                        "source_abbreviation": ABBREVIATIONS[source_city],
                        "target_city": target_city,
                        "target_abbreviation": ABBREVIATIONS[target_city],
                        "direction": f"{source_city} -> {target_city}",
                        "seed": seed,
                        "scope": "Overall",
                        "class": pd.NA,
                        "class_name": "All classes",
                        "feature_set": "AEF-192",
                        "normalization": "per-sample L2",
                        "metric": "cosine",
                        "k": K_NEIGHBORS,
                        "source_reference_samples": len(source_split.train),
                        "target_query_samples": len(values),
                        "support_distance_mean": float(values.mean()),
                        "support_distance_std": float(values.std(ddof=1)),
                        "support_distance_median": float(np.median(values)),
                        "support_distance_p90": float(np.quantile(values, 0.90)),
                        "target_test_index_sha256": index_hash(target_split.test),
                    }
                )
            print(f"SUPPORT source={source_city} seed={seed} complete", flush=True)
    return pd.DataFrame(rows).sort_values(
        ["source_city", "target_city", "seed", "scope"]
    ).reset_index(drop=True)


def build_pair_mechanism(
    support: pd.DataFrame,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    primary = gaps.loc[gaps["model"] == PRIMARY_MODEL].copy()
    overall = (
        support.loc[support["scope"] == "Overall"]
        .groupby(["source_city", "target_city", "direction"], sort=False)
        ["support_distance_mean"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "support_distance_mean", "std": "support_distance_seed_std"})
        .reset_index()
    )
    class2 = (
        support.loc[support["scope"] == "Class 2"]
        .groupby(["source_city", "target_city", "direction"], sort=False)
        ["support_distance_mean"]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "class2_support_distance_mean",
                "std": "class2_support_distance_seed_std",
            }
        )
        .reset_index()
    )
    metric_map = {
        "macro_f1": "cross_macro_f1",
        "macro_f1_gap": "macro_f1_gap",
        "target_same_city_macro_f1": "target_same_city_macro_f1",
        "expansion_f1": "cross_expansion_f1",
        "expansion_f1_gap": "expansion_f1_gap",
        "target_same_city_expansion_f1": "target_same_city_expansion_f1",
        "accuracy": "cross_accuracy",
    }
    pieces: list[pd.DataFrame] = []
    for source_column, output_prefix in metric_map.items():
        part = (
            primary.groupby(["source_city", "target_city", "direction"], sort=False)
            [source_column]
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": f"{output_prefix}_mean",
                    "std": f"{output_prefix}_std",
                }
            )
            .reset_index()
        )
        pieces.append(part)
    result = overall.merge(class2, on=["source_city", "target_city", "direction"])
    for part in pieces:
        result = result.merge(part, on=["source_city", "target_city", "direction"])
    result["cross_error_rate_mean"] = 1.0 - result["cross_accuracy_mean"]
    result["source_abbreviation"] = result["source_city"].map(ABBREVIATIONS)
    result["target_abbreviation"] = result["target_city"].map(ABBREVIATIONS)
    result["pair_label"] = (
        result["source_abbreviation"] + "→" + result["target_abbreviation"]
    )
    result["model"] = PRIMARY_MODEL
    result["feature_set"] = "AEF-192"
    if len(result) != 20:
        raise AssertionError(f"Expected 20 directed pairs, got {len(result)}")
    return result.sort_values(["source_city", "target_city"]).reset_index(drop=True)


def correlation_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != 20 or len(y) != 20:
        raise AssertionError("Pair-level correlation unit must be n=20")
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    pearson_boot: list[float] = []
    spearman_boot: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(x), size=len(x))
        bx = x[indices]
        by = y[indices]
        if np.ptp(bx) == 0 or np.ptp(by) == 0:
            continue
        pearson_value = float(pearsonr(bx, by).statistic)
        spearman_value = float(spearmanr(bx, by).statistic)
        if np.isfinite(pearson_value):
            pearson_boot.append(pearson_value)
        if np.isfinite(spearman_value):
            spearman_boot.append(spearman_value)
    if len(pearson_boot) < 950 or len(spearman_boot) < 950:
        raise AssertionError("Too many invalid bootstrap correlation replicates")
    return {
        "n_pairs": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "pearson_ci_low": float(np.quantile(pearson_boot, 0.025)),
        "pearson_ci_high": float(np.quantile(pearson_boot, 0.975)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "spearman_ci_low": float(np.quantile(spearman_boot, 0.025)),
        "spearman_ci_high": float(np.quantile(spearman_boot, 0.975)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "directed city pair",
    }


def judge_hypothesis(result: dict[str, float], expected_positive: bool = True) -> str:
    direction_ok = result["spearman_rho"] > 0 if expected_positive else result["spearman_rho"] < 0
    ci_excludes_zero = (
        result["spearman_ci_low"] > 0 if expected_positive
        else result["spearman_ci_high"] < 0
    )
    if direction_ok and result["spearman_p"] < 0.05 and ci_excludes_zero:
        return "supported"
    if direction_ok:
        return "partially supported"
    return "not supported"


def source_target_summaries(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = (
        pair.groupby(["source_city", "source_abbreviation"], sort=False)
        .agg(
            target_cities=("target_city", "nunique"),
            mean_outgoing_macro_f1=("cross_macro_f1_mean", "mean"),
            std_outgoing_macro_f1=("cross_macro_f1_mean", "std"),
            mean_outgoing_expansion_f1=("cross_expansion_f1_mean", "mean"),
            std_outgoing_expansion_f1=("cross_expansion_f1_mean", "std"),
            mean_outgoing_macro_gap=("macro_f1_gap_mean", "mean"),
            mean_outgoing_expansion_gap=("expansion_f1_gap_mean", "mean"),
            mean_outgoing_support_distance=("support_distance_mean", "mean"),
            mean_outgoing_class2_distance=("class2_support_distance_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_outgoing_macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    source["source_rank_macro_f1"] = np.arange(1, len(source) + 1)
    source["model"] = PRIMARY_MODEL
    source["feature_set"] = "AEF-192"

    target = (
        pair.groupby(["target_city", "target_abbreviation"], sort=False)
        .agg(
            source_cities=("source_city", "nunique"),
            mean_incoming_macro_f1=("cross_macro_f1_mean", "mean"),
            std_incoming_macro_f1=("cross_macro_f1_mean", "std"),
            mean_incoming_expansion_f1=("cross_expansion_f1_mean", "mean"),
            std_incoming_expansion_f1=("cross_expansion_f1_mean", "std"),
            mean_incoming_macro_gap=("macro_f1_gap_mean", "mean"),
            std_incoming_macro_gap=("macro_f1_gap_mean", "std"),
            mean_incoming_expansion_gap=("expansion_f1_gap_mean", "mean"),
            std_incoming_expansion_gap=("expansion_f1_gap_mean", "std"),
            mean_incoming_support_distance=("support_distance_mean", "mean"),
            mean_incoming_class2_distance=("class2_support_distance_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_incoming_macro_gap", ascending=False)
        .reset_index(drop=True)
    )
    target["target_difficulty_rank_macro_gap"] = np.arange(1, len(target) + 1)
    target["model"] = PRIMARY_MODEL
    target["feature_set"] = "AEF-192"
    return source, target


def run_domain_classifier(
    stage03: Any,
    city_frames: dict[str, pd.DataFrame],
    features: list[str],
    splits: dict[tuple[str, int], Any],
    stage_config: dict[str, Any],
    warnings: list[str],
) -> pd.DataFrame:
    seeds = [int(value) for value in stage_config["random_seeds"]]
    matrix = np.vstack(
        [city_frames[city][features].to_numpy(dtype=np.float32, copy=False) for city in CITIES]
    )
    labels = np.concatenate(
        [np.full(len(city_frames[city]), position, dtype=np.int64)
         for position, city in enumerate(CITIES)]
    )
    offsets = {city: position * 18000 for position, city in enumerate(CITIES)}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_indices = np.concatenate(
            [splits[(city, seed)].train + offsets[city] for city in CITIES]
        )
        validation_indices = np.concatenate(
            [splits[(city, seed)].validation + offsets[city] for city in CITIES]
        )
        test_indices = np.concatenate(
            [splits[(city, seed)].test + offsets[city] for city in CITIES]
        )
        model = stage03.make_model("Logistic Regression", "domain", seed, stage_config)
        model, seconds, iterations = stage03.fit_model(
            model,
            "Logistic Regression",
            matrix[train_indices],
            labels[train_indices],
            matrix[validation_indices],
            labels[validation_indices],
            warnings,
            f"Stage 07 five-city domain classifier seed={seed}",
        )
        prediction = model.predict(matrix[test_indices]).astype(int)
        cm = confusion_matrix(labels[test_indices], prediction, labels=np.arange(5))
        row: dict[str, Any] = {
            "seed": seed,
            "model": "Logistic Regression",
            "feature_set": "AEF-192",
            "split": "per-city group-aware spatial blocks",
            "scaler_fit_scope": "training set only",
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "accuracy": accuracy_score(labels[test_indices], prediction),
            "macro_f1": f1_score(labels[test_indices], prediction, average="macro"),
            "training_seconds": seconds,
            "fitted_iterations": iterations,
        }
        for true_index, true_city in enumerate(CITIES):
            for pred_index, pred_city in enumerate(CITIES):
                row[f"cm_{ABBREVIATIONS[true_city]}_{ABBREVIATIONS[pred_city]}"] = int(
                    cm[true_index, pred_index]
                )
        rows.append(row)
        print(f"DOMAIN five-city seed={seed} complete", flush=True)
        del model
        gc.collect()
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_matrix(matrix: pd.DataFrame, title: str, path: Path) -> None:
    values = matrix[[ABBREVIATIONS[city] for city in CITIES]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    image = ax.imshow(values, cmap="viridis", vmin=max(0.0, values.min() - 0.03), vmax=1.0)
    ax.grid(False)
    ax.set_xticks(np.arange(5), [ABBREVIATIONS[city] for city in CITIES])
    ax.set_yticks(np.arange(5), [ABBREVIATIONS[city] for city in CITIES])
    ax.set_xlabel("Target city")
    ax.set_ylabel("Source city")
    ax.set_title(title, pad=14, weight="bold")
    for row in range(5):
        for column in range(5):
            color = "white" if values[row, column] < np.median(values) else "black"
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("5-seed mean F1")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    pair: pd.DataFrame,
    x_column: str,
    y_column: str,
    correlation: dict[str, float],
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    x = pair[x_column].to_numpy(dtype=float)
    y = pair[y_column].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(x, y, s=64, color="#2474b5", alpha=0.86, edgecolor="white", linewidth=0.7)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, slope * line_x + intercept, color="#d95f02", linewidth=2.0)
    for item in pair.itertuples(index=False):
        ax.annotate(
            item.pair_label,
            (getattr(item, x_column), getattr(item, y_column)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.4,
        )
    annotation = (
        f"Spearman ρ={correlation['spearman_rho']:.3f} "
        f"(95% CI {correlation['spearman_ci_low']:.3f}, {correlation['spearman_ci_high']:.3f})\n"
        f"Pearson r={correlation['pearson_r']:.3f} "
        f"(95% CI {correlation['pearson_ci_low']:.3f}, {correlation['pearson_ci_high']:.3f})"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_source(source: pd.DataFrame, path: Path) -> None:
    ordered = source.sort_values("mean_outgoing_macro_f1", ascending=False)
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.bar(x - width / 2, ordered["mean_outgoing_macro_f1"], width, label="Macro-F1", color="#2c7fb8")
    ax.bar(x + width / 2, ordered["mean_outgoing_expansion_f1"], width, label="Expansion-F1", color="#f28e2b")
    ax.set_xticks(x, ordered["source_abbreviation"])
    ax.set_ylim(max(0.0, min(ordered["mean_outgoing_expansion_f1"].min(), ordered["mean_outgoing_macro_f1"].min()) - 0.08), 1.01)
    ax.set_xlabel("Source city")
    ax.set_ylabel("Mean zero-shot F1 across four targets")
    ax.set_title("Source-city transferability (XGBoost, AEF-192)", weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_target(target: pd.DataFrame, path: Path) -> None:
    ordered = target.sort_values("mean_incoming_macro_gap", ascending=False)
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.bar(x - width / 2, ordered["mean_incoming_macro_gap"], width, label="Macro-F1 gap", color="#8c6bb1")
    ax.bar(x + width / 2, ordered["mean_incoming_expansion_gap"], width, label="Expansion-F1 gap", color="#e6550d")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, ordered["target_abbreviation"])
    ax.set_xlabel("Target city")
    ax.set_ylabel("Mean incoming target-normalized gap")
    ax.set_title("Target-city transfer difficulty (XGBoost, AEF-192)", weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_domain_confusion(domain: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=float)
    for true_index, true_city in enumerate(CITIES):
        for pred_index, pred_city in enumerate(CITIES):
            column = f"cm_{ABBREVIATIONS[true_city]}_{ABBREVIATIONS[pred_city]}"
            matrix[true_index, pred_index] = domain[column].mean()
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def plot_domain(domain: pd.DataFrame, path: Path) -> None:
    values = mean_domain_confusion(domain)
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.grid(False)
    labels = [ABBREVIATIONS[city] for city in CITIES]
    ax.set_xticks(np.arange(5), labels)
    ax.set_yticks(np.arange(5), labels)
    ax.set_xlabel("Predicted city")
    ax.set_ylabel("True city")
    ax.set_title("Five-city domain classifier confusion matrix", pad=12, weight="bold")
    for row in range(5):
        for column in range(5):
            ax.text(
                column,
                row,
                f"{100 * values[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if values[row, column] > 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def consistency_with_stage03b(gaps: pd.DataFrame) -> tuple[list[str], list[str]]:
    prior_path = TABLES_DIR / "target_normalized_transfer_gap.csv"
    prior = pd.read_csv(prior_path)
    checks: list[str] = []
    warnings: list[str] = []
    for source, target in [("Shenzhen", "Kunming"), ("Kunming", "Shenzhen")]:
        for metric in ["macro_f1_gap", "expansion_f1_gap"]:
            current_values = gaps.loc[
                (gaps["model"] == PRIMARY_MODEL)
                & (gaps["source_city"] == source)
                & (gaps["target_city"] == target),
                metric,
            ].sort_index().to_numpy(dtype=float)
            prior_values = prior.loc[
                (prior["model"] == PRIMARY_MODEL)
                & (prior["source_city"] == source)
                & (prior["target_city"] == target),
                metric,
            ].to_numpy(dtype=float)
            difference = float(abs(current_values.mean() - prior_values.mean()))
            checks.append(
                f"{source}->{target} {metric}: Stage07={current_values.mean():.10f}, "
                f"Stage03B={prior_values.mean():.10f}, abs diff={difference:.3g}"
            )
            if difference > 1e-6:
                warnings.append(
                    f"Stage 03B consistency difference exceeds 1e-6 for "
                    f"{source}->{target} {metric}: {difference:.6g}"
                )
    return checks, warnings


def format_correlation(name: str, result: dict[str, float]) -> list[str]:
    return [
        name,
        f"  n directed pairs: {result['n_pairs']}",
        f"  Spearman rho={result['spearman_rho']:.6f}, p={result['spearman_p']:.6g}, "
        f"pair-bootstrap 95% CI=[{result['spearman_ci_low']:.6f}, {result['spearman_ci_high']:.6f}]",
        f"  Pearson r={result['pearson_r']:.6f}, p={result['pearson_p']:.6g}, "
        f"pair-bootstrap 95% CI=[{result['pearson_ci_low']:.6f}, {result['pearson_ci_high']:.6f}]",
        f"  Bootstrap: {BOOTSTRAP_RESAMPLES} resamples; unit=directed city pair",
    ]


def report_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> list[str]:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.{decimals}f}")
    return view.to_string(index=False).splitlines()


def write_report(
    input_hash: str,
    stage_config: dict[str, Any],
    split_confirmations: list[str],
    same: pd.DataFrame,
    cross: pd.DataFrame,
    gaps: pd.DataFrame,
    macro_matrix: pd.DataFrame,
    expansion_matrix: pd.DataFrame,
    support: pd.DataFrame,
    pair: pd.DataFrame,
    source: pd.DataFrame,
    target: pd.DataFrame,
    domain: pd.DataFrame,
    correlations: dict[str, dict[str, float]],
    h1_judgment: str,
    h2_judgment: str,
    consistency_checks: list[str],
    warnings: list[str],
    elapsed_seconds: float,
) -> None:
    primary_same = same.loc[same["model"] == PRIMARY_MODEL]
    primary_cross = cross.loc[cross["model"] == PRIMARY_MODEL]
    primary_gaps = gaps.loc[gaps["model"] == PRIMARY_MODEL]
    same_summary = (
        primary_same.groupby("source_city")[["macro_f1", "expansion_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cross_summary = (
        primary_cross.groupby(["source_city", "target_city"])[["macro_f1", "expansion_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    gap_summary = (
        primary_gaps.groupby(["source_city", "target_city"])[["macro_f1_gap", "expansion_f1_gap"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    support_summary = (
        support.groupby(["source_city", "target_city", "scope"])["support_distance_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    domain_accuracy_mean = float(domain["accuracy"].mean())
    domain_accuracy_std = float(domain["accuracy"].std(ddof=1))
    domain_f1_mean = float(domain["macro_f1"].mean())
    domain_f1_std = float(domain["macro_f1"].std(ddof=1))
    sorted_pair = pair.sort_values("macro_f1_gap_mean", ascending=False)

    lines: list[str] = [
        "Stage 07 - Five-City Transfer Matrix and Representation-Distance Generalization",
        "=" * 82,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Runtime: {elapsed_seconds:.2f} seconds",
        "",
        "1. Scope and frozen protocol",
        "- Cities (fixed order): Shenzhen (SZ), Kunming (KM), Wuhan (WH), Xian (XA), Beijing (BJ).",
        "- Input: data/processed/FiveCity_AEF_195D.parquet (90,000 rows x 205 columns).",
        "- Features: AEF-192 only; temporal distance features and all metadata are excluded.",
        "- Seeds: " + ", ".join(str(value) for value in stage_config["random_seeds"]),
        "- Spatial split: exact Stage 03 optimized group-stratified 2 km block algorithm.",
        "- Models: fixed Stage 03 XGBoost (primary) and Logistic Regression (secondary).",
        "- XGBoost parameters: n_estimators=400, max_depth=6, learning_rate=0.08, "
        "subsample=0.85, colsample_bytree=0.85, min_child_weight=1.0, reg_lambda=1.0, "
        "early_stopping_rounds=30, tree_method=hist, n_jobs=4.",
        "- Logistic Regression parameters: C=1.0, solver=lbfgs, max_iter=2000; "
        "StandardScaler fitted on source training samples only.",
        "- No few-shot adaptation and no target-label model selection were performed.",
        "",
        "2. Reproducibility and leakage controls",
        f"- Input SHA-256: {input_hash}",
        f"- Python: {platform.python_version()}; pandas: {package_version('pandas')}; "
        f"scikit-learn: {package_version('scikit-learn')}; xgboost: {package_version('xgboost')}",
        "- Logistic Regression StandardScaler was fitted inside the training-only pipeline.",
        "- Cross-city models were evaluated on the exact target test indices used by target same-city models.",
        "- Test equality was asserted with both index SHA-256 and sorted sample_id SHA-256.",
        "- Domain classifier used the same per-city group-aware train/validation/test blocks.",
        "- Forbidden feature audit: PASS (metadata, class, temporal distances absent from model input).",
        "",
        "3. Stage 03 split reproduction for Shenzhen and Kunming",
    ]
    lines.extend(f"- {item}" for item in split_confirmations)
    lines.extend(
        [
            "",
            "4. Same-city held-out performance (XGBoost, five-seed mean/std)",
            same_summary.to_string(index=False),
            "",
            "5. Directed cross-city zero-shot performance (XGBoost, five-seed mean/std)",
            cross_summary.to_string(index=False),
            "",
            "6. Macro-F1 transfer matrix (row=source, column=target; XGBoost means)",
            macro_matrix.to_string(index=False),
            "",
            "7. Expansion-F1 transfer matrix (row=source, column=target; XGBoost means)",
            expansion_matrix.to_string(index=False),
            "",
            "8. Target-normalized gaps (same-target same-city minus source-on-target)",
            gap_summary.to_string(index=False),
            "",
            "9. Directed same-class 5-NN cosine support distances",
            "- Each AEF-192 sample was L2-normalized before distance calculation.",
            "- Reference set: source training samples of the same land-cover class.",
            "- Query set: target held-out test samples; no target sample entered the reference set.",
            support_summary.to_string(index=False),
            "",
            "10. Pair-level mechanism table (n=20 directed pairs)",
            pair[[
                "pair_label", "support_distance_mean", "class2_support_distance_mean",
                "cross_macro_f1_mean", "macro_f1_gap_mean",
                "cross_expansion_f1_mean", "expansion_f1_gap_mean",
            ]].to_string(index=False),
            "",
            "11. H1 test: overall support distance vs target-normalized Macro-F1 gap",
        ]
    )
    lines.extend(format_correlation("Primary H1 association", correlations["h1_gap"]))
    lines.extend(
        [
            f"  Judgment: H1 is {h1_judgment}.",
            "  Rule: supported requires positive Spearman rho, p<0.05, and a pair-bootstrap 95% CI excluding zero;",
            "  a positive direction without both inferential conditions is partial support.",
            "",
            "12. H2 test: Class-2 support distance vs target-normalized Expansion-F1 gap",
        ]
    )
    lines.extend(format_correlation("Primary H2 association", correlations["h2_gap"]))
    lines.extend(
        [
            f"  Judgment: H2 is {h2_judgment}.",
            "  Rule: identical to H1, with Class-2 distance and Expansion-F1 gap.",
            "",
            "13. Overall distance vs target-normalized Expansion-F1 gap",
        ]
    )
    lines.extend(format_correlation(
        "Overall support distance vs Expansion-F1 gap",
        correlations["overall_expansion_gap"],
    ))
    lines.extend(["", "14. Auxiliary performance/error correlations"])
    lines.extend(format_correlation("Overall distance vs cross-city Macro-F1", correlations["overall_performance"]))
    lines.extend(format_correlation("Class-2 distance vs cross-city Expansion-F1", correlations["class2_performance"]))
    lines.extend(format_correlation("Overall distance vs cross-city error rate", correlations["overall_error"]))
    lines.extend(
        [
            "",
            "15. Source transferability ranking (XGBoost)",
            source.to_string(index=False),
            "",
            "16. Target difficulty ranking (XGBoost)",
            target.to_string(index=False),
            "",
            "17. Best and worst directed transfer pairs",
            f"- Best Macro-F1 pair: {pair.loc[pair['cross_macro_f1_mean'].idxmax(), 'pair_label']} "
            f"({pair['cross_macro_f1_mean'].max():.6f}).",
            f"- Worst Macro-F1 pair: {pair.loc[pair['cross_macro_f1_mean'].idxmin(), 'pair_label']} "
            f"({pair['cross_macro_f1_mean'].min():.6f}).",
            f"- Largest Macro-F1 gap: {sorted_pair.iloc[0]['pair_label']} "
            f"({sorted_pair.iloc[0]['macro_f1_gap_mean']:.6f}).",
            f"- Smallest Macro-F1 gap: {sorted_pair.iloc[-1]['pair_label']} "
            f"({sorted_pair.iloc[-1]['macro_f1_gap_mean']:.6f}).",
            "",
            "18. Five-class city domain classifier",
            f"- Accuracy: {domain_accuracy_mean:.6f} +/- {domain_accuracy_std:.6f}.",
            f"- Macro-F1: {domain_f1_mean:.6f} +/- {domain_f1_std:.6f}.",
            "- Confusion matrix columns are stored per seed in stage07_domain_classifier.csv;",
            "  Figure 25 displays the five-seed mean row-normalized matrix.",
            "",
            "19. Consistency with Stage 03B / Stage 05 SZ-KM findings",
        ]
    )
    lines.extend(f"- {item}" for item in consistency_checks)
    lines.extend(
        [
            "- The original SZ-KM direction comparison is therefore retained inside the five-city design.",
            "",
            "20. Interpretation and limitations",
            "- Directed support distance is a geometry diagnostic, not a causal estimate.",
            "- The inferential unit is the directed city pair (n=20), not individual pixels or seeds.",
            "- Five seeds quantify split sensitivity, while pair-level bootstrap intervals quantify uncertainty across city pairs.",
            "- Correlated directions sharing a city can narrow naive uncertainty; conclusions should be treated as exploratory.",
            "- High city-domain separability indicates substantial representation shift, but does not by itself prove poor transfer.",
            "- Only five Chinese cities are included; geographic and sensor-season generalization remains limited.",
            "",
            "21. Warnings",
        ]
    )
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "22. Next-step recommendation",
            "- Treat the observed distance-gap links as multi-city empirical associations. "
            "If Stage 08 is authorized later, pre-register a source-selection or reweighting rule "
            "using only source-side geometry, then evaluate it on untouched target test blocks.",
            "- Do not alter the frozen labels, ROIs, city set, seeds or Stage 07 test blocks.",
            "",
            "23. Outputs",
            *[f"- {path.relative_to(PROJECT_ROOT)}" for path in [
                SPATIAL_SPLITS_PATH, SAME_CITY_PATH, CROSS_CITY_PATH, MACRO_MATRIX_PATH,
                EXPANSION_MATRIX_PATH, GAP_PATH, SUPPORT_PATH, MECHANISM_PATH, SOURCE_PATH,
                TARGET_PATH, DOMAIN_PATH, *FIGURE_PATHS.values(), REPORT_PATH,
            ]],
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def validate_outputs() -> None:
    table_paths = [
        SPATIAL_SPLITS_PATH,
        SAME_CITY_PATH,
        CROSS_CITY_PATH,
        MACRO_MATRIX_PATH,
        EXPANSION_MATRIX_PATH,
        GAP_PATH,
        SUPPORT_PATH,
        MECHANISM_PATH,
        SOURCE_PATH,
        TARGET_PATH,
        DOMAIN_PATH,
    ]
    for path in table_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing/empty output: {path}")
        parsed = pd.read_csv(path)
        if parsed.empty:
            raise AssertionError(f"CSV has no data rows: {path}")
    for path in FIGURE_PATHS.values():
        if not path.is_file() or path.stat().st_size < 10_000:
            raise AssertionError(f"Missing/suspicious figure: {path}")
    if not REPORT_PATH.is_file() or REPORT_PATH.stat().st_size < 1000:
        raise AssertionError("Report missing or unexpectedly short")


def main() -> None:
    started = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stage_config = config["stage_03"]
    seeds = [int(value) for value in stage_config["random_seeds"]]
    if seeds != [20261008, 20261009, 20261010, 20261011, 20261012]:
        raise AssertionError(f"Stage 03 seeds changed: {seeds}")
    stage03 = load_stage03_module()
    features = feature_columns(config)
    input_hash = file_sha256(DATA_PATH)
    frame = validate_input(config, features)
    city_frames = build_city_frames(frame)
    del frame
    gc.collect()

    warnings: list[str] = []
    splits, split_table, split_confirmations = make_spatial_splits(
        stage03, city_frames, stage_config, warnings
    )
    save_csv(split_table, SPATIAL_SPLITS_PATH)
    print("SPLITS complete", flush=True)

    same, cross, gaps = run_land_cover_models(
        stage03, city_frames, features, splits, stage_config, warnings
    )
    macro_matrix = make_matrix(same, cross, "macro_f1")
    expansion_matrix = make_matrix(same, cross, "expansion_f1")
    save_csv(same, SAME_CITY_PATH)
    save_csv(cross, CROSS_CITY_PATH)
    save_csv(macro_matrix, MACRO_MATRIX_PATH)
    save_csv(expansion_matrix, EXPANSION_MATRIX_PATH)
    save_csv(gaps, GAP_PATH)

    support = run_support_distance(city_frames, features, splits, seeds)
    save_csv(support, SUPPORT_PATH)
    pair = build_pair_mechanism(support, gaps)
    source, target = source_target_summaries(pair)
    save_csv(pair, MECHANISM_PATH)
    save_csv(source, SOURCE_PATH)
    save_csv(target, TARGET_PATH)

    correlations = {
        "h1_gap": correlation_bootstrap(
            pair["support_distance_mean"].to_numpy(),
            pair["macro_f1_gap_mean"].to_numpy(),
            BOOTSTRAP_SEED,
        ),
        "h2_gap": correlation_bootstrap(
            pair["class2_support_distance_mean"].to_numpy(),
            pair["expansion_f1_gap_mean"].to_numpy(),
            BOOTSTRAP_SEED + 1,
        ),
        "overall_expansion_gap": correlation_bootstrap(
            pair["support_distance_mean"].to_numpy(),
            pair["expansion_f1_gap_mean"].to_numpy(),
            BOOTSTRAP_SEED + 2,
        ),
        "overall_performance": correlation_bootstrap(
            pair["support_distance_mean"].to_numpy(),
            pair["cross_macro_f1_mean"].to_numpy(),
            BOOTSTRAP_SEED + 3,
        ),
        "class2_performance": correlation_bootstrap(
            pair["class2_support_distance_mean"].to_numpy(),
            pair["cross_expansion_f1_mean"].to_numpy(),
            BOOTSTRAP_SEED + 4,
        ),
        "overall_error": correlation_bootstrap(
            pair["support_distance_mean"].to_numpy(),
            pair["cross_error_rate_mean"].to_numpy(),
            BOOTSTRAP_SEED + 5,
        ),
    }
    h1_judgment = judge_hypothesis(correlations["h1_gap"], expected_positive=True)
    h2_judgment = judge_hypothesis(correlations["h2_gap"], expected_positive=True)

    domain = run_domain_classifier(
        stage03, city_frames, features, splits, stage_config, warnings
    )
    save_csv(domain, DOMAIN_PATH)
    consistency_checks, consistency_warnings = consistency_with_stage03b(gaps)
    warnings.extend(consistency_warnings)

    set_plot_style()
    plot_matrix(macro_matrix, "Five-city transfer matrix: Macro-F1", FIGURE_PATHS["macro"])
    plot_matrix(
        expansion_matrix,
        "Five-city transfer matrix: Expansion-F1",
        FIGURE_PATHS["expansion"],
    )
    plot_scatter(
        pair,
        "support_distance_mean",
        "macro_f1_gap_mean",
        correlations["h1_gap"],
        "Directed same-class 5-NN cosine support distance",
        "Target-normalized Macro-F1 gap",
        "Representation support distance vs Macro-F1 degradation",
        FIGURE_PATHS["h1"],
    )
    plot_scatter(
        pair,
        "class2_support_distance_mean",
        "expansion_f1_gap_mean",
        correlations["h2_gap"],
        "Class-2 directed 5-NN cosine support distance",
        "Target-normalized Expansion-F1 gap",
        "Expansion support distance vs Expansion-F1 degradation",
        FIGURE_PATHS["h2"],
    )
    plot_source(source, FIGURE_PATHS["source"])
    plot_target(target, FIGURE_PATHS["target"])
    plot_domain(domain, FIGURE_PATHS["domain"])

    elapsed = time.perf_counter() - started
    write_report(
        input_hash,
        stage_config,
        split_confirmations,
        same,
        cross,
        gaps,
        macro_matrix,
        expansion_matrix,
        support,
        pair,
        source,
        target,
        domain,
        correlations,
        h1_judgment,
        h2_judgment,
        consistency_checks,
        warnings,
        elapsed,
    )
    validate_outputs()
    if file_sha256(DATA_PATH) != input_hash:
        raise RuntimeError("Input hash changed during Stage 07")
    print(f"Stage 07 complete in {elapsed:.2f}s", flush=True)
    print(f"H1: {h1_judgment}; H2: {h2_judgment}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

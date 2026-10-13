"""Stage 08: label-free transfer-risk estimation and multi-source baselines.

Risk features are constructed from source-labelled training data, target AEF-192
features, and source-model probabilities.  Target labels are deliberately absent
from every risk-feature function and are joined only after the features are frozen.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
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
CONFIG = ROOT / "config.json"
BASELINE = ROOT / "scripts" / "02_baseline_experiment.py"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
LOGS = ROOT / "outputs" / "logs"
REPORT = LOGS / "stage08_label_free_risk_report.txt"

SPLITS_PATH = TABLES / "stage07_spatial_splits.csv"
SAME_PATH = TABLES / "stage07_same_city_runs.csv"
CROSS_PATH = TABLES / "stage07_cross_city_runs.csv"
MECHANISM_PATH = TABLES / "stage07_pair_mechanism.csv"
DIVERSITY_PATH = TABLES / "stage07b_source_representation_diversity.csv"
DIRECT_SELECTION_PATH = TABLES / "stage07b_source_selection_hits.csv"

PAIR_SEED_PATH = TABLES / "stage08_label_free_pair_seed_features.csv"
PAIR_PATH = TABLES / "stage08_label_free_pair_features.csv"
PROXY_PATH = TABLES / "stage08_proxy_validation.csv"
OOF_PATH = TABLES / "stage08_risk_oof_predictions.csv"
RISK_METRICS_PATH = TABLES / "stage08_risk_model_metrics.csv"
HIGH_RISK_PATH = TABLES / "stage08_high_risk_classification.csv"
RANKING_PATH = TABLES / "stage08_fold_ranking.csv"
EXPANSION_PATH = TABLES / "stage08_expansion_risk_metrics.csv"
MULTI_RUNS_PATH = TABLES / "stage08_multisource_runs.csv"
MULTI_SUMMARY_PATH = TABLES / "stage08_multisource_summary.csv"
MULTI_COMPARISON_PATH = TABLES / "stage08_multisource_comparison.csv"

FIGURE_PATHS = {
    "proxy": FIGURES / "32_label_free_proxy_vs_trueclass_distance.png",
    "oof": FIGURES / "33_oof_predicted_vs_actual_transfer_gap.png",
    "risk": FIGURES / "34_high_risk_transfer_calibration.png",
    "pooled": FIGURES / "35_multisource_vs_single_source.png",
    "improvement": FIGURES / "36_targetwise_multisource_improvement.png",
    "kunming": FIGURES / "37_kunming_multisource_comparison.png",
    "expansion": FIGURES / "38_expansion_label_free_risk.png",
    "ensemble": FIGURES / "39_pooled_vs_uniform_ensemble.png",
}

CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
ABBR = {"Shenzhen": "SZ", "Kunming": "KM", "Wuhan": "WH", "Xian": "XA", "Beijing": "BJ"}
SEEDS = [20261008, 20261009, 20261010, 20261011, 20261012]
K = 5
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "class0_f1", "class1_f1", "expansion_f1"]
FEATURE_COLUMNS = [f"A{i:02d}_{year}" for year in (2020, 2022, 2024) for i in range(64)]
PAIR_FEATURES = [
    "PCD_5NN",
    "Global_5NN",
    "mean_entropy",
    "median_entropy",
    "mean_one_minus_max_probability",
    "fraction_pred_class0",
    "fraction_pred_class1",
    "fraction_pred_class2",
    "PredExpansion_5NN",
    "predicted_class2_samples",
]
ACTUAL_COLUMNS = [
    "actual_cross_macro_f1",
    "actual_macro_gap",
    "actual_cross_expansion_f1",
    "actual_expansion_gap",
]
MODEL_TOLERANCE = 1e-6
TARGET_LABEL_USED_IN_RISK_FEATURES = False


@dataclass
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_blocks: int
    validation_blocks: int
    test_blocks: int


def load_stage03() -> Any:
    spec = importlib.util.spec_from_file_location("stage03_baseline_for_stage08", BASELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Stage 03 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def sample_hash(frame: pd.DataFrame, indices: np.ndarray) -> str:
    values = sorted(frame.iloc[np.asarray(indices, dtype=np.int64)]["sample_id"].astype(str))
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def load_inputs() -> tuple[
    dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray],
    dict[tuple[str, int], Split], dict[str, Any], pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    required = [DATA, CONFIG, SPLITS_PATH, SAME_PATH, CROSS_PATH, MECHANISM_PATH, DIVERSITY_PATH, DIRECT_SELECTION_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 08 inputs: {missing}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["selected_multicity_cities"] != CITIES:
        raise AssertionError("Frozen city order changed")
    if [int(v) for v in config["stage_03"]["random_seeds"]] != SEEDS:
        raise AssertionError("Frozen seeds changed")
    expected_xgb = {
        "n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
        "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0,
        "reg_lambda": 1.0, "early_stopping_rounds": 30, "tree_method": "hist", "n_jobs": 4,
    }
    if config["stage_03"]["models"]["XGBoost"] != expected_xgb:
        raise AssertionError("Frozen XGBoost parameters changed")
    if len(FEATURE_COLUMNS) != 192 or any(c.startswith("D_") for c in FEATURE_COLUMNS):
        raise AssertionError("AEF-192 definition changed")

    full = pd.read_parquet(DATA, engine="pyarrow")
    if full.shape != (90000, 205):
        raise AssertionError(f"Unexpected data shape {full.shape}")
    frames = {city: full.loc[full["city"] == city].copy().reset_index(drop=True) for city in CITIES}
    matrices = {city: frames[city][FEATURE_COLUMNS].to_numpy(np.float32, copy=False) for city in CITIES}
    labels = {city: frames[city]["class"].to_numpy(np.int64, copy=False) for city in CITIES}
    for city in CITIES:
        if matrices[city].shape != (18000, 192) or np.bincount(labels[city], minlength=3).tolist() != [6000, 6000, 6000]:
            raise AssertionError(f"{city}: frozen data invariant failed")
        if not np.isfinite(matrices[city]).all():
            raise AssertionError(f"{city}: non-finite AEF-192")

    split_table = pd.read_csv(SPLITS_PATH, dtype={"block_id": str})
    splits: dict[tuple[str, int], Split] = {}
    for city in CITIES:
        block_values = frames[city]["block_id"].astype(str)
        for seed in SEEDS:
            part = split_table.loc[(split_table.city == city) & (split_table.seed == seed)]
            arrays: dict[str, np.ndarray] = {}
            block_counts: dict[str, int] = {}
            for split_name in ("train", "validation", "test"):
                blocks = set(part.loc[part.split == split_name, "block_id"].astype(str))
                arrays[split_name] = np.flatnonzero(block_values.isin(blocks).to_numpy())
                block_counts[split_name] = len(blocks)
            union = np.concatenate(list(arrays.values()))
            if len(union) != 18000 or len(np.unique(union)) != 18000:
                raise AssertionError(f"{city} seed={seed}: split coverage/overlap failure")
            splits[(city, seed)] = Split(
                arrays["train"], arrays["validation"], arrays["test"],
                block_counts["train"], block_counts["validation"], block_counts["test"],
            )

    same = pd.read_csv(SAME_PATH)
    cross = pd.read_csv(CROSS_PATH)
    same = same.loc[(same.model == "XGBoost") & (same.feature_set == "AEF-192")].copy()
    cross = cross.loc[(cross.model == "XGBoost") & (cross.feature_set == "AEF-192")].copy()
    if len(same) != 25 or len(cross) != 100:
        raise AssertionError(f"Expected 25 same and 100 cross XGBoost runs; got {len(same)}, {len(cross)}")
    for row in same.itertuples(index=False):
        split = splits[(row.target_city, int(row.seed))]
        if index_hash(split.test) != row.target_test_index_sha256 or sample_hash(frames[row.target_city], split.test) != row.target_test_sample_id_sha256:
            raise AssertionError(f"Frozen test hash mismatch: {row.target_city} seed={row.seed}")

    same_lookup = same[["target_city", "seed", "macro_f1", "expansion_f1", "target_test_index_sha256", "target_test_sample_id_sha256"]].rename(
        columns={"macro_f1": "target_same_macro_f1", "expansion_f1": "target_same_expansion_f1", "target_test_index_sha256": "same_index_hash", "target_test_sample_id_sha256": "same_sample_hash"}
    )
    outcomes = cross.merge(same_lookup, on=["target_city", "seed"], how="left", validate="many_to_one")
    if not (outcomes.target_test_index_sha256 == outcomes.same_index_hash).all() or not (outcomes.target_test_sample_id_sha256 == outcomes.same_sample_hash).all():
        raise AssertionError("Stage 07 target test sets are not identical")
    outcomes["actual_cross_macro_f1"] = outcomes["macro_f1"]
    outcomes["actual_macro_gap"] = outcomes["target_same_macro_f1"] - outcomes["macro_f1"]
    outcomes["actual_cross_expansion_f1"] = outcomes["expansion_f1"]
    outcomes["actual_expansion_gap"] = outcomes["target_same_expansion_f1"] - outcomes["expansion_f1"]
    outcomes = outcomes[["source_city", "target_city", "seed"] + ACTUAL_COLUMNS]

    mechanism = pd.read_csv(MECHANISM_PATH)
    diversity = pd.read_csv(DIVERSITY_PATH)
    direct = pd.read_csv(DIRECT_SELECTION_PATH)
    return frames, matrices, labels, splits, config, same, cross, outcomes, mechanism, diversity, direct


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise AssertionError("Zero/non-finite AEF vector norm")
    output = x / norms
    if np.max(np.abs(np.linalg.norm(output, axis=1) - 1.0)) > 3e-6:
        raise AssertionError("L2 normalization failed")
    return output.astype(np.float32, copy=False)


def exact_mean_cosine_knn(reference: np.ndarray, query: np.ndarray, k: int = K, batch_size: int = 256) -> np.ndarray:
    """Exact batched cosine kNN for already L2-normalized rows."""
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise AssertionError("Invalid kNN shapes")
    if len(reference) < k:
        raise AssertionError(f"Reference has fewer than {k} samples")
    result = np.empty(len(query), dtype=np.float64)
    reference_t = np.ascontiguousarray(reference.T)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        similarity = np.asarray(query[start:stop] @ reference_t, dtype=np.float32)
        top = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
        result[start:stop] = 1.0 - np.clip(top, -1.0, 1.0).mean(axis=1)
    return result


def compute_label_free_features(
    x_target: np.ndarray,
    predicted_probabilities: np.ndarray,
    x_source_train: np.ndarray,
    y_source_train: np.ndarray,
) -> dict[str, float | int]:
    """Construct risk features; intentionally has no y_target parameter."""
    if x_target.ndim != 2 or x_target.shape[1] != 192:
        raise AssertionError("Risk function requires target AEF-192 only")
    if predicted_probabilities.shape != (len(x_target), 3):
        raise AssertionError("Invalid target probability shape")
    if y_source_train.shape != (len(x_source_train),):
        raise AssertionError("Invalid source training label shape")
    x_t = l2_normalize_rows(x_target)
    x_s = l2_normalize_rows(x_source_train)
    y_hat = np.argmax(predicted_probabilities, axis=1).astype(np.int64)

    global_distance = exact_mean_cosine_knn(x_s, x_t)
    pseudo_distance = np.empty(len(x_t), dtype=np.float64)
    for cls in (0, 1, 2):
        target_mask = y_hat == cls
        if target_mask.any():
            source_reference = x_s[y_source_train == cls]
            pseudo_distance[target_mask] = exact_mean_cosine_knn(source_reference, x_t[target_mask])

    probabilities = np.clip(np.asarray(predicted_probabilities, dtype=np.float64), 1e-15, 1.0)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1) / math.log(3.0)
    counts = np.bincount(y_hat, minlength=3)
    class2_mask = y_hat == 2
    pred_expansion = float(np.mean(pseudo_distance[class2_mask])) if class2_mask.any() else np.nan
    return {
        "PCD_5NN": float(np.mean(pseudo_distance)),
        "Global_5NN": float(np.mean(global_distance)),
        "mean_entropy": float(np.mean(entropy)),
        "median_entropy": float(np.median(entropy)),
        "mean_one_minus_max_probability": float(np.mean(1.0 - probabilities.max(axis=1))),
        "fraction_pred_class0": float(counts[0] / len(y_hat)),
        "fraction_pred_class1": float(counts[1] / len(y_hat)),
        "fraction_pred_class2": float(counts[2] / len(y_hat)),
        "PredExpansion_5NN": pred_expansion,
        "predicted_class2_samples": int(counts[2]),
    }


def metric_difference(current: dict[str, Any], reference: pd.Series) -> float:
    return max(abs(float(current[name]) - float(reference[name])) for name in METRICS)


def train_source_models_and_features(
    stage03: Any,
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    splits: dict[tuple[str, int], Split],
    stage_config: dict[str, Any],
    same: pd.DataFrame,
    cross: pd.DataFrame,
    outcomes: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]], dict[str, float]]:
    feature_rows: list[dict[str, Any]] = []
    ensemble_metrics: dict[tuple[str, int], dict[str, Any]] = {}
    validation_max = {"same_city": 0.0, "cross_city": 0.0}

    for seed in SEEDS:
        models: dict[str, Any] = {}
        probability_cache: dict[tuple[str, str], np.ndarray] = {}
        for source in CITIES:
            split = splits[(source, seed)]
            model = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
            model, seconds, iterations = stage03.fit_model(
                model, "XGBoost", matrices[source][split.train], labels[source][split.train],
                matrices[source][split.validation], labels[source][split.validation], warnings,
                f"Stage 08 frozen source {source} seed={seed}",
            )
            models[source] = model
            print(f"SOURCE MODEL seed={seed} city={source} seconds={seconds:.1f} iterations={iterations}", flush=True)

        # This block freezes all label-free pair features before target outcomes are accessed.
        seed_start = len(feature_rows)
        for source in CITIES:
            source_split = splits[(source, seed)]
            for target in CITIES:
                if source == target:
                    continue
                target_index = splits[(target, seed)].test
                x_target_only = matrices[target][target_index].copy()
                probability = models[source].predict_proba(x_target_only)
                probability_cache[(source, target)] = probability
                risk = compute_label_free_features(
                    x_target=x_target_only,
                    predicted_probabilities=probability,
                    x_source_train=matrices[source][source_split.train],
                    y_source_train=labels[source][source_split.train],
                )
                if risk["predicted_class2_samples"] == 0:
                    warnings.append(f"{source}->{target} seed={seed}: zero predicted Class2; PredExpansion-5NN is NaN")
                feature_rows.append({
                    "source_city": source, "source_abbreviation": ABBR[source],
                    "target_city": target, "target_abbreviation": ABBR[target],
                    "direction": f"{source} -> {target}", "seed": seed,
                    "feature_set": "AEF-192", "knn_k": K,
                    "target_unlabeled_samples": len(target_index),
                    "source_reference_samples": len(source_split.train),
                    "target_test_index_sha256": index_hash(target_index),
                    "target_label_used_in_risk_features": False,
                    "risk_features_frozen_before_outcome_join": True,
                    **risk,
                })
        if len(feature_rows) - seed_start != 20:
            raise AssertionError("Feature freeze did not produce 20 directed pairs for seed")

        # Evaluation and model reproduction checks begin only after the seed's risk features are frozen.
        for source in CITIES:
            same_index = splits[(source, seed)].test
            same_prediction = models[source].predict(matrices[source][same_index]).astype(int)
            same_metrics = stage03.land_cover_metrics(labels[source][same_index], same_prediction)
            same_ref = same.loc[(same.source_city == source) & (same.seed == seed)].iloc[0]
            validation_max["same_city"] = max(validation_max["same_city"], metric_difference(same_metrics, same_ref))
            for target in CITIES:
                if source == target:
                    continue
                target_index = splits[(target, seed)].test
                prediction = np.argmax(probability_cache[(source, target)], axis=1)
                current = stage03.land_cover_metrics(labels[target][target_index], prediction)
                reference = cross.loc[(cross.source_city == source) & (cross.target_city == target) & (cross.seed == seed)].iloc[0]
                validation_max["cross_city"] = max(validation_max["cross_city"], metric_difference(current, reference))

        # Uniform soft-voting is convenient because all five frozen source models are in memory.
        for target in CITIES:
            target_index = splits[(target, seed)].test
            probabilities = [probability_cache[(source, target)] for source in CITIES if source != target]
            ensemble_prediction = np.argmax(np.mean(probabilities, axis=0), axis=1)
            ensemble_metrics[(target, seed)] = stage03.land_cover_metrics(labels[target][target_index], ensemble_prediction)
        del models, probability_cache
        gc.collect()

    if validation_max["same_city"] > MODEL_TOLERANCE or validation_max["cross_city"] > MODEL_TOLERANCE:
        raise RuntimeError(f"Major Stage 07 model reproduction mismatch: {validation_max}")

    features_only = pd.DataFrame(feature_rows)
    if len(features_only) != 100 or features_only.duplicated(["source_city", "target_city", "seed"]).any():
        raise AssertionError("Expected exactly 100 unique pair-seed feature rows")
    if "class" in features_only.columns:
        raise AssertionError("Target class leaked into feature table")
    pair_seed = features_only.merge(outcomes, on=["source_city", "target_city", "seed"], how="left", validate="one_to_one")
    if pair_seed[ACTUAL_COLUMNS].isna().any().any():
        raise AssertionError("Outcome join failed after risk-feature freeze")
    return pair_seed, ensemble_metrics, validation_max


def aggregate_pairs(pair_seed: pd.DataFrame, same: pd.DataFrame, diversity: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    numeric = PAIR_FEATURES + ACTUAL_COLUMNS
    rows: list[dict[str, Any]] = []
    same_quality = same.groupby("source_city")["macro_f1"].mean().to_dict()
    diversity_mean = diversity.loc[diversity.class_scope == "Mean across classes"].set_index("city")["participation_ratio"].to_dict()
    for (source, target), group in pair_seed.groupby(["source_city", "target_city"], sort=False):
        row: dict[str, Any] = {
            "source_city": source, "source_abbreviation": ABBR[source],
            "target_city": target, "target_abbreviation": ABBR[target],
            "direction": f"{source} -> {target}", "seed_count": len(group),
            "source_same_city_error": 1.0 - float(same_quality[source]),
            "source_participation_ratio": float(diversity_mean[source]),
            "target_label_used_in_risk_features": False,
        }
        for name in numeric:
            row[name] = float(group[name].mean())
            row[f"{name}_seed_std"] = float(group[name].std(ddof=1))
        rows.append(row)
    pair = pd.DataFrame(rows)
    if len(pair) != 20:
        raise AssertionError("Pair aggregation must yield n=20")
    check = pair.merge(mechanism[["source_city", "target_city", "macro_f1_gap_mean", "expansion_f1_gap_mean"]], on=["source_city", "target_city"], validate="one_to_one")
    if np.max(np.abs(check.actual_macro_gap - check.macro_f1_gap_mean)) > 1e-9 or np.max(np.abs(check.actual_expansion_gap - check.expansion_f1_gap_mean)) > 1e-9:
        raise AssertionError("Stage 08 outcomes do not reproduce Stage 07 pair gaps")
    return pair.sort_values(["source_city", "target_city"]).reset_index(drop=True)


def safe_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[int, float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    if keep.sum() < 3:
        return int(keep.sum()), np.nan, np.nan
    return int(keep.sum()), float(spearmanr(x_array[keep], y_array[keep]).statistic), float(pearsonr(x_array[keep], y_array[keep]).statistic)


def pair_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    index = {city: i for i, city in enumerate(CITIES)}
    matrix = np.full((5, 5), np.nan, dtype=float)
    for row in frame.itertuples(index=False):
        matrix[index[row.source_city], index[row.target_city]] = float(getattr(row, column))
    if not np.isfinite(matrix[~np.eye(5, dtype=bool)]).all():
        raise ValueError(f"Incomplete directed matrix for {column}")
    return matrix


def exact_qap(frame: pd.DataFrame, feature: str, outcome: str) -> dict[str, float | int]:
    try:
        x = pair_matrix(frame, feature)
        y = pair_matrix(frame, outcome)
    except ValueError:
        return {"qap_permutations": 0, "qap_observed_spearman": np.nan, "qap_one_sided_p": np.nan, "qap_two_sided_p": np.nan}
    mask = ~np.eye(5, dtype=bool)
    observed = float(spearmanr(x[mask], y[mask]).statistic)
    null = []
    for permutation in itertools.permutations(range(5)):
        permuted = x[np.ix_(permutation, permutation)]
        null.append(float(spearmanr(permuted[mask], y[mask]).statistic))
    null_array = np.asarray(null)
    return {
        "qap_permutations": 120,
        "qap_observed_spearman": observed,
        "qap_one_sided_p": float(np.mean(null_array >= observed - 1e-12)),
        "qap_two_sided_p": float(np.mean(np.abs(null_array) >= abs(observed) - 1e-12)),
    }


def correlation_audits(pair: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    merged = pair.merge(mechanism[["source_city", "target_city", "support_distance_mean", "class2_support_distance_mean"]], on=["source_city", "target_city"], validate="one_to_one")
    specifications = [
        ("proxy_validation", "PCD-5NN vs Stage07 true-class support distance", "support_distance_mean", "PCD_5NN", False),
        ("proxy_validation", "Global-5NN vs Stage07 true-class support distance", "support_distance_mean", "Global_5NN", False),
        ("proxy_validation", "PredExpansion-5NN vs Stage07 Class2 support distance", "class2_support_distance_mean", "PredExpansion_5NN", False),
        ("direct_risk_association", "PCD-5NN vs actual Macro gap", "PCD_5NN", "actual_macro_gap", True),
        ("direct_risk_association", "Global-5NN vs actual Macro gap", "Global_5NN", "actual_macro_gap", True),
        ("direct_risk_association", "Mean entropy vs actual Macro gap", "mean_entropy", "actual_macro_gap", True),
        ("direct_risk_association", "PredExpansion-5NN vs actual Expansion gap", "PredExpansion_5NN", "actual_expansion_gap", True),
    ]
    rows = []
    for group, label, x_col, y_col, use_qap in specifications:
        n, rho, r = safe_correlation(merged[x_col], merged[y_col])
        qap = exact_qap(merged, x_col, y_col) if use_qap else {
            "qap_permutations": np.nan, "qap_observed_spearman": np.nan,
            "qap_one_sided_p": np.nan, "qap_two_sided_p": np.nan,
        }
        rows.append({"analysis_group": group, "analysis": label, "x_column": x_col, "y_column": y_col, "n_directed_pairs": n, "spearman_rho": rho, "pearson_r": r, **qap})
    return pd.DataFrame(rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], outcome: str) -> np.ndarray:
    forbidden = {"class", "actual_cross_macro_f1", "actual_macro_gap", "actual_cross_expansion_f1", "actual_expansion_gap"}
    if forbidden.intersection(features):
        raise AssertionError(f"Outcome/target label leaked into risk design: {features}")
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        return np.full(len(test), np.nan)
    model = LinearRegression().fit(x_train, train[outcome].to_numpy(dtype=float))
    return model.predict(x_test)


def node_loco(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    macro_models = {
        "R1": ["PCD_5NN"],
        "R2": ["PCD_5NN", "source_same_city_error"],
        "R3": ["Global_5NN"],
        "R4": ["mean_entropy"],
        "RX_EXPLORATORY": ["PCD_5NN", "source_participation_ratio"],
    }
    for held_out in CITIES:
        train = pair.loc[(pair.source_city != held_out) & (pair.target_city != held_out)].copy()
        test = pair.loc[pair.target_city == held_out].copy()
        if len(train) != 12 or len(test) != 4 or held_out in set(train.source_city) or held_out in set(train.target_city):
            raise AssertionError(f"Invalid node-LOCO fold {held_out}")
        predictions: dict[str, np.ndarray] = {"R0": np.repeat(train.actual_macro_gap.mean(), len(test))}
        for model, features in macro_models.items():
            predictions[model] = fit_predict(train, test, features, "actual_macro_gap")
        expansion_predictions = {"E0": np.repeat(train.actual_expansion_gap.mean(), len(test))}
        expansion_predictions["E1"] = fit_predict(train, test, ["PredExpansion_5NN"], "actual_expansion_gap")
        for position, (_, item) in enumerate(test.iterrows()):
            rows.append({
                "held_out_target_city": held_out,
                "source_city": item.source_city, "target_city": item.target_city,
                "direction": item.direction,
                "actual_macro_gap": item.actual_macro_gap,
                "actual_expansion_gap": item.actual_expansion_gap,
                **{f"predicted_macro_gap_{name}": values[position] for name, values in predictions.items()},
                **{f"predicted_expansion_gap_{name}": values[position] for name, values in expansion_predictions.items()},
                "calibration_train_pairs": 12, "fold_test_pairs": 4,
                "held_out_city_absent_as_source_and_target": True,
                "target_label_used_in_risk_features": False,
            })
    output = pd.DataFrame(rows)
    if len(output) != 20 or output.duplicated(["source_city", "target_city"]).any():
        raise AssertionError("Node-LOCO must produce one OOF prediction for every directed pair")
    return output.sort_values(["target_city", "source_city"]).reset_index(drop=True)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    tss = float(np.sum((actual - actual.mean()) ** 2))
    rss = float(np.sum((actual - predicted) ** 2))
    return {
        "n_oof": len(actual), "mae": mae, "rmse": rmse,
        "oof_r_squared": 1.0 - rss / tss if tss > 0 else np.nan,
        "spearman_rho": float(spearmanr(predicted, actual).statistic) if len(actual) >= 3 else np.nan,
        "pearson_r": float(pearsonr(predicted, actual).statistic) if len(actual) >= 3 else np.nan,
    }


def classification_metrics(actual_gap: np.ndarray, predicted_gap: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual_gap = np.asarray(actual_gap, dtype=float)
    predicted_gap = np.asarray(predicted_gap, dtype=float)
    keep = np.isfinite(actual_gap) & np.isfinite(predicted_gap)
    actual = (actual_gap[keep] >= threshold).astype(int)
    predicted = (predicted_gap[keep] >= threshold).astype(int)
    return {
        "n": len(actual), "threshold": threshold,
        "actual_high_risk_count": int(actual.sum()), "predicted_high_risk_count": int(predicted.sum()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(actual, predicted_gap[keep])) if len(np.unique(actual)) == 2 else np.nan,
    }


def evaluate_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    rows = []
    formulas = {
        "R0": "training-pair mean", "R1": "Macro gap ~ PCD_5NN",
        "R2": "Macro gap ~ PCD_5NN + source_same_city_error",
        "R3": "Macro gap ~ Global_5NN", "R4": "Macro gap ~ mean_entropy",
        "RX_EXPLORATORY": "Macro gap ~ PCD_5NN + source_participation_ratio",
    }
    for model, formula in formulas.items():
        values = regression_metrics(oof.actual_macro_gap, oof[f"predicted_macro_gap_{model}"])
        rows.append({"model": model, "formula": formula, "status": "EXPLORATORY" if model.startswith("RX") else "predefined", **values})
    metrics = pd.DataFrame(rows)
    r0_mae = float(metrics.loc[metrics.model == "R0", "mae"].iloc[0])
    metrics["relative_mae_improvement_vs_R0"] = (r0_mae - metrics.mae) / r0_mae

    macro_high = {"outcome": "Macro-F1 gap", "model": "R1", **classification_metrics(oof.actual_macro_gap, oof.predicted_macro_gap_R1, 0.05)}
    expansion_high = {"outcome": "Expansion-F1 gap", "model": "E1", **classification_metrics(oof.actual_expansion_gap, oof.predicted_expansion_gap_E1, 0.07)}
    high = pd.DataFrame([macro_high, expansion_high])

    ranking_rows = []
    for target, group in oof.groupby("held_out_target_city", sort=False):
        rho = float(spearmanr(group.predicted_macro_gap_R1, group.actual_macro_gap).statistic)
        tau = float(kendalltau(group.predicted_macro_gap_R1, group.actual_macro_gap).statistic)
        actual_order = group.sort_values("actual_macro_gap").source_city.tolist()
        predicted_order = group.sort_values("predicted_macro_gap_R1").source_city.tolist()
        ranking_rows.append({
            "held_out_target_city": target, "candidate_sources": 4,
            "spearman_rho": rho, "kendall_tau": tau,
            "predicted_safest_source": predicted_order[0], "actual_safest_source": actual_order[0],
            "predicted_top2_sources": " | ".join(predicted_order[:2]),
            "top1_safest_source_hit": predicted_order[0] == actual_order[0],
            "actual_safest_in_predicted_top2": actual_order[0] in predicted_order[:2],
            "predicted_ranking": " < ".join(predicted_order), "actual_ranking": " < ".join(actual_order),
        })
    ranking = pd.DataFrame(ranking_rows)

    expansion_rows = []
    for model in ("E0", "E1"):
        values = regression_metrics(oof.actual_expansion_gap, oof[f"predicted_expansion_gap_{model}"])
        expansion_rows.append({"model": model, "formula": "training-pair mean" if model == "E0" else "Expansion gap ~ PredExpansion_5NN", **values})
    expansion = pd.DataFrame(expansion_rows)
    e0_mae = float(expansion.loc[expansion.model == "E0", "mae"].iloc[0])
    expansion["relative_mae_improvement_vs_E0"] = (e0_mae - expansion.mae) / e0_mae
    for key, value in expansion_high.items():
        if key not in {"outcome", "model"}:
            expansion[f"high_risk_{key}"] = value

    r1 = metrics.loc[metrics.model == "R1"].iloc[0]
    macro_judgment = (
        "SUPPORTED" if r1.spearman_rho >= 0.50 and r1.relative_mae_improvement_vs_R0 >= 0.15 and macro_high["auroc"] >= 0.75
        else "PARTIALLY SUPPORTED" if r1.spearman_rho > 0 and (r1.relative_mae_improvement_vs_R0 >= 0.15 or macro_high["auroc"] >= 0.75)
        else "NOT SUPPORTED"
    )
    e1 = expansion.loc[expansion.model == "E1"].iloc[0]
    expansion_judgment = (
        "SUPPORTED" if e1.spearman_rho >= 0.50 and e1.relative_mae_improvement_vs_E0 >= 0.15 and expansion_high["auroc"] >= 0.75
        else "PARTIALLY SUPPORTED" if e1.spearman_rho > 0 and (e1.relative_mae_improvement_vs_E0 >= 0.15 or expansion_high["auroc"] >= 0.75)
        else "NOT SUPPORTED"
    )
    metrics["predefined_stage08a_judgment"] = macro_judgment
    expansion["predefined_expansion_judgment"] = expansion_judgment
    return metrics, high, ranking, expansion, macro_judgment, expansion_judgment


def train_multisource(
    stage03: Any,
    matrices: dict[str, np.ndarray], labels: dict[str, np.ndarray],
    splits: dict[tuple[str, int], Split], stage_config: dict[str, Any],
    same: pd.DataFrame, cross: pd.DataFrame, pair_seed: pd.DataFrame,
    direct: pd.DataFrame, ensemble_metrics: dict[tuple[str, int], dict[str, Any]],
    warnings: list[str],
) -> pd.DataFrame:
    rows = []
    direct_lookup = direct.set_index("target_city")
    for seed in SEEDS:
        for target in CITIES:
            source_cities = [city for city in CITIES if city != target]
            x_train = np.concatenate([matrices[city][splits[(city, seed)].train] for city in source_cities])
            y_train = np.concatenate([labels[city][splits[(city, seed)].train] for city in source_cities])
            x_validation = np.concatenate([matrices[city][splits[(city, seed)].validation] for city in source_cities])
            y_validation = np.concatenate([labels[city][splits[(city, seed)].validation] for city in source_cities])
            target_index = splits[(target, seed)].test
            model = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
            model, seconds, iterations = stage03.fit_model(
                model, "XGBoost", x_train, y_train, x_validation, y_validation, warnings,
                f"Stage 08 pooled target={target} seed={seed}",
            )
            prediction = model.predict(matrices[target][target_index]).astype(int)
            pooled = stage03.land_cover_metrics(labels[target][target_index], prediction)
            same_row = same.loc[(same.target_city == target) & (same.seed == seed)].iloc[0]
            source_runs = cross.loc[(cross.target_city == target) & (cross.seed == seed)].copy()
            if len(source_runs) != 4:
                raise AssertionError("Each target/seed needs four Stage 07 source runs")
            pcd_group = pair_seed.loc[(pair_seed.target_city == target) & (pair_seed.seed == seed)]
            pcd_source = pcd_group.sort_values("PCD_5NN").iloc[0].source_city
            pcd_run = source_runs.loc[source_runs.source_city == pcd_source].iloc[0]
            macro_oracle = source_runs.sort_values("macro_f1", ascending=False).iloc[0]
            macro_worst = source_runs.sort_values("macro_f1").iloc[0]
            expansion_oracle = source_runs.sort_values("expansion_f1", ascending=False).iloc[0]
            expansion_worst = source_runs.sort_values("expansion_f1").iloc[0]
            direct_macro_source = direct_lookup.loc[target, "macro_predicted_best_source"]
            direct_exp_source = direct_lookup.loc[target, "expansion_predicted_best_source"]
            direct_macro_run = source_runs.loc[source_runs.source_city == direct_macro_source].iloc[0]
            direct_exp_run = source_runs.loc[source_runs.source_city == direct_exp_source].iloc[0]
            ensemble = ensemble_metrics[(target, seed)]
            rows.append({
                "target_city": target, "target_abbreviation": ABBR[target], "seed": seed,
                "source_cities": " | ".join(source_cities), "source_city_count": 4,
                "train_samples": len(x_train), "validation_samples": len(x_validation),
                "target_test_samples": len(target_index),
                "target_test_index_sha256": index_hash(target_index),
                "target_test_sample_id_sha256": sample_hash(pd.DataFrame({"sample_id": np.arange(len(matrices[target]))}), target_index) if False else same_row.target_test_sample_id_sha256,
                "training_seconds": seconds, "fitted_iterations": iterations,
                **{f"pooled_{name}": pooled[name] for name in METRICS},
                "target_same_city_macro_f1": same_row.macro_f1,
                "target_same_city_expansion_f1": same_row.expansion_f1,
                "pooled_macro_gap": same_row.macro_f1 - pooled["macro_f1"],
                "pooled_expansion_gap": same_row.expansion_f1 - pooled["expansion_f1"],
                "mean_single_source_macro_f1": source_runs.macro_f1.mean(),
                "mean_single_source_expansion_f1": source_runs.expansion_f1.mean(),
                "oracle_best_macro_source": macro_oracle.source_city,
                "oracle_best_macro_f1": macro_oracle.macro_f1,
                "oracle_best_expansion_source": expansion_oracle.source_city,
                "oracle_best_expansion_f1": expansion_oracle.expansion_f1,
                "worst_macro_source": macro_worst.source_city,
                "worst_macro_f1": macro_worst.macro_f1,
                "worst_expansion_source": expansion_worst.source_city,
                "worst_expansion_f1": expansion_worst.expansion_f1,
                "PCD_selected_source": pcd_source,
                "PCD_selected_macro_f1": pcd_run.macro_f1,
                "PCD_selected_expansion_f1": pcd_run.expansion_f1,
                "stage07b_direct_macro_selected_source": direct_macro_source,
                "stage07b_direct_macro_selected_f1": direct_macro_run.macro_f1,
                "stage07b_direct_expansion_selected_source": direct_exp_source,
                "stage07b_direct_expansion_selected_f1": direct_exp_run.expansion_f1,
                "pooled_minus_mean_single_macro_f1": pooled["macro_f1"] - source_runs.macro_f1.mean(),
                "pooled_minus_mean_single_expansion_f1": pooled["expansion_f1"] - source_runs.expansion_f1.mean(),
                "pooled_minus_PCD_selected_macro_f1": pooled["macro_f1"] - pcd_run.macro_f1,
                "uniform_ensemble_macro_f1": ensemble["macro_f1"],
                "uniform_ensemble_expansion_f1": ensemble["expansion_f1"],
                "uniform_ensemble_accuracy": ensemble["accuracy"],
                "uniform_ensemble_balanced_accuracy": ensemble["balanced_accuracy"],
                "uniform_ensemble_class0_f1": ensemble["class0_f1"],
                "uniform_ensemble_class1_f1": ensemble["class1_f1"],
                "uniform_ensemble_class2_f1": ensemble["expansion_f1"],
                "target_labels_used_for_training_or_early_stopping": False,
                "oracle_is_retrospective_not_deployable": True,
            })
            print(f"POOLED seed={seed} target={target} macro={pooled['macro_f1']:.4f} seconds={seconds:.1f}", flush=True)
            del model, x_train, y_train, x_validation, y_validation
            gc.collect()
    output = pd.DataFrame(rows)
    if len(output) != 25 or output.duplicated(["target_city", "seed"]).any():
        raise AssertionError("Expected 25 pooled target-seed runs")
    return output.sort_values(["target_city", "seed"]).reset_index(drop=True)


def summarize_multisource(runs: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    metrics = [c for c in runs.columns if c.startswith("pooled_") or c.startswith("mean_single_") or c.startswith("PCD_selected_") or c.startswith("uniform_ensemble_")]
    summary_rows = []
    for target, group in runs.groupby("target_city", sort=False):
        row: dict[str, Any] = {"target_city": target, "target_abbreviation": ABBR[target], "seed_count": len(group)}
        for column in metrics:
            if pd.api.types.is_numeric_dtype(group[column]):
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    incoming_risk = oof.groupby("target_city")["predicted_macro_gap_R1"].mean().to_dict()
    comparison_rows = []
    for target, group in runs.groupby("target_city", sort=False):
        comparison_rows.append({
            "target_city": target, "target_abbreviation": ABBR[target],
            "pooled_macro_f1": float(group.pooled_macro_f1.mean()),
            "pooled_expansion_f1": float(group.pooled_expansion_f1.mean()),
            "mean_single_source_macro_f1": float(group.mean_single_source_macro_f1.mean()),
            "mean_single_source_expansion_f1": float(group.mean_single_source_expansion_f1.mean()),
            "oracle_best_single_macro_f1": float(group.oracle_best_macro_f1.mean()),
            "oracle_best_single_expansion_f1": float(group.oracle_best_expansion_f1.mean()),
            "worst_single_macro_f1": float(group.worst_macro_f1.mean()),
            "PCD_selected_macro_f1": float(group.PCD_selected_macro_f1.mean()),
            "stage07b_direct_selected_macro_f1": float(group.stage07b_direct_macro_selected_f1.mean()),
            "uniform_ensemble_macro_f1": float(group.uniform_ensemble_macro_f1.mean()),
            "uniform_ensemble_expansion_f1": float(group.uniform_ensemble_expansion_f1.mean()),
            "pooled_minus_mean_single_macro_f1": float(group.pooled_minus_mean_single_macro_f1.mean()),
            "pooled_minus_mean_single_expansion_f1": float(group.pooled_minus_mean_single_expansion_f1.mean()),
            "pooled_minus_PCD_selected_macro_f1": float(group.pooled_minus_PCD_selected_macro_f1.mean()),
            "mean_R1_predicted_incoming_macro_gap": float(incoming_risk[target]),
            "paired_seed_count": len(group),
        })
    comparison = pd.DataFrame(comparison_rows)
    improved = comparison.pooled_minus_mean_single_macro_f1 > 0
    count = int(improved.sum())
    mean_improvement = float(comparison.pooled_minus_mean_single_macro_f1.mean())
    judgment = "SUPPORTED" if count >= 4 and mean_improvement > 0 else "PARTIALLY SUPPORTED" if count == 3 and mean_improvement > 0 else "NOT SUPPORTED"
    comparison["targets_improved_over_mean_single"] = count
    comparison["mean_target_improvement"] = mean_improvement
    comparison["median_target_improvement"] = float(comparison.pooled_minus_mean_single_macro_f1.median())
    comparison["worst_target_improvement"] = float(comparison.pooled_minus_mean_single_macro_f1.min())
    comparison["stage08b_predefined_judgment"] = judgment
    return summary, comparison, judgment


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)


def create_figures(pair: pd.DataFrame, mechanism: pd.DataFrame, oof: pd.DataFrame, comparison: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    merged = pair.merge(mechanism[["source_city", "target_city", "support_distance_mean"]], on=["source_city", "target_city"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(merged.support_distance_mean, merged.PCD_5NN, s=48, color="#176B87")
    for row in merged.itertuples(): ax.annotate(f"{ABBR[row.source_city]}→{ABBR[row.target_city]}", (row.support_distance_mean, row.PCD_5NN), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Stage07 true-class support distance", ylabel="Stage08 PCD-5NN", title="Label-free proxy vs true-class mechanism distance")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["proxy"], dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(oof.actual_macro_gap, oof.predicted_macro_gap_R1, s=55, c=[CITIES.index(x) for x in oof.target_city], cmap="viridis")
    limits = [min(oof.actual_macro_gap.min(), oof.predicted_macro_gap_R1.min()), max(oof.actual_macro_gap.max(), oof.predicted_macro_gap_R1.max())]
    ax.plot(limits, limits, "--", color="0.35", linewidth=1)
    ax.set(xlabel="Actual target-normalized Macro-F1 gap", ylabel="Node-LOCO R1 predicted gap", title="Out-of-city label-free transfer-risk predictions")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["oof"], dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    actual_high = oof.actual_macro_gap >= 0.05
    ax.scatter(oof.predicted_macro_gap_R1, oof.actual_macro_gap, c=np.where(actual_high, "#C73E1D", "#2A9D8F"), s=60)
    ax.axvline(0.05, color="0.3", linestyle="--"); ax.axhline(0.05, color="0.3", linestyle="--")
    ax.set(xlabel="R1 predicted Macro-F1 gap", ylabel="Actual Macro-F1 gap", title="Fixed-threshold high-risk calibration (0.05)")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["risk"], dpi=300); plt.close(fig)

    positions = np.arange(5); width = 0.35
    ordered = comparison.set_index("target_city").loc[CITIES].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(positions - width/2, ordered.mean_single_source_macro_f1, width, label="Mean single source", color="#9ECAE1")
    ax.bar(positions + width/2, ordered.pooled_macro_f1, width, label="Pooled", color="#176B87")
    ax.set_xticks(positions, [ABBR[x] for x in CITIES]); ax.set_ylabel("Macro-F1"); ax.set_title("Multi-source pooled vs mean single-source transfer"); ax.legend(frameon=False)
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["pooled"], dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = np.where(ordered.pooled_minus_mean_single_macro_f1 >= 0, "#2A9D8F", "#C73E1D")
    ax.bar([ABBR[x] for x in ordered.target_city], ordered.pooled_minus_mean_single_macro_f1, color=colors)
    ax.axhline(0, color="0.25", linewidth=1); ax.set(ylabel="Pooled − mean single-source Macro-F1", title="Target-wise multi-source improvement")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["improvement"], dpi=300); plt.close(fig)

    km = ordered.loc[ordered.target_city == "Kunming"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    labels_plot = ["Mean single", "PCD selected", "Pooled", "Oracle"]
    values = [km.mean_single_source_macro_f1, km.PCD_selected_macro_f1, km.pooled_macro_f1, km.oracle_best_single_macro_f1]
    ax.bar(labels_plot, values, color=["#9ECAE1", "#F4A261", "#176B87", "#777777"])
    ax.set_ylim(max(0, min(values)-0.08), 1.0); ax.set_ylabel("Macro-F1"); ax.set_title("Kunming transfer comparison")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["kunming"], dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(oof.actual_expansion_gap, oof.predicted_expansion_gap_E1, s=55, color="#6A4C93")
    limits = [min(oof.actual_expansion_gap.min(), oof.predicted_expansion_gap_E1.min()), max(oof.actual_expansion_gap.max(), oof.predicted_expansion_gap_E1.max())]
    ax.plot(limits, limits, "--", color="0.35"); ax.axvline(0.07, color="0.5", linestyle=":"); ax.axhline(0.07, color="0.5", linestyle=":")
    ax.set(xlabel="Actual Expansion-F1 gap", ylabel="Node-LOCO E1 predicted gap", title="Label-free Expansion risk")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["expansion"], dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(ordered.uniform_ensemble_macro_f1, ordered.pooled_macro_f1, s=65, color="#264653")
    for row in ordered.itertuples(): ax.annotate(ABBR[row.target_city], (row.uniform_ensemble_macro_f1, row.pooled_macro_f1), xytext=(4, 4), textcoords="offset points")
    limits = [min(ordered.uniform_ensemble_macro_f1.min(), ordered.pooled_macro_f1.min()), max(ordered.uniform_ensemble_macro_f1.max(), ordered.pooled_macro_f1.max())]
    ax.plot(limits, limits, "--", color="0.35"); ax.set(xlabel="Uniform source ensemble Macro-F1", ylabel="Pooled Macro-F1", title="Pooled training vs uniform soft-voting")
    style_axis(ax); fig.tight_layout(); fig.savefig(FIGURE_PATHS["ensemble"], dpi=300); plt.close(fig)


def build_report(
    pair_seed: pd.DataFrame, pair: pd.DataFrame, proxy: pd.DataFrame, oof: pd.DataFrame,
    risk_metrics: pd.DataFrame, high: pd.DataFrame, ranking: pd.DataFrame,
    expansion: pd.DataFrame, multi_summary: pd.DataFrame, comparison: pd.DataFrame,
    validation_max: dict[str, float], macro_judgment: str, expansion_judgment: str,
    pooled_judgment: str, warnings: list[str], elapsed: float,
) -> None:
    p = proxy.set_index("analysis")
    lines = [
        "Stage 08 — Label-free Transfer-Risk Estimation and Multi-source Transfer Baselines",
        "=" * 82,
        "",
        "1. Scope",
        "- Five frozen cities; 20 directed transfer pairs; AEF-192 only.",
        "- Stage08A evaluates genuinely label-free pair risk features; Stage08B evaluates pooled four-source XGBoost.",
        "",
        "2. Frozen protocol",
        f"- Cities: {', '.join(CITIES)}.",
        f"- Seeds: {', '.join(map(str, SEEDS))}; 2 km group-aware spatial blocks and Stage07 test sets reused.",
        "- XGBoost: n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, min_child_weight=1, reg_lambda=1, early_stopping_rounds=30, tree_method=hist, n_jobs=4.",
        f"- Reproduction maximum absolute metric difference: same-city={validation_max['same_city']:.10g}; cross-city={validation_max['cross_city']:.10g} (tolerance={MODEL_TOLERANCE:g}).",
        "",
        "3. No-target-label leakage audit",
        f"- target_label_used_in_risk_features = {TARGET_LABEL_USED_IN_RISK_FEATURES}",
        "- PASS: risk functions receive X_target, source probabilities, X_source_train, and y_source_train; no y_target argument exists.",
        "- PASS: outcomes were joined only after 100 risk-feature records were frozen.",
        "- PASS: risk calibration design columns contain no target class, target F1, or target gap predictors.",
        "",
        "4. Label-free feature definitions",
        "- PCD-5NN: predicted-class-conditioned exact cosine 5NN on sample-wise L2-normalized AEF-192.",
        "- Global-5NN: exact cosine 5NN against all source-train representations.",
        "- Uncertainty: normalized entropy, 1-max probability, and predicted class fractions.",
        "- PredExpansion-5NN: predicted Class2 target samples against source-train Class2 references.",
        "",
        "5. Proxy-vs-Stage07 mechanism distance",
        proxy.loc[proxy.analysis_group == "proxy_validation"].to_string(index=False),
        "",
        "6. Direct proxy-gap association",
        proxy.loc[proxy.analysis_group == "direct_risk_association"].to_string(index=False),
        "",
        "7. Node-level LOCO design",
        "- Each fold removes every pair containing held-out city H as source or target (12 calibration pairs) and tests four incoming S→H pairs.",
        "- Five folds yield exactly 20 out-of-city predictions; pair means, not 100 seed rows, are the statistical unit.",
        "",
        "8. R0-R4 Macro risk calibration",
        risk_metrics.to_string(index=False),
        "- RX is explicitly exploratory and is excluded from the predefined Stage08A judgment.",
        "",
        "9. OOF risk metrics",
        risk_metrics[["model", "mae", "rmse", "oof_r_squared", "spearman_rho", "pearson_r", "relative_mae_improvement_vs_R0"]].to_string(index=False),
        "",
        "10. High-risk classification",
        high.to_string(index=False),
        "- Macro threshold=0.05 and Expansion threshold=0.07 were fixed before evaluation.",
        "",
        "11. Fold-level rankings",
        ranking.to_string(index=False),
        "- Ranking is secondary and does not redefine Stage08 as a source-selection method.",
        "",
        "12. Expansion risk results",
        expansion.to_string(index=False),
        f"- Predefined secondary judgment: {expansion_judgment}.",
        "",
        "13. Multi-source pooled protocol",
        "- For every target×seed, train and validation sets are the unions of frozen blocks from the other four cities; target labels are excluded from training and early stopping.",
        "- Uniform samples, no weights, no target tuning; evaluation uses the exact Stage07 target test blocks.",
        "",
        "14. Multi-source results for five targets",
        multi_summary.to_string(index=False),
        "",
        "15. Comparison with single-source baselines",
        comparison.to_string(index=False),
        "- ORACLE uses target labels retrospectively and is not deployable; it is not used for method claims.",
        "",
        "16. Kunming-specific result",
        comparison.loc[comparison.target_city == "Kunming"].to_string(index=False),
        "",
        "17. Optional uniform ensemble result",
        "- RUN: uniform soft-voting averaged probabilities from the four already-reproduced frozen source models; no learned weights or target tuning.",
        comparison[["target_city", "uniform_ensemble_macro_f1", "pooled_macro_f1", "uniform_ensemble_expansion_f1", "pooled_expansion_f1"]].to_string(index=False),
        "",
        "18. Exploratory risk-vs-pooling-benefit",
        comparison[["target_city", "mean_R1_predicted_incoming_macro_gap", "pooled_minus_mean_single_macro_f1"]].to_string(index=False),
        "- n=5 descriptive only; no inferential claim.",
        "",
        "19. Stage08A predefined judgment",
        f"- Label-free Macro risk estimation: {macro_judgment}.",
        "- Rule: supported only if R1 Spearman≥0.50, MAE improvement≥15%, and high-risk AUROC≥0.75; partial requires positive Spearman plus at least one other criterion.",
        "",
        "20. Stage08B predefined judgment",
        f"- Multi-source pooling: {pooled_judgment}.",
        "- Rule: supported if pooled Macro-F1 improves over mean single source in ≥4/5 targets and mean improvement>0; partial if 3/5 and mean>0.",
        "",
        "21. Limitations",
        "- Only five cities and 20 directed pair outcomes.",
        "- The risk calibrator was developed after observing Stage07/07B; node-LOCO reduces but does not eliminate method-development bias.",
        "- Target pools are benchmark samples constructed from the frozen class-balanced dataset; this is not independent external validation.",
        "- Pseudo-class-conditioned distance depends on source-model predictions; no causal interpretation is justified.",
        "- Chengdu/Hangzhou have not been used for transfer-outcome validation.",
        "",
        "22. Warnings",
    ]
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None beyond the stated small-n and post-development limitations.")
    lines.extend([
        "",
        "23. Recommendation for external validation",
        "- If Stage08A or Stage08B is promising under the predefined rules, stop method development and validate once on untouched Chengdu and/or Hangzhou outcomes. Do not tune the proxy, threshold, or pooling protocol on those cities.",
        "",
        f"Execution time: {elapsed / 60:.2f} minutes.",
        f"Pair-seed records: {len(pair_seed)}; pair-level records: {len(pair)}; OOF predictions: {len(oof)}.",
    ])
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    expected_rows = {
        PAIR_SEED_PATH: 100, PAIR_PATH: 20, PROXY_PATH: 7, OOF_PATH: 20,
        RISK_METRICS_PATH: 6, HIGH_RISK_PATH: 2, RANKING_PATH: 5,
        EXPANSION_PATH: 2, MULTI_RUNS_PATH: 25, MULTI_SUMMARY_PATH: 5,
        MULTI_COMPARISON_PATH: 5,
    }
    for path, count in expected_rows.items():
        frame = pd.read_csv(path)
        if len(frame) != count:
            raise AssertionError(f"{path.name}: expected {count} rows, got {len(frame)}")
        if frame.columns.duplicated().any():
            raise AssertionError(f"{path.name}: duplicate columns")
    for path in FIGURE_PATHS.values():
        if not path.is_file() or path.stat().st_size < 20_000:
            raise AssertionError(f"Missing/too-small figure: {path}")
    if not REPORT.is_file() or REPORT.stat().st_size < 3000:
        raise AssertionError("Stage 08 report missing or incomplete")


def main() -> None:
    start = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    stage03 = load_stage03()
    frames, matrices, labels, splits, config, same, cross, outcomes, mechanism, diversity, direct = load_inputs()
    stage_config = config["stage_03"]
    pair_seed, ensemble_metrics, validation_max = train_source_models_and_features(
        stage03, matrices, labels, splits, stage_config, same, cross, outcomes, warnings
    )
    pair = aggregate_pairs(pair_seed, same, diversity, mechanism)
    proxy = correlation_audits(pair, mechanism)
    oof = node_loco(pair)
    risk_metrics, high, ranking, expansion, macro_judgment, expansion_judgment = evaluate_oof(oof)
    save_csv(pair_seed, PAIR_SEED_PATH)
    save_csv(pair, PAIR_PATH)
    save_csv(proxy, PROXY_PATH)
    save_csv(oof, OOF_PATH)
    save_csv(risk_metrics, RISK_METRICS_PATH)
    save_csv(high, HIGH_RISK_PATH)
    save_csv(ranking, RANKING_PATH)
    save_csv(expansion, EXPANSION_PATH)
    print("STAGE08A outputs saved; starting 25 pooled models", flush=True)

    multi_runs = train_multisource(
        stage03, matrices, labels, splits, stage_config, same, cross, pair_seed,
        direct, ensemble_metrics, warnings,
    )
    multi_summary, comparison, pooled_judgment = summarize_multisource(multi_runs, oof)
    save_csv(multi_runs, MULTI_RUNS_PATH)
    save_csv(multi_summary, MULTI_SUMMARY_PATH)
    save_csv(comparison, MULTI_COMPARISON_PATH)
    create_figures(pair, mechanism, oof, comparison)
    build_report(
        pair_seed, pair, proxy, oof, risk_metrics, high, ranking, expansion,
        multi_summary, comparison, validation_max, macro_judgment,
        expansion_judgment, pooled_judgment, warnings, time.perf_counter() - start,
    )
    validate_outputs()
    print(f"STAGE08 COMPLETE minutes={(time.perf_counter()-start)/60:.2f}")
    print(f"Stage08A={macro_judgment}; Expansion={expansion_judgment}; Stage08B={pooled_judgment}")


if __name__ == "__main__":
    main()

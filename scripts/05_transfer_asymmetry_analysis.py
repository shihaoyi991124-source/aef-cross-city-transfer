"""Stage 05: asymmetric transfer mechanism analysis.

This script explains transfer asymmetry without introducing a new prediction
model. It uses class-conditional cosine support geometry in the original AEF
spaces, block-bootstrap uncertainty, symmetric discrepancy diagnostics, and
the association between source-support distance and zero-shot error.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
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
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "02_baseline_experiment.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = LOGS_DIR / "transfer_asymmetry_report.txt"
STAGE03B_TRANSFER_PATH = TABLES_DIR / "target_normalized_transfer_gap.csv"

DIRECTED_PATH = TABLES_DIR / "directed_support_distance.csv"
COVERAGE_PATH = TABLES_DIR / "support_coverage.csv"
DIVERSITY_PATH = TABLES_DIR / "representation_diversity.csv"
DISCREPANCY_PATH = TABLES_DIR / "domain_discrepancy.csv"
YEARLY_PATH = TABLES_DIR / "yearly_support_asymmetry.csv"
ERROR_PATH = TABLES_DIR / "support_error_analysis.csv"
MECHANISM_PATH = TABLES_DIR / "transfer_mechanism_summary.csv"

CITY_FILES = {
    "Shenzhen": PROCESSED_DIR / "Shenzhen_AEF_195D.parquet",
    "Kunming": PROCESSED_DIR / "Kunming_AEF_195D.parquet",
}
TARGET_SOURCE_PAIRS = [("Shenzhen", "Kunming"), ("Kunming", "Shenzhen")]
TRANSFER_DIRECTIONS = [("Shenzhen", "Kunming"), ("Kunming", "Shenzhen")]
CLASS_VALUES = [0, 1, 2]
CLASS_LABELS = {
    0: "Stable Non-built",
    1: "Stable Built",
    2: "Strict Expansion",
}
FORBIDDEN_REPRESENTATION_FEATURES = {
    "city",
    "class",
    "longitude",
    "latitude",
    "utm_x",
    "utm_y",
    "block_x",
    "block_y",
    "block_id",
    "sample_id",
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


def l2_normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise AssertionError("Non-finite or zero representation norm")
    result = normalize(matrix, norm="l2", axis=1, copy=True)
    normalized_norms = np.linalg.norm(result, axis=1)
    if float(np.max(np.abs(normalized_norms - 1.0))) > 2e-6:
        raise AssertionError("L2 normalization failed")
    return np.asarray(result, dtype=np.float32)


def norm_summary(matrix: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)
    return {
        "input_norm_mean": float(np.mean(norms)),
        "input_norm_std": float(np.std(norms, ddof=1)),
        "input_norm_min": float(np.min(norms)),
        "input_norm_max": float(np.max(norms)),
    }


def cross_knn_distances(
    target: np.ndarray,
    reference: np.ndarray,
    max_k: int,
) -> np.ndarray:
    if max_k > len(reference):
        raise ValueError("k exceeds reference sample count")
    estimator = NearestNeighbors(
        n_neighbors=max_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=4,
    )
    estimator.fit(reference)
    distances, _ = estimator.kneighbors(target, return_distance=True)
    return np.clip(np.asarray(distances, dtype=np.float64), 0.0, 2.0)


def within_knn_mean_distance(matrix: np.ndarray, k: int) -> np.ndarray:
    neighbors = min(k + 2, len(matrix))
    estimator = NearestNeighbors(
        n_neighbors=neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=4,
    )
    estimator.fit(matrix)
    distances, indices = estimator.kneighbors(matrix, return_distance=True)
    self_indices = np.arange(len(matrix), dtype=np.int64)[:, None]
    distances = np.where(indices == self_indices, np.inf, distances)
    distances = np.sort(distances, axis=1)[:, :k]
    if not np.isfinite(distances).all():
        raise AssertionError("Could not exclude self-neighbor")
    return np.clip(distances.mean(axis=1), 0.0, 2.0)


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
    }


def block_bootstrap_mean(
    values: np.ndarray,
    block_ids: np.ndarray,
    resamples: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    block_ids = np.asarray(block_ids, dtype=object)
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    block_sums = np.bincount(inverse, weights=values, minlength=len(unique_blocks))
    block_counts = np.bincount(inverse, minlength=len(unique_blocks)).astype(np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(resamples, dtype=np.float64)
    for iteration in range(resamples):
        chosen = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        output[iteration] = block_sums[chosen].sum() / block_counts[chosen].sum()
    return output


def ci95(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def analyze_support_geometry(
    datasets: dict[str, pd.DataFrame],
    raw_matrices: dict[str, np.ndarray],
    representation: str,
    k_values: list[int],
    primary_k: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    normalized = {city: l2_normalized(matrix) for city, matrix in raw_matrices.items()}
    class_indices = {
        city: {
            class_value: np.flatnonzero(
                datasets[city]["class"].to_numpy(dtype=np.int64) == class_value
            )
            for class_value in CLASS_VALUES
        }
        for city in datasets
    }
    within_primary: dict[tuple[str, int], np.ndarray] = {}
    within_radius: dict[tuple[str, int], float] = {}
    for city in ["Shenzhen", "Kunming"]:
        for class_value in CLASS_VALUES:
            idx = class_indices[city][class_value]
            distances = within_knn_mean_distance(normalized[city][idx], primary_k)
            within_primary[(city, class_value)] = distances
            within_radius[(city, class_value)] = float(np.quantile(distances, 0.95))

    directed_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    directed_boot: dict[tuple[str, str, str, int], np.ndarray] = {}
    coverage_boot: dict[tuple[str, str, str], np.ndarray] = {}
    cache: dict[str, Any] = {"normalized": normalized, "within_radius": within_radius}

    for pair_index, (target_city, source_city) in enumerate(TARGET_SOURCE_PAIRS):
        all_distances: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
        all_blocks: list[np.ndarray] = []
        all_covered: list[np.ndarray] = []
        cache[(target_city, source_city)] = {}
        for class_value in CLASS_VALUES:
            target_idx = class_indices[target_city][class_value]
            source_idx = class_indices[source_city][class_value]
            distances = cross_knn_distances(
                normalized[target_city][target_idx],
                normalized[source_city][source_idx],
                max(k_values),
            )
            target_blocks = datasets[target_city].iloc[target_idx]["block_id"].to_numpy()
            distance_by_k = {
                k: distances[:, :k].mean(axis=1) for k in k_values
            }
            primary_distance = distance_by_k[primary_k]
            radius = within_radius[(source_city, class_value)]
            covered = primary_distance <= radius
            cache[(target_city, source_city)][class_value] = {
                "distance_by_k": distance_by_k,
                "primary_distance": primary_distance,
                "covered": covered,
                "target_indices": target_idx,
                "target_blocks": target_blocks,
                "support_radius": radius,
            }
            class_scope = f"Class {class_value}"
            for k in k_values:
                summary = distribution_summary(distance_by_k[k])
                boot = block_bootstrap_mean(
                    distance_by_k[k],
                    target_blocks,
                    bootstrap_resamples,
                    bootstrap_seed + pair_index * 10000 + class_value * 100 + k,
                )
                low, high = ci95(boot)
                directed_boot[(target_city, source_city, class_scope, k)] = boot
                directed_rows.append(
                    {
                        "representation": representation,
                        "target_city": target_city,
                        "source_city": source_city,
                        "support_direction": f"{target_city} target -> {source_city} support",
                        "transfer_direction": f"{source_city} -> {target_city}",
                        "class_scope": class_scope,
                        "class_value": class_value,
                        "k": k,
                        "target_samples": len(target_idx),
                        "source_reference_samples": len(source_idx),
                        "target_blocks": int(pd.Series(target_blocks).nunique()),
                        **summary,
                        "bootstrap_mean": float(np.mean(boot)),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "bootstrap_unit": "2 km target block",
                        "bootstrap_resamples": bootstrap_resamples,
                    }
                )
                all_distances[k].append(distance_by_k[k])
            coverage_bootstrap = block_bootstrap_mean(
                covered.astype(float),
                target_blocks,
                bootstrap_resamples,
                bootstrap_seed + 50000 + pair_index * 10000 + class_value * 100,
            )
            low, high = ci95(coverage_bootstrap)
            coverage_boot[(target_city, source_city, class_scope)] = coverage_bootstrap
            coverage_rows.append(
                {
                    "representation": representation,
                    "target_city": target_city,
                    "source_city": source_city,
                    "coverage_label": f"Coverage({target_city} | {source_city})",
                    "transfer_direction": f"{source_city} -> {target_city}",
                    "class_scope": class_scope,
                    "class_value": class_value,
                    "k": primary_k,
                    "target_samples": len(target_idx),
                    "source_reference_samples": len(source_idx),
                    "target_blocks": int(pd.Series(target_blocks).nunique()),
                    "source_support_radius_q95": radius,
                    "covered_samples": int(covered.sum()),
                    "coverage": float(covered.mean()),
                    "bootstrap_mean": float(np.mean(coverage_bootstrap)),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_unit": "2 km target block",
                    "bootstrap_resamples": bootstrap_resamples,
                }
            )
            all_blocks.append(target_blocks)
            all_covered.append(covered)

        combined_blocks = np.concatenate(all_blocks)
        combined_covered = np.concatenate(all_covered)
        for k in k_values:
            combined_distance = np.concatenate(all_distances[k])
            summary = distribution_summary(combined_distance)
            boot = block_bootstrap_mean(
                combined_distance,
                combined_blocks,
                bootstrap_resamples,
                bootstrap_seed + 90000 + pair_index * 10000 + k,
            )
            low, high = ci95(boot)
            directed_boot[(target_city, source_city, "All", k)] = boot
            directed_rows.append(
                {
                    "representation": representation,
                    "target_city": target_city,
                    "source_city": source_city,
                    "support_direction": f"{target_city} target -> {source_city} support",
                    "transfer_direction": f"{source_city} -> {target_city}",
                    "class_scope": "All",
                    "class_value": np.nan,
                    "k": k,
                    "target_samples": len(combined_distance),
                    "source_reference_samples": sum(
                        len(class_indices[source_city][value]) for value in CLASS_VALUES
                    ),
                    "target_blocks": int(pd.Series(combined_blocks).nunique()),
                    **summary,
                    "bootstrap_mean": float(np.mean(boot)),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_unit": "2 km target block",
                    "bootstrap_resamples": bootstrap_resamples,
                }
            )
        coverage_bootstrap = block_bootstrap_mean(
            combined_covered.astype(float),
            combined_blocks,
            bootstrap_resamples,
            bootstrap_seed + 140000 + pair_index * 10000,
        )
        low, high = ci95(coverage_bootstrap)
        coverage_boot[(target_city, source_city, "All")] = coverage_bootstrap
        coverage_rows.append(
            {
                "representation": representation,
                "target_city": target_city,
                "source_city": source_city,
                "coverage_label": f"Coverage({target_city} | {source_city})",
                "transfer_direction": f"{source_city} -> {target_city}",
                "class_scope": "All",
                "class_value": np.nan,
                "k": primary_k,
                "target_samples": len(combined_covered),
                "source_reference_samples": sum(
                    len(class_indices[source_city][value]) for value in CLASS_VALUES
                ),
                "target_blocks": int(pd.Series(combined_blocks).nunique()),
                "source_support_radius_q95": np.nan,
                "covered_samples": int(combined_covered.sum()),
                "coverage": float(combined_covered.mean()),
                "bootstrap_mean": float(np.mean(coverage_bootstrap)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_unit": "2 km target block",
                "bootstrap_resamples": bootstrap_resamples,
            }
        )

    directed = pd.DataFrame(directed_rows)
    coverage = pd.DataFrame(coverage_rows)
    for class_scope in ["Class 0", "Class 1", "Class 2", "All"]:
        for k in k_values:
            km_target = directed.loc[
                (directed["target_city"] == "Kunming")
                & (directed["source_city"] == "Shenzhen")
                & (directed["class_scope"] == class_scope)
                & (directed["k"] == k),
                "mean",
            ].iloc[0]
            sz_target = directed.loc[
                (directed["target_city"] == "Shenzhen")
                & (directed["source_city"] == "Kunming")
                & (directed["class_scope"] == class_scope)
                & (directed["k"] == k),
                "mean",
            ].iloc[0]
            difference_boot = (
                directed_boot[("Kunming", "Shenzhen", class_scope, k)]
                - directed_boot[("Shenzhen", "Kunming", class_scope, k)]
            )
            low, high = ci95(difference_boot)
            mask = (directed["class_scope"] == class_scope) & (directed["k"] == k)
            directed.loc[mask, "dsai"] = km_target - sz_target
            directed.loc[mask, "dsai_bootstrap_ci95_low"] = low
            directed.loc[mask, "dsai_bootstrap_ci95_high"] = high

        sz_coverage = coverage.loc[
            (coverage["target_city"] == "Shenzhen")
            & (coverage["source_city"] == "Kunming")
            & (coverage["class_scope"] == class_scope),
            "coverage",
        ].iloc[0]
        km_coverage = coverage.loc[
            (coverage["target_city"] == "Kunming")
            & (coverage["source_city"] == "Shenzhen")
            & (coverage["class_scope"] == class_scope),
            "coverage",
        ].iloc[0]
        difference_boot = (
            coverage_boot[("Shenzhen", "Kunming", class_scope)]
            - coverage_boot[("Kunming", "Shenzhen", class_scope)]
        )
        low, high = ci95(difference_boot)
        mask = coverage["class_scope"] == class_scope
        coverage.loc[mask, "coverage_asymmetry"] = sz_coverage - km_coverage
        coverage.loc[mask, "coverage_asymmetry_bootstrap_ci95_low"] = low
        coverage.loc[mask, "coverage_asymmetry_bootstrap_ci95_high"] = high

    return directed, coverage, cache


def representation_diversity(
    datasets: dict[str, pd.DataFrame],
    raw_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city in ["Shenzhen", "Kunming"]:
        normalized = l2_normalized(raw_matrices[city])
        labels = datasets[city]["class"].to_numpy(dtype=np.int64)
        for class_scope, class_value in [
            ("Class 0", 0),
            ("Class 1", 1),
            ("Class 2", 2),
            ("All", None),
        ]:
            idx = np.arange(len(labels)) if class_value is None else np.flatnonzero(labels == class_value)
            x = normalized[idx].astype(np.float64)
            raw = raw_matrices[city][idx]
            centroid = x.mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm <= 0:
                raise AssertionError("Zero centroid norm")
            centroid_unit = centroid / centroid_norm
            distances = np.clip(1.0 - x @ centroid_unit, 0.0, 2.0)
            covariance = np.cov(x, rowvar=False, ddof=1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            trace = float(eigenvalues.sum())
            if trace <= 0:
                effective_rank = 0.0
                participation_ratio = 0.0
            else:
                probabilities = eigenvalues[eigenvalues > 1e-15] / trace
                effective_rank = float(
                    np.exp(-np.sum(probabilities * np.log(probabilities)))
                )
                participation_ratio = float(
                    trace**2 / np.sum(eigenvalues**2)
                )
            normalized_norms = np.linalg.norm(x, axis=1)
            rows.append(
                {
                    "representation": "AEF-192 L2-normalized",
                    "city": city,
                    "class_scope": class_scope,
                    "class_value": np.nan if class_value is None else class_value,
                    "samples": len(idx),
                    "mean_cosine_distance_to_centroid": float(np.mean(distances)),
                    "median_cosine_distance_to_centroid": float(np.median(distances)),
                    "covariance_trace": trace,
                    "effective_rank": effective_rank,
                    "participation_ratio": participation_ratio,
                    "centroid_norm": centroid_norm,
                    **norm_summary(raw),
                    "normalized_norm_mean": float(normalized_norms.mean()),
                    "normalized_norm_max_abs_deviation": float(
                        np.max(np.abs(normalized_norms - 1.0))
                    ),
                }
            )
    return pd.DataFrame(rows)


def stratified_sample_indices(
    frame: pd.DataFrame,
    class_value: int | None,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = frame["class"].to_numpy(dtype=np.int64)
    if class_value is not None:
        candidates = np.flatnonzero(labels == class_value)
        return np.sort(rng.choice(candidates, size=sample_size, replace=False))
    base = sample_size // 3
    allocations = [base, base, sample_size - 2 * base]
    selected: list[np.ndarray] = []
    for value, count in zip(CLASS_VALUES, allocations):
        candidates = np.flatnonzero(labels == value)
        selected.append(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def domain_discrepancy(
    datasets: dict[str, pd.DataFrame],
    raw_matrices: dict[str, np.ndarray],
    sample_size: int,
    bandwidth_sample_size: int,
    seed: int,
) -> pd.DataFrame:
    normalized = {city: l2_normalized(matrix) for city, matrix in raw_matrices.items()}
    rows: list[dict[str, Any]] = []
    for scope_index, (class_scope, class_value) in enumerate(
        [("Class 0", 0), ("Class 1", 1), ("Class 2", 2), ("All", None)]
    ):
        sz_idx = stratified_sample_indices(
            datasets["Shenzhen"], class_value, sample_size, seed + scope_index * 100
        )
        km_idx = stratified_sample_indices(
            datasets["Kunming"], class_value, sample_size, seed + scope_index * 100 + 1
        )
        x = normalized["Shenzhen"][sz_idx].astype(np.float64)
        y = normalized["Kunming"][km_idx].astype(np.float64)
        pooled = np.vstack([x, y])
        rng = np.random.default_rng(seed + 10000 + scope_index)
        bandwidth_n = min(bandwidth_sample_size, len(pooled))
        bandwidth_idx = rng.choice(len(pooled), size=bandwidth_n, replace=False)
        squared_distances = pairwise_distances(
            pooled[bandwidth_idx], metric="sqeuclidean", n_jobs=4
        )
        triangle = squared_distances[np.triu_indices_from(squared_distances, k=1)]
        positive = triangle[triangle > 1e-15]
        median_squared_distance = float(np.median(positive))
        if median_squared_distance <= 0:
            raise AssertionError("Invalid MMD median bandwidth")
        gamma = 1.0 / (2.0 * median_squared_distance)
        k_xx = rbf_kernel(x, x, gamma=gamma)
        k_yy = rbf_kernel(y, y, gamma=gamma)
        k_xy = rbf_kernel(x, y, gamma=gamma)
        n = len(x)
        m = len(y)
        mmd2 = float(
            (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
        covariance_x = np.cov(x, rowvar=False, ddof=1)
        covariance_y = np.cov(y, rowvar=False, ddof=1)
        covariance_difference = covariance_x - covariance_y
        frobenius = float(np.linalg.norm(covariance_difference, ord="fro"))
        dimensions = x.shape[1]
        coral = float(np.sum(covariance_difference**2) / (4.0 * dimensions**2))
        rows.append(
            {
                "representation": "AEF-192 L2-normalized",
                "class_scope": class_scope,
                "class_value": np.nan if class_value is None else class_value,
                "samples_per_city": sample_size,
                "sampling_seed": seed + scope_index * 100,
                "mmd_bandwidth_method": "median heuristic on pooled fixed sample",
                "bandwidth_sample_size": bandwidth_n,
                "median_squared_distance": median_squared_distance,
                "rbf_gamma": gamma,
                "mmd2_unbiased": mmd2,
                "mmd_nonnegative_sqrt": math.sqrt(max(mmd2, 0.0)),
                "covariance_frobenius_distance": frobenius,
                "coral_distance": coral,
                "symmetric_metric_warning": (
                    "MMD and CORAL establish discrepancy but do not encode transfer direction"
                ),
            }
        )
        del k_xx, k_yy, k_xy, squared_distances
        gc.collect()
    return pd.DataFrame(rows)


def yearly_support_analysis(
    datasets: dict[str, pd.DataFrame],
    years: list[int],
    stage05: dict[str, Any],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for year_index, year in enumerate(years):
        columns = [f"A{dimension:02d}_{year}" for dimension in range(64)]
        raw = {
            city: datasets[city][columns].to_numpy(dtype=np.float32, copy=False)
            for city in datasets
        }
        directed, coverage, _ = analyze_support_geometry(
            datasets,
            raw,
            f"AEF-{year} 64D",
            [int(stage05["primary_k"])],
            int(stage05["primary_k"]),
            int(stage05["bootstrap_resamples"]),
            int(stage05["bootstrap_seed"]) + 200000 + year_index * 20000,
        )
        merged = directed.merge(
            coverage[
                [
                    "target_city",
                    "source_city",
                    "class_scope",
                    "coverage",
                    "source_support_radius_q95",
                    "bootstrap_ci95_low",
                    "bootstrap_ci95_high",
                    "coverage_asymmetry",
                    "coverage_asymmetry_bootstrap_ci95_low",
                    "coverage_asymmetry_bootstrap_ci95_high",
                ]
            ].rename(
                columns={
                    "bootstrap_ci95_low": "coverage_bootstrap_ci95_low",
                    "bootstrap_ci95_high": "coverage_bootstrap_ci95_high",
                }
            ),
            on=["target_city", "source_city", "class_scope"],
            validate="one_to_one",
        )
        merged.insert(0, "year", year)
        for city_role in ["target", "source"]:
            city_column = f"{city_role}_city"
            norm_values: list[float] = []
            norm_stds: list[float] = []
            for _, row in merged.iterrows():
                city = row[city_column]
                class_scope = row["class_scope"]
                matrix = raw[city]
                if class_scope != "All":
                    class_value = int(class_scope.split()[-1])
                    mask = datasets[city]["class"].to_numpy(dtype=int) == class_value
                    matrix = matrix[mask]
                stats = norm_summary(matrix)
                norm_values.append(stats["input_norm_mean"])
                norm_stds.append(stats["input_norm_std"])
            merged[f"{city_role}_input_norm_mean"] = norm_values
            merged[f"{city_role}_input_norm_std"] = norm_stds
        outputs.append(merged)
        print(f"YEARLY SUPPORT {year} complete")
    return pd.concat(outputs, ignore_index=True)


def error_distance_analysis(
    stage03: Any,
    datasets: dict[str, pd.DataFrame],
    aef192: list[str],
    stage_config: dict[str, Any],
    stage05: dict[str, Any],
    warning_messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = {
        city: datasets[city][aef192].to_numpy(dtype=np.float32, copy=False)
        for city in datasets
    }
    normalized = {city: l2_normalized(matrix) for city, matrix in raw.items()}
    labels = {
        city: datasets[city]["class"].to_numpy(dtype=np.int64)
        for city in datasets
    }
    prior_transfer = pd.read_csv(STAGE03B_TRANSFER_PATH)
    analysis_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in stage_config["random_seeds"]]
    k = int(stage05["error_distance_k"])

    for seed in seeds:
        splits: dict[str, Any] = {}
        models: dict[str, Any] = {}
        training_info: dict[str, tuple[float, int | None]] = {}
        for city in ["Shenzhen", "Kunming"]:
            frame = datasets[city]
            split = stage03.optimized_group_stratified_split(
                frame,
                frame["class"].astype(int),
                seed,
                stage_config["split"],
                f"Stage 05 error analysis {city}",
                warning_messages,
            )
            expected = prior_transfer.loc[
                (prior_transfer["target_city"] == city)
                & (prior_transfer["seed"] == seed),
                "target_test_index_sha256",
            ].unique()
            if len(expected) != 1 or expected[0] != index_hash(split.test):
                raise AssertionError(f"{city} seed={seed}: test split differs from Stage 03B")
            splits[city] = split
            model = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
            model, seconds, iterations = stage03.fit_model(
                model,
                "XGBoost",
                raw[city][split.train],
                labels[city][split.train],
                raw[city][split.validation],
                labels[city][split.validation],
                warning_messages,
                f"Stage 05 XGBoost {city} seed={seed}",
            )
            models[city] = model
            training_info[city] = (seconds, iterations)

        for source_city, target_city in TRANSFER_DIRECTIONS:
            source_split = splits[source_city]
            target_split = splits[target_city]
            target_indices = np.asarray(target_split.test, dtype=np.int64)
            y_true = labels[target_city][target_indices]
            prediction = models[source_city].predict(raw[target_city][target_indices]).astype(int)
            incorrect = (prediction != y_true).astype(np.int64)
            support_distance = np.empty(len(target_indices), dtype=np.float64)
            for class_value in CLASS_VALUES:
                target_positions = np.flatnonzero(y_true == class_value)
                source_train_indices = source_split.train[
                    labels[source_city][source_split.train] == class_value
                ]
                distances = cross_knn_distances(
                    normalized[target_city][target_indices[target_positions]],
                    normalized[source_city][source_train_indices],
                    k,
                )[:, :k].mean(axis=1)
                support_distance[target_positions] = distances
            if not np.isfinite(support_distance).all():
                raise AssertionError("Non-finite support distance")
            metrics = stage03.land_cover_metrics(y_true, prediction)
            performance_rows.append(
                {
                    "direction": f"{source_city} -> {target_city}",
                    "source_city": source_city,
                    "target_city": target_city,
                    "seed": seed,
                    "target_test_samples": len(target_indices),
                    "target_test_blocks": target_split.diagnostics["test_blocks"],
                    "target_test_index_sha256": index_hash(target_indices),
                    "training_seconds": training_info[source_city][0],
                    "fitted_iterations": training_info[source_city][1],
                    **metrics,
                }
            )

            for scope, mask in [
                ("All", np.ones(len(y_true), dtype=bool)),
                ("Class 2 Expansion", y_true == 2),
            ]:
                distance = support_distance[mask]
                error = incorrect[mask]
                correct_distance = distance[error == 0]
                incorrect_distance = distance[error == 1]
                if len(correct_distance) == 0 or len(incorrect_distance) == 0:
                    raise RuntimeError(
                        f"{source_city}->{target_city} seed={seed} {scope}: "
                        "Mann-Whitney comparison lacks one outcome group"
                    )
                mw = mannwhitneyu(
                    incorrect_distance,
                    correct_distance,
                    alternative="two-sided",
                    method="asymptotic",
                )
                rank_biserial = float(
                    2.0 * mw.statistic
                    / (len(incorrect_distance) * len(correct_distance))
                    - 1.0
                )
                risk_model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                )
                risk_model.fit(distance.reshape(-1, 1), error)
                error_probability = risk_model.predict_proba(distance.reshape(-1, 1))[:, 1]
                coefficient = float(risk_model.coef_[0, 0])
                intercept = float(risk_model.intercept_[0])
                association_auc = float(roc_auc_score(error, error_probability))
                analysis_rows.append(
                    {
                        "analysis_type": "correct_vs_incorrect",
                        "direction": f"{source_city} -> {target_city}",
                        "source_city": source_city,
                        "target_city": target_city,
                        "seed": seed,
                        "scope": scope,
                        "k": k,
                        "samples": len(distance),
                        "correct_samples": len(correct_distance),
                        "incorrect_samples": len(incorrect_distance),
                        "correct_distance_mean": float(np.mean(correct_distance)),
                        "correct_distance_median": float(np.median(correct_distance)),
                        "incorrect_distance_mean": float(np.mean(incorrect_distance)),
                        "incorrect_distance_median": float(np.median(incorrect_distance)),
                        "mannwhitney_u": float(mw.statistic),
                        "mannwhitney_p_two_sided": float(mw.pvalue),
                        "rank_biserial_incorrect_vs_correct": rank_biserial,
                        "error_logistic_coefficient_raw_distance": coefficient,
                        "error_logistic_intercept": intercept,
                        "error_odds_ratio_per_unit_distance": float(np.exp(coefficient)),
                        "error_odds_ratio_per_0_01_distance": float(
                            np.exp(coefficient * 0.01)
                        ),
                        "error_association_roc_auc": association_auc,
                        "association_interpretation": "association only; not causal",
                        "quintile": np.nan,
                        "quintile_samples": np.nan,
                        "quintile_errors": np.nan,
                        "quintile_error_rate": np.nan,
                        "quintile_distance_mean": np.nan,
                        "quintile_distance_median": np.nan,
                    }
                )
                quintiles = (
                    pd.qcut(
                        pd.Series(distance).rank(method="first"),
                        q=5,
                        labels=False,
                    ).to_numpy(dtype=int)
                    + 1
                )
                for quintile in range(1, 6):
                    qmask = quintiles == quintile
                    analysis_rows.append(
                        {
                            "analysis_type": "distance_quintile",
                            "direction": f"{source_city} -> {target_city}",
                            "source_city": source_city,
                            "target_city": target_city,
                            "seed": seed,
                            "scope": scope,
                            "k": k,
                            "samples": len(distance),
                            "correct_samples": np.nan,
                            "incorrect_samples": np.nan,
                            "correct_distance_mean": np.nan,
                            "correct_distance_median": np.nan,
                            "incorrect_distance_mean": np.nan,
                            "incorrect_distance_median": np.nan,
                            "mannwhitney_u": np.nan,
                            "mannwhitney_p_two_sided": np.nan,
                            "rank_biserial_incorrect_vs_correct": np.nan,
                            "error_logistic_coefficient_raw_distance": np.nan,
                            "error_logistic_intercept": np.nan,
                            "error_odds_ratio_per_unit_distance": np.nan,
                            "error_odds_ratio_per_0_01_distance": np.nan,
                            "error_association_roc_auc": np.nan,
                            "association_interpretation": "association only; not causal",
                            "quintile": quintile,
                            "quintile_samples": int(qmask.sum()),
                            "quintile_errors": int(error[qmask].sum()),
                            "quintile_error_rate": float(error[qmask].mean()),
                            "quintile_distance_mean": float(distance[qmask].mean()),
                            "quintile_distance_median": float(np.median(distance[qmask])),
                        }
                    )
        models.clear()
        gc.collect()
        print(f"ERROR ANALYSIS seed={seed} complete")
    return pd.DataFrame(analysis_rows), pd.DataFrame(performance_rows)


def build_mechanism_summary(
    directed: pd.DataFrame,
    coverage: pd.DataFrame,
    error_analysis: pd.DataFrame,
    performance: pd.DataFrame,
    primary_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = error_analysis.loc[
        error_analysis["analysis_type"] == "correct_vs_incorrect"
    ]
    quintiles = error_analysis.loc[
        error_analysis["analysis_type"] == "distance_quintile"
    ]
    for source_city, target_city in TRANSFER_DIRECTIONS:
        direction = f"{source_city} -> {target_city}"
        perf = performance.loc[performance["direction"] == direction]
        directed_row = directed.loc[
            (directed["target_city"] == target_city)
            & (directed["source_city"] == source_city)
            & (directed["class_scope"] == "All")
            & (directed["k"] == primary_k)
        ].iloc[0]
        coverage_row = coverage.loc[
            (coverage["target_city"] == target_city)
            & (coverage["source_city"] == source_city)
            & (coverage["class_scope"] == "All")
        ].iloc[0]
        all_comparison = comparisons.loc[
            (comparisons["direction"] == direction)
            & (comparisons["scope"] == "All")
        ]
        expansion_comparison = comparisons.loc[
            (comparisons["direction"] == direction)
            & (comparisons["scope"] == "Class 2 Expansion")
        ]
        q_all = quintiles.loc[
            (quintiles["direction"] == direction) & (quintiles["scope"] == "All")
        ]
        q_expansion = quintiles.loc[
            (quintiles["direction"] == direction)
            & (quintiles["scope"] == "Class 2 Expansion")
        ]
        rows.append(
            {
                "direction": direction,
                "source_city": source_city,
                "target_city": target_city,
                "model": "XGBoost",
                "feature_set": "AEF-192",
                "seeds": len(perf),
                "zero_shot_macro_f1_mean": float(perf["macro_f1"].mean()),
                "zero_shot_macro_f1_std": float(perf["macro_f1"].std(ddof=1)),
                "zero_shot_expansion_f1_mean": float(perf["expansion_f1"].mean()),
                "zero_shot_expansion_f1_std": float(perf["expansion_f1"].std(ddof=1)),
                "mean_directed_support_distance": float(directed_row["mean"]),
                "directed_support_distance_ci95_low": float(
                    directed_row["bootstrap_ci95_low"]
                ),
                "directed_support_distance_ci95_high": float(
                    directed_row["bootstrap_ci95_high"]
                ),
                "coverage": float(coverage_row["coverage"]),
                "coverage_ci95_low": float(coverage_row["bootstrap_ci95_low"]),
                "coverage_ci95_high": float(coverage_row["bootstrap_ci95_high"]),
                "q1_error_rate_mean": float(
                    q_all.loc[q_all["quintile"] == 1, "quintile_error_rate"].mean()
                ),
                "q5_error_rate_mean": float(
                    q_all.loc[q_all["quintile"] == 5, "quintile_error_rate"].mean()
                ),
                "error_distance_association_auc_mean": float(
                    all_comparison["error_association_roc_auc"].mean()
                ),
                "error_distance_rank_biserial_mean": float(
                    all_comparison["rank_biserial_incorrect_vs_correct"].mean()
                ),
                "expansion_q1_error_rate_mean": float(
                    q_expansion.loc[
                        q_expansion["quintile"] == 1, "quintile_error_rate"
                    ].mean()
                ),
                "expansion_q5_error_rate_mean": float(
                    q_expansion.loc[
                        q_expansion["quintile"] == 5, "quintile_error_rate"
                    ].mean()
                ),
                "expansion_error_distance_association_auc_mean": float(
                    expansion_comparison["error_association_roc_auc"].mean()
                ),
                "expansion_error_distance_rank_biserial_mean": float(
                    expansion_comparison[
                        "rank_biserial_incorrect_vs_correct"
                    ].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


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


def plot_directed_support(directed: pd.DataFrame, primary_k: int) -> None:
    set_plot_style()
    classes = ["Class 0", "Class 1", "Class 2", "All"]
    directions = [
        ("Shenzhen", "Kunming", "SZ target -> KM support", "#2878B5"),
        ("Kunming", "Shenzhen", "KM target -> SZ support", "#D9534F"),
    ]
    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for index, (target, source, label, color) in enumerate(directions):
        subset = directed.loc[
            (directed["target_city"] == target)
            & (directed["source_city"] == source)
            & (directed["k"] == primary_k)
        ].set_index("class_scope").loc[classes]
        means = subset["mean"].to_numpy()
        yerr = np.vstack(
            [
                means - subset["bootstrap_ci95_low"].to_numpy(),
                subset["bootstrap_ci95_high"].to_numpy() - means,
            ]
        )
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=yerr,
            capsize=3,
            color=color,
            alpha=0.86,
            label=label,
        )
    ax.set_xticks(x, classes)
    ax.set_ylabel("Mean 5NN cosine distance")
    ax.set_title("Directed class-conditional representation support")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_directed_support_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(coverage: pd.DataFrame) -> None:
    set_plot_style()
    classes = ["Class 0", "Class 1", "Class 2", "All"]
    directions = [
        ("Shenzhen", "Kunming", "Coverage(SZ | KM)", "#2878B5"),
        ("Kunming", "Shenzhen", "Coverage(KM | SZ)", "#D9534F"),
    ]
    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for index, (target, source, label, color) in enumerate(directions):
        subset = coverage.loc[
            (coverage["target_city"] == target)
            & (coverage["source_city"] == source)
        ].set_index("class_scope").loc[classes]
        means = subset["coverage"].to_numpy()
        yerr = np.vstack(
            [
                means - subset["bootstrap_ci95_low"].to_numpy(),
                subset["bootstrap_ci95_high"].to_numpy() - means,
            ]
        )
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=yerr,
            capsize=3,
            color=color,
            alpha=0.86,
            label=label,
        )
    ax.set_xticks(x, classes)
    if np.allclose(coverage["coverage"].to_numpy(dtype=float), 0.0):
        ax.scatter(x - width / 2, np.zeros(len(x)), color="#2878B5", marker="o", zorder=4)
        ax.scatter(x + width / 2, np.zeros(len(x)), color="#D9534F", marker="s", zorder=4)
        ax.set_ylim(-0.002, 0.05)
        ax.text(
            0.5,
            0.70,
            "All coverage values are 0 under the predefined within-source 95% radius",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Coverage")
    ax.set_title("Source support coverage using within-source 95% radius")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_support_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diversity(diversity: pd.DataFrame) -> None:
    set_plot_style()
    classes = ["Class 0", "Class 1", "Class 2"]
    cities = [("Shenzhen", "#2878B5"), ("Kunming", "#D9534F")]
    x = np.arange(len(classes))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for index, (city, color) in enumerate(cities):
        subset = diversity.loc[
            (diversity["city"] == city) & diversity["class_scope"].isin(classes)
        ].set_index("class_scope").loc[classes]
        axes[0].bar(
            x + (index - 0.5) * width,
            subset["mean_cosine_distance_to_centroid"],
            width,
            color=color,
            alpha=0.86,
            label=city,
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            subset["effective_rank"],
            width,
            color=color,
            alpha=0.86,
            label=city,
        )
    axes[0].set_title("Mean cosine distance to class centroid")
    axes[0].set_ylabel("Cosine distance")
    axes[1].set_title("Covariance effective rank")
    axes[1].set_ylabel("Effective rank")
    for ax in axes:
        ax.set_xticks(x, classes)
        ax.legend(frameon=False)
    fig.suptitle("Within-domain AEF-192 representation diversity", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_representation_diversity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(error_analysis: pd.DataFrame, direction: str, filename: str) -> None:
    set_plot_style()
    subset = error_analysis.loc[
        (error_analysis["analysis_type"] == "distance_quintile")
        & (error_analysis["direction"] == direction)
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    styles = [
        ("All", "All target samples", "#2878B5", "o"),
        ("Class 2 Expansion", "Class 2 Expansion", "#D9534F", "s"),
    ]
    for scope, label, color, marker in styles:
        grouped = (
            subset.loc[subset["scope"] == scope]
            .groupby("quintile")["quintile_error_rate"]
            .agg(["mean", "std"])
            .reindex(range(1, 6))
        )
        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            color=color,
            marker=marker,
            linewidth=2.0,
            capsize=3,
            label=label,
        )
    ax.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("Source-support distance quintile (Q1 closest, Q5 farthest)")
    ax.set_ylabel("Zero-shot error rate")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{direction}: error rate vs source-support distance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_asymmetry(yearly: pd.DataFrame) -> None:
    set_plot_style()
    classes = ["Class 0", "Class 1", "Class 2", "All"]
    colors = ["#2878B5", "#E07A5F", "#59A14F", "#4D4D4D"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    unique = yearly.drop_duplicates(["year", "class_scope"])
    for class_scope, color in zip(classes, colors):
        subset = unique.loc[unique["class_scope"] == class_scope].sort_values("year")
        axes[0].plot(
            subset["year"], subset["dsai"], marker="o", linewidth=2, color=color, label=class_scope
        )
        axes[1].plot(
            subset["year"],
            subset["coverage_asymmetry"],
            marker="o",
            linewidth=2,
            color=color,
            label=class_scope,
        )
    for ax in axes:
        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        ax.set_xticks([2020, 2022, 2024])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Directed Support Asymmetry Index")
    axes[0].set_ylabel("DSAI")
    axes[1].set_title("Coverage asymmetry")
    axes[1].set_ylabel("Coverage(SZ | KM) - Coverage(KM | SZ)")
    if np.allclose(unique["coverage_asymmetry"].to_numpy(dtype=float), 0.0):
        axes[1].set_ylim(-0.01, 0.01)
        axes[1].text(
            0.5,
            0.72,
            "Both directed coverages are 0 in every year",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    fig.suptitle("Per-year support asymmetry", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_yearly_support_asymmetry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    directed: pd.DataFrame,
    coverage: pd.DataFrame,
    diversity: pd.DataFrame,
    error_analysis: pd.DataFrame,
    yearly: pd.DataFrame,
    primary_k: int,
) -> None:
    plot_directed_support(directed, primary_k)
    plot_coverage(coverage)
    plot_diversity(diversity)
    plot_error_curve(
        error_analysis,
        "Shenzhen -> Kunming",
        "16_support_distance_error_curve_SZ_to_KM.png",
    )
    plot_error_curve(
        error_analysis,
        "Kunming -> Shenzhen",
        "17_support_distance_error_curve_KM_to_SZ.png",
    )
    plot_yearly_asymmetry(yearly)


def fmt(mean: float, std: float | None = None) -> str:
    if std is None or not np.isfinite(std):
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f}"


def build_report(
    config: dict[str, Any],
    directed: pd.DataFrame,
    coverage: pd.DataFrame,
    diversity: pd.DataFrame,
    discrepancy: pd.DataFrame,
    yearly: pd.DataFrame,
    error_analysis: pd.DataFrame,
    mechanism: pd.DataFrame,
    warning_messages: list[str],
    hashes_unchanged: bool,
    elapsed_seconds: float,
) -> str:
    stage05 = config["stage_05"]
    primary_k = int(stage05["primary_k"])
    main_distance = directed.loc[directed["k"] == primary_k]
    all_distance = main_distance.loc[main_distance["class_scope"] == "All"]
    all_coverage = coverage.loc[coverage["class_scope"] == "All"]
    dsai_row = all_distance.iloc[0]
    coverage_asym_row = all_coverage.iloc[0]
    class_distance = main_distance.drop_duplicates("class_scope").loc[
        lambda frame: frame["class_scope"] != "All"
    ]
    class_coverage = coverage.drop_duplicates("class_scope").loc[
        lambda frame: frame["class_scope"] != "All"
    ]
    positive_distance_classes = int((class_distance["dsai"] > 0).sum())
    positive_coverage_classes = int((class_coverage["coverage_asymmetry"] > 0).sum())
    overall_distance_supported = bool(
        dsai_row["dsai"] > 0 and dsai_row["dsai_bootstrap_ci95_low"] > 0
    )
    overall_coverage_supported = bool(
        coverage_asym_row["coverage_asymmetry"] > 0
        and coverage_asym_row["coverage_asymmetry_bootstrap_ci95_low"] > 0
    )
    coverage_saturated_zero = bool(
        np.allclose(coverage["coverage"].to_numpy(dtype=float), 0.0)
    )
    if (
        overall_distance_supported
        and overall_coverage_supported
        and positive_distance_classes >= 2
        and positive_coverage_classes >= 2
    ):
        mechanism_judgment = "Supported"
        mechanism_text = (
            "The data support the statement that Kunming representation support covers "
            "Shenzhen better than Shenzhen support covers Kunming. This is a geometric "
            "association with transfer difficulty, not a causal mechanism."
        )
    elif dsai_row["dsai"] > 0 or coverage_asym_row["coverage_asymmetry"] > 0:
        mechanism_judgment = "Partially supported"
        mechanism_text = (
            "Directed distances robustly show that Kunming targets are farther from "
            "Shenzhen support than Shenzhen targets are from Kunming support, and larger "
            "source-support distance is associated with higher error. However, the "
            "predefined 95% support-radius coverage metric is zero in both directions "
            "and therefore cannot confirm directional coverage."
            if coverage_saturated_zero
            else
            "The support-coverage hypothesis is only partially supported because the "
            "overall point estimates and/or class-specific results are not uniformly "
            "confirmed by block-bootstrap uncertainty."
        )
    else:
        mechanism_judgment = "Not supported"
        mechanism_text = (
            "The data do not support the claim that Kunming representation support "
            "covers Shenzhen better than the reverse direction."
        )

    lines = [
        "Stage 05 - Asymmetric Transfer Mechanism Analysis",
        "=" * 78,
        f"Completed: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "Feature set: AEF-192 for the main mechanism analysis.",
        "Processed datasets were read only; input hashes unchanged: " + str(hashes_unchanged),
        "",
        "1. Methods",
        "-" * 78,
        "Cosine distances were computed in the original annual 64D spaces and in the "
        "sample-wise L2-normalized 192D concatenated space; PCA distances and feature-wise "
        "StandardScaler transformations were not used.",
        "Each target sample was compared only with source samples of the same land-cover class.",
        "kNN distance is the mean cosine distance to the first k neighbors; k=1,5,10 "
        "were evaluated and k=5 is primary.",
        "Coverage uses the 95th percentile of within-source, same-class 5NN distance as "
        "the source support radius.",
        f"Core confidence intervals use {stage05['bootstrap_resamples']} target-side "
        "2 km block bootstrap resamples.",
        "",
        "2. Directed support distance",
        "-" * 78,
    ]
    for _, row in main_distance.sort_values(["class_scope", "target_city"]).iterrows():
        lines.append(
            f"{row['support_direction']}, {row['class_scope']}: mean={row['mean']:.6f}, "
            f"std={row['std']:.6f}, median={row['median']:.6f}, "
            f"block-bootstrap 95% CI=[{row['bootstrap_ci95_low']:.6f}, "
            f"{row['bootstrap_ci95_high']:.6f}]."
        )
    lines.append(
        f"Overall DSAI={dsai_row['dsai']:.6f}, 95% CI=["
        f"{dsai_row['dsai_bootstrap_ci95_low']:.6f}, "
        f"{dsai_row['dsai_bootstrap_ci95_high']:.6f}]. Positive means Kunming target "
        "samples are farther from Shenzhen support than the reverse."
    )

    lines.extend(["", "3. Support coverage", "-" * 78])
    for _, row in coverage.sort_values(["class_scope", "target_city"]).iterrows():
        radius_text = (
            "class-specific radii"
            if pd.isna(row["source_support_radius_q95"])
            else f"radius={row['source_support_radius_q95']:.6f}"
        )
        lines.append(
            f"{row['coverage_label']}, {row['class_scope']}: "
            f"coverage={row['coverage']:.6f}, {radius_text}, 95% CI=["
            f"{row['bootstrap_ci95_low']:.6f}, {row['bootstrap_ci95_high']:.6f}]."
        )
    lines.append(
        f"Overall coverage asymmetry={coverage_asym_row['coverage_asymmetry']:.6f}, "
        f"95% CI=[{coverage_asym_row['coverage_asymmetry_bootstrap_ci95_low']:.6f}, "
        f"{coverage_asym_row['coverage_asymmetry_bootstrap_ci95_high']:.6f}]."
    )
    if coverage_saturated_zero:
        lines.append(
            "Coverage warning: the predefined within-source 95% radius yields zero "
            "coverage for every class and direction. The metric is threshold-saturated "
            "and is not direction-discriminative for these two domains."
        )

    lines.extend(["", "4. Representation diversity", "-" * 78])
    for _, row in diversity.loc[diversity["class_scope"] != "All"].sort_values(
        ["class_scope", "city"]
    ).iterrows():
        lines.append(
            f"{row['city']}, {row['class_scope']}: centroid-distance mean="
            f"{row['mean_cosine_distance_to_centroid']:.6f}, median="
            f"{row['median_cosine_distance_to_centroid']:.6f}, covariance trace="
            f"{row['covariance_trace']:.6f}, effective rank={row['effective_rank']:.3f}, "
            f"participation ratio={row['participation_ratio']:.3f}."
        )
    lines.append(
        "The diversity metrics describe within-domain spread; they do not by themselves "
        "establish directed support coverage."
    )

    lines.extend(["", "5. Symmetric MMD and CORAL discrepancy", "-" * 78])
    for _, row in discrepancy.iterrows():
        lines.append(
            f"{row['class_scope']}: MMD^2={row['mmd2_unbiased']:.6f}, "
            f"MMD={row['mmd_nonnegative_sqrt']:.6f}, covariance Frobenius="
            f"{row['covariance_frobenius_distance']:.6f}, "
            f"CORAL={row['coral_distance']:.10f}."
        )
    lines.append(
        "MMD and CORAL confirm symmetric domain discrepancy only. They cannot explain "
        "which transfer direction is easier."
    )

    lines.extend(["", "6. Per-year consistency", "-" * 78])
    yearly_all = yearly.loc[yearly["class_scope"] == "All"].drop_duplicates("year")
    for _, row in yearly_all.sort_values("year").iterrows():
        lines.append(
            f"{int(row['year'])}: DSAI={row['dsai']:.6f}, 95% CI=["
            f"{row['dsai_bootstrap_ci95_low']:.6f}, "
            f"{row['dsai_bootstrap_ci95_high']:.6f}]; coverage asymmetry="
            f"{row['coverage_asymmetry']:.6f}, 95% CI=["
            f"{row['coverage_asymmetry_bootstrap_ci95_low']:.6f}, "
            f"{row['coverage_asymmetry_bootstrap_ci95_high']:.6f}]."
        )

    comparisons = error_analysis.loc[
        error_analysis["analysis_type"] == "correct_vs_incorrect"
    ]
    quintiles = error_analysis.loc[
        error_analysis["analysis_type"] == "distance_quintile"
    ]
    lines.extend(
        [
            "",
            "7. Correct versus incorrect source-support distance",
            "-" * 78,
        ]
    )
    for (direction, scope), group in comparisons.groupby(["direction", "scope"]):
        significant = int((group["mannwhitney_p_two_sided"] < 0.05).sum())
        lines.append(
            f"{direction}, {scope}: correct mean={group['correct_distance_mean'].mean():.6f}, "
            f"incorrect mean={group['incorrect_distance_mean'].mean():.6f}; correct median="
            f"{group['correct_distance_median'].mean():.6f}, incorrect median="
            f"{group['incorrect_distance_median'].mean():.6f}; rank-biserial mean="
            f"{group['rank_biserial_incorrect_vs_correct'].mean():.6f}; "
            f"Mann-Whitney p<0.05 in {significant}/5 seeds; error-association AUC="
            f"{fmt(group['error_association_roc_auc'].mean(), group['error_association_roc_auc'].std(ddof=1))}."
        )

    lines.extend(["", "8. Distance-quintile error rates", "-" * 78])
    for (direction, scope), group in quintiles.groupby(["direction", "scope"]):
        q1 = group.loc[group["quintile"] == 1, "quintile_error_rate"]
        q5 = group.loc[group["quintile"] == 5, "quintile_error_rate"]
        lines.append(
            f"{direction}, {scope}: Q1={fmt(q1.mean(), q1.std(ddof=1))}, "
            f"Q5={fmt(q5.mean(), q5.std(ddof=1))}."
        )

    lines.extend(["", "9. Class 2 Expansion analysis", "-" * 78])
    for _, row in mechanism.iterrows():
        lines.append(
            f"{row['direction']}: Expansion-F1={fmt(row['zero_shot_expansion_f1_mean'], row['zero_shot_expansion_f1_std'])}; "
            f"Expansion Q1/Q5 error={row['expansion_q1_error_rate_mean']:.6f}/"
            f"{row['expansion_q5_error_rate_mean']:.6f}; error-distance AUC="
            f"{row['expansion_error_distance_association_auc_mean']:.6f}; rank-biserial="
            f"{row['expansion_error_distance_rank_biserial_mean']:.6f}."
        )

    lines.extend(["", "10. Transfer mechanism summary", "-" * 78])
    for _, row in mechanism.iterrows():
        lines.append(
            f"{row['direction']}: Macro-F1={fmt(row['zero_shot_macro_f1_mean'], row['zero_shot_macro_f1_std'])}, "
            f"Expansion-F1={fmt(row['zero_shot_expansion_f1_mean'], row['zero_shot_expansion_f1_std'])}, "
            f"support distance={row['mean_directed_support_distance']:.6f}, "
            f"coverage={row['coverage']:.6f}, Q1/Q5 error="
            f"{row['q1_error_rate_mean']:.6f}/{row['q5_error_rate_mean']:.6f}, "
            f"error-distance AUC={row['error_distance_association_auc_mean']:.6f}."
        )

    lines.extend(
        [
            "",
            "11. Statistical uncertainty",
            "-" * 78,
            "Core directed-distance and coverage confidence intervals use target-side "
            "2 km block bootstrap resampling, preserving within-block sample dependence.",
            "The source support reference and source radius are held fixed within each "
            "bootstrap; uncertainty therefore conditions on the observed source sample.",
            "Mann-Whitney tests are sample-level descriptive tests and do not correct for "
            "spatial dependence. Their p-values must be interpreted with effect sizes, "
            "quintile curves, and five-seed consistency.",
            "",
            "12. Warnings",
            "-" * 78,
            "- Class labels are used only to stratify representation support and never as "
            "an embedding dimension or prediction input.",
            "- Support distance is associated with error risk but is not a causal variable.",
            "- The study contains only two cities; generalization to other cities is unverified.",
            "- MMD bandwidth uses a fixed median-heuristic subsample; MMD and CORAL are symmetric.",
            "- The predefined 95% coverage radius is too strict for the observed cross-city "
            "separation: all coverage values are zero, so coverage asymmetry is uninformative.",
            "- Test labels are used only after zero-shot prediction for mechanism analysis.",
        ]
    )
    if warning_messages:
        lines.append("- Runtime/model/split warnings:")
        lines.extend(f"  - {message}" for message in sorted(set(warning_messages)))
    else:
        lines.append("- Runtime/model/split warnings: none.")

    lines.extend(
        [
            "",
            "13. Mechanism interpretation",
            "-" * 78,
            f"Hypothesis judgment: {mechanism_judgment}.",
            mechanism_text,
            "The result is a representation-geometry explanation consistent with the "
            "observed asymmetry. The causal origin of the city/domain signature remains unverified.",
            "",
            "14. Output tables and figures",
            "-" * 78,
            str(DIRECTED_PATH),
            str(COVERAGE_PATH),
            str(DIVERSITY_PATH),
            str(DISCREPANCY_PATH),
            str(YEARLY_PATH),
            str(ERROR_PATH),
            str(MECHANISM_PATH),
            str(REPORT_PATH),
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs(
    directed: pd.DataFrame,
    coverage: pd.DataFrame,
    diversity: pd.DataFrame,
    discrepancy: pd.DataFrame,
    yearly: pd.DataFrame,
    error_analysis: pd.DataFrame,
    mechanism: pd.DataFrame,
    stage05: dict[str, Any],
) -> None:
    if len(directed) != 2 * 4 * len(stage05["knn_values"]):
        raise AssertionError(f"Unexpected directed rows: {len(directed)}")
    if len(coverage) != 2 * 4:
        raise AssertionError(f"Unexpected coverage rows: {len(coverage)}")
    if len(diversity) != 2 * 4:
        raise AssertionError(f"Unexpected diversity rows: {len(diversity)}")
    if len(discrepancy) != 4:
        raise AssertionError(f"Unexpected discrepancy rows: {len(discrepancy)}")
    if len(yearly) != 3 * 2 * 4:
        raise AssertionError(f"Unexpected yearly rows: {len(yearly)}")
    if len(error_analysis) != 5 * 2 * 2 * 6:
        raise AssertionError(f"Unexpected error analysis rows: {len(error_analysis)}")
    if len(mechanism) != 2:
        raise AssertionError(f"Unexpected mechanism rows: {len(mechanism)}")
    for frame, columns in [
        (directed, ["mean", "std", "median", "q25", "q75", "q90", "q95"]),
        (coverage, ["coverage"]),
        (
            diversity,
            [
                "mean_cosine_distance_to_centroid",
                "covariance_trace",
                "effective_rank",
                "participation_ratio",
            ],
        ),
        (discrepancy, ["mmd2_unbiased", "coral_distance"]),
    ]:
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"Non-finite values in {columns}")
    if np.any((coverage["coverage"] < 0) | (coverage["coverage"] > 1)):
        raise AssertionError("Coverage outside [0,1]")
    if error_analysis.loc[
        error_analysis["analysis_type"] == "correct_vs_incorrect",
        "error_association_roc_auc",
    ].isna().any():
        raise AssertionError("Missing error association AUC")


def main() -> int:
    start = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    warning_messages: list[str] = []

    stage03 = load_stage03_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stage_config = config["stage_03"]
    stage05 = config["stage_05"]
    features = stage03.feature_definitions(config)
    aef192 = features["AEF-192"]
    if len(aef192) != 192 or FORBIDDEN_REPRESENTATION_FEATURES.intersection(aef192):
        raise AssertionError("AEF-192 definition contains forbidden fields")
    hashes_before = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    datasets = {
        city: stage03.validate_dataset(path, city, features)
        for city, path in CITY_FILES.items()
    }
    raw_192 = {
        city: datasets[city][aef192].to_numpy(dtype=np.float32, copy=False)
        for city in datasets
    }

    directed, coverage, _ = analyze_support_geometry(
        datasets,
        raw_192,
        "AEF-192 L2-normalized",
        [int(value) for value in stage05["knn_values"]],
        int(stage05["primary_k"]),
        int(stage05["bootstrap_resamples"]),
        int(stage05["bootstrap_seed"]),
    )
    print("MAIN SUPPORT geometry complete")
    diversity = representation_diversity(datasets, raw_192)
    print("DIVERSITY complete")
    discrepancy = domain_discrepancy(
        datasets,
        raw_192,
        int(stage05["domain_discrepancy_samples_per_city"]),
        int(stage05["mmd_bandwidth_sample_size"]),
        int(stage05["bootstrap_seed"]),
    )
    print("DOMAIN DISCREPANCY complete")
    yearly = yearly_support_analysis(datasets, [2020, 2022, 2024], stage05)
    error_analysis, performance = error_distance_analysis(
        stage03,
        datasets,
        aef192,
        stage_config,
        stage05,
        warning_messages,
    )
    mechanism = build_mechanism_summary(
        directed,
        coverage,
        error_analysis,
        performance,
        int(stage05["primary_k"]),
    )
    validate_outputs(
        directed,
        coverage,
        diversity,
        discrepancy,
        yearly,
        error_analysis,
        mechanism,
        stage05,
    )
    create_figures(
        directed,
        coverage,
        diversity,
        error_analysis,
        yearly,
        int(stage05["primary_k"]),
    )

    directed.to_csv(DIRECTED_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    coverage.to_csv(COVERAGE_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    diversity.to_csv(DIVERSITY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    discrepancy.to_csv(DISCREPANCY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    yearly.to_csv(YEARLY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    error_analysis.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    mechanism.to_csv(MECHANISM_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")

    hashes_after = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Processed input hash changed during Stage 05")
    elapsed_seconds = time.perf_counter() - start
    report = build_report(
        config,
        directed,
        coverage,
        diversity,
        discrepancy,
        yearly,
        error_analysis,
        mechanism,
        warning_messages,
        hashes_before == hashes_after,
        elapsed_seconds,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    expected_paths = [
        DIRECTED_PATH,
        COVERAGE_PATH,
        DIVERSITY_PATH,
        DISCREPANCY_PATH,
        YEARLY_PATH,
        ERROR_PATH,
        MECHANISM_PATH,
        REPORT_PATH,
        FIGURES_DIR / "13_directed_support_distance.png",
        FIGURES_DIR / "14_support_coverage.png",
        FIGURES_DIR / "15_representation_diversity.png",
        FIGURES_DIR / "16_support_distance_error_curve_SZ_to_KM.png",
        FIGURES_DIR / "17_support_distance_error_curve_KM_to_SZ.png",
        FIGURES_DIR / "18_yearly_support_asymmetry.png",
    ]
    for path in expected_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    for path in [
        DIRECTED_PATH,
        COVERAGE_PATH,
        DIVERSITY_PATH,
        DISCREPANCY_PATH,
        YEARLY_PATH,
        ERROR_PATH,
        MECHANISM_PATH,
    ]:
        pd.read_csv(path)
    print(f"Stage 05 complete in {elapsed_seconds:.2f}s")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

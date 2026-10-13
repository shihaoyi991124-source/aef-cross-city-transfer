"""Stage 07B: robustness and decomposition of multi-city transfer asymmetry.

This script is a statistical post-analysis of frozen Stage 07 outputs.  It
does not train classifiers, rerun predictions, or alter splits, labels, cities,
or seeds.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, pearsonr, spearmanr, t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "FiveCity_AEF_195D.parquet"

PAIR_PATH = TABLES_DIR / "stage07_pair_mechanism.csv"
CROSS_RUNS_PATH = TABLES_DIR / "stage07_cross_city_runs.csv"
SAME_RUNS_PATH = TABLES_DIR / "stage07_same_city_runs.csv"
SUPPORT_RUNS_PATH = TABLES_DIR / "stage07_directed_support_distance.csv"
SOURCE_PATH = TABLES_DIR / "stage07_source_transferability.csv"
TARGET_PATH = TABLES_DIR / "stage07_target_difficulty.csv"

WITHIN_TARGET_PATH = TABLES_DIR / "stage07b_within_target_ranking.csv"
SELECTION_PATH = TABLES_DIR / "stage07b_source_selection_hits.csv"
WITHIN_SOURCE_PATH = TABLES_DIR / "stage07b_within_source_target_difficulty.csv"
CENTERED_PATH = TABLES_DIR / "stage07b_centered_correlations.csv"
FIXED_PATH = TABLES_DIR / "stage07b_fixed_effect_models.csv"
QAP_PATH = TABLES_DIR / "stage07b_qap_results.csv"
LOCO_PATH = TABLES_DIR / "stage07b_leave_one_city_out.csv"
ASYMMETRY_PATH = TABLES_DIR / "stage07b_pair_asymmetry.csv"
DIVERSITY_PATH = TABLES_DIR / "stage07b_source_representation_diversity.csv"
REPORT_PATH = LOGS_DIR / "stage07b_transfer_decomposition_report.txt"

FIGURE_PATHS = {
    "within_target": FIGURES_DIR / "26_within_target_source_ranking.png",
    "loco": FIGURES_DIR / "27_leave_one_city_out_robustness.png",
    "qap": FIGURES_DIR / "28_qap_null_distribution_macro.png",
    "asymmetry": FIGURES_DIR / "29_pairwise_directional_asymmetry.png",
    "fixed": FIGURES_DIR / "30_fixed_effect_distance_relationship.png",
    "diversity": FIGURES_DIR / "31_source_representation_diversity.png",
}

CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
ABBREVIATIONS = {
    "Shenzhen": "SZ",
    "Kunming": "KM",
    "Wuhan": "WH",
    "Xian": "XA",
    "Beijing": "BJ",
}
REQUIRED_PAIR_COLUMNS = {
    "source_city",
    "target_city",
    "support_distance_mean",
    "class2_support_distance_mean",
    "cross_macro_f1_mean",
    "macro_f1_gap_mean",
    "cross_expansion_f1_mean",
    "expansion_f1_gap_mean",
}
HEADLINE_MACRO_RHO = 0.6210526315789473
HEADLINE_EXPANSION_RHO = 0.5278195488721804
TOLERANCE = 1e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def load_and_validate_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    input_paths = [
        PAIR_PATH,
        CROSS_RUNS_PATH,
        SAME_RUNS_PATH,
        SUPPORT_RUNS_PATH,
        SOURCE_PATH,
        TARGET_PATH,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in input_paths}
    tables = {path.stem: pd.read_csv(path) for path in input_paths}
    pair = tables[PAIR_PATH.stem].copy()
    missing = REQUIRED_PAIR_COLUMNS.difference(pair.columns)
    if missing:
        raise AssertionError(f"Stage 07 pair table is missing columns: {sorted(missing)}")
    if len(pair) != 20 or len(pair[["source_city", "target_city"]].drop_duplicates()) != 20:
        raise AssertionError("Stage 07 pair table must contain 20 unique directed pairs")
    if (pair["source_city"] == pair["target_city"]).any():
        raise AssertionError("Stage 07 pair table contains self-pairs")
    observed_cities = set(pair["source_city"]).union(pair["target_city"])
    if observed_cities != set(CITIES):
        raise AssertionError(f"Unexpected city set: {sorted(observed_cities)}")
    pair_order = {(source, target) for source in CITIES for target in CITIES if source != target}
    observed_pairs = set(map(tuple, pair[["source_city", "target_city"]].to_numpy()))
    if observed_pairs != pair_order:
        raise AssertionError("Directed-pair coverage is incomplete")
    numeric = pair[list(REQUIRED_PAIR_COLUMNS.difference({"source_city", "target_city"}))]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("Stage 07 pair table contains missing/non-finite required values")

    macro = spearmanr(pair["support_distance_mean"], pair["macro_f1_gap_mean"])
    expansion = spearmanr(
        pair["class2_support_distance_mean"], pair["expansion_f1_gap_mean"]
    )
    if not np.isclose(macro.statistic, HEADLINE_MACRO_RHO, rtol=0.0, atol=TOLERANCE):
        raise RuntimeError(
            f"Stage 07 Macro headline failed reproduction: {macro.statistic}"
        )
    if not np.isclose(
        expansion.statistic, HEADLINE_EXPANSION_RHO, rtol=0.0, atol=TOLERANCE
    ):
        raise RuntimeError(
            f"Stage 07 Expansion headline failed reproduction: {expansion.statistic}"
        )
    return pair, tables, hashes


def exact_vector_permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    """Positive one-sided exact Spearman p over all 4! source-label permutations."""
    observed = float(spearmanr(x, y).statistic)
    statistics = []
    for permutation in itertools.permutations(range(len(y))):
        statistics.append(float(spearmanr(x, y[list(permutation)]).statistic))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def ranking_text(frame: pd.DataFrame, column: str) -> str:
    ordered = frame.sort_values([column, "source_city"])["source_city"]
    return " > ".join(ABBREVIATIONS[city] for city in ordered)


def within_target_analysis(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    for target in CITIES:
        subset = pair.loc[pair["target_city"] == target].copy()
        if len(subset) != 4 or subset["source_city"].nunique() != 4:
            raise AssertionError(f"{target}: expected four candidate sources")
        macro_rho = spearmanr(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        macro_tau = kendalltau(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        expansion_rho = spearmanr(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        expansion_tau = kendalltau(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        rows.append(
            {
                "target_city": target,
                "target_abbreviation": ABBREVIATIONS[target],
                "candidate_sources": 4,
                "macro_spearman_rho": float(macro_rho.statistic),
                "macro_kendall_tau": float(macro_tau.statistic),
                "macro_exact_permutation_p_positive": exact_vector_permutation_p(
                    subset["support_distance_mean"].to_numpy(),
                    subset["macro_f1_gap_mean"].to_numpy(),
                ),
                "macro_distance_ranking": ranking_text(subset, "support_distance_mean"),
                "macro_gap_ranking": ranking_text(subset, "macro_f1_gap_mean"),
                "expansion_spearman_rho": float(expansion_rho.statistic),
                "expansion_kendall_tau": float(expansion_tau.statistic),
                "expansion_exact_permutation_p_positive": exact_vector_permutation_p(
                    subset["class2_support_distance_mean"].to_numpy(),
                    subset["expansion_f1_gap_mean"].to_numpy(),
                ),
                "expansion_distance_ranking": ranking_text(
                    subset, "class2_support_distance_mean"
                ),
                "expansion_gap_ranking": ranking_text(subset, "expansion_f1_gap_mean"),
                "permutation_count": 24,
                "inference_note": "effect direction/ranking is primary because n=4",
            }
        )

        macro_predicted = subset.sort_values(
            ["support_distance_mean", "source_city"]
        )["source_city"].tolist()
        macro_actual = subset.sort_values(["macro_f1_gap_mean", "source_city"])[
            "source_city"
        ].tolist()
        expansion_predicted = subset.sort_values(
            ["class2_support_distance_mean", "source_city"]
        )["source_city"].tolist()
        expansion_actual = subset.sort_values(
            ["expansion_f1_gap_mean", "source_city"]
        )["source_city"].tolist()
        hit_rows.append(
            {
                "target_city": target,
                "target_abbreviation": ABBREVIATIONS[target],
                "macro_predicted_best_source": macro_predicted[0],
                "macro_actual_best_source": macro_actual[0],
                "macro_predicted_top2": " | ".join(macro_predicted[:2]),
                "macro_top1_hit": macro_predicted[0] == macro_actual[0],
                "macro_top2_hit": macro_actual[0] in macro_predicted[:2],
                "expansion_predicted_best_source": expansion_predicted[0],
                "expansion_actual_best_source": expansion_actual[0],
                "expansion_predicted_top2": " | ".join(expansion_predicted[:2]),
                "expansion_top1_hit": expansion_predicted[0] == expansion_actual[0],
                "expansion_top2_hit": expansion_actual[0] in expansion_predicted[:2],
                "top2_definition": "actual best source appears in two lowest-distance sources",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(hit_rows)


def within_source_analysis(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in CITIES:
        subset = pair.loc[pair["source_city"] == source].copy()
        if len(subset) != 4 or subset["target_city"].nunique() != 4:
            raise AssertionError(f"{source}: expected four target cities")
        macro_rho = spearmanr(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        macro_tau = kendalltau(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        expansion_rho = spearmanr(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        expansion_tau = kendalltau(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        rows.append(
            {
                "source_city": source,
                "source_abbreviation": ABBREVIATIONS[source],
                "candidate_targets": 4,
                "macro_spearman_rho": float(macro_rho.statistic),
                "macro_kendall_tau": float(macro_tau.statistic),
                "macro_exact_permutation_p_positive": exact_vector_permutation_p(
                    subset["support_distance_mean"].to_numpy(),
                    subset["macro_f1_gap_mean"].to_numpy(),
                ),
                "expansion_spearman_rho": float(expansion_rho.statistic),
                "expansion_kendall_tau": float(expansion_tau.statistic),
                "expansion_exact_permutation_p_positive": exact_vector_permutation_p(
                    subset["class2_support_distance_mean"].to_numpy(),
                    subset["expansion_f1_gap_mean"].to_numpy(),
                ),
                "permutation_count": 24,
                "inference_note": "effect direction/ranking is primary because n=4",
            }
        )
    return pd.DataFrame(rows)


def centered_correlations(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = [
        (
            "target-centered",
            "Macro-F1 gap",
            "support_distance_mean",
            "macro_f1_gap_mean",
            "target_city",
        ),
        (
            "target-centered",
            "Expansion-F1 gap",
            "class2_support_distance_mean",
            "expansion_f1_gap_mean",
            "target_city",
        ),
        (
            "source-centered",
            "Macro-F1 gap",
            "support_distance_mean",
            "macro_f1_gap_mean",
            "source_city",
        ),
        (
            "source-centered",
            "Expansion-F1 gap",
            "class2_support_distance_mean",
            "expansion_f1_gap_mean",
            "source_city",
        ),
    ]
    for centering, outcome, distance_column, gap_column, group_column in specifications:
        distance = pair[distance_column] - pair.groupby(group_column)[distance_column].transform("mean")
        gap = pair[gap_column] - pair.groupby(group_column)[gap_column].transform("mean")
        spearman = spearmanr(distance, gap)
        pearson = pearsonr(distance, gap)
        rows.append(
            {
                "centering": centering,
                "outcome": outcome,
                "distance_column": distance_column,
                "gap_column": gap_column,
                "n_pairs": len(pair),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "group_means_removed": group_column,
            }
        )
    return pd.DataFrame(rows)


def design_matrix(
    frame: pd.DataFrame,
    include_source: bool,
    include_target: bool,
    distance_column: str | None,
) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    if include_source:
        for city in CITIES[1:]:
            columns.append((frame["source_city"] == city).to_numpy(dtype=float))
            names.append(f"source[{city}]")
    if include_target:
        for city in CITIES[1:]:
            columns.append((frame["target_city"] == city).to_numpy(dtype=float))
            names.append(f"target[{city}]")
    if distance_column is not None:
        columns.append(frame[distance_column].to_numpy(dtype=float))
        names.append("distance")
    return np.column_stack(columns), names


def ols_fit(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n, parameters = x.shape
    rank = int(np.linalg.matrix_rank(x))
    if rank != parameters:
        raise AssertionError(f"OLS design is rank deficient ({rank}/{parameters})")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    rss = float(residual @ residual)
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0
    df_resid = n - parameters
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / df_resid
    sigma2 = rss / df_resid
    covariance = sigma2 * np.linalg.inv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = beta / standard_errors
    p_values = 2.0 * student_t.sf(np.abs(t_statistics), df=df_resid)
    rss_safe = max(rss, np.finfo(float).tiny)
    log_likelihood = -0.5 * n * (
        math.log(2.0 * math.pi) + 1.0 + math.log(rss_safe / n)
    )
    return {
        "n_pairs": n,
        "parameters": parameters,
        "rank": rank,
        "df_resid": df_resid,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "aic": 2.0 * parameters - 2.0 * log_likelihood,
        "bic": math.log(n) * parameters - 2.0 * log_likelihood,
        "rss": rss,
        "coefficients": dict(zip(names, beta)),
        "standard_errors": dict(zip(names, standard_errors)),
        "t_statistics": dict(zip(names, t_statistics)),
        "p_values": dict(zip(names, p_values)),
        "residuals": residual,
        "fitted": fitted,
    }


def fixed_effect_models(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    outcomes = [
        ("Macro-F1 gap", "macro_f1_gap_mean", "support_distance_mean"),
        ("Expansion-F1 gap", "expansion_f1_gap_mean", "class2_support_distance_mean"),
    ]
    specifications = [
        ("Model 0", False, False, False),
        ("Model S", True, False, False),
        ("Model T", False, True, False),
        ("Model ST", True, True, False),
        ("Model STD", True, True, True),
    ]
    for outcome_name, outcome_column, distance_column in outcomes:
        fit_by_model: dict[str, Any] = {}
        for model_name, source_effects, target_effects, include_distance in specifications:
            x, names = design_matrix(
                pair,
                source_effects,
                target_effects,
                distance_column if include_distance else None,
            )
            fit = ols_fit(pair[outcome_column].to_numpy(), x, names)
            fit_by_model[model_name] = fit
            rows.append(
                {
                    "outcome": outcome_name,
                    "outcome_column": outcome_column,
                    "model": model_name,
                    "formula": {
                        "Model 0": "gap ~ 1",
                        "Model S": "gap ~ source_city",
                        "Model T": "gap ~ target_city",
                        "Model ST": "gap ~ source_city + target_city",
                        "Model STD": "gap ~ source_city + target_city + distance",
                    }[model_name],
                    "distance_column": distance_column if include_distance else "",
                    "n_pairs": fit["n_pairs"],
                    "parameters": fit["parameters"],
                    "df_resid": fit["df_resid"],
                    "r_squared": fit["r_squared"],
                    "adjusted_r_squared": fit["adjusted_r_squared"],
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                    "support_distance_beta": fit["coefficients"].get("distance", np.nan),
                    "support_distance_standard_error": fit["standard_errors"].get(
                        "distance", np.nan
                    ),
                    "support_distance_t": fit["t_statistics"].get("distance", np.nan),
                    "support_distance_ordinary_p": fit["p_values"].get(
                        "distance", np.nan
                    ),
                    "delta_r_squared_vs_st": np.nan,
                    "ordinary_p_is_primary": False,
                }
            )
        delta = fit_by_model["Model STD"]["r_squared"] - fit_by_model["Model ST"][
            "r_squared"
        ]
        rows[-1]["delta_r_squared_vs_st"] = delta
        cache[outcome_name] = {
            "fits": fit_by_model,
            "outcome_column": outcome_column,
            "distance_column": distance_column,
        }
    return pd.DataFrame(rows), cache


def matrix_from_pair(pair: pd.DataFrame, column: str, cities: list[str]) -> np.ndarray:
    index = {city: position for position, city in enumerate(cities)}
    matrix = np.full((len(cities), len(cities)), np.nan, dtype=float)
    for row in pair.itertuples(index=False):
        if row.source_city in index and row.target_city in index:
            matrix[index[row.source_city], index[row.target_city]] = float(
                getattr(row, column)
            )
    off_diagonal = ~np.eye(len(cities), dtype=bool)
    if not np.isfinite(matrix[off_diagonal]).all():
        raise AssertionError(f"Incomplete matrix for {column}")
    return matrix


def qap_association(
    pair: pd.DataFrame,
    distance_column: str,
    outcome_column: str,
    expected_direction: str,
    label: str,
    cities: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    cities = CITIES if cities is None else cities
    distance = matrix_from_pair(pair, distance_column, cities)
    outcome = matrix_from_pair(pair, outcome_column, cities)
    mask = ~np.eye(len(cities), dtype=bool)
    observed = float(spearmanr(distance[mask], outcome[mask]).statistic)
    null = []
    for permutation in itertools.permutations(range(len(cities))):
        permuted = distance[np.ix_(permutation, permutation)]
        null.append(float(spearmanr(permuted[mask], outcome[mask]).statistic))
    null_array = np.asarray(null)
    if expected_direction == "positive":
        one_sided = float(np.mean(null_array >= observed - 1e-12))
    elif expected_direction == "negative":
        one_sided = float(np.mean(null_array <= observed + 1e-12))
    else:
        raise ValueError(expected_direction)
    result = {
        "test": label,
        "test_type": "association QAP",
        "distance_column": distance_column,
        "outcome_column": outcome_column,
        "expected_direction": expected_direction,
        "cities": len(cities),
        "directed_pairs": int(mask.sum()),
        "permutations": math.factorial(len(cities)),
        "observed_statistic": observed,
        "statistic_name": "Spearman rho",
        "exact_qap_one_sided_p": one_sided,
        "exact_qap_two_sided_p": float(
            np.mean(np.abs(null_array) >= abs(observed) - 1e-12)
        ),
        "permutation_mean": float(null_array.mean()),
        "permutation_std": float(null_array.std(ddof=1)),
        "permutation_min": float(null_array.min()),
        "permutation_max": float(null_array.max()),
        "permutation_ci95_low": float(np.quantile(null_array, 0.025)),
        "permutation_ci95_high": float(np.quantile(null_array, 0.975)),
        "p_value_convention": (
            "count as-or-more-extreme / all permutations; identity included; "
            "two-sided uses absolute statistic"
        ),
    }
    return result, null_array


def permuted_distance_column(
    pair: pd.DataFrame,
    distance_column: str,
    permutation: tuple[int, ...],
) -> np.ndarray:
    matrix = matrix_from_pair(pair, distance_column, CITIES)
    permuted = matrix[np.ix_(permutation, permutation)]
    index = {city: position for position, city in enumerate(CITIES)}
    return np.asarray(
        [permuted[index[s], index[t]] for s, t in pair[["source_city", "target_city"]].itertuples(index=False, name=None)],
        dtype=float,
    )


def qap_fixed_effect_beta(
    pair: pd.DataFrame,
    outcome_column: str,
    distance_column: str,
    label: str,
) -> tuple[dict[str, Any], np.ndarray]:
    y = pair[outcome_column].to_numpy(dtype=float)
    x, names = design_matrix(pair, True, True, distance_column)
    observed_fit = ols_fit(y, x, names)
    observed = float(observed_fit["coefficients"]["distance"])
    betas = []
    for permutation in itertools.permutations(range(5)):
        working = pair.copy()
        working["_permuted_distance"] = permuted_distance_column(
            pair, distance_column, permutation
        )
        px, pnames = design_matrix(working, True, True, "_permuted_distance")
        betas.append(float(ols_fit(y, px, pnames)["coefficients"]["distance"]))
    null = np.asarray(betas)
    result = {
        "test": label,
        "test_type": "fixed-effect beta QAP",
        "distance_column": distance_column,
        "outcome_column": outcome_column,
        "expected_direction": "positive",
        "cities": 5,
        "directed_pairs": 20,
        "permutations": 120,
        "observed_statistic": observed,
        "statistic_name": "OLS distance beta",
        "exact_qap_one_sided_p": float(np.mean(null >= observed - 1e-12)),
        "exact_qap_two_sided_p": float(
            np.mean(np.abs(null) >= abs(observed) - 1e-12)
        ),
        "permutation_mean": float(null.mean()),
        "permutation_std": float(null.std(ddof=1)),
        "permutation_min": float(null.min()),
        "permutation_max": float(null.max()),
        "permutation_ci95_low": float(np.quantile(null, 0.025)),
        "permutation_ci95_high": float(np.quantile(null, 0.975)),
        "p_value_convention": (
            "count as-or-more-extreme / 120; identity included; "
            "two-sided uses absolute beta"
        ),
    }
    return result, null


def leave_one_city_out(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for removed in CITIES:
        remaining = [city for city in CITIES if city != removed]
        subset = pair.loc[
            (pair["source_city"] != removed) & (pair["target_city"] != removed)
        ].copy()
        if len(subset) != 12:
            raise AssertionError(f"LOCO {removed}: expected 12 directed pairs")
        macro_spearman = spearmanr(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        macro_pearson = pearsonr(
            subset["support_distance_mean"], subset["macro_f1_gap_mean"]
        )
        macro_qap, _ = qap_association(
            subset,
            "support_distance_mean",
            "macro_f1_gap_mean",
            "positive",
            f"LOCO Macro remove {removed}",
            remaining,
        )
        expansion_spearman = spearmanr(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        expansion_pearson = pearsonr(
            subset["class2_support_distance_mean"], subset["expansion_f1_gap_mean"]
        )
        expansion_qap, _ = qap_association(
            subset,
            "class2_support_distance_mean",
            "expansion_f1_gap_mean",
            "positive",
            f"LOCO Expansion remove {removed}",
            remaining,
        )
        rows.append(
            {
                "removed_city": removed,
                "removed_abbreviation": ABBREVIATIONS[removed],
                "remaining_cities": " | ".join(remaining),
                "directed_pairs": len(subset),
                "qap_permutations": 24,
                "macro_spearman_rho": float(macro_spearman.statistic),
                "macro_spearman_descriptive_p": float(macro_spearman.pvalue),
                "macro_pearson_r": float(macro_pearson.statistic),
                "macro_pearson_descriptive_p": float(macro_pearson.pvalue),
                "macro_qap_one_sided_p": macro_qap["exact_qap_one_sided_p"],
                "macro_qap_two_sided_p": macro_qap["exact_qap_two_sided_p"],
                "expansion_spearman_rho": float(expansion_spearman.statistic),
                "expansion_spearman_descriptive_p": float(expansion_spearman.pvalue),
                "expansion_pearson_r": float(expansion_pearson.statistic),
                "expansion_pearson_descriptive_p": float(expansion_pearson.pvalue),
                "expansion_qap_one_sided_p": expansion_qap[
                    "exact_qap_one_sided_p"
                ],
                "expansion_qap_two_sided_p": expansion_qap[
                    "exact_qap_two_sided_p"
                ],
            }
        )
    return pd.DataFrame(rows)


def pair_asymmetry(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = pair.set_index(["source_city", "target_city"])
    rows: list[dict[str, Any]] = []
    for city_a, city_b in itertools.combinations(CITIES, 2):
        forward = indexed.loc[(city_a, city_b)]
        reverse = indexed.loc[(city_b, city_a)]
        delta_d = float(forward["support_distance_mean"] - reverse["support_distance_mean"])
        delta_g = float(forward["macro_f1_gap_mean"] - reverse["macro_f1_gap_mean"])
        delta_d2 = float(
            forward["class2_support_distance_mean"]
            - reverse["class2_support_distance_mean"]
        )
        delta_g2 = float(
            forward["expansion_f1_gap_mean"] - reverse["expansion_f1_gap_mean"]
        )
        rows.append(
            {
                "city_a": city_a,
                "city_b": city_b,
                "pair_label": f"{ABBREVIATIONS[city_a]}/{ABBREVIATIONS[city_b]}",
                "delta_support_distance_a_to_b_minus_b_to_a": delta_d,
                "delta_macro_gap_a_to_b_minus_b_to_a": delta_g,
                "macro_direction_concordant": np.sign(delta_d) == np.sign(delta_g),
                "abs_support_distance_asymmetry": abs(delta_d),
                "abs_macro_gap_asymmetry": abs(delta_g),
                "delta_class2_distance_a_to_b_minus_b_to_a": delta_d2,
                "delta_expansion_gap_a_to_b_minus_b_to_a": delta_g2,
                "expansion_direction_concordant": np.sign(delta_d2) == np.sign(delta_g2),
                "abs_class2_distance_asymmetry": abs(delta_d2),
                "abs_expansion_gap_asymmetry": abs(delta_g2),
            }
        )
    output = pd.DataFrame(rows)
    macro_hits = int(output["macro_direction_concordant"].sum())
    expansion_hits = int(output["expansion_direction_concordant"].sum())
    macro_abs = spearmanr(
        output["abs_support_distance_asymmetry"], output["abs_macro_gap_asymmetry"]
    )
    expansion_abs = spearmanr(
        output["abs_class2_distance_asymmetry"],
        output["abs_expansion_gap_asymmetry"],
    )
    summary = {
        "macro_concordant": macro_hits,
        "macro_proportion": macro_hits / 10.0,
        "macro_binomial_two_sided_p": float(
            binomtest(macro_hits, 10, 0.5, alternative="two-sided").pvalue
        ),
        "macro_abs_asymmetry_spearman": float(macro_abs.statistic),
        "macro_abs_asymmetry_descriptive_p": float(macro_abs.pvalue),
        "expansion_concordant": expansion_hits,
        "expansion_proportion": expansion_hits / 10.0,
        "expansion_binomial_two_sided_p": float(
            binomtest(expansion_hits, 10, 0.5, alternative="two-sided").pvalue
        ),
        "expansion_abs_asymmetry_spearman": float(expansion_abs.statistic),
        "expansion_abs_asymmetry_descriptive_p": float(expansion_abs.pvalue),
    }
    for key, value in summary.items():
        output[key] = value
    return output, summary


def aef192_columns() -> list[str]:
    return [f"A{dimension:02d}_{year}" for year in [2020, 2022, 2024] for dimension in range(64)]


def representation_diversity(source_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = aef192_columns()
    columns = ["city", "class"] + features
    frame = pd.read_parquet(DATA_PATH, columns=columns, engine="pyarrow")
    if len(frame) != 90000 or set(frame["city"].unique()) != set(CITIES):
        raise AssertionError("Five-city Parquet failed diversity input validation")
    rows: list[dict[str, Any]] = []
    for city in CITIES:
        city_frame = frame.loc[frame["city"] == city]
        for class_value in [0, 1, 2]:
            raw = city_frame.loc[city_frame["class"] == class_value, features].to_numpy(
                dtype=np.float64, copy=True
            )
            if raw.shape != (6000, 192) or not np.isfinite(raw).all():
                raise AssertionError(f"{city} class {class_value}: invalid diversity matrix")
            norms = np.linalg.norm(raw, axis=1)
            if np.any(norms <= 0):
                raise AssertionError("Zero representation norm")
            x = raw / norms[:, None]
            centroid = x.mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            centroid_unit = centroid / centroid_norm
            distances = np.clip(1.0 - x @ centroid_unit, 0.0, 2.0)
            covariance = np.cov(x, rowvar=False, ddof=1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            trace = float(eigenvalues.sum())
            probabilities = eigenvalues[eigenvalues > 1e-15] / trace
            effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
            participation_ratio = float(trace**2 / np.sum(eigenvalues**2))
            rows.append(
                {
                    "city": city,
                    "city_abbreviation": ABBREVIATIONS[city],
                    "class_scope": f"Class {class_value}",
                    "class_value": class_value,
                    "aggregation": "class-specific",
                    "samples": len(x),
                    "representation": "AEF-192 L2-normalized",
                    "mean_cosine_distance_to_centroid": float(distances.mean()),
                    "median_cosine_distance_to_centroid": float(np.median(distances)),
                    "covariance_trace": trace,
                    "effective_rank": effective_rank,
                    "participation_ratio": participation_ratio,
                    "centroid_norm": centroid_norm,
                    "mean_outgoing_macro_f1": np.nan,
                    "mean_outgoing_macro_gap": np.nan,
                }
            )
    class_rows = pd.DataFrame(rows)
    average_rows = (
        class_rows.groupby(["city", "city_abbreviation"], sort=False)
        [[
            "mean_cosine_distance_to_centroid",
            "median_cosine_distance_to_centroid",
            "covariance_trace",
            "effective_rank",
            "participation_ratio",
            "centroid_norm",
        ]]
        .mean()
        .reset_index()
    )
    average_rows["class_scope"] = "Mean across classes"
    average_rows["class_value"] = pd.NA
    average_rows["aggregation"] = "unweighted mean of three class statistics"
    average_rows["samples"] = 18000
    average_rows["representation"] = "AEF-192 L2-normalized"
    source_metrics = source_summary.set_index("source_city")
    average_rows["mean_outgoing_macro_f1"] = average_rows["city"].map(
        source_metrics["mean_outgoing_macro_f1"]
    )
    average_rows["mean_outgoing_macro_gap"] = average_rows["city"].map(
        source_metrics["mean_outgoing_macro_gap"]
    )
    average_rows["effective_rank_descending_rank"] = average_rows[
        "effective_rank"
    ].rank(method="min", ascending=False).astype(int)
    class_rows["effective_rank_descending_rank"] = pd.NA
    output = pd.concat([class_rows, average_rows], ignore_index=True, sort=False)
    output = output.sort_values(["city", "class_scope"]).reset_index(drop=True)
    correlations: dict[str, Any] = {}
    for metric in [
        "mean_cosine_distance_to_centroid",
        "covariance_trace",
        "effective_rank",
        "participation_ratio",
    ]:
        for outcome in ["mean_outgoing_macro_f1", "mean_outgoing_macro_gap"]:
            result = spearmanr(average_rows[metric], average_rows[outcome])
            correlations[f"{metric}__{outcome}"] = {
                "rho": float(result.statistic),
                "p": float(result.pvalue),
                "n_cities": 5,
                "interpretation": "descriptive/exploratory only",
            }
    return output, correlations


def source_selection_judgment(within_target: pd.DataFrame, hits: pd.DataFrame, prefix: str) -> str:
    rhos = within_target[f"{prefix}_spearman_rho"]
    median = float(rhos.median())
    positives = int((rhos > 0).sum())
    top1 = int(hits[f"{prefix}_top1_hit"].sum())
    if median > 0.5 and positives >= 4 and top1 >= 3:
        return "SUPPORTED"
    if median > 0 and positives >= 3:
        return "PARTIALLY SUPPORTED"
    return "NOT SUPPORTED"


def robust_global_judgment(qap_row: dict[str, Any], loco: pd.DataFrame, prefix: str) -> str:
    positives = int((loco[f"{prefix}_spearman_rho"] > 0).sum())
    expected_sign = qap_row["observed_statistic"] > 0
    if expected_sign and qap_row["exact_qap_one_sided_p"] < 0.05 and positives >= 4:
        return "SUPPORTED"
    if expected_sign:
        return "PARTIALLY SUPPORTED"
    return "NOT SUPPORTED"


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


def plot_within_target(pair: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    for axis, target in zip(axes.flat, CITIES):
        subset = pair.loc[pair["target_city"] == target].sort_values(
            "support_distance_mean"
        )
        x = np.arange(4)
        labels = [ABBREVIATIONS[value] for value in subset["source_city"]]
        bars = axis.bar(x, subset["support_distance_mean"], color="#4c78a8", alpha=0.85)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Candidate source")
        axis.set_ylabel("Support distance", color="#2b5d8a")
        axis.tick_params(axis="y", labelcolor="#2b5d8a")
        axis.set_title(f"Target {ABBREVIATIONS[target]}", weight="bold")
        axis2 = axis.twinx()
        axis2.plot(x, subset["macro_f1_gap_mean"], color="#e45756", marker="o", linewidth=2)
        axis2.set_ylabel("Macro-F1 gap", color="#b23b3a")
        axis2.tick_params(axis="y", labelcolor="#b23b3a")
        axis2.grid(False)
        for bar, value in zip(bars, subset["support_distance_mean"]):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Within-target source ranking: distance and Macro-F1 gap", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_PATHS["within_target"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loco(loco: pd.DataFrame) -> None:
    labels = ["Original"] + [f"−{value}" for value in loco["removed_abbreviation"]]
    macro = [HEADLINE_MACRO_RHO] + loco["macro_spearman_rho"].tolist()
    expansion = [HEADLINE_EXPANSION_RHO] + loco["expansion_spearman_rho"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.bar(x - width / 2, macro, width, label="Macro", color="#4c78a8")
    ax.bar(x + width / 2, expansion, width, label="Expansion", color="#f58518")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman rho")
    ax.set_xlabel("Full analysis or city removed")
    ax.set_title("Leave-one-city-out robustness", weight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["loco"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qap(null: np.ndarray, observed: float, one_sided_p: float) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.hist(null, bins=16, color="#72b7b2", edgecolor="white", alpha=0.9)
    ax.axvline(observed, color="#d62728", linewidth=2.5, label=f"Observed rho={observed:.3f}")
    ax.set_xlabel("Spearman rho under node-label QAP")
    ax.set_ylabel("Permutation count")
    ax.set_title("Exact QAP null distribution: support distance vs Macro-F1 gap", weight="bold")
    ax.text(0.02, 0.95, f"120 permutations\none-sided p={one_sided_p:.4f}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["qap"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_asymmetry(asymmetry: pd.DataFrame) -> None:
    x = asymmetry["delta_support_distance_a_to_b_minus_b_to_a"]
    y = asymmetry["delta_macro_gap_a_to_b_minus_b_to_a"]
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    ax.scatter(x, y, s=70, color="#4c78a8", edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axvline(0, color="#666666", linewidth=0.8)
    for row in asymmetry.itertuples(index=False):
        ax.annotate(
            row.pair_label,
            (
                row.delta_support_distance_a_to_b_minus_b_to_a,
                row.delta_macro_gap_a_to_b_minus_b_to_a,
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Directional support-distance asymmetry")
    ax.set_ylabel("Directional Macro-F1-gap asymmetry")
    ax.set_title("Directional asymmetry across 10 unordered city pairs", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["asymmetry"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def residualize(pair: pd.DataFrame, column: str) -> np.ndarray:
    x, names = design_matrix(pair, True, True, None)
    return ols_fit(pair[column].to_numpy(), x, names)["residuals"]


def plot_fixed_effect(pair: pd.DataFrame, beta: float) -> None:
    x = residualize(pair, "support_distance_mean")
    y = residualize(pair, "macro_f1_gap_mean")
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    ax.scatter(x, y, s=64, color="#4c78a8", edgecolor="white", linewidth=0.8)
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, beta * line_x, color="#e45756", linewidth=2.2)
    for row, xv, yv in zip(pair.itertuples(index=False), x, y):
        ax.annotate(
            f"{ABBREVIATIONS[row.source_city]}→{ABBREVIATIONS[row.target_city]}",
            (xv, yv),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.axvline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("Support-distance residual after source/target effects")
    ax.set_ylabel("Macro-gap residual after source/target effects")
    ax.set_title("Source/target-adjusted distance relationship", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["fixed"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diversity(diversity: pd.DataFrame) -> None:
    data = diversity.loc[diversity["class_scope"] == "Mean across classes"].copy()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.scatter(
        data["effective_rank"],
        data["mean_outgoing_macro_f1"],
        s=80,
        color="#54a24b",
        edgecolor="white",
        linewidth=0.8,
    )
    for row in data.itertuples(index=False):
        ax.annotate(
            row.city_abbreviation,
            (row.effective_rank, row.mean_outgoing_macro_f1),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean class effective rank")
    ax.set_ylabel("Mean outgoing zero-shot Macro-F1")
    ax.set_title("Source representation diversity (exploratory, n=5)", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["diversity"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    pair: pd.DataFrame,
    input_hashes: dict[str, str],
    within_target: pd.DataFrame,
    hits: pd.DataFrame,
    within_source: pd.DataFrame,
    centered: pd.DataFrame,
    fixed: pd.DataFrame,
    qap: pd.DataFrame,
    loco: pd.DataFrame,
    asymmetry: pd.DataFrame,
    asymmetry_summary: dict[str, Any],
    diversity: pd.DataFrame,
    diversity_correlations: dict[str, Any],
    macro_global_judgment: str,
    expansion_global_judgment: str,
    macro_selection_judgment: str,
    expansion_selection_judgment: str,
    scenario: str,
    recommendation: str,
    elapsed_seconds: float,
) -> None:
    qap_index = qap.set_index("test")
    macro_qap = qap_index.loc["Overall distance vs Macro gap"]
    expansion_qap = qap_index.loc["Class2 distance vs Expansion gap"]
    macro_beta = qap_index.loc["Macro fixed-effect distance beta"]
    expansion_beta = qap_index.loc["Expansion fixed-effect distance beta"]
    macro_std = fixed.loc[
        (fixed["outcome"] == "Macro-F1 gap") & (fixed["model"] == "Model STD")
    ].iloc[0]
    expansion_std = fixed.loc[
        (fixed["outcome"] == "Expansion-F1 gap") & (fixed["model"] == "Model STD")
    ].iloc[0]
    macro_target_centered = centered.loc[
        (centered["centering"] == "target-centered")
        & (centered["outcome"] == "Macro-F1 gap")
    ].iloc[0]
    macro_source_centered = centered.loc[
        (centered["centering"] == "source-centered")
        & (centered["outcome"] == "Macro-F1 gap")
    ].iloc[0]
    diversity_means = diversity.loc[diversity["class_scope"] == "Mean across classes"].sort_values(
        "effective_rank", ascending=False
    )
    macro_top1 = int(hits["macro_top1_hit"].sum())
    macro_top2 = int(hits["macro_top2_hit"].sum())
    expansion_top1 = int(hits["expansion_top1_hit"].sum())
    expansion_top2 = int(hits["expansion_top2_hit"].sum())

    lines = [
        "Stage 07B - Robustness and Decomposition of Multi-city Transfer Asymmetry",
        "=" * 79,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Runtime: {elapsed_seconds:.2f} seconds",
        "",
        "1. Inputs and reproducibility",
        "- Statistical post-analysis only. No classifier was trained and no Stage 07 prediction was rerun.",
        "- Frozen cities: Shenzhen, Kunming, Wuhan, Xian, Beijing.",
        "- Pair table: 20 unique off-diagonal directed city pairs; no self-pairs.",
        "- Stage 07 splits, labels, city set and seeds were not modified.",
        "- Input hashes:",
        *[f"  {path}: {digest}" for path, digest in input_hashes.items()],
        "",
        "2. Stage 07 headline reproduction",
        f"- Overall support distance vs Macro-F1 gap: Spearman rho={HEADLINE_MACRO_RHO:.12f} (PASS).",
        f"- Class-2 support distance vs Expansion-F1 gap: Spearman rho={HEADLINE_EXPANSION_RHO:.12f} (PASS).",
        "",
        "3. Within-target source ranking (four candidate sources per target)",
        within_target.to_string(index=False),
        f"- Macro median rho={within_target['macro_spearman_rho'].median():.6f}; positive={int((within_target['macro_spearman_rho'] > 0).sum())}/5.",
        f"- Expansion median rho={within_target['expansion_spearman_rho'].median():.6f}; positive={int((within_target['expansion_spearman_rho'] > 0).sum())}/5.",
        "- Exact permutation p values enumerate all 4!=24 source-label permutations including identity.",
        "",
        "4. Source-selection top-1/top-2 results",
        hits.to_string(index=False),
        f"- Macro top-1={macro_top1}/5; top-2={macro_top2}/5.",
        f"- Expansion top-1={expansion_top1}/5; top-2={expansion_top2}/5.",
        "- Top-2 means the actual minimum-gap source appears among the two minimum-distance sources.",
        "",
        "5. Within-source target difficulty",
        within_source.to_string(index=False),
        f"- Macro median rho={within_source['macro_spearman_rho'].median():.6f}; positive={int((within_source['macro_spearman_rho'] > 0).sum())}/5.",
        f"- Expansion median rho={within_source['expansion_spearman_rho'].median():.6f}; positive={int((within_source['expansion_spearman_rho'] > 0).sum())}/5.",
        "",
        "6. Target-centered correlations",
        centered.loc[centered['centering'] == 'target-centered'].to_string(index=False),
        "",
        "7. Source-centered correlations",
        centered.loc[centered['centering'] == 'source-centered'].to_string(index=False),
        "",
        "8. Two-way source/target fixed-effect regression",
        fixed.to_string(index=False),
        f"- Macro STD beta={macro_std['support_distance_beta']:.6f}, SE={macro_std['support_distance_standard_error']:.6f}, t={macro_std['support_distance_t']:.6f}, ordinary p={macro_std['support_distance_ordinary_p']:.6g}.",
        f"- Expansion STD beta={expansion_std['support_distance_beta']:.6f}, SE={expansion_std['support_distance_standard_error']:.6f}, t={expansion_std['support_distance_t']:.6f}, ordinary p={expansion_std['support_distance_ordinary_p']:.6g}.",
        "- Ordinary coefficient p values are secondary; node-label QAP is primary.",
        "",
        "9. Incremental explanatory value",
        f"- Macro Delta R2 = {macro_std['delta_r_squared_vs_st']:.6f}.",
        f"- Expansion Delta R2 = {expansion_std['delta_r_squared_vs_st']:.6f}.",
        "- These R2 increments are descriptive with n=20.",
        "",
        "10. Exact 120-permutation association QAP",
        qap.loc[qap['test_type'] == 'association QAP'].to_string(index=False),
        "- Convention: all 5!=120 simultaneous row/column node-label permutations of the distance matrix, including identity.",
        "- One-sided p counts permutation statistics as or more extreme in the expected direction; two-sided p uses absolute rho.",
        "",
        "11. Fixed-effect beta QAP",
        qap.loc[qap['test_type'] == 'fixed-effect beta QAP'].to_string(index=False),
        f"- Macro beta={macro_beta['observed_statistic']:.6f}, one-sided p={macro_beta['exact_qap_one_sided_p']:.6f}, two-sided p={macro_beta['exact_qap_two_sided_p']:.6f}.",
        f"- Expansion beta={expansion_beta['observed_statistic']:.6f}, one-sided p={expansion_beta['exact_qap_one_sided_p']:.6f}, two-sided p={expansion_beta['exact_qap_two_sided_p']:.6f}.",
        "",
        "12. Leave-one-city-out analysis",
        loco.to_string(index=False),
        f"- Macro positive rho count={int((loco['macro_spearman_rho'] > 0).sum())}/5.",
        f"- Expansion positive rho count={int((loco['expansion_spearman_rho'] > 0).sum())}/5.",
        f"- After removing Kunming: Macro rho={loco.loc[loco['removed_city'] == 'Kunming', 'macro_spearman_rho'].iloc[0]:.6f}.",
        "",
        "13. Directional asymmetry concordance",
        asymmetry.to_string(index=False),
        f"- Macro: {asymmetry_summary['macro_concordant']}/10 concordant; exact two-sided binomial p={asymmetry_summary['macro_binomial_two_sided_p']:.6f}; absolute-asymmetry Spearman rho={asymmetry_summary['macro_abs_asymmetry_spearman']:.6f}.",
        f"- Expansion: {asymmetry_summary['expansion_concordant']}/10 concordant; exact two-sided binomial p={asymmetry_summary['expansion_binomial_two_sided_p']:.6f}; absolute-asymmetry Spearman rho={asymmetry_summary['expansion_abs_asymmetry_spearman']:.6f}.",
        "",
        "14. Source representation diversity (exploratory only)",
        diversity_means.to_string(index=False),
        "- Effective-rank ranking: " + " > ".join(diversity_means['city_abbreviation']),
        "- Descriptive correlations (n=5; no inferential claim):",
        *[
            f"  {key}: rho={value['rho']:.6f}, descriptive p={value['p']:.6g}"
            for key, value in diversity_correlations.items()
        ],
        "- No combined diversity index was created.",
        "",
        "15. Macro conclusion",
        f"- Robust global association judgment: {macro_global_judgment}.",
        f"- Within-target source-selection judgment: {macro_selection_judgment}.",
        f"- Target-centered Macro rho={macro_target_centered['spearman_rho']:.6f}; source-centered Macro rho={macro_source_centered['spearman_rho']:.6f}.",
        "",
        "16. Expansion conclusion",
        f"- Robust global Expansion association judgment: {expansion_global_judgment}.",
        f"- Within-target Expansion source-selection judgment: {expansion_selection_judgment}.",
        "",
        "17. Scenario judgment",
        f"- Scenario {scenario}.",
        "- The global and within-source results are stronger than the within-target top-1 source-selection results.",
        "",
        "18. Limitations",
        "- Only five cities and 20 directed pairs are available.",
        "- Each within-target ranking has only four candidate sources.",
        "- Source and target directions are dyadically dependent.",
        "- QAP handles node-label dependence more appropriately than treating all pairs as independent, but five nodes still provide limited inferential resolution.",
        "- Source diversity analysis has n=5 cities and is descriptive/exploratory only.",
        "- Fixed-effect OLS has only 10 residual degrees of freedom in Model STD.",
        "- No causal claim is supported.",
        "",
        "19. Warnings",
        "- No data-integrity, headline-reproduction or execution warning occurred.",
        "- Exact within-target/within-source p values have coarse 1/24 resolution; five-city QAP p values have 1/120 resolution.",
        "- Ordinary p values assume independent pairs and are secondary to QAP.",
        "",
        "20. Recommendation for Stage 08",
        f"- {recommendation}",
        "- Stage 08 was not started.",
        "",
        "Outputs",
        *[
            f"- {path.relative_to(PROJECT_ROOT)}"
            for path in [
                WITHIN_TARGET_PATH,
                SELECTION_PATH,
                WITHIN_SOURCE_PATH,
                CENTERED_PATH,
                FIXED_PATH,
                QAP_PATH,
                LOCO_PATH,
                ASYMMETRY_PATH,
                DIVERSITY_PATH,
                *FIGURE_PATHS.values(),
                REPORT_PATH,
            ]
        ],
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    expected = {
        WITHIN_TARGET_PATH: 5,
        SELECTION_PATH: 5,
        WITHIN_SOURCE_PATH: 5,
        CENTERED_PATH: 4,
        FIXED_PATH: 10,
        QAP_PATH: 5,
        LOCO_PATH: 5,
        ASYMMETRY_PATH: 10,
        DIVERSITY_PATH: 20,
    }
    for path, rows in expected.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing/empty output: {path}")
        frame = pd.read_csv(path)
        if len(frame) != rows:
            raise AssertionError(f"{path.name}: expected {rows} rows, got {len(frame)}")
        text = frame.astype(str)
        if text.apply(
            lambda column: column.str.contains(
                r"#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
                regex=True,
            ).any()
        ).any():
            raise AssertionError(f"Spreadsheet-style error token found in {path}")
    for path in FIGURE_PATHS.values():
        if not path.is_file() or path.stat().st_size < 10_000:
            raise AssertionError(f"Missing/suspicious figure: {path}")
    if not REPORT_PATH.is_file() or REPORT_PATH.stat().st_size < 2000:
        raise AssertionError("Report missing or unexpectedly short")


def main() -> None:
    started = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pair, tables, input_hashes = load_and_validate_inputs()
    print("Stage 07 headline reproduction: PASS", flush=True)

    within_target, hits = within_target_analysis(pair)
    within_source = within_source_analysis(pair)
    centered = centered_correlations(pair)
    fixed, fixed_cache = fixed_effect_models(pair)

    macro_qap, macro_qap_null = qap_association(
        pair,
        "support_distance_mean",
        "macro_f1_gap_mean",
        "positive",
        "Overall distance vs Macro gap",
    )
    expansion_qap, _ = qap_association(
        pair,
        "class2_support_distance_mean",
        "expansion_f1_gap_mean",
        "positive",
        "Class2 distance vs Expansion gap",
    )
    performance_qap, _ = qap_association(
        pair,
        "support_distance_mean",
        "cross_macro_f1_mean",
        "negative",
        "Overall distance vs cross-city Macro-F1",
    )
    macro_beta_qap, _ = qap_fixed_effect_beta(
        pair,
        "macro_f1_gap_mean",
        "support_distance_mean",
        "Macro fixed-effect distance beta",
    )
    expansion_beta_qap, _ = qap_fixed_effect_beta(
        pair,
        "expansion_f1_gap_mean",
        "class2_support_distance_mean",
        "Expansion fixed-effect distance beta",
    )
    qap = pd.DataFrame(
        [macro_qap, expansion_qap, performance_qap, macro_beta_qap, expansion_beta_qap]
    )
    loco = leave_one_city_out(pair)
    asymmetry, asymmetry_summary = pair_asymmetry(pair)
    diversity, diversity_correlations = representation_diversity(tables[SOURCE_PATH.stem])

    macro_global = robust_global_judgment(macro_qap, loco, "macro")
    expansion_global = robust_global_judgment(expansion_qap, loco, "expansion")
    macro_selection = source_selection_judgment(within_target, hits, "macro")
    expansion_selection = source_selection_judgment(within_target, hits, "expansion")
    scenario = "C" if macro_selection == "SUPPORTED" else "A"
    recommendation = (
        "Prioritize preregistered leave-one-city-out validation of target-difficulty "
        "calibration and a multi-source pooled baseline. Do not use minimum support "
        "distance as a standalone single-source selector until it is validated on more cities."
        if scenario == "A"
        else
        "Preregister and compare distance-guided source selection with a multi-source pooled "
        "baseline under leave-one-city-out validation before any adaptation experiment."
    )

    save_csv(within_target, WITHIN_TARGET_PATH)
    save_csv(hits, SELECTION_PATH)
    save_csv(within_source, WITHIN_SOURCE_PATH)
    save_csv(centered, CENTERED_PATH)
    save_csv(fixed, FIXED_PATH)
    save_csv(qap, QAP_PATH)
    save_csv(loco, LOCO_PATH)
    save_csv(asymmetry, ASYMMETRY_PATH)
    save_csv(diversity, DIVERSITY_PATH)

    set_plot_style()
    plot_within_target(pair)
    plot_loco(loco)
    plot_qap(
        macro_qap_null,
        macro_qap["observed_statistic"],
        macro_qap["exact_qap_one_sided_p"],
    )
    plot_asymmetry(asymmetry)
    macro_beta = fixed.loc[
        (fixed["outcome"] == "Macro-F1 gap") & (fixed["model"] == "Model STD"),
        "support_distance_beta",
    ].iloc[0]
    plot_fixed_effect(pair, float(macro_beta))
    plot_diversity(diversity)

    elapsed = time.perf_counter() - started
    write_report(
        pair,
        input_hashes,
        within_target,
        hits,
        within_source,
        centered,
        fixed,
        qap,
        loco,
        asymmetry,
        asymmetry_summary,
        diversity,
        diversity_correlations,
        macro_global,
        expansion_global,
        macro_selection,
        expansion_selection,
        scenario,
        recommendation,
        elapsed,
    )
    validate_outputs()
    for relative_path, original_hash in input_hashes.items():
        if file_sha256(PROJECT_ROOT / relative_path) != original_hash:
            raise RuntimeError(f"Frozen Stage 07 input changed: {relative_path}")
    print(f"Stage 07B complete in {elapsed:.2f}s", flush=True)
    print(
        f"Macro global={macro_global}; source selection={macro_selection}; scenario={scenario}",
        flush=True,
    )
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

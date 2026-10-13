"""Stage 03B: domain-shift robustness and target-normalized transfer audit.

This script intentionally does not rerun the full Stage 03 baseline. It reuses
the Stage 03 feature, split, model, fitting, and metric helpers, then fits only
the models required for the robustness checks requested in Stage 03B.
"""

from __future__ import annotations

import gc
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "02_baseline_experiment.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

YEARLY_PATH = TABLES_DIR / "domain_yearly_audit.csv"
PER_CLASS_PATH = TABLES_DIR / "domain_per_class_audit.csv"
TRANSFER_PATH = TABLES_DIR / "target_normalized_transfer_gap.csv"
REPORT_PATH = LOGS_DIR / "domain_shift_audit_report.txt"
PCA_CITY_PATH = FIGURES_DIR / "07_domain_PCA_by_city.png"
PCA_CITY_CLASS_PATH = FIGURES_DIR / "08_domain_PCA_by_city_and_class.png"

CITY_FILES = {
    "Shenzhen": PROCESSED_DIR / "Shenzhen_AEF_195D.parquet",
    "Kunming": PROCESSED_DIR / "Kunming_AEF_195D.parquet",
}
CITY_LABELS = {"Shenzhen": 0, "Kunming": 1}
CLASS_LABELS = {
    0: "Stable Non-built",
    1: "Stable Built",
    2: "Strict Expansion",
}
FORBIDDEN_DOMAIN_INPUTS = {
    "city",
    "longitude",
    "latitude",
    "utm_x",
    "utm_y",
    "block_x",
    "block_y",
    "block_id",
    "sample_id",
    "class",
}
PCA_SAMPLES_PER_CITY_CLASS = 1000


def load_stage03_module() -> Any:
    """Load the prior-stage script as a module without running its main()."""
    spec = importlib.util.spec_from_file_location("stage03_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage 03 helpers from {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_group_summary(
    runs: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Attach five-seed mean/std columns to every auditable run row."""
    grouped = runs.groupby(group_columns, as_index=False)[metrics].agg(["mean", "std"])
    grouped.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return runs.merge(grouped, on=group_columns, how="left", validate="many_to_one")


def assert_domain_features(feature_sets: dict[str, list[str]]) -> None:
    for name, columns in feature_sets.items():
        if len(columns) != len(set(columns)):
            raise AssertionError(f"{name}: duplicate feature names")
        leaked = sorted(FORBIDDEN_DOMAIN_INPUTS.intersection(columns))
        if leaked:
            raise AssertionError(f"{name}: forbidden metadata inputs present: {leaked}")


def balanced_domain_frame(
    stage03: Any,
    datasets: dict[str, pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    balanced = stage03.balance_domain_data(datasets, seed).copy()
    expected = {
        (city, class_value): 6000
        for city in ["Shenzhen", "Kunming"]
        for class_value in [0, 1, 2]
    }
    observed = balanced.groupby(["city", "class"]).size().to_dict()
    if observed != expected:
        raise AssertionError(f"Unexpected city x class balance: {observed}")
    mapped = balanced["city"].map(CITY_LABELS).to_numpy(dtype=np.int64)
    if not np.array_equal(mapped, balanced["domain_label"].to_numpy(dtype=np.int64)):
        raise AssertionError("Domain labels do not match Shenzhen=0, Kunming=1")
    return balanced


def run_domain_audits(
    stage03: Any,
    datasets: dict[str, pd.DataFrame],
    aef_features: dict[str, list[str]],
    stage_config: dict[str, Any],
    warning_messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [int(seed) for seed in stage_config["random_seeds"]]
    balanced = balanced_domain_frame(stage03, datasets, seeds[0])
    year_features = {
        f"AEF-{year}": [f"A{index:02d}_{year}" for index in range(64)]
        for year in [2020, 2022, 2024]
    }
    yearly_features = {**year_features, "Temporal-only": ["D_20_22", "D_22_24", "D_20_24"]}
    assert_domain_features(yearly_features)
    assert all(column in aef_features["AEF-195"] for columns in yearly_features.values() for column in columns)

    y = balanced["domain_label"].to_numpy(dtype=np.int64)
    combined_strata = balanced["domain_label"].astype(int) * 10 + balanced["class"].astype(int)
    yearly_rows: list[dict[str, Any]] = []

    for seed in seeds:
        split = stage03.optimized_group_stratified_split(
            balanced,
            combined_strata,
            seed + 100000,
            stage_config["split"],
            "Stage 03B yearly domain classifier",
            warning_messages,
        )
        for feature_set, columns in yearly_features.items():
            matrix = balanced[columns].to_numpy(dtype=np.float32, copy=False)
            context = f"Stage 03B domain {feature_set} seed={seed}"
            model = stage03.make_model("Logistic Regression", "domain", seed, stage_config)
            model, training_seconds, iterations = stage03.fit_model(
                model,
                "Logistic Regression",
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
                "audit_scope": "temporal-only" if feature_set == "Temporal-only" else "single-year",
                "feature_set": feature_set,
                "year": "temporal-only" if feature_set == "Temporal-only" else int(feature_set.split("-")[1]),
                "seed": seed,
                "model": "Logistic Regression",
                "feature_count": len(columns),
                "train_samples": len(split.train),
                "validation_samples": len(split.validation),
                "test_samples": len(split.test),
                "train_blocks": split.diagnostics["train_blocks"],
                "validation_blocks": split.diagnostics["validation_blocks"],
                "test_blocks": split.diagnostics["test_blocks"],
                "training_seconds": training_seconds,
                "fitted_iterations": iterations,
                "scaler_fit_scope": "training set only",
            }
            row.update(stage03.domain_metrics(y[split.test], prediction, probability))
            yearly_rows.append(row)
            print(
                f"DOMAIN {feature_set} seed={seed}: "
                f"AUC={row['roc_auc']:.6f}, ACC={row['accuracy']:.6f}"
            )
            del model, matrix
            gc.collect()

    yearly_runs = pd.DataFrame(yearly_rows)
    yearly_output = add_group_summary(
        yearly_runs,
        ["audit_scope", "feature_set", "year", "model"],
        ["roc_auc", "accuracy", "balanced_accuracy"],
    ).sort_values(["audit_scope", "feature_set", "seed"]).reset_index(drop=True)

    assert_domain_features({"AEF-192": aef_features["AEF-192"]})
    class_rows: list[dict[str, Any]] = []
    for class_value, class_name in CLASS_LABELS.items():
        subset = balanced.loc[balanced["class"] == class_value].reset_index(drop=True)
        y_class = subset["domain_label"].to_numpy(dtype=np.int64)
        matrix = subset[aef_features["AEF-192"]].to_numpy(dtype=np.float32, copy=False)
        for seed in seeds:
            split = stage03.optimized_group_stratified_split(
                subset,
                subset["domain_label"].astype(int),
                seed + 200000 + class_value * 1000,
                stage_config["split"],
                f"Stage 03B class {class_value} domain classifier",
                warning_messages,
            )
            context = f"Stage 03B class {class_value} domain seed={seed}"
            model = stage03.make_model("Logistic Regression", "domain", seed, stage_config)
            model, training_seconds, iterations = stage03.fit_model(
                model,
                "Logistic Regression",
                matrix[split.train],
                y_class[split.train],
                matrix[split.validation],
                y_class[split.validation],
                warning_messages,
                context,
            )
            prediction = model.predict(matrix[split.test]).astype(int)
            probability = model.predict_proba(matrix[split.test])[:, 1]
            row = {
                "class": class_value,
                "class_name": class_name,
                "seed": seed,
                "model": "Logistic Regression",
                "feature_set": "AEF-192",
                "feature_count": 192,
                "train_samples": len(split.train),
                "validation_samples": len(split.validation),
                "test_samples": len(split.test),
                "train_blocks": split.diagnostics["train_blocks"],
                "validation_blocks": split.diagnostics["validation_blocks"],
                "test_blocks": split.diagnostics["test_blocks"],
                "training_seconds": training_seconds,
                "fitted_iterations": iterations,
                "scaler_fit_scope": "training set only",
            }
            row.update(stage03.domain_metrics(y_class[split.test], prediction, probability))
            class_rows.append(row)
            print(
                f"DOMAIN class={class_value} seed={seed}: "
                f"AUC={row['roc_auc']:.6f}, ACC={row['accuracy']:.6f}"
            )
            del model
            gc.collect()
        del matrix

    class_runs = pd.DataFrame(class_rows)
    class_output = add_group_summary(
        class_runs,
        ["class", "class_name", "model", "feature_set"],
        ["roc_auc", "accuracy"],
    ).sort_values(["class", "seed"]).reset_index(drop=True)
    return yearly_output, class_output


def run_target_normalized_transfer(
    stage03: Any,
    datasets: dict[str, pd.DataFrame],
    aef192: list[str],
    stage_config: dict[str, Any],
    warning_messages: list[str],
) -> pd.DataFrame:
    seeds = [int(seed) for seed in stage_config["random_seeds"]]
    model_names = ["Logistic Regression", "XGBoost"]
    rows: list[dict[str, Any]] = []
    assert_domain_features({"AEF-192": aef192})

    for seed in seeds:
        splits: dict[str, Any] = {}
        matrices: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for city, frame in datasets.items():
            split = stage03.optimized_group_stratified_split(
                frame,
                frame["class"].astype(int),
                seed,
                stage_config["split"],
                f"Stage 03B transfer {city}",
                warning_messages,
            )
            splits[city] = split
            matrices[city] = frame[aef192].to_numpy(dtype=np.float32, copy=False)
            labels[city] = frame["class"].to_numpy(dtype=np.int64)

        for model_name in model_names:
            fitted: dict[str, Any] = {}
            training_info: dict[str, tuple[float, int | None]] = {}
            for city in ["Shenzhen", "Kunming"]:
                split = splits[city]
                model = stage03.make_model(model_name, "land_cover", seed, stage_config)
                model, training_seconds, iterations = stage03.fit_model(
                    model,
                    model_name,
                    matrices[city][split.train],
                    labels[city][split.train],
                    matrices[city][split.validation],
                    labels[city][split.validation],
                    warning_messages,
                    f"Stage 03B transfer {city} {model_name} seed={seed}",
                )
                fitted[city] = model
                training_info[city] = (training_seconds, iterations)

            for source_city, target_city in [
                ("Shenzhen", "Kunming"),
                ("Kunming", "Shenzhen"),
            ]:
                target_split = splits[target_city]
                target_test_index = target_split.test
                target_x = matrices[target_city][target_test_index]
                target_y = labels[target_city][target_test_index]

                same_prediction = fitted[target_city].predict(target_x).astype(int)
                cross_prediction = fitted[source_city].predict(target_x).astype(int)
                same_metrics = stage03.land_cover_metrics(target_y, same_prediction)
                cross_metrics = stage03.land_cover_metrics(target_y, cross_prediction)

                test_block_ids = datasets[target_city].iloc[target_test_index]["block_id"]
                if test_block_ids.nunique() != target_split.diagnostics["test_blocks"]:
                    raise AssertionError("Target test block count mismatch")

                rows.append(
                    {
                        "direction": f"{source_city} -> {target_city}",
                        "source_city": source_city,
                        "target_city": target_city,
                        "seed": seed,
                        "model": model_name,
                        "feature_set": "AEF-192",
                        "target_test_samples": len(target_test_index),
                        "target_test_blocks": int(test_block_ids.nunique()),
                        "target_test_index_sha256": stage03.hashlib.sha256(
                            np.asarray(target_test_index, dtype=np.int64).tobytes()
                        ).hexdigest(),
                        "shared_target_test_set_assertion": True,
                        "target_same_city_macro_f1": same_metrics["macro_f1"],
                        "source_on_target_macro_f1": cross_metrics["macro_f1"],
                        "macro_f1_gap": same_metrics["macro_f1"] - cross_metrics["macro_f1"],
                        "target_same_city_expansion_f1": same_metrics["expansion_f1"],
                        "source_on_target_expansion_f1": cross_metrics["expansion_f1"],
                        "expansion_f1_gap": same_metrics["expansion_f1"] - cross_metrics["expansion_f1"],
                        "source_training_seconds": training_info[source_city][0],
                        "target_training_seconds": training_info[target_city][0],
                        "source_fitted_iterations": training_info[source_city][1],
                        "target_fitted_iterations": training_info[target_city][1],
                    }
                )
                latest = rows[-1]
                print(
                    f"TRANSFER {latest['direction']} {model_name} seed={seed}: "
                    f"Macro gap={latest['macro_f1_gap']:.6f}, "
                    f"Expansion gap={latest['expansion_f1_gap']:.6f}"
                )

            fitted.clear()
            gc.collect()

    runs = pd.DataFrame(rows)
    output = add_group_summary(
        runs,
        ["direction", "source_city", "target_city", "model", "feature_set"],
        ["macro_f1_gap", "expansion_f1_gap"],
    ).sort_values(["direction", "model", "seed"]).reset_index(drop=True)
    if not output["shared_target_test_set_assertion"].all():
        raise AssertionError("Same-city and cross-city evaluations did not share target test sets")
    return output


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


def create_pca_figures(
    datasets: dict[str, pd.DataFrame],
    aef192: list[str],
    seed: int,
) -> tuple[float, float, int]:
    sampled_parts: list[pd.DataFrame] = []
    for city_index, city in enumerate(["Shenzhen", "Kunming"]):
        for class_value in [0, 1, 2]:
            subset = datasets[city].loc[datasets[city]["class"] == class_value]
            sample_count = min(PCA_SAMPLES_PER_CITY_CLASS, len(subset))
            sampled_parts.append(
                subset.sample(
                    n=sample_count,
                    replace=False,
                    random_state=seed + city_index * 100 + class_value,
                )
            )
    sampled = pd.concat(sampled_parts, ignore_index=True)
    check = sampled.groupby(["city", "class"]).size()
    if check.nunique() != 1:
        raise AssertionError(f"PCA sample is not city x class balanced: {check.to_dict()}")

    matrix = sampled[aef192].to_numpy(dtype=np.float64, copy=False)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(scaled)
    sampled = sampled[["city", "class"]].copy()
    sampled["PC1"] = coordinates[:, 0]
    sampled["PC2"] = coordinates[:, 1]
    variance = pca.explained_variance_ratio_

    set_plot_style()
    colors = {"Shenzhen": "#2878B5", "Kunming": "#D9534F"}
    markers = {0: "o", 1: "^", 2: "s"}

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    for city in ["Shenzhen", "Kunming"]:
        mask = sampled["city"] == city
        ax.scatter(
            sampled.loc[mask, "PC1"],
            sampled.loc[mask, "PC2"],
            s=10,
            alpha=0.35,
            c=colors[city],
            linewidths=0,
            label=city,
        )
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.set_title("AEF-192 PCA by city")
    ax.legend(frameon=False, markerscale=2.0)
    fig.tight_layout()
    fig.savefig(PCA_CITY_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    for city in ["Shenzhen", "Kunming"]:
        for class_value, class_name in CLASS_LABELS.items():
            mask = (sampled["city"] == city) & (sampled["class"] == class_value)
            ax.scatter(
                sampled.loc[mask, "PC1"],
                sampled.loc[mask, "PC2"],
                s=13,
                alpha=0.32,
                c=colors[city],
                marker=markers[class_value],
                linewidths=0,
                label=f"{city}, Class {class_value} {class_name}",
            )
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.set_title("AEF-192 PCA by city and land-cover class")
    ax.legend(frameon=False, fontsize=8, markerscale=1.7, ncol=2)
    fig.tight_layout()
    fig.savefig(PCA_CITY_CLASS_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(variance[0]), float(variance[1]), len(sampled)


def metric_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    result = frame.groupby(group_columns, as_index=False)[metrics].agg(["mean", "std"])
    result.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in result.columns
    ]
    return result


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.6f} +/- {std:.6f}"


def build_report(
    config: dict[str, Any],
    stage03: Any,
    yearly: pd.DataFrame,
    per_class: pd.DataFrame,
    transfer: pd.DataFrame,
    pca_info: tuple[float, float, int],
    warning_messages: list[str],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    elapsed_seconds: float,
) -> str:
    yearly_summary = metric_summary(
        yearly.drop_duplicates(["feature_set", "seed"]),
        ["audit_scope", "feature_set", "year"],
        ["roc_auc", "accuracy", "balanced_accuracy"],
    )
    class_summary = metric_summary(
        per_class.drop_duplicates(["class", "seed"]),
        ["class", "class_name"],
        ["roc_auc", "accuracy"],
    )
    transfer_summary = metric_summary(
        transfer.drop_duplicates(["direction", "model", "seed"]),
        ["direction", "model", "feature_set"],
        ["macro_f1_gap", "expansion_f1_gap"],
    )
    single_year = yearly_summary.loc[yearly_summary["audit_scope"] == "single-year"]
    class_auc = class_summary["roc_auc_mean"]
    robust = bool(
        np.allclose(single_year["roc_auc_mean"].to_numpy(), 1.0, atol=1e-12)
        and np.allclose(class_auc.to_numpy(), 1.0, atol=1e-12)
    )
    unchanged = hashes_before == hashes_after
    pc1, pc2, pca_samples = pca_info

    lines = [
        "Stage 03B - Domain shift robustness audit",
        "=" * 78,
        f"Completed: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "Scope: focused Stage 03B models only; full Stage 03 baseline not rerun.",
        "Few-shot adaptation: not run.",
        "",
        "1. Reproducibility and safeguards",
        "-" * 78,
        f"Random seeds: {json.dumps(config['stage_03']['random_seeds'])}",
        "Spatial split: Stage 03 optimized group-aware 70/15/15 block split.",
        "Domain labels: Shenzhen=0, Kunming=1.",
        "Logistic Regression StandardScaler fit scope: training set only.",
        "Forbidden metadata excluded from every domain classifier: "
        + ", ".join(sorted(FORBIDDEN_DOMAIN_INPUTS)),
        "Metadata leakage assertion: PASSED; no direct metadata input leakage found.",
        "Target-normalized assertion: PASSED; same-city and cross-city predictions were "
        "evaluated on identical target test indices and blocks for each seed.",
        f"Processed input hashes unchanged: {unchanged}.",
        "",
        "2. Single-year domain classifier (Logistic Regression)",
        "-" * 78,
    ]
    for _, row in single_year.sort_values("year").iterrows():
        lines.append(
            f"{row['feature_set']}: ROC-AUC "
            f"{format_mean_std(row['roc_auc_mean'], row['roc_auc_std'])}; "
            f"Accuracy {format_mean_std(row['accuracy_mean'], row['accuracy_std'])}; "
            f"Balanced Accuracy "
            f"{format_mean_std(row['balanced_accuracy_mean'], row['balanced_accuracy_std'])}."
        )

    lines.extend(["", "3. Per-class domain classifier (AEF-192)", "-" * 78])
    for _, row in class_summary.sort_values("class").iterrows():
        lines.append(
            f"Class {int(row['class'])} {row['class_name']}: ROC-AUC "
            f"{format_mean_std(row['roc_auc_mean'], row['roc_auc_std'])}; "
            f"Accuracy {format_mean_std(row['accuracy_mean'], row['accuracy_std'])}."
        )

    temporal = yearly_summary.loc[yearly_summary["audit_scope"] == "temporal-only"].iloc[0]
    lines.extend(
        [
            "",
            "4. Temporal-only domain classifier",
            "-" * 78,
            "Features: D_20_22, D_22_24, D_20_24.",
            f"ROC-AUC {format_mean_std(temporal['roc_auc_mean'], temporal['roc_auc_std'])}; "
            f"Accuracy {format_mean_std(temporal['accuracy_mean'], temporal['accuracy_std'])}; "
            f"Balanced Accuracy "
            f"{format_mean_std(temporal['balanced_accuracy_mean'], temporal['balanced_accuracy_std'])}.",
            "",
            "5. Target-normalized transfer gaps (AEF-192)",
            "-" * 78,
            "Gap = target-city same-city held-out performance minus source-trained "
            "performance on the same target held-out test blocks.",
        ]
    )
    for _, row in transfer_summary.sort_values(["direction", "model"]).iterrows():
        lines.append(
            f"{row['direction']}, {row['model']}: Macro-F1 gap "
            f"{format_mean_std(row['macro_f1_gap_mean'], row['macro_f1_gap_std'])}; "
            f"Expansion-F1 gap "
            f"{format_mean_std(row['expansion_f1_gap_mean'], row['expansion_f1_gap_std'])}."
        )

    robustness_text = (
        "YES. Exact mean ROC-AUC=1.0 persisted for every single-year and every "
        "within-class AEF-192 audit. This rules out dependence on one acquisition "
        "year or unequal land-cover class composition as the sole explanation."
        if robust
        else
        "NO, not as an exact universal result. At least one single-year or within-class "
        "audit fell below exact mean ROC-AUC=1.0; the city signal remains strong only "
        "to the degree shown by the reported metrics."
    )
    lines.extend(
        [
            "",
            "6. PCA visualization",
            "-" * 78,
            f"Balanced sample: {pca_samples} rows "
            f"({PCA_SAMPLES_PER_CITY_CLASS} per city x class cell).",
            f"Explained variance: PC1={pc1:.6f}, PC2={pc2:.6f}, total={pc1 + pc2:.6f}.",
            "PCA was used only for visualization and never as a classifier input.",
            "",
            "7. Robustness judgment",
            "-" * 78,
            robustness_text,
            "No direct metadata leakage was found: city, coordinates, UTM fields, "
            "block identifiers, sample identifiers, and class labels were excluded "
            "from classifier inputs.",
            "",
            "8. Warnings and interpretation",
            "-" * 78,
            "- A persistent perfect city AUC is evidence of a strong domain signature "
            "inside the embeddings, not proof of causal environmental differences.",
            "- The domain signature may include geographic, acquisition, atmospheric, "
            "seasonal, preprocessing, or sensor-context effects that are not represented "
            "by the explicit metadata columns.",
            "- Group-aware blocks reduce local spatial leakage but do not guarantee "
            "independence across all broader spatial or acquisition processes.",
            "- Positive and negative target-normalized gaps should be interpreted by "
            "direction and model; a negative gap means the source-trained model exceeded "
            "the target-trained model on that exact target test set.",
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
            "9. Outputs",
            "-" * 78,
            str(YEARLY_PATH),
            str(PER_CLASS_PATH),
            str(TRANSFER_PATH),
            str(PCA_CITY_PATH),
            str(PCA_CITY_CLASS_PATH),
            str(REPORT_PATH),
            "",
            "10. Input hashes",
            "-" * 78,
        ]
    )
    for city in ["Shenzhen", "Kunming"]:
        lines.append(f"{city}: {hashes_after[city]}")
    lines.append("")
    return "\n".join(lines)


def validate_outputs(
    yearly: pd.DataFrame,
    per_class: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:
    if len(yearly) != 20:
        raise AssertionError(f"Expected 20 yearly/temporal runs, got {len(yearly)}")
    if len(per_class) != 15:
        raise AssertionError(f"Expected 15 per-class runs, got {len(per_class)}")
    if len(transfer) != 20:
        raise AssertionError(f"Expected 20 transfer runs, got {len(transfer)}")
    for frame, metrics in [
        (yearly, ["roc_auc", "accuracy", "balanced_accuracy"]),
        (per_class, ["roc_auc", "accuracy"]),
    ]:
        values = frame[metrics].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise AssertionError("Invalid domain metric values")
    gap_values = transfer[["macro_f1_gap", "expansion_f1_gap"]].to_numpy(dtype=float)
    if not np.isfinite(gap_values).all() or np.any(gap_values < -1) or np.any(gap_values > 1):
        raise AssertionError("Invalid transfer gap values")
    if yearly.groupby("feature_set")["seed"].nunique().to_dict() != {
        "AEF-2020": 5,
        "AEF-2022": 5,
        "AEF-2024": 5,
        "Temporal-only": 5,
    }:
        raise AssertionError("Yearly audit does not contain all five seeds")
    if per_class.groupby("class")["seed"].nunique().to_dict() != {0: 5, 1: 5, 2: 5}:
        raise AssertionError("Per-class audit does not contain all five seeds")
    if not transfer["shared_target_test_set_assertion"].all():
        raise AssertionError("Target test set equality assertion failed")


def main() -> int:
    start = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    warning_messages: list[str] = []

    stage03 = load_stage03_module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stage_config = config["stage_03"]
    features = stage03.feature_definitions(config)
    hashes_before = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    datasets = {
        city: stage03.validate_dataset(path, city, features)
        for city, path in CITY_FILES.items()
    }

    yearly, per_class = run_domain_audits(
        stage03, datasets, features, stage_config, warning_messages
    )
    transfer = run_target_normalized_transfer(
        stage03, datasets, features["AEF-192"], stage_config, warning_messages
    )
    pca_info = create_pca_figures(
        datasets, features["AEF-192"], int(stage_config["random_seeds"][0])
    )

    validate_outputs(yearly, per_class, transfer)
    yearly.to_csv(YEARLY_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    per_class.to_csv(PER_CLASS_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")
    transfer.to_csv(TRANSFER_PATH, index=False, encoding="utf-8-sig", float_format="%.10f")

    hashes_after = {city: stage03.sha256(path) for city, path in CITY_FILES.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Processed input hash changed during Stage 03B")
    elapsed_seconds = time.perf_counter() - start
    report = build_report(
        config,
        stage03,
        yearly,
        per_class,
        transfer,
        pca_info,
        warning_messages,
        hashes_before,
        hashes_after,
        elapsed_seconds,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    for path in [YEARLY_PATH, PER_CLASS_PATH, TRANSFER_PATH, REPORT_PATH, PCA_CITY_PATH, PCA_CITY_CLASS_PATH]:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    pd.read_csv(YEARLY_PATH)
    pd.read_csv(PER_CLASS_PATH)
    pd.read_csv(TRANSFER_PATH)
    print(f"Stage 03B complete in {elapsed_seconds:.2f}s")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

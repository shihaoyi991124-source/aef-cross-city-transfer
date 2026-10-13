"""Stage 09B Phase A: label-blind external prediction freezing.

This process never reads external target labels.  It verifies the frozen inputs,
creates label-blind external spatial splits, reproduces the frozen Stage 07
source models, freezes all external predictions and risk estimates, and writes
a cryptographic manifest for the physically separate Phase B process.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "FiveCity_AEF_195D.parquet"
EXTERNAL = {
    "Chengdu": ROOT / "data" / "processed" / "external" / "Chengdu_External_AEF_195D.parquet",
    "Hangzhou": ROOT / "data" / "processed" / "external" / "Hangzhou_External_AEF_195D.parquet",
}
CONFIG = ROOT / "config.json"
BASELINE = ROOT / "scripts" / "02_baseline_experiment.py"
TABLES = ROOT / "outputs" / "tables"
FROZEN = ROOT / "outputs" / "frozen_external"

SPLITS_PATH = TABLES / "stage09b_external_block_splits.csv"
CALIBRATORS_PATH = TABLES / "stage09b_frozen_risk_calibrators.csv"
SAMPLE_PREDICTIONS_PATH = FROZEN / "stage09b_external_sample_predictions.parquet"
PAIR_PREDICTIONS_PATH = FROZEN / "stage09b_external_pair_predictions.csv"
MANIFEST_PATH = FROZEN / "stage09b_phaseA_manifest.txt"

STAGE07_SPLITS = TABLES / "stage07_spatial_splits.csv"
STAGE07_SAME = TABLES / "stage07_same_city_runs.csv"
STAGE07_CROSS = TABLES / "stage07_cross_city_runs.csv"
STAGE08_PAIRS = TABLES / "stage08_label_free_pair_features.csv"

CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
TARGETS = ["Chengdu", "Hangzhou"]
ABBR = {
    "Shenzhen": "SZ", "Kunming": "KM", "Wuhan": "WH", "Xian": "XA",
    "Beijing": "BJ", "Chengdu": "CD", "Hangzhou": "HZ",
}
SEEDS = [20261008, 20261009, 20261010, 20261011, 20261012]
FEATURES = [f"A{i:02d}_{year}" for year in (2020, 2022, 2024) for i in range(64)]
EXPECTED_HASHES = {
    "data/processed/external/Chengdu_External_AEF_195D.parquet": "1B286B9FA32E59FDC7470B09FA3A53B5D7B550C971AC7D92FDCC971C4C0286A1",
    "data/processed/external/Hangzhou_External_AEF_195D.parquet": "A66C5930FD227DD9E048C7F1CF0D3DD1E8546F36D6E79572E06190AA23DC9D70",
    "data/processed/FiveCity_AEF_195D.parquet": "C68147608896E9CF0A522625496872E7540AFBE959FF665E8EFBD08A7A7D9FAE",
}
METRICS = [
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "class0_f1", "class1_f1", "expansion_f1",
]
K = 5
TOLERANCE = 1e-6


@dataclass
class DevelopmentSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def load_stage03() -> Any:
    spec = importlib.util.spec_from_file_location("stage03_baseline_for_stage09b_a", BASELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Stage 03 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def label_blind_block_split(city: str, block_ids: list[str], seed: int) -> dict[str, str]:
    """Assign blocks using only city, block IDs, and seed; no class input exists."""
    if city not in TARGETS or seed not in SEEDS:
        raise AssertionError("Unexpected external split request")
    unique = sorted(set(str(value) for value in block_ids))
    ordered = sorted(
        unique,
        key=lambda block: (hashlib.sha256(f"{seed}|{block}".encode("utf-8")).hexdigest(), block),
    )
    n_train = round(0.70 * len(ordered))
    n_validation = round(0.15 * len(ordered))
    cuts = (n_train, n_train + n_validation)
    assignment: dict[str, str] = {}
    for position, block in enumerate(ordered):
        assignment[block] = "train" if position < cuts[0] else "validation" if position < cuts[1] else "test"
    groups = {name: {b for b, value in assignment.items() if value == name} for name in ("train", "validation", "test")}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise AssertionError(f"{city} seed={seed}: overlapping external blocks")
    if set().union(*groups.values()) != set(unique):
        raise AssertionError(f"{city} seed={seed}: incomplete external block coverage")
    return assignment


def load_external_label_blind() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    split_rows: list[dict[str, Any]] = []
    columns = ["sample_id", "city", "block_id"] + FEATURES
    forbidden = {"class", "D_20_22", "D_22_24", "D_20_24", "external_target", "longitude", "latitude"}
    for city, path in EXTERNAL.items():
        frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
        if forbidden.intersection(frame.columns):
            raise AssertionError(f"{city}: forbidden external columns entered Phase A")
        if list(frame.columns) != columns or len(frame) != 18000 or frame["sample_id"].duplicated().any():
            raise AssertionError(f"{city}: invalid label-blind target object")
        if set(frame["city"].astype(str)) != {city} or frame["block_id"].isna().any():
            raise AssertionError(f"{city}: invalid city/block metadata")
        matrix = frame[FEATURES].to_numpy(np.float32, copy=False)
        if matrix.shape != (18000, 192) or not np.isfinite(matrix).all():
            raise AssertionError(f"{city}: invalid AEF-192 matrix")
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["block_id"] = frame["block_id"].astype(str)
        frames[city] = frame
        blocks = frame["block_id"].unique().tolist()
        for seed in SEEDS:
            mapping = label_blind_block_split(city, blocks, seed)
            split_rows.extend(
                {"city": city, "seed": seed, "block_id": block, "split": split}
                for block, split in mapping.items()
            )
    split_table = pd.DataFrame(split_rows).sort_values(["city", "seed", "split", "block_id"]).reset_index(drop=True)
    if list(split_table.columns) != ["city", "seed", "block_id", "split"]:
        raise AssertionError("Phase A split table contains unexpected fields")
    return frames, split_table


def load_development() -> tuple[
    dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, np.ndarray],
    dict[tuple[str, int], DevelopmentSplit], dict[str, Any], pd.DataFrame, pd.DataFrame,
]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("selected_multicity_cities") != CITIES:
        raise AssertionError("Frozen development city order changed")
    stage_config = config["stage_03"]
    if [int(value) for value in stage_config["random_seeds"]] != SEEDS:
        raise AssertionError("Frozen seeds changed")
    expected_xgb = {
        "n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
        "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0,
        "reg_lambda": 1.0, "early_stopping_rounds": 30, "tree_method": "hist", "n_jobs": 4,
    }
    if stage_config["models"]["XGBoost"] != expected_xgb:
        raise AssertionError("Frozen XGBoost parameters changed")

    full = pd.read_parquet(DATA, columns=["sample_id", "city", "class", "block_id"] + FEATURES, engine="pyarrow")
    frames = {city: full.loc[full.city == city].copy().reset_index(drop=True) for city in CITIES}
    matrices = {city: frames[city][FEATURES].to_numpy(np.float32, copy=False) for city in CITIES}
    labels = {city: frames[city]["class"].to_numpy(np.int64, copy=False) for city in CITIES}
    for city in CITIES:
        if matrices[city].shape != (18000, 192) or np.bincount(labels[city], minlength=3).tolist() != [6000, 6000, 6000]:
            raise AssertionError(f"{city}: frozen development invariant failed")
        frames[city]["block_id"] = frames[city]["block_id"].astype(str)

    table = pd.read_csv(STAGE07_SPLITS, dtype={"block_id": str})
    splits: dict[tuple[str, int], DevelopmentSplit] = {}
    for city in CITIES:
        values = frames[city]["block_id"]
        for seed in SEEDS:
            part = table.loc[(table.city == city) & (table.seed == seed)]
            arrays = {
                name: np.flatnonzero(values.isin(set(part.loc[part.split == name, "block_id"])).to_numpy())
                for name in ("train", "validation", "test")
            }
            union = np.concatenate(list(arrays.values()))
            if len(union) != 18000 or len(np.unique(union)) != 18000:
                raise AssertionError(f"{city} seed={seed}: development split mismatch")
            splits[(city, seed)] = DevelopmentSplit(**arrays)

    same = pd.read_csv(STAGE07_SAME)
    cross = pd.read_csv(STAGE07_CROSS)
    same = same.loc[(same.model == "XGBoost") & (same.feature_set == "AEF-192")].copy()
    cross = cross.loc[(cross.model == "XGBoost") & (cross.feature_set == "AEF-192")].copy()
    if len(same) != 25 or len(cross) != 100:
        raise AssertionError("Frozen Stage 07 XGBoost reference tables are incomplete")
    return frames, matrices, labels, splits, stage_config, same, cross


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise AssertionError("Zero/non-finite AEF vector norm")
    return (values / norms).astype(np.float32, copy=False)


def exact_mean_cosine_knn(reference: np.ndarray, query: np.ndarray, k: int = K, batch_size: int = 256) -> np.ndarray:
    """Exact Stage 08 batched cosine kNN for already L2-normalized rows."""
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1] or len(reference) < k:
        raise AssertionError("Invalid exact kNN inputs")
    result = np.empty(len(query), dtype=np.float64)
    reference_t = np.ascontiguousarray(reference.T)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        similarity = np.asarray(query[start:stop] @ reference_t, dtype=np.float32)
        top = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
        result[start:stop] = 1.0 - np.clip(top, -1.0, 1.0).mean(axis=1)
    return result


def label_free_risk_samples(
    x_target: np.ndarray,
    predicted_probabilities: np.ndarray,
    x_source_train: np.ndarray,
    y_source_train: np.ndarray,
) -> dict[str, Any]:
    """Compute Stage 08 risk quantities; intentionally has no y_target argument."""
    if x_target.shape[1] != 192 or predicted_probabilities.shape != (len(x_target), 3):
        raise AssertionError("Invalid label-free risk input")
    x_t = l2_normalize_rows(x_target)
    x_s = l2_normalize_rows(x_source_train)
    y_hat = np.argmax(predicted_probabilities, axis=1).astype(np.int64)
    global_distance = exact_mean_cosine_knn(x_s, x_t)
    pseudo_distance = np.empty(len(x_t), dtype=np.float64)
    for cls in (0, 1, 2):
        mask = y_hat == cls
        if mask.any():
            pseudo_distance[mask] = exact_mean_cosine_knn(x_s[y_source_train == cls], x_t[mask])
    probability = np.clip(np.asarray(predicted_probabilities, dtype=np.float64), 1e-15, 1.0)
    entropy = -(probability * np.log(probability)).sum(axis=1) / math.log(3.0)
    exp_mask = y_hat == 2
    return {
        "predicted_class": y_hat,
        "PCD_5NN_sample": pseudo_distance,
        "Global_5NN_sample": global_distance,
        "entropy": entropy,
        "PCD_5NN": float(pseudo_distance.mean()),
        "Global_5NN": float(global_distance.mean()),
        "PredExpansion_5NN": float(pseudo_distance[exp_mask].mean()) if exp_mask.any() else np.nan,
        "mean_entropy": float(entropy.mean()),
    }


def prediction_rows(
    target: pd.DataFrame,
    indices: np.ndarray,
    seed: int,
    model_name: str,
    probabilities: np.ndarray,
    risk: dict[str, Any] | None,
) -> pd.DataFrame:
    probability = np.asarray(probabilities, dtype=np.float64)
    entropy = -(np.clip(probability, 1e-15, 1.0) * np.log(np.clip(probability, 1e-15, 1.0))).sum(axis=1) / math.log(3.0)
    return pd.DataFrame({
        "sample_id": target.iloc[indices]["sample_id"].to_numpy(),
        "target_city": target.iloc[indices]["city"].to_numpy(),
        "seed": seed,
        "source_city_or_model": model_name,
        "prob_class0": probability[:, 0],
        "prob_class1": probability[:, 1],
        "prob_class2": probability[:, 2],
        "predicted_class": np.argmax(probability, axis=1).astype(np.int8),
        "PCD_5NN_sample": np.nan if risk is None else risk["PCD_5NN_sample"],
        "Global_5NN_sample": np.nan if risk is None else risk["Global_5NN_sample"],
        "entropy": entropy,
    })


def metric_error(current: dict[str, Any], reference: pd.Series) -> float:
    return max(abs(float(current[name]) - float(reference[name])) for name in METRICS)


def main() -> None:
    required = [DATA, CONFIG, BASELINE, STAGE07_SPLITS, STAGE07_SAME, STAGE07_CROSS, STAGE08_PAIRS, *EXTERNAL.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 09B Phase A inputs: {missing}")
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / name)
        print(f"HASH {name} {actual}", flush=True)
        if actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {name}: {actual} != {expected}")

    FROZEN.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    external_frames, split_table = load_external_label_blind()
    save_csv(split_table, SPLITS_PATH)
    stage03 = load_stage03()
    _, matrices, labels, dev_splits, stage_config, same_ref, cross_ref = load_development()

    development_pairs = pd.read_csv(STAGE08_PAIRS)
    if len(development_pairs) != 20 or development_pairs[["PCD_5NN", "mean_entropy", "actual_macro_gap"]].isna().any().any():
        raise AssertionError("Frozen Stage 08 development pair table is invalid")
    calibrator_rows: list[dict[str, Any]] = []
    calibrators: dict[str, LinearRegression] = {}
    for model_name, predictor, status in [
        ("R1", "PCD_5NN", "PREDEFINED GEOMETRY CANDIDATE"),
        ("R4", "mean_entropy", "POST-HOC DEVELOPMENT CANDIDATE"),
    ]:
        model = LinearRegression().fit(development_pairs[[predictor]], development_pairs["actual_macro_gap"])
        calibrators[model_name] = model
        calibrator_rows.append({
            "risk_model": model_name, "predictor": predictor, "outcome": "actual_macro_gap",
            "intercept": float(model.intercept_), "coefficient": float(model.coef_[0]),
            "training_n": len(development_pairs), "development_table_path": relative(STAGE08_PAIRS),
            "development_table_sha256": sha256(STAGE08_PAIRS), "candidate_status": status,
            "external_outcomes_used": False,
        })
    calibrators_table = pd.DataFrame(calibrator_rows)
    save_csv(calibrators_table, CALIBRATORS_PATH)

    warnings: list[str] = []
    frozen_sample_parts: list[pd.DataFrame] = []
    pair_seed_rows: list[dict[str, Any]] = []
    reproduction_max = {"same_city": 0.0, "cross_city": 0.0}

    for seed in SEEDS:
        models: dict[str, Any] = {}
        for source in CITIES:
            split = dev_splits[(source, seed)]
            model = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
            model, seconds, iterations = stage03.fit_model(
                model, "XGBoost", matrices[source][split.train], labels[source][split.train],
                matrices[source][split.validation], labels[source][split.validation], warnings,
                f"Stage 09B Phase A source={source} seed={seed}",
            )
            models[source] = model
            print(f"SOURCE_MODEL seed={seed} source={source} seconds={seconds:.1f} iterations={iterations}", flush=True)

        probability_cache: dict[tuple[str, str], np.ndarray] = {}
        target_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for target_city in TARGETS:
            target = external_frames[target_city]
            part = split_table.loc[(split_table.city == target_city) & (split_table.seed == seed)]
            test_blocks = set(part.loc[part.split == "test", "block_id"])
            target_indices = np.flatnonzero(target["block_id"].isin(test_blocks).to_numpy())
            x_target = target.iloc[target_indices][FEATURES].to_numpy(np.float32, copy=False)
            target_cache[target_city] = (target_indices, x_target)
            for source in CITIES:
                probability = models[source].predict_proba(x_target)
                probability_cache[(source, target_city)] = probability
                split = dev_splits[(source, seed)]
                risk = label_free_risk_samples(
                    x_target, probability, matrices[source][split.train], labels[source][split.train]
                )
                frozen_sample_parts.append(prediction_rows(target, target_indices, seed, source, probability, risk))
                pair_seed_rows.append({
                    "source_city": source, "source_abbreviation": ABBR[source],
                    "target_city": target_city, "target_abbreviation": ABBR[target_city],
                    "direction": f"{source} -> {target_city}", "seed": seed,
                    "target_test_samples": len(target_indices), "source_reference_samples": len(split.train),
                    "PCD_5NN": risk["PCD_5NN"], "Global_5NN": risk["Global_5NN"],
                    "PredExpansion_5NN": risk["PredExpansion_5NN"], "mean_entropy": risk["mean_entropy"],
                    "target_label_used_in_risk_features": False,
                })

            ensemble_probability = np.mean([probability_cache[(source, target_city)] for source in CITIES], axis=0)
            frozen_sample_parts.append(prediction_rows(target, target_indices, seed, "ENSEMBLE5", ensemble_probability, None))

        # The pooled model uses only frozen development train/validation blocks.
        x_train = np.concatenate([matrices[city][dev_splits[(city, seed)].train] for city in CITIES])
        y_train = np.concatenate([labels[city][dev_splits[(city, seed)].train] for city in CITIES])
        x_validation = np.concatenate([matrices[city][dev_splits[(city, seed)].validation] for city in CITIES])
        y_validation = np.concatenate([labels[city][dev_splits[(city, seed)].validation] for city in CITIES])
        pooled = stage03.make_model("XGBoost", "land_cover", seed, stage_config)
        pooled, seconds, iterations = stage03.fit_model(
            pooled, "XGBoost", x_train, y_train, x_validation, y_validation, warnings,
            f"Stage 09B Phase A POOLED5 seed={seed}",
        )
        for target_city in TARGETS:
            target_indices, x_target = target_cache[target_city]
            probability = pooled.predict_proba(x_target)
            frozen_sample_parts.append(prediction_rows(external_frames[target_city], target_indices, seed, "POOLED5", probability, None))
        print(f"POOLED_MODEL seed={seed} seconds={seconds:.1f} iterations={iterations}", flush=True)

        # All external source predictions/risk features for this seed are frozen in memory above.
        # Only development labels are used below to verify exact Stage 07 reproduction.
        for source in CITIES:
            source_test = dev_splits[(source, seed)].test
            current = stage03.land_cover_metrics(labels[source][source_test], models[source].predict(matrices[source][source_test]).astype(int))
            reference = same_ref.loc[(same_ref.source_city == source) & (same_ref.seed == seed)].iloc[0]
            reproduction_max["same_city"] = max(reproduction_max["same_city"], metric_error(current, reference))
            for target_city in CITIES:
                if target_city == source:
                    continue
                target_test = dev_splits[(target_city, seed)].test
                current = stage03.land_cover_metrics(labels[target_city][target_test], models[source].predict(matrices[target_city][target_test]).astype(int))
                reference = cross_ref.loc[
                    (cross_ref.source_city == source) & (cross_ref.target_city == target_city) & (cross_ref.seed == seed)
                ].iloc[0]
                reproduction_max["cross_city"] = max(reproduction_max["cross_city"], metric_error(current, reference))
        if max(reproduction_max.values()) > TOLERANCE:
            raise RuntimeError(f"Stage 07 reproduction exceeds tolerance: {reproduction_max}")
        del models, probability_cache, pooled, x_train, y_train, x_validation, y_validation, target_cache
        gc.collect()

    sample_predictions = pd.concat(frozen_sample_parts, ignore_index=True)
    required_columns = [
        "sample_id", "target_city", "seed", "source_city_or_model", "prob_class0", "prob_class1",
        "prob_class2", "predicted_class", "PCD_5NN_sample", "Global_5NN_sample", "entropy",
    ]
    if list(sample_predictions.columns) != required_columns or "class" in sample_predictions.columns:
        raise AssertionError("Frozen sample prediction schema violates Phase A protocol")
    if sample_predictions.duplicated(["sample_id", "target_city", "seed", "source_city_or_model"]).any():
        raise AssertionError("Duplicate frozen sample predictions")
    sample_predictions.to_parquet(SAMPLE_PREDICTIONS_PATH, index=False, engine="pyarrow", compression="zstd")

    pair_seed = pd.DataFrame(pair_seed_rows)
    if len(pair_seed) != 50 or pair_seed.duplicated(["source_city", "target_city", "seed"]).any():
        raise AssertionError("Expected 50 unique source-target-seed risk rows")
    pair_rows: list[dict[str, Any]] = []
    for (source, target), group in pair_seed.groupby(["source_city", "target_city"], sort=False):
        row: dict[str, Any] = {
            "source_city": source, "source_abbreviation": ABBR[source],
            "target_city": target, "target_abbreviation": ABBR[target],
            "direction": f"{source} -> {target}", "seed_count": len(group),
            "PCD_5NN": float(group["PCD_5NN"].mean()),
            "PCD_5NN_seed_std": float(group["PCD_5NN"].std(ddof=1)),
            "Global_5NN": float(group["Global_5NN"].mean()),
            "Global_5NN_seed_std": float(group["Global_5NN"].std(ddof=1)),
            "PredExpansion_5NN": float(group["PredExpansion_5NN"].mean()),
            "PredExpansion_5NN_seed_std": float(group["PredExpansion_5NN"].std(ddof=1)),
            "mean_entropy": float(group["mean_entropy"].mean()),
            "mean_entropy_seed_std": float(group["mean_entropy"].std(ddof=1)),
            "target_label_used_in_risk_features": False,
        }
        row["R1_predicted_macro_gap"] = float(calibrators["R1"].predict(pd.DataFrame({"PCD_5NN": [row["PCD_5NN"]]}))[0])
        row["R4_predicted_macro_gap"] = float(calibrators["R4"].predict(pd.DataFrame({"mean_entropy": [row["mean_entropy"]]}))[0])
        pair_rows.append(row)
    pair_predictions = pd.DataFrame(pair_rows).sort_values(["target_city", "source_city"]).reset_index(drop=True)
    if len(pair_predictions) != 10 or any(column.startswith("actual_") for column in pair_predictions.columns):
        raise AssertionError("Frozen pair predictions violate Phase A protocol")
    save_csv(pair_predictions, PAIR_PREDICTIONS_PATH)

    manifest_files = [
        EXTERNAL["Chengdu"], EXTERNAL["Hangzhou"], DATA, SPLITS_PATH, CALIBRATORS_PATH,
        SAMPLE_PREDICTIONS_PATH, PAIR_PREDICTIONS_PATH,
    ]
    manifest = {
        "stage": "Stage 09B Phase A",
        "status": "FROZEN_LABEL_BLIND",
        "phaseA_freeze_timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "files": {relative(path): sha256(path) for path in manifest_files},
        "target_labels_read_for_prediction": False,
        "target_f1_computed": False,
        "target_gap_computed": False,
        "threshold_changed": False,
        "model_changed": False,
        "risk_calibrators_fit_on_development_pairs_only": True,
        "stage07_reproduction_max_absolute_error": reproduction_max,
        "warnings": warnings,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PHASE_A_FROZEN sample_rows={len(sample_predictions)} pair_rows={len(pair_predictions)}", flush=True)
    print(f"SAMPLE_SHA256={sha256(SAMPLE_PREDICTIONS_PATH)}", flush=True)
    print(f"MANIFEST={MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()

"""Audit annual AlphaEarth CSV files and build city-level 195D datasets.

This script never writes to data/raw. Derived artifacts and a local run record
are written under data/processed or outputs.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import sys
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
CONFIG_PATH = PROJECT_ROOT / "config.json"
EXPERIMENT_LOG_PATH = LOGS_DIR / "reproduction_runs.md"
AUDIT_PATH = LOGS_DIR / "dataset_audit.txt"

EXPECTED_CLASSES = {0, 1, 2}
COORD_TOLERANCE = 1e-8
FORBIDDEN_MODEL_FIELDS = {
    "sample_id",
    "city",
    "longitude",
    "latitude",
    "utm_x",
    "utm_y",
    "block_x",
    "block_y",
    "block_id",
    "class",
}


class AuditLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def log(self, message: object = "") -> None:
        text = str(message)
        print(text)
        self.handle.write(text + "\n")
        self.handle.flush()

    def section(self, title: str) -> None:
        self.log()
        self.log("=" * 78)
        self.log(title)
        self.log("=" * 78)

    def close(self) -> None:
        self.handle.close()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def format_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    return ", ".join(f"{key}: {int(value)}" for key, value in counts.items())


def format_describe(values: Iterable[float]) -> str:
    stats = pd.Series(values, dtype="float64").describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    ordered = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    return ", ".join(f"{key}={stats[key]:.10f}" for key in ordered)


def append_experiment_log(status: str, details: list[str]) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## Run — {timestamp}\n\n")
        handle.write(f"Status: {status}\n\n")
        handle.write("Command: `python scripts/01_build_dataset.py`\n\n")
        if details:
            handle.write("Results:\n\n")
            for detail in details:
                handle.write(f"- {detail}\n")


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def detect_embedding_columns(
    columns: list[str], year: int, dimensions: int
) -> tuple[list[str], dict[str, str]]:
    detected: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    pattern = re.compile(r"^A(\d{2})(?:_(\d{4}))?$")

    for column in columns:
        match = pattern.fullmatch(str(column))
        if not match:
            continue
        index = int(match.group(1))
        if 0 <= index < dimensions:
            if index in detected:
                duplicates.setdefault(index, [detected[index]]).append(column)
            else:
                detected[index] = column

    if duplicates:
        raise ValueError(f"ambiguous AEF columns for indices: {duplicates}")

    missing = sorted(set(range(dimensions)) - set(detected))
    if missing:
        raise ValueError(f"missing AEF indices: {missing}")

    ordered = [detected[index] for index in range(dimensions)]
    rename_map = {
        source: f"A{index:02d}_{year}" for index, source in enumerate(ordered)
    }
    return ordered, rename_map


def audit_csv(
    path: Path,
    expected_city: str,
    year: int,
    dimensions: int,
    logger: AuditLogger,
    errors: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    logger.section(f"RAW CSV AUDIT — {path.name}")
    logger.log(f"filename: {path.name}")
    logger.log(f"path: {path}")
    logger.log(f"size_bytes: {path.stat().st_size}")

    header = read_header(path)
    header_duplicates = sorted({name for name in header if header.count(name) > 1})
    logger.log(f"header_column_count: {len(header)}")
    logger.log(f"column_names: {json.dumps(header, ensure_ascii=False)}")
    logger.log(f"duplicate_column_names: {header_duplicates}")
    if header_duplicates:
        errors.append(f"{path.name}: duplicate column names {header_duplicates}")

    dataframe = pd.read_csv(path, low_memory=False)
    logger.log(f"number_of_rows: {len(dataframe)}")
    logger.log(f"number_of_columns: {len(dataframe.columns)}")

    try:
        source_bands, rename_map = detect_embedding_columns(
            list(dataframe.columns), year, dimensions
        )
    except ValueError as exc:
        errors.append(f"{path.name}: {exc}")
        logger.log(f"AEF_band_detection: ERROR — {exc}")
        return dataframe, []

    canonical_bands = [f"A{index:02d}_{year}" for index in range(dimensions)]
    noncanonical = {
        source: target for source, target in rename_map.items() if source != target
    }
    logger.log(f"AEF_band_count: {len(source_bands)}")
    logger.log(f"AEF_band_detection: recognized A00-A63 by numeric band index")
    if noncanonical:
        logger.log(
            "AEF_column_mapping: " + json.dumps(noncanonical, ensure_ascii=False)
        )
    else:
        logger.log("AEF_column_mapping: source names already match canonical names")

    dataframe = dataframe.rename(columns=rename_map)
    required_metadata = ["sample_id", "city", "class", "longitude", "latitude"]
    missing_metadata = [name for name in required_metadata if name not in dataframe]
    logger.log(f"required_metadata_present: {not missing_metadata}")
    if missing_metadata:
        errors.append(f"{path.name}: missing metadata columns {missing_metadata}")
        return dataframe, canonical_bands

    dataframe["sample_id"] = dataframe["sample_id"].astype("string")
    null_sample_ids = int(dataframe["sample_id"].isna().sum())
    duplicate_sample_ids = int(dataframe["sample_id"].duplicated(keep=False).sum())
    logger.log(f"sample_id_present: True")
    logger.log(f"sample_id_null_count: {null_sample_ids}")
    logger.log(f"sample_id_duplicate_rows: {duplicate_sample_ids}")
    logger.log(f"sample_id_unique: {null_sample_ids == 0 and duplicate_sample_ids == 0}")
    if null_sample_ids:
        errors.append(f"{path.name}: {null_sample_ids} null sample_id values")
    if duplicate_sample_ids:
        errors.append(f"{path.name}: {duplicate_sample_ids} rows have duplicate sample_id")

    observed_cities = sorted(dataframe["city"].dropna().astype(str).unique().tolist())
    logger.log(f"expected_city: {expected_city}")
    logger.log(f"observed_city_values: {observed_cities}")
    if observed_cities != [expected_city]:
        errors.append(
            f"{path.name}: expected city {expected_city}, observed {observed_cities}"
        )

    numeric_class = pd.to_numeric(dataframe["class"], errors="coerce")
    invalid_class = int(numeric_class.isna().sum() - dataframe["class"].isna().sum())
    if invalid_class:
        errors.append(f"{path.name}: {invalid_class} non-numeric class values")
    dataframe["class"] = numeric_class
    observed_classes = set(dataframe["class"].dropna().astype(int).unique().tolist())
    logger.log(f"class_distribution: {format_counts(dataframe['class'])}")
    if observed_classes != EXPECTED_CLASSES:
        errors.append(
            f"{path.name}: expected classes {sorted(EXPECTED_CLASSES)}, "
            f"observed {sorted(observed_classes)}"
        )

    for coordinate in ["longitude", "latitude"]:
        original = dataframe[coordinate]
        converted = pd.to_numeric(original, errors="coerce")
        conversion_failures = int((converted.isna() & original.notna()).sum())
        dataframe[coordinate] = converted
        logger.log(f"{coordinate}_conversion_failures: {conversion_failures}")
        if conversion_failures:
            errors.append(
                f"{path.name}: {conversion_failures} invalid {coordinate} values"
            )

    longitude_outside = int((~dataframe["longitude"].between(-180, 180)).sum())
    latitude_outside = int((~dataframe["latitude"].between(-90, 90)).sum())
    logger.log(f"longitude_outside_valid_range: {longitude_outside}")
    logger.log(f"latitude_outside_valid_range: {latitude_outside}")
    if longitude_outside or latitude_outside:
        errors.append(
            f"{path.name}: invalid coordinate range, longitude={longitude_outside}, "
            f"latitude={latitude_outside}"
        )

    nonnumeric_bands = [
        column
        for column in canonical_bands
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    logger.log(f"embedding_numeric_dtype: {not nonnumeric_bands}")
    logger.log(f"embedding_nonnumeric_columns: {nonnumeric_bands}")
    if nonnumeric_bands:
        errors.append(f"{path.name}: non-numeric embedding columns {nonnumeric_bands}")

    for column in canonical_bands:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    nan_count = int(dataframe.isna().sum().sum())
    numeric_values = dataframe.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64, copy=False
    )
    inf_count = int(np.isinf(numeric_values).sum())
    duplicate_rows = int(dataframe.duplicated().sum())
    logger.log(f"NaN_count_all_columns: {nan_count}")
    logger.log(f"Inf_count_numeric_columns: {inf_count}")
    logger.log(f"duplicate_full_rows: {duplicate_rows}")
    if nan_count:
        errors.append(f"{path.name}: {nan_count} NaN values")
    if inf_count:
        errors.append(f"{path.name}: {inf_count} Inf values")
    if duplicate_rows:
        errors.append(f"{path.name}: {duplicate_rows} duplicate full rows")

    vectors = dataframe[canonical_bands].to_numpy(dtype=np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    finite_norms = norms[np.isfinite(norms)]
    logger.log(f"L2_norm_statistics: {format_describe(finite_norms)}")
    near_one = float(np.mean(np.abs(finite_norms - 1.0) < 0.01) * 100.0)
    logger.log(f"percent_abs_norm_minus_1_lt_0.01: {near_one:.6f}%")
    zero_norms = int(np.sum(norms == 0))
    nonfinite_norms = int(np.sum(~np.isfinite(norms)))
    logger.log(f"zero_norm_count: {zero_norms}")
    logger.log(f"nonfinite_norm_count: {nonfinite_norms}")
    if zero_norms or nonfinite_norms:
        errors.append(
            f"{path.name}: zero norms={zero_norms}, non-finite norms={nonfinite_norms}"
        )

    return dataframe, canonical_bands


def audit_alignment(
    city: str,
    yearly: dict[int, pd.DataFrame],
    years: list[int],
    logger: AuditLogger,
    errors: list[str],
) -> None:
    logger.section(f"SAMPLE ALIGNMENT — {city}")
    for left_year, right_year in [(years[0], years[1]), (years[0], years[2]), (years[1], years[2])]:
        left = yearly[left_year].set_index("sample_id", drop=False)
        right = yearly[right_year].set_index("sample_id", drop=False)
        left_ids = pd.Index(left.index)
        right_ids = pd.Index(right.index)
        common = left_ids.intersection(right_ids, sort=False)
        left_only = left_ids.difference(right_ids, sort=False)
        right_only = right_ids.difference(left_ids, sort=False)

        logger.log(
            f"{left_year} vs {right_year}: left={len(left_ids)}, right={len(right_ids)}, "
            f"intersection={len(common)}, missing_from_{right_year}={len(left_only)}, "
            f"extra_in_{right_year}={len(right_only)}"
        )
        if len(left_only):
            logger.log(f"  sample missing examples: {left_only[:10].tolist()}")
            errors.append(
                f"{city} {left_year} vs {right_year}: {len(left_only)} IDs missing from {right_year}"
            )
        if len(right_only):
            logger.log(f"  extra ID examples: {right_only[:10].tolist()}")
            errors.append(
                f"{city} {left_year} vs {right_year}: {len(right_only)} extra IDs in {right_year}"
            )

        left_common = left.loc[common]
        right_common = right.loc[common]
        class_mismatch = int(
            np.sum(left_common["class"].to_numpy() != right_common["class"].to_numpy())
        )
        city_mismatch = int(
            np.sum(
                left_common["city"].astype(str).to_numpy()
                != right_common["city"].astype(str).to_numpy()
            )
        )
        longitude_diff = np.abs(
            left_common["longitude"].to_numpy(dtype=float)
            - right_common["longitude"].to_numpy(dtype=float)
        )
        latitude_diff = np.abs(
            left_common["latitude"].to_numpy(dtype=float)
            - right_common["latitude"].to_numpy(dtype=float)
        )
        longitude_mismatch = int(np.sum(longitude_diff > COORD_TOLERANCE))
        latitude_mismatch = int(np.sum(latitude_diff > COORD_TOLERANCE))
        logger.log(
            f"  metadata mismatches: class={class_mismatch}, city={city_mismatch}, "
            f"longitude={longitude_mismatch}, latitude={latitude_mismatch}"
        )
        logger.log(
            f"  maximum coordinate differences: longitude={longitude_diff.max(initial=0):.12g}, "
            f"latitude={latitude_diff.max(initial=0):.12g}, tolerance={COORD_TOLERANCE}"
        )
        if class_mismatch or city_mismatch or longitude_mismatch or latitude_mismatch:
            errors.append(
                f"{city} {left_year} vs {right_year}: metadata mismatch "
                f"class={class_mismatch}, city={city_mismatch}, "
                f"longitude={longitude_mismatch}, latitude={latitude_mismatch}"
            )


def temporal_distance(
    first: np.ndarray,
    second: np.ndarray,
    name: str,
    logger: AuditLogger,
    errors: list[str],
) -> np.ndarray:
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    denominator = first_norm * second_norm
    invalid_denominator = int(np.sum((denominator == 0) | ~np.isfinite(denominator)))
    if invalid_denominator:
        errors.append(f"{name}: {invalid_denominator} invalid cosine denominators")

    similarity = np.full(len(first), np.nan, dtype=np.float64)
    valid = (denominator > 0) & np.isfinite(denominator)
    similarity[valid] = np.einsum("ij,ij->i", first[valid], second[valid]) / denominator[valid]
    raw_distance = 1.0 - similarity
    out_of_range = int(
        np.sum((raw_distance < -1e-10) | (raw_distance > 2.0 + 1e-10))
    )
    nonfinite = int(np.sum(~np.isfinite(raw_distance)))
    logger.log(f"{name}_raw_outside_[0,2]: {out_of_range}")
    logger.log(f"{name}_NaN_or_Inf: {nonfinite}")
    if out_of_range:
        errors.append(f"{name}: {out_of_range} values outside valid cosine distance range")
    if nonfinite:
        errors.append(f"{name}: {nonfinite} non-finite values")

    distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    logger.log(f"{name}_statistics: {format_describe(distance[np.isfinite(distance)])}")
    return distance


def build_city_dataset(
    city: str,
    yearly: dict[int, pd.DataFrame],
    yearly_bands: dict[int, list[str]],
    config: dict,
    logger: AuditLogger,
    errors: list[str],
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    years = config["years"]
    logger.section(f"MASTER DATASET — {city}")
    id_sets = [set(yearly[year]["sample_id"].tolist()) for year in years]
    intersection = set.intersection(*id_sets)
    ordered_ids = [
        sample_id
        for sample_id in yearly[years[0]]["sample_id"].tolist()
        if sample_id in intersection
    ]
    logger.log(
        f"counts: {years[0]}={len(yearly[years[0]])}, "
        f"{years[1]}={len(yearly[years[1]])}, {years[2]}={len(yearly[years[2]])}, "
        f"intersection={len(ordered_ids)}"
    )
    for year in years:
        loss = len(yearly[year]) - len(ordered_ids)
        logger.log(f"samples_lost_from_{year}_after_inner_join: {loss}")

    first = yearly[years[0]].set_index("sample_id", drop=False).loc[ordered_ids]
    master = first[["sample_id", "city", "class", "longitude", "latitude"]].reset_index(
        drop=True
    )
    for year in years:
        source = yearly[year].set_index("sample_id").loc[ordered_ids]
        master = pd.concat(
            [master, source[yearly_bands[year]].reset_index(drop=True)], axis=1
        )

    arrays = {
        year: master[yearly_bands[year]].to_numpy(dtype=np.float64, copy=False)
        for year in years
    }
    master["D_20_22"] = temporal_distance(
        arrays[2020], arrays[2022], f"{city}_D_20_22", logger, errors
    )
    master["D_22_24"] = temporal_distance(
        arrays[2022], arrays[2024], f"{city}_D_22_24", logger, errors
    )
    master["D_20_24"] = temporal_distance(
        arrays[2020], arrays[2024], f"{city}_D_20_24", logger, errors
    )

    crs = config["cities"][city]["crs"]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    utm_x, utm_y = transformer.transform(
        master["longitude"].to_numpy(dtype=float),
        master["latitude"].to_numpy(dtype=float),
    )
    master["utm_x"] = utm_x
    master["utm_y"] = utm_y
    block_size = int(config["spatial_block_size_m"])
    master["block_x"] = np.floor(master["utm_x"] / block_size).astype("int64")
    master["block_y"] = np.floor(master["utm_y"] / block_size).astype("int64")
    master["block_id"] = (
        city
        + "_"
        + master["block_x"].astype(str)
        + "_"
        + master["block_y"].astype(str)
    )

    logger.section(f"SPATIAL BLOCK AUDIT — {city}")
    total_blocks = int(master["block_id"].nunique())
    logger.log(f"crs: {crs}")
    logger.log(f"spatial_block_size_m: {block_size}")
    logger.log(f"total_blocks: {total_blocks}")
    for class_value in sorted(EXPECTED_CLASSES):
        subset = master.loc[master["class"] == class_value]
        class_blocks = int(subset["block_id"].nunique())
        largest_share = (
            float(subset.groupby("block_id").size().max() / len(subset))
            if len(subset)
            else float("nan")
        )
        logger.log(
            f"class_{class_value}: samples={len(subset)}, blocks={class_blocks}, "
            f"largest_single_block_share={largest_share:.6%}"
        )
        if class_blocks < 5 or largest_share > 0.5:
            warnings.append(
                f"{city} class {class_value} is extremely concentrated: "
                f"blocks={class_blocks}, largest block share={largest_share:.2%}"
            )

    block_sizes = master.groupby("block_id").size().astype(float)
    logger.log(f"samples_per_block_statistics: {format_describe(block_sizes)}")

    feature_columns = [
        column for year in years for column in yearly_bands[year]
    ] + list(config["temporal_features"])
    expected_count = int(config["final_feature_count"])
    logger.section(f"FINAL FEATURE AUDIT — {city}")
    logger.log(f"feature_count: {len(feature_columns)}")
    logger.log(f"expected_feature_count: {expected_count}")
    logger.log(f"forbidden_fields_in_feature_list: {sorted(FORBIDDEN_MODEL_FIELDS & set(feature_columns))}")
    if len(feature_columns) != expected_count:
        errors.append(
            f"{city}: feature count {len(feature_columns)} != {expected_count}"
        )
    if FORBIDDEN_MODEL_FIELDS & set(feature_columns):
        errors.append(f"{city}: forbidden metadata included in model feature list")

    metadata_columns = [
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
    master = master[metadata_columns + feature_columns]
    summary = {
        "city": city,
        "total_samples": int(len(master)),
        "class_0_count": int((master["class"] == 0).sum()),
        "class_1_count": int((master["class"] == 1).sum()),
        "class_2_count": int((master["class"] == 2).sum()),
        "spatial_blocks": total_blocks,
        "feature_count": len(feature_columns),
    }
    return master, summary


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(AUDIT_PATH)
    errors: list[str] = []
    warnings: list[str] = []
    run_details: list[str] = []

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        years = [int(year) for year in config["years"]]
        dimensions = int(config["embedding_dimensions_per_year"])

        logger.section("RUN INFORMATION")
        logger.log(f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.log(f"project_root: {PROJECT_ROOT}")
        logger.log(f"python_version: {platform.python_version()}")
        logger.log(f"python_full: {sys.version.replace(chr(10), ' ')}")
        logger.log(f"python_executable: {sys.executable}")
        for package in [
            "pandas",
            "numpy",
            "pyarrow",
            "pyproj",
            "scikit-learn",
            "matplotlib",
            "xgboost",
            "joblib",
        ]:
            logger.log(f"package_{package}: {package_version(package)}")

        expected_files = {
            (city, year): RAW_DIR / f"AEF_{city}_{year}.csv"
            for city in config["cities"]
            for year in years
        }
        missing_files = [str(path) for path in expected_files.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"missing expected raw CSV files: {missing_files}")

        data: dict[str, dict[int, pd.DataFrame]] = {
            city: {} for city in config["cities"]
        }
        bands: dict[str, dict[int, list[str]]] = {
            city: {} for city in config["cities"]
        }
        for (city, year), path in expected_files.items():
            dataframe, band_columns = audit_csv(
                path, city, year, dimensions, logger, errors
            )
            data[city][year] = dataframe
            bands[city][year] = band_columns

        if errors:
            raise ValueError("raw CSV audit found data integrity errors")

        for city in config["cities"]:
            audit_alignment(city, data[city], years, logger, errors)

        if errors:
            raise ValueError("cross-year sample alignment found data integrity errors")

        outputs: list[Path] = []
        summaries: list[dict[str, int]] = []
        for city in config["cities"]:
            master, summary = build_city_dataset(
                city,
                data[city],
                bands[city],
                config,
                logger,
                errors,
                warnings,
            )
            if errors:
                raise ValueError(f"{city} dataset construction found errors")
            output_path = PROCESSED_DIR / f"{city}_AEF_195D.parquet"
            master.to_parquet(output_path, index=False, engine="pyarrow")
            verification = pd.read_parquet(output_path, engine="pyarrow")
            if verification.shape != master.shape:
                raise RuntimeError(
                    f"Parquet verification failed for {output_path}: "
                    f"expected {master.shape}, got {verification.shape}"
                )
            outputs.append(output_path)
            summaries.append(summary)
            logger.log(f"verified_output: {output_path}")
            logger.log(f"verified_shape: {verification.shape}")

        summary_path = TABLES_DIR / "dataset_summary.csv"
        summary_frame = pd.DataFrame(summaries)
        summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
        summary_check = pd.read_csv(summary_path)
        if len(summary_check) != len(config["cities"]):
            raise RuntimeError("dataset_summary.csv verification failed")

        logger.section("WARNINGS")
        if warnings:
            for warning in warnings:
                logger.log(f"WARNING: {warning}")
        else:
            logger.log("None")

        logger.section("OUTPUTS")
        for output in outputs:
            logger.log(output)
        logger.log(summary_path)
        logger.log(AUDIT_PATH)
        logger.log("final_status: SUCCESS")

        run_details.extend(
            [
                "Six annual CSV files audited successfully.",
                "Sample IDs and metadata aligned across 2020, 2022 and 2024.",
                "Temporal cosine distances and 2 km spatial blocks generated.",
                "Two 195-feature Parquet datasets and dataset_summary.csv written and verified.",
                f"Warnings: {len(warnings)}.",
            ]
        )
        append_experiment_log("Completed", run_details)
        return 0
    except Exception as exc:
        logger.section("ERRORS")
        if errors:
            for error in errors:
                logger.log(f"ERROR: {error}")
        logger.log(f"FATAL: {type(exc).__name__}: {exc}")
        logger.log(traceback.format_exc())
        logger.log("final_status: FAILED")
        append_experiment_log(
            "Failed", [f"{type(exc).__name__}: {exc}", f"Audit log: {AUDIT_PATH}"]
        )
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())


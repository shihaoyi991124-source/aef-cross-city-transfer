"""Stage 06C: integrate and audit Wuhan, Xian, and Beijing AEF exports.

This stage performs data-only work. It copies the nine downloaded CSV files
into the project's immutable raw-data area, audits them, constructs compatible
195D Parquet datasets, and creates a five-city combined dataset. It does not
train or evaluate any model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "imports" / "development"
RAW_MULTICITY_DIR = PROJECT_ROOT / "data" / "raw" / "multicity"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
CONFIG_PATH = PROJECT_ROOT / "config.json"
AUDIT_PATH = LOGS_DIR / "stage06c_multicity_audit.txt"

NEW_CITY_CRS = {
    "Wuhan": "EPSG:32650",
    "Xian": "EPSG:32649",
    "Beijing": "EPSG:32650",
}
ALL_CITIES = ["Shenzhen", "Kunming", "Wuhan", "Xian", "Beijing"]
YEARS = [2020, 2022, 2024]
EXPECTED_CLASSES = [0, 1, 2]
EXPECTED_ROWS_PER_FILE = 18_000
EXPECTED_SAMPLES_PER_CLASS = 6_000
EMBEDDING_DIMENSIONS = 64
AEF_FEATURE_COUNT = 192
FINAL_FEATURE_COUNT = 195
COORDINATE_TOLERANCE = 1e-8
BLOCK_SIZE_M = 2_000
TEMPORAL_FEATURES = ["D_20_22", "D_22_24", "D_20_24"]
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
FORBIDDEN_MODEL_FIELDS = set(METADATA_COLUMNS)


class AuditLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def log(self, value: object = "") -> None:
        text = str(value)
        print(text)
        self.handle.write(text + "\n")
        self.handle.flush()

    def section(self, title: str) -> None:
        self.log()
        self.log("=" * 88)
        self.log(title)
        self.log("=" * 88)

    def close(self) -> None:
        self.handle.close()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def format_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    return ", ".join(f"{key}: {int(value)}" for key, value in counts.items())


def describe_values(values: Iterable[float]) -> dict[str, float]:
    stats = pd.Series(values, dtype="float64").describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "min": float(stats["min"]),
        "25%": float(stats["25%"]),
        "median": float(stats["50%"]),
        "75%": float(stats["75%"]),
        "max": float(stats["max"]),
    }


def format_statistics(stats: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.10f}" for key, value in stats.items())


def normalized_filename_stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_source_files(
    logger: AuditLogger,
) -> dict[tuple[str, int], Path]:
    logger.section("1. NINE RAW CSV PATHS")
    if not DOWNLOAD_DIR.is_dir():
        raise FileNotFoundError(f"Stage 06C download directory not found: {DOWNLOAD_DIR}")

    all_csvs = sorted(DOWNLOAD_DIR.glob("*.csv"))
    resolved: dict[tuple[str, int], Path] = {}
    missing: list[str] = []
    ambiguous: list[str] = []

    for city in NEW_CITY_CRS:
        for year in YEARS:
            expected_name = f"AEF_{city}_{year}.csv"
            exact = DOWNLOAD_DIR / expected_name
            if exact.is_file():
                resolved[(city, year)] = exact
                logger.log(f"{city} {year}: exact match -> {exact}")
                continue

            expected_key = normalized_filename_stem(Path(expected_name).stem)
            matches = [
                path
                for path in all_csvs
                if normalized_filename_stem(path.stem) == expected_key
            ]
            if len(matches) == 1:
                resolved[(city, year)] = matches[0]
                logger.log(
                    f"{city} {year}: uniquely normalized match -> {matches[0].name}"
                )
            elif not matches:
                missing.append(expected_name)
            else:
                ambiguous.append(
                    f"{expected_name}: {[path.name for path in matches]}"
                )

    if missing or ambiguous:
        if missing:
            logger.log(f"missing_files: {missing}")
        if ambiguous:
            logger.log(f"ambiguous_matches: {ambiguous}")
        raise FileNotFoundError(
            f"required CSV resolution failed; missing={missing}, ambiguous={ambiguous}"
        )

    logger.log(f"resolved_file_count: {len(resolved)}")
    return resolved


def copy_raw_files(
    source_files: dict[tuple[str, int], Path],
    logger: AuditLogger,
) -> tuple[dict[tuple[str, int], Path], list[dict[str, object]]]:
    logger.section("2. RAW COPY AND SHA256 VERIFICATION")
    RAW_MULTICITY_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[tuple[str, int], Path] = {}
    records: list[dict[str, object]] = []

    for key, source in source_files.items():
        destination = RAW_MULTICITY_DIR / source.name
        source_size = source.stat().st_size
        source_hash = sha256_file(source)

        if destination.exists():
            destination_size = destination.stat().st_size
            destination_hash = sha256_file(destination)
            if destination_size != source_size or destination_hash != source_hash:
                raise RuntimeError(
                    f"existing raw copy differs from source and was not overwritten: {destination}"
                )
            copy_action = "existing copy verified"
        else:
            shutil.copy2(source, destination)
            destination_size = destination.stat().st_size
            destination_hash = sha256_file(destination)
            copy_action = "copied"

        match = (
            source.name == destination.name
            and source_size == destination_size
            and source_hash == destination_hash
        )
        if not match:
            raise RuntimeError(f"raw copy verification failed for {source.name}")

        logger.log(f"filename: {source.name}")
        logger.log(f"  source_path: {source}")
        logger.log(f"  destination_path: {destination}")
        logger.log(f"  action: {copy_action}")
        logger.log(f"  source_size_bytes: {source_size}")
        logger.log(f"  destination_size_bytes: {destination_size}")
        logger.log(f"  source_sha256: {source_hash}")
        logger.log(f"  destination_sha256: {destination_hash}")
        logger.log(f"  copy_identical: {match}")

        copied[key] = destination
        records.append(
            {
                "filename": source.name,
                "source_path": str(source),
                "destination_path": str(destination),
                "source_size_bytes": source_size,
                "destination_size_bytes": destination_size,
                "source_sha256": source_hash,
                "destination_sha256": destination_hash,
                "copy_identical": match,
            }
        )

    return copied, records


def detect_embedding_columns(
    columns: list[str], year: int
) -> tuple[list[str], dict[str, str], list[str]]:
    detected: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    mismatched_year_columns: list[str] = []
    pattern = re.compile(r"^A(\d{2})(?:_(\d{4}))?$")

    for column in columns:
        match = pattern.fullmatch(str(column))
        if not match:
            continue
        index = int(match.group(1))
        suffix = match.group(2)
        if suffix is not None and int(suffix) != year:
            mismatched_year_columns.append(str(column))
            continue
        if 0 <= index < EMBEDDING_DIMENSIONS:
            if index in detected:
                duplicates.setdefault(index, [detected[index]]).append(str(column))
            else:
                detected[index] = str(column)

    if mismatched_year_columns:
        raise ValueError(
            f"AEF columns contain a different year suffix: {mismatched_year_columns}"
        )
    if duplicates:
        raise ValueError(f"ambiguous AEF columns for indices: {duplicates}")
    missing = sorted(set(range(EMBEDDING_DIMENSIONS)) - set(detected))
    if missing:
        raise ValueError(f"missing AEF indices: {missing}")

    ordered = [detected[index] for index in range(EMBEDDING_DIMENSIONS)]
    canonical = [f"A{index:02d}_{year}" for index in range(EMBEDDING_DIMENSIONS)]
    rename_map = {source: target for source, target in zip(ordered, canonical)}
    noncanonical = [source for source, target in rename_map.items() if source != target]
    return ordered, rename_map, noncanonical


def audit_csv(
    path: Path,
    expected_city: str,
    year: int,
    logger: AuditLogger,
    errors: list[str],
    warnings: list[str],
    norm_records: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[str]]:
    logger.section(f"4–6. RAW CSV AND AEF NORM AUDIT — {path.name}")
    logger.log(f"filename: {path.name}")
    logger.log(f"year: {year}")
    logger.log(f"expected_city: {expected_city}")
    logger.log(f"path: {path}")
    logger.log(f"size_bytes: {path.stat().st_size}")

    header = read_header(path)
    header_duplicates = sorted({name for name in header if header.count(name) > 1})
    logger.log(f"header_columns: {len(header)}")
    logger.log(f"column_names: {json.dumps(header, ensure_ascii=False)}")
    logger.log(f"duplicate_column_names: {header_duplicates}")
    if header_duplicates:
        errors.append(f"{path.name}: duplicate column names {header_duplicates}")

    frame = pd.read_csv(path, low_memory=False)
    logger.log(f"rows: {len(frame)}")
    logger.log(f"columns: {len(frame.columns)}")
    if len(frame) != EXPECTED_ROWS_PER_FILE:
        errors.append(
            f"{path.name}: row count {len(frame)} != {EXPECTED_ROWS_PER_FILE}"
        )

    try:
        source_bands, rename_map, noncanonical = detect_embedding_columns(
            list(frame.columns), year
        )
    except ValueError as exc:
        logger.log(f"AEF_band_detection: ERROR — {exc}")
        errors.append(f"{path.name}: {exc}")
        return frame, []

    canonical_bands = [f"A{index:02d}_{year}" for index in range(64)]
    logger.log(f"AEF_band_count: {len(source_bands)}")
    logger.log(f"AEF_band_order: {canonical_bands[0]} ... {canonical_bands[-1]}")
    if noncanonical:
        logger.log(
            "AEF_noncanonical_source_columns: "
            + json.dumps(noncanonical, ensure_ascii=False)
        )
        warnings.append(
            f"{path.name}: safely canonicalized {len(noncanonical)} AEF column names"
        )
    else:
        logger.log("AEF_column_mapping: source names already canonical")
    frame = frame.rename(columns=rename_map)

    required = ["sample_id", "city", "class", "longitude", "latitude"]
    missing_metadata = [column for column in required if column not in frame.columns]
    logger.log(f"required_metadata_present: {not missing_metadata}")
    logger.log(f"missing_metadata: {missing_metadata}")
    if missing_metadata:
        errors.append(f"{path.name}: missing metadata {missing_metadata}")
        return frame, canonical_bands

    frame["sample_id"] = frame["sample_id"].astype("string")
    null_ids = int(frame["sample_id"].isna().sum())
    duplicate_id_rows = int(frame["sample_id"].duplicated(keep=False).sum())
    logger.log(f"sample_id_null_count: {null_ids}")
    logger.log(f"sample_id_duplicate_rows: {duplicate_id_rows}")
    logger.log(f"sample_id_unique: {null_ids == 0 and duplicate_id_rows == 0}")
    if null_ids or duplicate_id_rows:
        errors.append(
            f"{path.name}: null sample IDs={null_ids}, duplicate-ID rows={duplicate_id_rows}"
        )

    observed_cities = sorted(frame["city"].dropna().astype(str).unique().tolist())
    logger.log(f"observed_city_values: {observed_cities}")
    if observed_cities != [expected_city]:
        errors.append(
            f"{path.name}: expected city {expected_city}, observed {observed_cities}"
        )

    original_class = frame["class"]
    numeric_class = pd.to_numeric(original_class, errors="coerce")
    class_conversion_failures = int(
        (numeric_class.isna() & original_class.notna()).sum()
    )
    frame["class"] = numeric_class
    class_counts = frame["class"].value_counts(dropna=False).sort_index()
    logger.log(f"class_distribution: {format_counts(frame['class'])}")
    logger.log(f"class_numeric_conversion_failures: {class_conversion_failures}")
    if class_conversion_failures or frame["class"].isna().any():
        errors.append(f"{path.name}: invalid or missing class values")
    observed_classes = set(frame["class"].dropna().astype(int).unique().tolist())
    if observed_classes != set(EXPECTED_CLASSES):
        errors.append(
            f"{path.name}: expected classes {EXPECTED_CLASSES}, observed {sorted(observed_classes)}"
        )
    for class_value in EXPECTED_CLASSES:
        count = int(class_counts.get(class_value, 0))
        if count != EXPECTED_SAMPLES_PER_CLASS:
            errors.append(
                f"{path.name}: class {class_value} count {count} != {EXPECTED_SAMPLES_PER_CLASS}"
            )

    for coordinate in ["longitude", "latitude"]:
        original = frame[coordinate]
        converted = pd.to_numeric(original, errors="coerce")
        conversion_failures = int((converted.isna() & original.notna()).sum())
        frame[coordinate] = converted.astype("float64")
        logger.log(f"{coordinate}_numeric_conversion_failures: {conversion_failures}")
        if conversion_failures:
            errors.append(
                f"{path.name}: {conversion_failures} invalid {coordinate} values"
            )
    invalid_longitude = int((~frame["longitude"].between(-180, 180)).sum())
    invalid_latitude = int((~frame["latitude"].between(-90, 90)).sum())
    logger.log(f"longitude_outside_valid_range: {invalid_longitude}")
    logger.log(f"latitude_outside_valid_range: {invalid_latitude}")
    if invalid_longitude or invalid_latitude:
        errors.append(
            f"{path.name}: invalid coordinates lon={invalid_longitude}, lat={invalid_latitude}"
        )

    nonnumeric_before = [
        column
        for column in canonical_bands
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    logger.log(f"AEF_numeric_dtype_before_coercion: {not nonnumeric_before}")
    logger.log(f"AEF_nonnumeric_columns: {nonnumeric_before}")
    for column in canonical_bands:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        failures = int((converted.isna() & original.notna()).sum())
        if failures:
            errors.append(f"{path.name}: {column} has {failures} non-numeric values")
        frame[column] = converted.astype("float64")

    nan_count = int(frame.isna().sum().sum())
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64, copy=False
    )
    inf_count = int(np.isinf(numeric).sum())
    duplicate_rows = int(frame.duplicated().sum())
    logger.log(f"NaN_count_all_columns: {nan_count}")
    logger.log(f"Inf_count_numeric_columns: {inf_count}")
    logger.log(f"duplicate_full_rows: {duplicate_rows}")
    if nan_count:
        errors.append(f"{path.name}: {nan_count} NaN values")
    if inf_count:
        errors.append(f"{path.name}: {inf_count} Inf values")
    if duplicate_rows:
        errors.append(f"{path.name}: {duplicate_rows} duplicate full rows")

    vectors = frame[canonical_bands].to_numpy(dtype=np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    finite_norms = norms[np.isfinite(norms)]
    if len(finite_norms):
        norm_stats = describe_values(finite_norms)
        near_one_percent = float(np.mean(np.abs(finite_norms - 1.0) < 0.01) * 100)
    else:
        norm_stats = {key: float("nan") for key in ["mean", "std", "min", "25%", "median", "75%", "max"]}
        near_one_percent = float("nan")
    zero_norms = int(np.sum(norms == 0))
    nan_norms = int(np.sum(np.isnan(norms)))
    inf_norms = int(np.sum(np.isinf(norms)))
    logger.log(f"AEF_L2_norm: {format_statistics(norm_stats)}")
    logger.log(f"percent_abs_norm_minus_1_lt_0.01: {near_one_percent:.6f}")
    logger.log(f"zero_norm_count: {zero_norms}")
    logger.log(f"NaN_norm_count: {nan_norms}")
    logger.log(f"Inf_norm_count: {inf_norms}")
    if zero_norms or nan_norms or inf_norms:
        errors.append(
            f"{path.name}: zero norms={zero_norms}, NaN norms={nan_norms}, Inf norms={inf_norms}"
        )
    if np.isfinite(near_one_percent) and near_one_percent < 99.0:
        warnings.append(
            f"{path.name}: only {near_one_percent:.2f}% of AEF norms are within 0.01 of 1"
        )

    norm_records.append(
        {
            "city": expected_city,
            "year": year,
            **norm_stats,
            "percent_abs_norm_minus_1_lt_0.01": near_one_percent,
            "zero_norm_count": zero_norms,
            "NaN_norm_count": nan_norms,
            "Inf_norm_count": inf_norms,
        }
    )

    if not frame["class"].isna().any():
        frame["class"] = frame["class"].astype("int64")
    frame["city"] = frame["city"].astype(str)
    retained = required + canonical_bands
    return frame[retained].copy(), canonical_bands


def audit_alignment(
    city: str,
    yearly: dict[int, pd.DataFrame],
    logger: AuditLogger,
    errors: list[str],
) -> None:
    logger.section(f"7. THREE-YEAR SAMPLE ALIGNMENT — {city}")
    pairs = [(2020, 2022), (2020, 2024), (2022, 2024)]
    for left_year, right_year in pairs:
        left = yearly[left_year].set_index("sample_id", drop=False)
        right = yearly[right_year].set_index("sample_id", drop=False)
        left_ids = pd.Index(left.index)
        right_ids = pd.Index(right.index)
        intersection = left_ids.intersection(right_ids, sort=False)
        left_only = left_ids.difference(right_ids, sort=False)
        right_only = right_ids.difference(left_ids, sort=False)

        logger.log(
            f"{left_year} vs {right_year}: left={len(left_ids)}, right={len(right_ids)}, "
            f"intersection={len(intersection)}, missing_from_{right_year}={len(left_only)}, "
            f"extra_in_{right_year}={len(right_only)}"
        )
        if len(left_only) or len(right_only):
            errors.append(
                f"{city} {left_year} vs {right_year}: missing={len(left_only)}, extra={len(right_only)}"
            )

        left_common = left.loc[intersection]
        right_common = right.loc[intersection]
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
        longitude_mismatch = int(np.sum(longitude_diff > COORDINATE_TOLERANCE))
        latitude_mismatch = int(np.sum(latitude_diff > COORDINATE_TOLERANCE))
        logger.log(
            f"  metadata_mismatches: class={class_mismatch}, city={city_mismatch}, "
            f"longitude={longitude_mismatch}, latitude={latitude_mismatch}"
        )
        logger.log(
            f"  maximum_coordinate_difference: longitude={longitude_diff.max(initial=0):.12g}, "
            f"latitude={latitude_diff.max(initial=0):.12g}, tolerance={COORDINATE_TOLERANCE}"
        )
        if class_mismatch or city_mismatch or longitude_mismatch or latitude_mismatch:
            errors.append(
                f"{city} {left_year} vs {right_year}: metadata mismatch "
                f"class={class_mismatch}, city={city_mismatch}, "
                f"longitude={longitude_mismatch}, latitude={latitude_mismatch}"
            )


def cosine_distance_rows(
    first: np.ndarray,
    second: np.ndarray,
    label: str,
    logger: AuditLogger,
    errors: list[str],
) -> np.ndarray:
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    denominator = first_norm * second_norm
    invalid_denominator = int(np.sum((denominator == 0) | ~np.isfinite(denominator)))
    similarity = np.full(len(first), np.nan, dtype=np.float64)
    valid = (denominator > 0) & np.isfinite(denominator)
    similarity[valid] = (
        np.einsum("ij,ij->i", first[valid], second[valid]) / denominator[valid]
    )
    raw_distance = 1.0 - similarity
    nonfinite = int(np.sum(~np.isfinite(raw_distance)))
    out_of_range = int(
        np.sum((raw_distance < -1e-10) | (raw_distance > 2.0 + 1e-10))
    )
    logger.log(f"{label}: invalid_denominator={invalid_denominator}")
    logger.log(f"{label}: NaN_or_Inf={nonfinite}")
    logger.log(f"{label}: outside_[0,2]={out_of_range}")
    if invalid_denominator or nonfinite or out_of_range:
        errors.append(
            f"{label}: invalid_denominator={invalid_denominator}, nonfinite={nonfinite}, outside={out_of_range}"
        )
    distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    logger.log(
        f"{label}: {format_statistics(describe_values(distance[np.isfinite(distance)]))}"
    )
    return distance


def spatial_summary(
    frame: pd.DataFrame,
    city: str,
    logger: AuditLogger,
    warnings: list[str],
) -> dict[str, object]:
    logger.section(f"11. SPATIAL BLOCK AUDIT — {city}")
    total_blocks = int(frame["block_id"].nunique())
    logger.log(f"total_blocks: {total_blocks}")
    summary: dict[str, object] = {
        "city": city,
        "total_samples": int(len(frame)),
        "total_blocks": total_blocks,
    }
    for class_value in EXPECTED_CLASSES:
        subset = frame.loc[frame["class"] == class_value]
        class_blocks = int(subset["block_id"].nunique())
        largest_share = float(subset.groupby("block_id").size().max() / len(subset))
        logger.log(
            f"Class {class_value}: samples={len(subset)}, blocks={class_blocks}, "
            f"largest_single_block_share={largest_share:.10f} ({largest_share:.4%})"
        )
        summary[f"class{class_value}_samples"] = int(len(subset))
        summary[f"class{class_value}_blocks"] = class_blocks
        summary[f"class{class_value}_largest_block_share"] = largest_share
        if class_blocks < 50:
            warnings.append(
                f"{city} Class {class_value} uses only {class_blocks} spatial blocks (<50 screening threshold)"
            )
        if largest_share > 0.10:
            warnings.append(
                f"{city} Class {class_value} largest block share is {largest_share:.2%} (>10% screening threshold)"
            )

    class2_blocks = int(summary["class2_blocks"])
    class2_share = float(summary["class2_largest_block_share"])
    logger.log(
        "Class 2 reference context: Shenzhen approximately 190 blocks; "
        "Kunming approximately 704 blocks."
    )
    if class2_blocks < 50 or class2_share > 0.10:
        warnings.append(
            f"{city} Class 2 spatial concentration requires attention: blocks={class2_blocks}, "
            f"largest share={class2_share:.2%}"
        )

    block_sizes = frame.groupby("block_id").size().astype(float)
    block_stats = describe_values(block_sizes)
    logger.log(f"samples_per_block: {format_statistics(block_stats)}")
    for key, value in block_stats.items():
        summary[f"samples_per_block_{key}"] = value
    summary["AEF_feature_count"] = AEF_FEATURE_COUNT
    summary["final_feature_count"] = FINAL_FEATURE_COUNT
    return summary


def build_city_dataset(
    city: str,
    yearly: dict[int, pd.DataFrame],
    bands: dict[int, list[str]],
    logger: AuditLogger,
    errors: list[str],
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger.section(f"8–12. MASTER DATASET CONSTRUCTION — {city}")
    base_ids = yearly[2020]["sample_id"].tolist()
    base_set = set(base_ids)
    for year in [2022, 2024]:
        if set(yearly[year]["sample_id"].tolist()) != base_set:
            raise RuntimeError(
                f"{city}: sample IDs are not identical across years; master construction stopped"
            )

    first = yearly[2020].set_index("sample_id", drop=False).loc[base_ids]
    master = first[["sample_id", "city", "class", "longitude", "latitude"]].reset_index(
        drop=True
    )
    for year in YEARS:
        annual = yearly[year].set_index("sample_id").loc[base_ids]
        master = pd.concat(
            [master, annual[bands[year]].reset_index(drop=True)], axis=1
        )

    annual_arrays = {
        year: master[bands[year]].to_numpy(dtype=np.float64, copy=False)
        for year in YEARS
    }
    logger.section(f"9. TEMPORAL COSINE FEATURES — {city}")
    master["D_20_22"] = cosine_distance_rows(
        annual_arrays[2020], annual_arrays[2022], f"{city} D_20_22", logger, errors
    )
    master["D_22_24"] = cosine_distance_rows(
        annual_arrays[2022], annual_arrays[2024], f"{city} D_22_24", logger, errors
    )
    master["D_20_24"] = cosine_distance_rows(
        annual_arrays[2020], annual_arrays[2024], f"{city} D_20_24", logger, errors
    )

    logger.section(f"10. UTM AND 2 KM BLOCKS — {city}")
    crs = NEW_CITY_CRS[city]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    utm_x, utm_y = transformer.transform(
        master["longitude"].to_numpy(dtype=float),
        master["latitude"].to_numpy(dtype=float),
    )
    master["utm_x"] = np.asarray(utm_x, dtype=np.float64)
    master["utm_y"] = np.asarray(utm_y, dtype=np.float64)
    master["block_x"] = np.floor(master["utm_x"] / BLOCK_SIZE_M).astype("int64")
    master["block_y"] = np.floor(master["utm_y"] / BLOCK_SIZE_M).astype("int64")
    master["block_id"] = (
        city
        + "_"
        + master["block_x"].astype(str)
        + "_"
        + master["block_y"].astype(str)
    )
    logger.log(f"source_crs: EPSG:4326")
    logger.log(f"target_crs: {crs}")
    logger.log(f"block_size_m: {BLOCK_SIZE_M}")

    aef_features = [column for year in YEARS for column in bands[year]]
    final_features = aef_features + TEMPORAL_FEATURES
    logger.section(f"12. FINAL FEATURE VALIDATION — {city}")
    logger.log(f"AEF_feature_count: {len(aef_features)}")
    logger.log(f"final_feature_count: {len(final_features)}")
    forbidden = sorted(FORBIDDEN_MODEL_FIELDS.intersection(final_features))
    logger.log(f"forbidden_fields_in_feature_list: {forbidden}")
    if len(aef_features) != AEF_FEATURE_COUNT:
        errors.append(f"{city}: AEF feature count {len(aef_features)} != 192")
    if len(final_features) != FINAL_FEATURE_COUNT:
        errors.append(f"{city}: final feature count {len(final_features)} != 195")
    if forbidden:
        errors.append(f"{city}: forbidden metadata in model features {forbidden}")

    master["sample_id"] = master["sample_id"].astype("string")
    master["city"] = master["city"].astype(str)
    master["class"] = master["class"].astype("int64")
    master["longitude"] = master["longitude"].astype("float64")
    master["latitude"] = master["latitude"].astype("float64")
    for column in final_features:
        master[column] = master[column].astype("float64")
    master = master[METADATA_COLUMNS + final_features]

    nan_count = int(master.isna().sum().sum())
    numeric = master.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    inf_count = int(np.isinf(numeric).sum())
    logger.log(f"master_shape: {master.shape}")
    logger.log(f"master_NaN_count: {nan_count}")
    logger.log(f"master_Inf_count: {inf_count}")
    if nan_count or inf_count:
        errors.append(f"{city}: master NaN={nan_count}, Inf={inf_count}")

    summary = spatial_summary(master, city, logger, warnings)
    return master, summary


def write_and_verify_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    table = pq.read_table(path)
    if table.num_rows != len(frame) or table.num_columns != len(frame.columns):
        raise RuntimeError(
            f"Parquet verification failed for {path}: "
            f"expected {frame.shape}, got {(table.num_rows, table.num_columns)}"
        )
    if table.column_names != list(frame.columns):
        raise RuntimeError(f"Parquet column order verification failed for {path}")


def dtype_signature(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return [(column, str(dtype)) for column, dtype in frame.dtypes.items()]


def arrow_signature(path: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(path)
    return [(field.name, str(field.type)) for field in schema]


def summary_from_processed(frame: pd.DataFrame, city: str) -> dict[str, object]:
    result: dict[str, object] = {
        "city": city,
        "total_samples": int(len(frame)),
        "total_blocks": int(frame["block_id"].nunique()),
    }
    for class_value in EXPECTED_CLASSES:
        subset = frame.loc[frame["class"] == class_value]
        result[f"class{class_value}_samples"] = int(len(subset))
        result[f"class{class_value}_blocks"] = int(subset["block_id"].nunique())
        result[f"class{class_value}_largest_block_share"] = float(
            subset.groupby("block_id").size().max() / len(subset)
        )
    result["AEF_feature_count"] = AEF_FEATURE_COUNT
    result["final_feature_count"] = FINAL_FEATURE_COUNT
    return result


def temporal_summary(frame: pd.DataFrame, city: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for feature in TEMPORAL_FEATURES:
        values = frame[feature].astype(float)
        records.append(
            {
                "city": city,
                "temporal_feature": feature,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "25%": float(values.quantile(0.25)),
                "75%": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return records


def validate_five_city_schema(
    paths: dict[str, Path],
    logger: AuditLogger,
    errors: list[str],
) -> dict[str, pd.DataFrame]:
    logger.section("14. FIVE-CITY SCHEMA COMPATIBILITY")
    frames: dict[str, pd.DataFrame] = {}
    reference_columns: list[str] | None = None
    reference_dtypes: list[tuple[str, str]] | None = None
    reference_arrow: list[tuple[str, str]] | None = None

    expected_aef = [
        f"A{index:02d}_{year}"
        for year in YEARS
        for index in range(EMBEDDING_DIMENSIONS)
    ]
    expected_columns = METADATA_COLUMNS + expected_aef + TEMPORAL_FEATURES

    for city in ALL_CITIES:
        path = paths[city]
        if not path.is_file():
            errors.append(f"missing processed dataset: {path}")
            continue
        frame = pd.read_parquet(path, engine="pyarrow")
        frames[city] = frame
        columns = list(frame.columns)
        dtypes = dtype_signature(frame)
        arrow_types = arrow_signature(path)
        aef_count = sum(
            bool(re.fullmatch(r"A\d{2}_\d{4}", column)) for column in columns
        )
        final_count = aef_count + sum(
            column in TEMPORAL_FEATURES for column in columns
        )
        logger.log(
            f"{city}: shape={frame.shape}, columns_match_expected={columns == expected_columns}, "
            f"AEF_features={aef_count}, final_features={final_count}"
        )
        if columns != expected_columns:
            errors.append(f"{city}: processed columns do not match expected 205-column schema")
        if reference_columns is None:
            reference_columns = columns
            reference_dtypes = dtypes
            reference_arrow = arrow_types
        else:
            if columns != reference_columns:
                errors.append(f"{city}: column names/order differ from Shenzhen")
            if dtypes != reference_dtypes:
                errors.append(f"{city}: pandas dtypes differ from Shenzhen")
            if arrow_types != reference_arrow:
                errors.append(f"{city}: Arrow field types differ from Shenzhen")

    logger.log(f"schema_compatible: {not errors}")
    return frames


def validate_csv_output(path: Path, expected_rows: int, expected_columns: list[str]) -> None:
    check = pd.read_csv(path)
    if len(check) != expected_rows:
        raise RuntimeError(
            f"CSV verification failed for {path}: rows={len(check)}, expected={expected_rows}"
        )
    if list(check.columns) != expected_columns:
        raise RuntimeError(f"CSV verification failed for {path}: unexpected columns")
    if check.isna().any().any():
        raise RuntimeError(f"CSV verification failed for {path}: contains blank values")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(AUDIT_PATH)
    errors: list[str] = []
    warnings: list[str] = []
    norm_records: list[dict[str, object]] = []

    try:
        logger.section("STAGE 06C — MULTI-CITY LOCAL INTEGRATION AND DATASET AUDIT")
        logger.log(f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.log(f"project_root: {PROJECT_ROOT}")
        logger.log(f"download_directory: {DOWNLOAD_DIR}")
        logger.log(f"raw_copy_directory: {RAW_MULTICITY_DIR}")
        logger.log("models_trained: none")
        logger.log("transfer_matrix_run: no")

        logger.section("PYTHON AND PACKAGE VERSIONS")
        logger.log(f"python_version: {platform.python_version()}")
        logger.log(f"python_full: {sys.version.replace(chr(10), ' ')}")
        logger.log(f"python_executable: {sys.executable}")
        for package in ["pandas", "numpy", "pyarrow", "pyproj"]:
            logger.log(f"package_{package}: {package_version(package)}")

        source_files = resolve_source_files(logger)
        raw_files, copy_records = copy_raw_files(source_files, logger)
        logger.log(f"verified_raw_copy_count: {len(copy_records)}")

        yearly_data: dict[str, dict[int, pd.DataFrame]] = {
            city: {} for city in NEW_CITY_CRS
        }
        yearly_bands: dict[str, dict[int, list[str]]] = {
            city: {} for city in NEW_CITY_CRS
        }
        for (city, year), path in raw_files.items():
            frame, bands = audit_csv(
                path,
                city,
                year,
                logger,
                errors,
                warnings,
                norm_records,
            )
            yearly_data[city][year] = frame
            yearly_bands[city][year] = bands

        if errors:
            raise ValueError("critical raw CSV audit errors detected")

        for city in NEW_CITY_CRS:
            audit_alignment(city, yearly_data[city], logger, errors)
        if errors:
            raise ValueError("three-year alignment or metadata consistency failed")

        new_frames: dict[str, pd.DataFrame] = {}
        for city in NEW_CITY_CRS:
            master, _ = build_city_dataset(
                city,
                yearly_data[city],
                yearly_bands[city],
                logger,
                errors,
                warnings,
            )
            new_frames[city] = master
        if errors:
            raise ValueError("new-city master dataset validation failed")

        logger.section("13. NEW-CITY PARQUET OUTPUTS")
        processed_paths = {
            city: PROCESSED_DIR / f"{city}_AEF_195D.parquet"
            for city in ALL_CITIES
        }
        for city, frame in new_frames.items():
            path = processed_paths[city]
            write_and_verify_parquet(frame, path)
            logger.log(f"{city}: {path}, shape={frame.shape}, verified=True")

        schema_errors: list[str] = []
        five_frames = validate_five_city_schema(
            processed_paths, logger, schema_errors
        )
        errors.extend(schema_errors)
        if errors:
            raise ValueError("five-city schema compatibility failed")

        logger.section("15. FIVE-CITY COMBINED DATASET")
        combined = pd.concat(
            [five_frames[city] for city in ALL_CITIES],
            axis=0,
            ignore_index=True,
        )
        combined["sample_id"] = combined["sample_id"].astype("string")
        global_unique = bool(combined["sample_id"].is_unique)
        city_counts = combined["city"].value_counts().sort_index()
        class_counts = combined["class"].value_counts().sort_index()
        city_class_counts = combined.groupby(["city", "class"]).size()
        combined_nan = int(combined.isna().sum().sum())
        combined_numeric = combined.select_dtypes(include=[np.number]).to_numpy(
            dtype=np.float64, copy=False
        )
        combined_inf = int(np.isinf(combined_numeric).sum())
        logger.log(f"FiveCity_shape: {combined.shape}")
        logger.log(f"global_sample_id_unique: {global_unique}")
        logger.log(f"city_distribution: {city_counts.to_dict()}")
        logger.log(f"class_distribution: {class_counts.to_dict()}")
        logger.log(f"city_by_class_distribution: {city_class_counts.to_dict()}")
        logger.log(f"NaN_count: {combined_nan}")
        logger.log(f"Inf_count: {combined_inf}")
        if combined.shape != (90_000, 205):
            errors.append(f"FiveCity shape {combined.shape} != (90000, 205)")
        if not global_unique:
            errors.append("FiveCity sample_id values are not globally unique")
        if combined_nan or combined_inf:
            errors.append(f"FiveCity contains NaN={combined_nan}, Inf={combined_inf}")
        for city in ALL_CITIES:
            if int(city_counts.get(city, 0)) != EXPECTED_ROWS_PER_FILE:
                errors.append(
                    f"FiveCity {city} count {int(city_counts.get(city, 0))} != 18000"
                )
            for class_value in EXPECTED_CLASSES:
                count = int(city_class_counts.get((city, class_value), 0))
                if count != EXPECTED_SAMPLES_PER_CLASS:
                    errors.append(
                        f"FiveCity {city} Class {class_value} count {count} != 6000"
                    )
        if errors:
            raise ValueError("FiveCity combined dataset validation failed")

        five_city_path = PROCESSED_DIR / "FiveCity_AEF_195D.parquet"
        write_and_verify_parquet(combined, five_city_path)
        logger.log(f"verified_output: {five_city_path}")
        logger.log(f"verified_shape: {combined.shape}")

        logger.section("16. FIVE-CITY DATASET SUMMARY")
        summary_columns = [
            "city",
            "total_samples",
            "class0_samples",
            "class1_samples",
            "class2_samples",
            "total_blocks",
            "class0_blocks",
            "class1_blocks",
            "class2_blocks",
            "class0_largest_block_share",
            "class1_largest_block_share",
            "class2_largest_block_share",
            "AEF_feature_count",
            "final_feature_count",
        ]
        summary_rows = [
            summary_from_processed(five_frames[city], city) for city in ALL_CITIES
        ]
        summary_frame = pd.DataFrame(summary_rows)[summary_columns]
        summary_path = TABLES_DIR / "multicity_dataset_summary.csv"
        summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
        validate_csv_output(summary_path, 5, summary_columns)
        logger.log(summary_frame.to_string(index=False))
        logger.log(f"verified_output: {summary_path}")

        logger.section("17. FIVE-CITY TEMPORAL FEATURE SUMMARY")
        temporal_columns = [
            "city",
            "temporal_feature",
            "mean",
            "std",
            "median",
            "25%",
            "75%",
            "min",
            "max",
        ]
        temporal_rows: list[dict[str, object]] = []
        for city in ALL_CITIES:
            temporal_rows.extend(temporal_summary(five_frames[city], city))
        temporal_frame = pd.DataFrame(temporal_rows)[temporal_columns]
        temporal_path = TABLES_DIR / "multicity_temporal_summary.csv"
        temporal_frame.to_csv(temporal_path, index=False, encoding="utf-8-sig")
        validate_csv_output(temporal_path, 15, temporal_columns)
        logger.log(temporal_frame.to_string(index=False))
        logger.log(f"verified_output: {temporal_path}")

        logger.section("AEF NORM SUMMARY")
        logger.log(pd.DataFrame(norm_records).to_string(index=False))

        logger.section("WARNINGS")
        if warnings:
            for warning in warnings:
                logger.log(f"WARNING: {warning}")
        else:
            logger.log("None")

        logger.section("OUTPUTS")
        for city in NEW_CITY_CRS:
            logger.log(processed_paths[city])
        logger.log(five_city_path)
        logger.log(summary_path)
        logger.log(temporal_path)
        logger.log(AUDIT_PATH)

        logger.section("18. FINAL STATUS")
        logger.log("final_status: SUCCESS")
        return 0

    except Exception as exc:
        logger.section("ERRORS")
        for error in errors:
            logger.log(f"ERROR: {error}")
        logger.log(f"FATAL: {type(exc).__name__}: {exc}")
        logger.log(traceback.format_exc())
        logger.section("18. FINAL STATUS")
        logger.log("final_status: FAILED")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

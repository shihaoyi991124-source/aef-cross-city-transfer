"""Stage 09A: external-target dataset integration and quality-control audit.

This data-only stage prepares Chengdu and Hangzhou as isolated external targets.
It never imports, trains, evaluates, or applies a transfer model and never reads
target outcomes beyond the frozen class labels required for data-integrity checks.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "imports" / "external"
RAW_EXTERNAL_DIR = ROOT / "data" / "raw" / "external"
PROCESSED_EXTERNAL_DIR = ROOT / "data" / "processed" / "external"
FIVE_CITY_PATH = ROOT / "data" / "processed" / "FiveCity_AEF_195D.parquet"
TABLES_DIR = ROOT / "outputs" / "tables"
LOGS_DIR = ROOT / "outputs" / "logs"

SUMMARY_PATH = TABLES_DIR / "stage09a_external_dataset_summary.csv"
TEMPORAL_PATH = TABLES_DIR / "stage09a_external_temporal_summary.csv"
NORM_PATH = TABLES_DIR / "stage09a_external_norm_summary.csv"
MANIFEST_PATH = TABLES_DIR / "stage09_external_manifest.csv"
REPORT_PATH = LOGS_DIR / "stage09a_external_dataset_audit.txt"

CITY_CRS = {"Chengdu": "EPSG:32648", "Hangzhou": "EPSG:32651"}
CITIES = list(CITY_CRS)
YEARS = [2020, 2022, 2024]
CLASSES = [0, 1, 2]
EXPECTED_ROWS = 18_000
EXPECTED_CLASS_ROWS = 6_000
DIMENSIONS = 64
AEF_COUNT = 192
FINAL_FEATURE_COUNT = 195
COORDINATE_TOLERANCE = 1e-8
BLOCK_SIZE_M = 2_000
TEMPORAL_FEATURES = ["D_20_22", "D_22_24", "D_20_24"]
DEVELOPMENT_METADATA = [
    "sample_id", "city", "class", "longitude", "latitude",
    "utm_x", "utm_y", "block_x", "block_y", "block_id",
]
EXTERNAL_METADATA = [
    "sample_id", "city", "class", "external_target", "longitude", "latitude",
    "utm_x", "utm_y", "block_x", "block_y", "block_id",
]
FORBIDDEN_FEATURES = set(EXTERNAL_METADATA)

OUTCOME_ACCESS = {
    "transfer_model_trained": False,
    "same_city_model_trained": False,
    "cross_city_transfer_evaluated": False,
    "target_macro_f1_computed": False,
    "target_expansion_f1_computed": False,
    "target_normalized_gap_computed": False,
    "stage08_risk_prediction_applied": False,
    "multi_source_external_model_run": False,
}


class AuditLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def log(self, value: object = "") -> None:
        text = str(value)
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def section(self, number: int, title: str) -> None:
        self.log()
        self.log(f"{number}. {title}")
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


def describe(values: Iterable[float]) -> dict[str, float]:
    stats = pd.Series(values, dtype="float64").describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "mean": float(stats["mean"]), "std": float(stats["std"]),
        "min": float(stats["min"]), "25%": float(stats["25%"]),
        "median": float(stats["50%"]), "75%": float(stats["75%"]),
        "max": float(stats["max"]),
    }


def normalized_stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_sources() -> dict[tuple[str, int], Path]:
    if not DOWNLOAD_DIR.is_dir():
        raise FileNotFoundError(f"External download directory not found: {DOWNLOAD_DIR}")
    available = sorted(DOWNLOAD_DIR.glob("*.csv"))
    resolved: dict[tuple[str, int], Path] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for city in CITIES:
        for year in YEARS:
            expected = f"AEF_{city}_{year}.csv"
            exact = DOWNLOAD_DIR / expected
            if exact.is_file():
                resolved[(city, year)] = exact
                continue
            key = normalized_stem(Path(expected).stem)
            matches = [path for path in available if normalized_stem(path.stem) == key]
            if len(matches) == 1:
                resolved[(city, year)] = matches[0]
            elif not matches:
                missing.append(expected)
            else:
                ambiguous.append(f"{expected}: {[p.name for p in matches]}")
    if missing or ambiguous or len(resolved) != 6:
        raise FileNotFoundError(f"CSV resolution failed; missing={missing}, ambiguous={ambiguous}")
    return resolved


def copy_and_verify(
    sources: dict[tuple[str, int], Path],
) -> tuple[dict[tuple[str, int], Path], pd.DataFrame]:
    RAW_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    copies: dict[tuple[str, int], Path] = {}
    records: list[dict[str, Any]] = []
    for key, source in sources.items():
        destination = RAW_EXTERNAL_DIR / source.name
        source_size = source.stat().st_size
        source_hash = sha256_file(source)
        if destination.exists():
            if destination.stat().st_size != source_size or sha256_file(destination) != source_hash:
                raise RuntimeError(f"Existing raw external copy differs and was not overwritten: {destination}")
            action = "existing copy verified"
        else:
            shutil.copy2(source, destination)
            action = "copied"
        destination_size = destination.stat().st_size
        destination_hash = sha256_file(destination)
        identical = source_size == destination_size and source_hash == destination_hash
        if not identical:
            raise RuntimeError(f"Raw copy verification failed: {source.name}")
        copies[key] = destination
        records.append({
            "city": key[0], "year": key[1], "filename": source.name,
            "source_path": str(source), "copied_path": str(destination), "copy_action": action,
            "source_size_bytes": source_size, "copied_size_bytes": destination_size,
            "source_sha256": source_hash, "copied_sha256": destination_hash,
            "size_and_sha256_identical": identical,
        })
    return copies, pd.DataFrame(records)


def detect_bands(columns: list[str], year: int) -> tuple[list[str], dict[str, str], list[str]]:
    found: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    wrong_year: list[str] = []
    pattern = re.compile(r"^A(\d{2})(?:_(\d{4}))?$")
    for raw in columns:
        match = pattern.fullmatch(str(raw))
        if not match:
            continue
        index = int(match.group(1))
        suffix = match.group(2)
        if suffix is not None and int(suffix) != year:
            wrong_year.append(str(raw))
            continue
        if 0 <= index < DIMENSIONS:
            if index in found:
                duplicates.setdefault(index, [found[index]]).append(str(raw))
            else:
                found[index] = str(raw)
    if wrong_year:
        raise ValueError(f"AEF columns have wrong year suffix: {wrong_year}")
    if duplicates:
        raise ValueError(f"Duplicate/ambiguous AEF indices: {duplicates}")
    missing = sorted(set(range(DIMENSIONS)) - set(found))
    if missing:
        raise ValueError(f"Missing AEF band indices: {missing}")
    source_order = [found[i] for i in range(DIMENSIONS)]
    canonical = [f"A{i:02d}_{year}" for i in range(DIMENSIONS)]
    file_band_order = [c for c in columns if c in set(source_order)]
    if file_band_order != source_order:
        raise ValueError("AEF bands are present but not in A00…A63 order")
    rename = dict(zip(source_order, canonical))
    noncanonical = [source for source, target in rename.items() if source != target]
    return source_order, rename, noncanonical


def audit_raw_file(
    path: Path,
    city: str,
    year: int,
    errors: list[str],
    warnings: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    header = read_header(path)
    duplicate_columns = sorted({name for name in header if header.count(name) > 1})
    if duplicate_columns:
        errors.append(f"{path.name}: duplicate header columns {duplicate_columns}")
    frame = pd.read_csv(path, low_memory=False)
    try:
        source_bands, rename, noncanonical = detect_bands(header, year)
    except ValueError as exc:
        errors.append(f"{path.name}: {exc}")
        return frame, [], {}, {}
    canonical = [f"A{i:02d}_{year}" for i in range(DIMENSIONS)]
    if noncanonical:
        warnings.append(f"{path.name}: safely canonicalized {len(noncanonical)} AEF band names")
    frame = frame.rename(columns=rename)
    required = ["sample_id", "city", "class", "external_target", "longitude", "latitude"]
    missing_metadata = [column for column in required if column not in frame]
    if missing_metadata:
        errors.append(f"{path.name}: missing metadata {missing_metadata}")
        return frame, canonical, {}, {}

    sample_null = int(frame.sample_id.isna().sum())
    sample_duplicate_rows = int(frame.sample_id.duplicated(keep=False).sum())
    duplicate_full_rows = int(frame.duplicated().sum())
    observed_city = sorted(frame.city.dropna().astype(str).unique().tolist())
    numeric_class = pd.to_numeric(frame["class"], errors="coerce")
    class_conversion_failures = int((numeric_class.isna() & frame["class"].notna()).sum())
    frame["class"] = numeric_class
    class_counts = {str(c): int((numeric_class == c).sum()) for c in CLASSES}
    observed_classes = sorted(numeric_class.dropna().astype(int).unique().tolist())
    external_values = sorted(frame.external_target.dropna().astype(str).str.strip().str.lower().unique().tolist())
    external_valid = bool(external_values) and set(external_values).issubset({"1", "true"})

    for coordinate in ("longitude", "latitude"):
        original = frame[coordinate]
        converted = pd.to_numeric(original, errors="coerce")
        failures = int((converted.isna() & original.notna()).sum())
        if failures:
            errors.append(f"{path.name}: {failures} non-numeric {coordinate} values")
        frame[coordinate] = converted.astype("float64")
    aef_nonnumeric_before = [c for c in canonical if not pd.api.types.is_numeric_dtype(frame[c])]
    for column in canonical:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        failures = int((converted.isna() & original.notna()).sum())
        if failures:
            errors.append(f"{path.name}: {column} has {failures} non-numeric values")
        frame[column] = converted.astype("float64")

    nan_count = int(frame.isna().sum().sum())
    numeric_values = frame.select_dtypes(include=[np.number]).to_numpy(np.float64, copy=False)
    inf_count = int(np.isinf(numeric_values).sum())
    longitude_invalid = int((~frame.longitude.between(-180, 180)).sum())
    latitude_invalid = int((~frame.latitude.between(-90, 90)).sum())

    if len(frame) != EXPECTED_ROWS:
        errors.append(f"{path.name}: rows={len(frame)} != {EXPECTED_ROWS}")
    if sample_null or sample_duplicate_rows:
        errors.append(f"{path.name}: sample_id null={sample_null}, duplicate rows={sample_duplicate_rows}")
    if observed_city != [city]:
        errors.append(f"{path.name}: observed city values {observed_city}, expected {city}")
    if class_conversion_failures or observed_classes != CLASSES:
        errors.append(f"{path.name}: invalid classes {observed_classes}")
    for cls in CLASSES:
        if class_counts[str(cls)] != EXPECTED_CLASS_ROWS:
            errors.append(f"{path.name}: Class {cls} rows={class_counts[str(cls)]} != {EXPECTED_CLASS_ROWS}")
    if not external_valid or frame.external_target.isna().any():
        errors.append(f"{path.name}: external_target values invalid: {external_values}")
    if nan_count or inf_count:
        errors.append(f"{path.name}: NaN={nan_count}, Inf={inf_count}")
    if duplicate_full_rows:
        errors.append(f"{path.name}: duplicate full rows={duplicate_full_rows}")
    if longitude_invalid or latitude_invalid:
        errors.append(f"{path.name}: invalid coordinates lon={longitude_invalid}, lat={latitude_invalid}")
    if aef_nonnumeric_before:
        errors.append(f"{path.name}: non-numeric AEF dtypes before coercion: {aef_nonnumeric_before}")

    vectors = frame[canonical].to_numpy(np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    finite = norms[np.isfinite(norms)]
    norm_stats = describe(finite) if len(finite) else {k: np.nan for k in ["mean", "std", "min", "25%", "median", "75%", "max"]}
    norm_record = {
        "city": city, "year": year, **norm_stats,
        "percent_abs_norm_minus_1_lt_0.01": float(np.mean(np.abs(finite - 1.0) < 0.01) * 100) if len(finite) else np.nan,
        "zero_norm_count": int(np.sum(norms == 0)),
        "NaN_norm_count": int(np.sum(np.isnan(norms))),
        "Inf_norm_count": int(np.sum(np.isinf(norms))),
    }
    if norm_record["zero_norm_count"] or norm_record["NaN_norm_count"] or norm_record["Inf_norm_count"]:
        errors.append(f"{path.name}: invalid AEF norms")
    if np.isfinite(norm_record["percent_abs_norm_minus_1_lt_0.01"]) and norm_record["percent_abs_norm_minus_1_lt_0.01"] < 99.0:
        warnings.append(f"{path.name}: only {norm_record['percent_abs_norm_minus_1_lt_0.01']:.2f}% of norms are within 0.01 of 1")

    raw_record = {
        "filename": path.name, "city": city, "year": year,
        "rows": len(frame), "columns": len(header), "AEF_band_count": len(canonical),
        "AEF_band_first": canonical[0], "AEF_band_last": canonical[-1],
        "AEF_band_order_valid": True, "duplicate_column_count": len(duplicate_columns),
        "sample_id_null_count": sample_null, "sample_id_duplicate_rows": sample_duplicate_rows,
        "sample_id_unique": sample_null == 0 and sample_duplicate_rows == 0,
        "city_values": " | ".join(observed_city), "class_distribution": json.dumps(class_counts, sort_keys=True),
        "external_target_values": " | ".join(external_values),
        "longitude_min": float(frame.longitude.min()), "longitude_max": float(frame.longitude.max()),
        "latitude_min": float(frame.latitude.min()), "latitude_max": float(frame.latitude.max()),
        "NaN_count_all_columns": nan_count, "Inf_count_numeric_columns": inf_count,
        "duplicate_full_rows": duplicate_full_rows,
        "AEF_all_numeric_dtype": len(aef_nonnumeric_before) == 0,
    }
    retained = required + canonical
    output = frame[retained].copy()
    output["sample_id"] = output.sample_id.astype("string")
    output["city"] = output.city.astype(str)
    output["class"] = output["class"].astype("int64")
    output["external_target"] = True
    return output, canonical, raw_record, norm_record


def audit_alignment(city: str, yearly: dict[int, pd.DataFrame], errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left_year, right_year in [(2020, 2022), (2020, 2024), (2022, 2024)]:
        left = yearly[left_year].set_index("sample_id", drop=False)
        right = yearly[right_year].set_index("sample_id", drop=False)
        left_ids, right_ids = pd.Index(left.index), pd.Index(right.index)
        common = left_ids.intersection(right_ids, sort=False)
        left_only = left_ids.difference(right_ids, sort=False)
        right_only = right_ids.difference(left_ids, sort=False)
        l = left.loc[common]
        r = right.loc[common]
        class_mismatch = int(np.sum(l["class"].to_numpy() != r["class"].to_numpy()))
        city_mismatch = int(np.sum(l.city.astype(str).to_numpy() != r.city.astype(str).to_numpy()))
        external_mismatch = int(np.sum(l.external_target.to_numpy() != r.external_target.to_numpy()))
        lon_diff = np.abs(l.longitude.to_numpy(float) - r.longitude.to_numpy(float))
        lat_diff = np.abs(l.latitude.to_numpy(float) - r.latitude.to_numpy(float))
        lon_mismatch = int(np.sum(lon_diff > COORDINATE_TOLERANCE))
        lat_mismatch = int(np.sum(lat_diff > COORDINATE_TOLERANCE))
        record = {
            "city": city, "comparison": f"{left_year} vs {right_year}",
            "left_count": len(left_ids), "right_count": len(right_ids),
            "intersection": len(common), "missing_from_right": len(left_only),
            "extra_in_right": len(right_only), "class_mismatch": class_mismatch,
            "city_mismatch": city_mismatch, "external_target_mismatch": external_mismatch,
            "longitude_mismatch": lon_mismatch, "latitude_mismatch": lat_mismatch,
            "maximum_longitude_difference": float(lon_diff.max(initial=0)),
            "maximum_latitude_difference": float(lat_diff.max(initial=0)),
            "fully_aligned": not any([len(left_only), len(right_only), class_mismatch, city_mismatch, external_mismatch, lon_mismatch, lat_mismatch]),
        }
        if not record["fully_aligned"]:
            errors.append(f"{city} {left_year} vs {right_year}: alignment/metadata mismatch")
        records.append(record)
    return records


def cosine_distance(first: np.ndarray, second: np.ndarray, label: str, errors: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = (denominator > 0) & np.isfinite(denominator)
    similarity = np.full(len(first), np.nan, dtype=np.float64)
    similarity[valid] = np.einsum("ij,ij->i", first[valid], second[valid]) / denominator[valid]
    raw_distance = 1.0 - similarity
    invalid_denominator = int(np.sum(~valid))
    nonfinite = int(np.sum(~np.isfinite(raw_distance)))
    outside = int(np.sum((raw_distance < -1e-10) | (raw_distance > 2.0 + 1e-10)))
    if invalid_denominator or nonfinite or outside:
        errors.append(f"{label}: invalid denominator={invalid_denominator}, nonfinite={nonfinite}, outside={outside}")
    distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    stats = describe(distance[np.isfinite(distance)])
    record = {
        "temporal_feature": label, "invalid_denominator_count": invalid_denominator,
        "NaN_or_Inf_count": nonfinite, "outside_0_2_count": outside,
        **stats,
    }
    return distance, record


def build_external_dataset(
    city: str,
    yearly: dict[int, pd.DataFrame],
    bands: dict[int, list[str]],
    errors: list[str],
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    base_ids = yearly[2020].sample_id.tolist()
    if any(set(yearly[year].sample_id) != set(base_ids) for year in YEARS[1:]):
        raise RuntimeError(f"{city}: sample IDs are not identical across years")
    first = yearly[2020].set_index("sample_id", drop=False).loc[base_ids]
    master = first[["sample_id", "city", "class", "external_target", "longitude", "latitude"]].reset_index(drop=True)
    for year in YEARS:
        annual = yearly[year].set_index("sample_id").loc[base_ids]
        master = pd.concat([master, annual[bands[year]].reset_index(drop=True)], axis=1)

    arrays = {year: master[bands[year]].to_numpy(np.float64, copy=False) for year in YEARS}
    temporal_records: list[dict[str, Any]] = []
    for feature, left, right in [("D_20_22", 2020, 2022), ("D_22_24", 2022, 2024), ("D_20_24", 2020, 2024)]:
        values, record = cosine_distance(arrays[left], arrays[right], feature, errors)
        master[feature] = values
        temporal_records.append({"city": city, **record})

    transformer = Transformer.from_crs("EPSG:4326", CITY_CRS[city], always_xy=True)
    utm_x, utm_y = transformer.transform(master.longitude.to_numpy(float), master.latitude.to_numpy(float))
    master["utm_x"] = np.asarray(utm_x, dtype=np.float64)
    master["utm_y"] = np.asarray(utm_y, dtype=np.float64)
    master["block_x"] = np.floor(master.utm_x / BLOCK_SIZE_M).astype("int64")
    master["block_y"] = np.floor(master.utm_y / BLOCK_SIZE_M).astype("int64")
    master["block_id"] = city + "_" + master.block_x.astype(str) + "_" + master.block_y.astype(str)

    aef = [column for year in YEARS for column in bands[year]]
    final = aef + TEMPORAL_FEATURES
    if len(aef) != AEF_COUNT or len(set(aef)) != AEF_COUNT:
        errors.append(f"{city}: invalid AEF-192 feature definition")
    if len(final) != FINAL_FEATURE_COUNT or len(set(final)) != FINAL_FEATURE_COUNT:
        errors.append(f"{city}: invalid AEF-195 compatibility definition")
    forbidden = sorted(FORBIDDEN_FEATURES.intersection(final))
    if forbidden:
        errors.append(f"{city}: forbidden metadata in feature list {forbidden}")

    master["sample_id"] = master.sample_id.astype("string")
    master["city"] = master.city.astype(str)
    master["class"] = master["class"].astype("int64")
    master["external_target"] = master.external_target.astype(bool)
    for column in ["longitude", "latitude", "utm_x", "utm_y"] + final:
        master[column] = master[column].astype("float64")
    master["block_x"] = master.block_x.astype("int64")
    master["block_y"] = master.block_y.astype("int64")
    master["block_id"] = master.block_id.astype(str)
    master = master[EXTERNAL_METADATA + final]
    nan_count = int(master.isna().sum().sum())
    inf_count = int(np.isinf(master.select_dtypes(include=[np.number]).to_numpy(np.float64, copy=False)).sum())
    if nan_count or inf_count:
        errors.append(f"{city}: external master NaN={nan_count}, Inf={inf_count}")

    summary: dict[str, Any] = {
        "city": city, "total_samples": len(master), "total_blocks": master.block_id.nunique(),
        "AEF_feature_count": len(aef), "final_feature_count": len(final), "external_target": True,
    }
    for cls in CLASSES:
        subset = master.loc[master["class"] == cls]
        block_sizes = subset.groupby("block_id").size()
        summary[f"class{cls}_samples"] = len(subset)
        summary[f"class{cls}_blocks"] = subset.block_id.nunique()
        summary[f"class{cls}_largest_block_share"] = float(block_sizes.max() / len(subset))
    overall_block_sizes = master.groupby("block_id").size().astype(float)
    summary["samples_per_block_mean"] = float(overall_block_sizes.mean())
    summary["samples_per_block_std"] = float(overall_block_sizes.std(ddof=1))
    summary["samples_per_block_median"] = float(overall_block_sizes.median())
    summary["samples_per_block_max"] = float(overall_block_sizes.max())
    if int(summary["class2_blocks"]) < 50:
        warnings.append(f"{city} Class2 uses only {summary['class2_blocks']} spatial blocks (<50 screening threshold)")
    if float(summary["class2_largest_block_share"]) > 0.10:
        warnings.append(f"{city} Class2 largest-block share is {summary['class2_largest_block_share']:.2%} (>10% screening threshold)")
    return master, summary, temporal_records


def write_verify_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    table = pq.read_table(path)
    if (table.num_rows, table.num_columns) != frame.shape or table.column_names != list(frame.columns):
        raise RuntimeError(f"Parquet round-trip verification failed: {path}")


def schema_audit(external_paths: dict[str, Path], errors: list[str]) -> pd.DataFrame:
    development = pd.read_parquet(FIVE_CITY_PATH, engine="pyarrow")
    expected_aef = [f"A{i:02d}_{year}" for year in YEARS for i in range(DIMENSIONS)]
    if list(development.columns) != DEVELOPMENT_METADATA + expected_aef + TEMPORAL_FEATURES:
        errors.append("Development FiveCity schema differs from frozen 205-column expectation")
    records = []
    for city, path in external_paths.items():
        external = pd.read_parquet(path, engine="pyarrow")
        external_without_role = [c for c in external.columns if c != "external_target"]
        columns_match = external_without_role == list(development.columns)
        aef_match = list(external[expected_aef].columns) == list(development[expected_aef].columns)
        temporal_match = list(external[TEMPORAL_FEATURES].columns) == list(development[TEMPORAL_FEATURES].columns)
        common_dtype_match = all(str(external[c].dtype) == str(development[c].dtype) for c in development.columns)
        arrow_external = {field.name: str(field.type) for field in pq.read_schema(path)}
        arrow_development = {field.name: str(field.type) for field in pq.read_schema(FIVE_CITY_PATH)}
        arrow_common_match = all(arrow_external.get(c) == arrow_development.get(c) for c in development.columns)
        compatible = all([columns_match, aef_match, temporal_match, common_dtype_match, arrow_common_match])
        if not compatible:
            errors.append(f"{city}: external schema is not compatible with development FiveCity")
        records.append({
            "city": city, "external_shape": str(external.shape), "development_shape": str(development.shape),
            "external_target_extra_field_allowed": "external_target" in external.columns,
            "common_column_names_and_order_match": columns_match,
            "AEF_192_names_match": aef_match, "temporal_names_match": temporal_match,
            "pandas_common_dtypes_match": common_dtype_match,
            "arrow_common_dtypes_match": arrow_common_match, "schema_compatible": compatible,
        })
    return pd.DataFrame(records)


def save_csv(frame: pd.DataFrame, path: Path, expected_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")
    check = pd.read_csv(path)
    if len(check) != expected_rows or list(check.columns) != list(frame.columns):
        raise RuntimeError(f"CSV structural verification failed: {path}")
    if check.columns.duplicated().any() or check.isna().any().any():
        raise RuntimeError(f"CSV duplicate-column or blank-value verification failed: {path}")


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(REPORT_PATH)
    errors: list[str] = []
    warnings: list[str] = []
    five_city_hash_before = ""
    try:
        logger.log("Stage 09A Local Audit — External Target Dataset Integration and Quality Control")
        logger.log("=" * 88)
        logger.section(1, "EXECUTION INFORMATION")
        logger.log(f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.log(f"project_root: {ROOT}")
        logger.log(f"python_executable: {sys.executable}")
        logger.log(f"python_version: {platform.python_version()}")
        for package in ["pandas", "numpy", "pyproj", "pyarrow"]:
            logger.log(f"{package}_version: {package_version(package)}")

        logger.section(2, "EXTERNAL-VALIDATION RULES")
        logger.log("external_targets: Chengdu, Hangzhou")
        logger.log("dataset_role: EXTERNAL_TARGET")
        logger.log("development_dataset_membership: prohibited")
        logger.log("Stage07/08 refitting, threshold changes, source-selection development, tuning: prohibited")
        logger.log("This stage performs data integrity, alignment, feature construction, block construction, schema audit, and saving only.")
        if not FIVE_CITY_PATH.is_file():
            raise FileNotFoundError(FIVE_CITY_PATH)
        five_city_hash_before = sha256_file(FIVE_CITY_PATH)
        logger.log(f"FiveCity_SHA256_before: {five_city_hash_before}")

        sources = resolve_sources()
        logger.section(3, "SIX RAW INPUT PATHS")
        for key in sorted(sources):
            logger.log(f"{key[0]} {key[1]}: {sources[key]}")
        logger.log(f"six_csvs_found: {len(sources) == 6}")

        copies, copy_records = copy_and_verify(sources)
        logger.section(4, "RAW INPUT SHA256")
        logger.log(copy_records[["city", "year", "filename", "source_size_bytes", "source_sha256"]].to_string(index=False))
        logger.section(5, "COPIED RAW FILES SHA256")
        logger.log(copy_records[["city", "year", "copied_path", "copied_size_bytes", "copied_sha256", "size_and_sha256_identical"]].to_string(index=False))

        yearly: dict[str, dict[int, pd.DataFrame]] = {city: {} for city in CITIES}
        bands: dict[str, dict[int, list[str]]] = {city: {} for city in CITIES}
        raw_records: list[dict[str, Any]] = []
        norm_records: list[dict[str, Any]] = []
        for (city, year), path in copies.items():
            frame, year_bands, raw_record, norm_record = audit_raw_file(path, city, year, errors, warnings)
            yearly[city][year] = frame
            bands[city][year] = year_bands
            raw_records.append(raw_record)
            norm_records.append(norm_record)
        raw_audit = pd.DataFrame(raw_records).sort_values(["city", "year"])
        norm_frame = pd.DataFrame(norm_records).sort_values(["city", "year"])
        if errors:
            raise ValueError("Critical raw CSV audit errors detected")

        logger.section(6, "ROW COUNTS")
        logger.log(raw_audit[["filename", "city", "year", "rows", "columns", "AEF_band_count", "AEF_band_first", "AEF_band_last", "AEF_band_order_valid", "duplicate_column_count"]].to_string(index=False))
        logger.section(7, "CLASS DISTRIBUTIONS")
        logger.log(raw_audit[["filename", "class_distribution", "external_target_values"]].to_string(index=False))
        logger.section(8, "SAMPLE_ID UNIQUENESS")
        logger.log(raw_audit[["filename", "sample_id_null_count", "sample_id_duplicate_rows", "sample_id_unique", "duplicate_full_rows"]].to_string(index=False))

        alignment_records: list[dict[str, Any]] = []
        for city in CITIES:
            alignment_records.extend(audit_alignment(city, yearly[city], errors))
        alignment = pd.DataFrame(alignment_records)
        logger.section(9, "THREE-YEAR ALIGNMENT")
        logger.log(alignment[["city", "comparison", "left_count", "right_count", "intersection", "missing_from_right", "extra_in_right", "fully_aligned"]].to_string(index=False))
        logger.section(10, "METADATA CONSISTENCY")
        logger.log(alignment[["city", "comparison", "class_mismatch", "city_mismatch", "external_target_mismatch", "longitude_mismatch", "latitude_mismatch", "maximum_longitude_difference", "maximum_latitude_difference"]].to_string(index=False))
        if errors:
            raise ValueError("Three-year alignment or metadata consistency failed")

        logger.section(11, "NaN / INF AUDIT")
        logger.log(raw_audit[["filename", "NaN_count_all_columns", "Inf_count_numeric_columns", "AEF_all_numeric_dtype", "longitude_min", "longitude_max", "latitude_min", "latitude_max"]].to_string(index=False))
        logger.section(12, "AEF NORM AUDIT")
        logger.log(norm_frame.to_string(index=False))

        external_frames: dict[str, pd.DataFrame] = {}
        summaries: list[dict[str, Any]] = []
        temporal_records: list[dict[str, Any]] = []
        for city in CITIES:
            master, summary, city_temporal = build_external_dataset(city, yearly[city], bands[city], errors, warnings)
            external_frames[city] = master
            summaries.append(summary)
            temporal_records.extend(city_temporal)
        temporal_frame = pd.DataFrame(temporal_records)
        if errors:
            raise ValueError("External feature/block construction failed")
        logger.section(13, "TEMPORAL COSINE STATISTICS")
        logger.log(temporal_frame.to_string(index=False))

        summary_frame = pd.DataFrame(summaries)
        summary_columns = [
            "city", "total_samples", "class0_samples", "class1_samples", "class2_samples",
            "total_blocks", "class0_blocks", "class1_blocks", "class2_blocks",
            "class0_largest_block_share", "class1_largest_block_share", "class2_largest_block_share",
            "samples_per_block_mean", "samples_per_block_std", "samples_per_block_median", "samples_per_block_max",
            "AEF_feature_count", "final_feature_count", "external_target",
        ]
        summary_frame = summary_frame[summary_columns].sort_values("city")
        logger.section(14, "SPATIAL BLOCK AUDIT")
        logger.log(summary_frame.to_string(index=False))
        logger.log("Chengdu CRS: EPSG:32648; Hangzhou CRS: EPSG:32651; block size: 2000 m")
        logger.section(15, "CLASS2 CONCENTRATION")
        logger.log(summary_frame[["city", "class2_samples", "class2_blocks", "class2_largest_block_share"]].to_string(index=False))
        logger.log("Concentration is recorded only; no resampling, label adjustment, or ROI change was performed.")

        PROCESSED_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        external_paths = {city: PROCESSED_EXTERNAL_DIR / f"{city}_External_AEF_195D.parquet" for city in CITIES}
        for city in CITIES:
            write_verify_parquet(external_frames[city], external_paths[city])
        schema_records = schema_audit(external_paths, errors)
        logger.section(16, "SCHEMA COMPATIBILITY WITH FIVE-CITY DEVELOPMENT SET")
        logger.log(schema_records.to_string(index=False))
        logger.log("external_target is an allowed external-only metadata field; it is excluded from AEF-192/195 feature lists.")
        if errors:
            raise ValueError("External-development schema compatibility failed")

        parquet_records = []
        logger.section(17, "EXTERNAL PARQUET OUTPUTS")
        for city in CITIES:
            path = external_paths[city]
            frame = pd.read_parquet(path, engine="pyarrow")
            logger.log(f"{city}: {path}; shape={frame.shape}; round_trip_verified=True")
            parquet_records.append({
                "city": city, "dataset_role": "EXTERNAL_TARGET",
                "parquet_path": path.relative_to(ROOT).as_posix(), "rows": len(frame),
                "sha256": sha256_file(path), "AEF_feature_count": AEF_COUNT,
                "external_validation_status": "DATA_READY_NOT_EVALUATED",
            })
        manifest = pd.DataFrame(parquet_records)
        logger.section(18, "EXTERNAL PARQUET SHA256")
        logger.log(manifest[["city", "parquet_path", "rows", "sha256"]].to_string(index=False))

        temporal_output = temporal_frame[["city", "temporal_feature", "mean", "std", "median", "25%", "75%", "min", "max"]].sort_values(["city", "temporal_feature"])
        norm_output = norm_frame[["city", "year", "mean", "std", "min", "25%", "median", "75%", "max", "percent_abs_norm_minus_1_lt_0.01", "zero_norm_count", "NaN_norm_count", "Inf_norm_count"]]
        save_csv(summary_frame, SUMMARY_PATH, 2)
        save_csv(temporal_output, TEMPORAL_PATH, 6)
        save_csv(norm_output, NORM_PATH, 6)
        save_csv(manifest, MANIFEST_PATH, 2)
        logger.section(19, "MANIFEST PATH")
        logger.log(f"manifest_path: {MANIFEST_PATH}")
        logger.log(manifest.to_string(index=False))

        logger.section(20, "LEAKAGE / OUTCOME-ACCESS AUDIT")
        for key, value in OUTCOME_ACCESS.items():
            logger.log(f"{key} = {value}")
        if any(OUTCOME_ACCESS.values()):
            raise RuntimeError("External-outcome access audit failed")

        five_city_hash_after = sha256_file(FIVE_CITY_PATH)
        logger.log(f"FiveCity_SHA256_after: {five_city_hash_after}")
        logger.log(f"FiveCity_unchanged: {five_city_hash_before == five_city_hash_after}")
        if five_city_hash_before != five_city_hash_after:
            raise RuntimeError("Development FiveCity dataset changed during Stage09A")
        if (ROOT / "data" / "processed" / "SevenCity_AEF_195D.parquet").exists():
            raise RuntimeError("Forbidden SevenCity_AEF_195D.parquet exists")

        logger.section(21, "WARNINGS")
        if warnings:
            for warning in warnings:
                logger.log(f"WARNING: {warning}")
        else:
            logger.log("None")
        logger.section(22, "FINAL STATUS")
        logger.log("SUCCESS")
        return 0
    except Exception as exc:
        logger.section(21, "WARNINGS AND ERRORS")
        for warning in warnings:
            logger.log(f"WARNING: {warning}")
        for error in errors:
            logger.log(f"ERROR: {error}")
        logger.log(f"FATAL: {type(exc).__name__}: {exc}")
        logger.log(traceback.format_exc())
        logger.section(22, "FINAL STATUS")
        logger.log("FAILED")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

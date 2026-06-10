"""
src/pipeline/validate.py
─────────────────────────────────────────────────────────────────────────────
Data quality checks for ERA5 grids, asset files, and feature tables.

Each check returns a ValidationResult with a pass/fail status and a
human-readable message. The run_all_checks() function is the entry point
used by the CLI and in notebooks.

Design philosophy: validation should be loud but non-fatal. We log all
failures as warnings and return a summary, letting the caller decide whether
to abort or proceed with degraded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ── ERA5 Dataset checks ───────────────────────────────────────────────────────

def check_era5_time_coverage(
    ds: xr.Dataset,
    year_start: int,
    year_end: int,
) -> ValidationResult:
    """Verify the dataset covers the expected year range."""
    name = "ERA5 time coverage"
    try:
        years = pd.to_datetime(ds.time.values).year
        actual_start, actual_end = int(years.min()), int(years.max())
        if actual_start > year_start:
            return ValidationResult(
                name, False,
                f"Data starts {actual_start}, expected ≤ {year_start}.",
                {"actual_start": actual_start, "actual_end": actual_end},
            )
        if actual_end < year_end:
            return ValidationResult(
                name, False,
                f"Data ends {actual_end}, expected ≥ {year_end}.",
                {"actual_start": actual_start, "actual_end": actual_end},
            )
        return ValidationResult(
            name, True,
            f"Covers {actual_start}–{actual_end}.",
            {"actual_start": actual_start, "actual_end": actual_end},
        )
    except Exception as e:
        return ValidationResult(name, False, f"Error reading time: {e}")


def check_era5_nan_fraction(
    ds: xr.Dataset,
    var_name: str,
    max_nan_fraction: float = 0.05,
) -> ValidationResult:
    """Flag if more than max_nan_fraction of values are NaN."""
    name = f"ERA5 NaN fraction ({var_name})"
    try:
        da = ds[var_name]
        nan_frac = float(da.isnull().mean())
        passed = nan_frac <= max_nan_fraction
        return ValidationResult(
            name,
            passed,
            f"NaN fraction = {nan_frac:.2%} (threshold {max_nan_fraction:.0%})",
            {"nan_fraction": nan_frac},
        )
    except KeyError:
        return ValidationResult(name, False, f"Variable '{var_name}' not found in dataset.")


def check_era5_physical_range(
    ds: xr.Dataset,
    var_name: str,
) -> ValidationResult:
    """Check that values fall within physically plausible ranges."""
    name = f"ERA5 physical range ({var_name})"
    # Physical bounds: (min_K, max_K) for temperature; (min, max) for others
    bounds: dict[str, tuple[float, float]] = {
        "t2m":               (180.0, 340.0),   # Kelvin
        "2m_temperature":    (180.0, 340.0),
        "tp":                (0.0,   1.0),      # metres per day
        "total_precipitation": (0.0, 1.0),
        "ws10":              (0.0,   120.0),    # m/s
        "u10":               (-100.0, 100.0),
        "v10":               (-100.0, 100.0),
    }
    try:
        da = ds[var_name]
        if var_name not in bounds:
            return ValidationResult(name, True, "No physical bounds defined — skipped.")
        lo, hi = bounds[var_name]
        vmin = float(da.min())
        vmax = float(da.max())
        if vmin < lo or vmax > hi:
            return ValidationResult(
                name, False,
                f"Values [{vmin:.2f}, {vmax:.2f}] outside [{lo}, {hi}].",
                {"vmin": vmin, "vmax": vmax},
            )
        return ValidationResult(
            name, True,
            f"Values [{vmin:.2f}, {vmax:.2f}] within [{lo}, {hi}].",
        )
    except Exception as e:
        return ValidationResult(name, False, str(e))


# ── Asset GeoDataFrame checks ─────────────────────────────────────────────────

def check_asset_coordinates(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Verify no assets have null or out-of-range lat/lon."""
    name = "Asset coordinates"
    issues: list[str] = []
    if gdf["latitude"].isna().any():
        issues.append(f"{gdf['latitude'].isna().sum()} null latitudes")
    if gdf["longitude"].isna().any():
        issues.append(f"{gdf['longitude'].isna().sum()} null longitudes")
    oob_lat = (~gdf["latitude"].between(-90, 90)).sum()
    oob_lon = (~gdf["longitude"].between(-180, 180)).sum()
    if oob_lat:
        issues.append(f"{oob_lat} latitudes outside [-90,90]")
    if oob_lon:
        issues.append(f"{oob_lon} longitudes outside [-180,180]")
    if issues:
        return ValidationResult(name, False, "; ".join(issues))
    return ValidationResult(name, True, f"All {len(gdf)} asset coordinates valid.")


def check_asset_duplicates(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Warn if any asset_id appears more than once."""
    name = "Asset ID uniqueness"
    dups = gdf["asset_id"].duplicated().sum()
    if dups:
        dup_ids = gdf.loc[gdf["asset_id"].duplicated(keep=False), "asset_id"].unique()
        return ValidationResult(
            name, False,
            f"{dups} duplicate asset_ids found.",
            {"examples": list(dup_ids[:5])},
        )
    return ValidationResult(name, True, "All asset_ids are unique.")


def check_asset_book_values(gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Check book_value is strictly positive and has no outliers (> 1000× median)."""
    name = "Asset book values"
    issues: list[str] = []
    non_positive = (gdf["book_value"] <= 0).sum()
    if non_positive:
        issues.append(f"{non_positive} non-positive book_value entries")
    median = gdf["book_value"].median()
    extreme = (gdf["book_value"] > 1000 * median).sum()
    if extreme:
        issues.append(f"{extreme} book_values > 1000× median (possible data error)")
    if issues:
        return ValidationResult(name, False, "; ".join(issues))
    return ValidationResult(
        name, True,
        f"All book_values positive. Range: [{gdf['book_value'].min():.1f}, "
        f"{gdf['book_value'].max():.1f}] USD M",
    )


# ── Feature table checks ──────────────────────────────────────────────────────

def check_feature_score_range(df: pd.DataFrame) -> ValidationResult:
    """Verify all hazard scores are in [0, 100]."""
    name = "Hazard score range"
    score_cols = ["flood_score", "heat_score", "cyclone_score", "composite_score"]
    present = [c for c in score_cols if c in df.columns]
    if not present:
        return ValidationResult(name, False, "No score columns found.")
    issues: list[str] = []
    for col in present:
        out = df[~df[col].between(0, 100)][col]
        if len(out):
            issues.append(f"{col}: {len(out)} values outside [0,100]")
    if issues:
        return ValidationResult(name, False, "; ".join(issues))
    return ValidationResult(name, True, f"All {len(present)} score columns in [0,100].")


def check_feature_join_coverage(df: pd.DataFrame) -> ValidationResult:
    """Warn if many assets failed to join to climate data."""
    name = "Feature join coverage"
    total = len(df)
    if "composite_score" not in df.columns:
        return ValidationResult(name, False, "'composite_score' column missing.")
    missing = df["composite_score"].isna().sum()
    coverage = 1 - missing / total
    if coverage < 0.80:
        return ValidationResult(
            name, False,
            f"Only {coverage:.0%} of assets have climate scores "
            f"({missing}/{total} missing). Check ERA5 geographic coverage.",
        )
    return ValidationResult(
        name, True,
        f"Join coverage: {coverage:.1%} ({total - missing}/{total} assets scored).",
    )


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_checks(
    assets: gpd.GeoDataFrame | None = None,
    era5_ds: xr.Dataset | None = None,
    era5_var: str | None = None,
    feature_df: pd.DataFrame | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> list[ValidationResult]:
    """Run all applicable validation checks and log a summary.

    Pass whichever objects you have; checks are skipped gracefully
    when their required inputs are not provided.

    Args:
        assets: GeoDataFrame from ingest.load_assets().
        era5_ds: xarray Dataset from ingest.load_era5_dataset().
        era5_var: Variable name in era5_ds to check ranges for.
        feature_df: Feature DataFrame from features.build_feature_table().
        year_start: Expected ERA5 start year.
        year_end: Expected ERA5 end year.

    Returns:
        List of ValidationResult objects.
    """
    from src.config import settings as cfg

    results: list[ValidationResult] = []

    # ERA5 checks
    if era5_ds is not None:
        _year_start = year_start or cfg.era5_year_start
        _year_end   = year_end   or cfg.era5_year_end
        results.append(check_era5_time_coverage(era5_ds, _year_start, _year_end))
        if era5_var:
            results.append(check_era5_nan_fraction(era5_ds, era5_var))
            results.append(check_era5_physical_range(era5_ds, era5_var))

    # Asset checks
    if assets is not None:
        results.append(check_asset_coordinates(assets))
        results.append(check_asset_duplicates(assets))
        results.append(check_asset_book_values(assets))

    # Feature checks
    if feature_df is not None:
        results.append(check_feature_score_range(feature_df))
        results.append(check_feature_join_coverage(feature_df))

    # Log summary
    passed = sum(r.passed for r in results)
    logger.info(f"\nValidation summary: {passed}/{len(results)} checks passed")
    for r in results:
        if r.passed:
            logger.info(str(r))
        else:
            logger.warning(str(r))

    return results

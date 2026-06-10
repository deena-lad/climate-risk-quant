"""
src/pipeline/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Converts raw ERA5 NetCDF datasets into tidy annual hazard maxima tables.

Pipeline per variable:
  1. Optional geographic clip to a bounding box (saves memory for regional work)
  2. Resample from hourly/6-hourly → monthly max
  3. Resample monthly → annual max (captures worst month per year)
  4. Convert units (K→°C, m→mm, m/s stays)
  5. Stack to tidy DataFrame: [lat, lon, year, variable]

Outputs are saved to data/interim/ as parquet for fast downstream loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from src.config import settings

# ── Unit conversion constants ─────────────────────────────────────────────────
KELVIN_OFFSET = 273.15          # K → °C
MM_PER_METRE = 1000.0          # m → mm precipitation


# ── Geographic clip ───────────────────────────────────────────────────────────

def clip_to_bbox(
    ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """Subset an ERA5 dataset to a geographic bounding box.

    ERA5 uses 'latitude' and 'longitude' coordinate names.
    Longitude is in [0, 360] for some ERA5 products and [-180, 180]
    for others; we normalise to [-180, 180] before slicing.

    Args:
        ds: Raw ERA5 Dataset.
        lat_min, lat_max: Latitude bounds (degrees North).
        lon_min, lon_max: Longitude bounds (degrees East, [-180, 180]).

    Returns:
        Clipped Dataset.
    """
    # Normalise longitude to [-180, 180]
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        ds = ds.sortby("longitude")

    ds = ds.sortby("latitude")
    ds = ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    logger.debug(
        f"Clipped to bbox [{lat_min},{lat_max}]×[{lon_min},{lon_max}]. "
        f"Grid: {ds.dims}"
    )
    return ds


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_annual_max(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Resample an ERA5 DataArray to annual maxima.

    ERA5 single-levels data is typically hourly or 6-hourly.
    We take: raw → monthly max → annual max of monthly maxima.
    This captures the single worst month in each year, which
    is the conventional approach for physical hazard scoring.

    Args:
        ds: ERA5 Dataset (after optional clip).
        var_name: Name of the DataArray inside the Dataset.

    Returns:
        DataArray with dims [year, latitude, longitude].
    """
    da: xr.DataArray = ds[var_name]

    # Drop time-bound variable if present (common in ERA5 NetCDF)
    for extra in ["time_bnds", "expver"]:
        if extra in ds:
            ds = ds.drop_vars(extra)

    logger.info(f"Resampling '{var_name}': {da.dims}, shape={da.shape}")

    # Monthly max first (handles 6-hourly or hourly input)
    monthly_max = da.resample(time="1ME").max(skipna=True)

    # Annual max from monthly maxima
    annual_max = monthly_max.resample(time="1YE").max(skipna=True)

    # Rename time → year and extract integer year for readability
    annual_max = annual_max.assign_coords(
        time=annual_max.time.dt.year.values
    ).rename({"time": "year"})

    logger.debug(f"Annual max shape: {annual_max.shape}")
    return annual_max


# ── Unit conversion ───────────────────────────────────────────────────────────

def convert_units(da: xr.DataArray, var_name: str) -> xr.DataArray:
    """Apply physical unit conversions appropriate for each ERA5 variable.

    Conversions applied:
    - Temperature (t2m, 2m_temperature): Kelvin → Celsius
    - Precipitation (tp, total_precipitation): metres → millimetres
    - Wind speed (ws10, u10, v10): no conversion (m/s retained)

    Args:
        da: DataArray of annual maxima.
        var_name: Original variable name used to determine conversion.

    Returns:
        Unit-converted DataArray with updated 'units' attribute.
    """
    name_lower = var_name.lower()

    if "temperature" in name_lower or var_name in ("t2m", "2m_temperature"):
        logger.debug(f"Converting {var_name}: K → °C")
        da = da - KELVIN_OFFSET
        da.attrs["units"] = "°C"
        da.attrs["long_name"] = "2m air temperature (°C)"

    elif "precipitation" in name_lower or var_name in ("tp", "total_precipitation"):
        logger.debug(f"Converting {var_name}: m → mm")
        da = da * MM_PER_METRE
        da.attrs["units"] = "mm"
        da.attrs["long_name"] = "Total precipitation (mm)"

    elif "wind" in name_lower or var_name in ("ws10", "u10", "v10"):
        da.attrs["units"] = "m/s"
        da.attrs["long_name"] = "10m wind speed (m/s)"

    return da


# ── Stack to tidy DataFrame ───────────────────────────────────────────────────

def to_tidy_dataframe(
    annual_max: xr.DataArray,
    var_label: str,
) -> pd.DataFrame:
    """Convert an annual-max DataArray to a tidy pandas DataFrame.

    Output columns: lat, lon, year, <var_label>

    Args:
        annual_max: DataArray with dims [year, latitude, longitude].
        var_label: Column name for the variable in the output DataFrame.

    Returns:
        Tidy DataFrame suitable for merging with asset data.
    """
    df = (
        annual_max
        .to_dataframe(name=var_label)
        .reset_index()
        .rename(columns={"latitude": "lat", "longitude": "lon"})
        .dropna(subset=[var_label])
    )
    # Round coordinates to 2 decimal places for consistent join keys
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)
    logger.debug(f"Tidy DataFrame shape: {df.shape}")
    return df


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_era5_variable(
    ds: xr.Dataset,
    var_name: str,
    var_label: str,
    bbox: tuple[float, float, float, float] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline for one ERA5 variable.

    Steps: clip → resample annual max → unit convert → tidy DataFrame → save.

    Args:
        ds: Raw xarray Dataset from load_era5_dataset().
        var_name: Name of the variable in the Dataset (e.g. 't2m').
        var_label: Output column label (e.g. 'heat_max_c').
        bbox: Optional (lat_min, lat_max, lon_min, lon_max) clip bounds.
        output_path: Save location for parquet. If None, saves to interim dir.

    Returns:
        Tidy DataFrame [lat, lon, year, var_label].
    """
    if bbox is not None:
        ds = clip_to_bbox(ds, *bbox)

    annual_max = resample_annual_max(ds, var_name)
    annual_max = convert_units(annual_max, var_name)
    df = to_tidy_dataframe(annual_max, var_label)

    if output_path is None:
        output_path = settings.data_interim_dir / f"{var_label}_annual_max.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Saved interim data → {output_path} ({len(df):,} rows)")
    return df


def compute_climatology(
    df: pd.DataFrame,
    var_col: str,
    baseline_year_start: int = 1990,
    baseline_year_end: int = 2020,
) -> pd.DataFrame:
    """Compute 30-year climatological statistics per grid cell.

    Statistics computed: mean (μ), std (σ), and 95th percentile.
    Used downstream to normalise annual maxima into hazard scores.

    Args:
        df: Tidy DataFrame from preprocess_era5_variable().
        var_col: Column name of the climate variable.
        baseline_year_start: First year of the climatological baseline.
        baseline_year_end: Last year of the climatological baseline.

    Returns:
        DataFrame indexed by (lat, lon) with columns:
        {var_col}_mean, {var_col}_std, {var_col}_p95.
    """
    baseline = df[
        df["year"].between(baseline_year_start, baseline_year_end)
    ]
    clim = (
        baseline
        .groupby(["lat", "lon"])[var_col]
        .agg(
            **{
                f"{var_col}_mean": "mean",
                f"{var_col}_std": "std",
                f"{var_col}_p95": lambda x: np.percentile(x.dropna(), 95),
            }
        )
        .reset_index()
    )
    # Guard against zero std (constant cells — e.g. desert grids)
    std_col = f"{var_col}_std"
    clim[std_col] = clim[std_col].replace(0.0, np.nan).fillna(1e-6)

    logger.info(
        f"Climatology computed for '{var_col}' over "
        f"{baseline_year_start}–{baseline_year_end}: {len(clim):,} grid cells"
    )
    return clim


def merge_climatology(
    df: pd.DataFrame,
    clim: pd.DataFrame,
    var_col: str,
) -> pd.DataFrame:
    """Join annual data with climatology and compute normalised anomaly.

    Z-score = (annual_value − μ) / σ
    This z-score feeds directly into the hazard scorer.

    Args:
        df: Annual tidy DataFrame.
        clim: Climatology DataFrame from compute_climatology().
        var_col: Variable column name.

    Returns:
        DataFrame with added columns: {var_col}_mean, {var_col}_std,
        {var_col}_p95, {var_col}_zscore.
    """
    merged = df.merge(clim, on=["lat", "lon"], how="left")
    merged[f"{var_col}_zscore"] = (
        (merged[var_col] - merged[f"{var_col}_mean"])
        / merged[f"{var_col}_std"]
    )
    return merged

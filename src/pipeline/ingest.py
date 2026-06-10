"""
src/pipeline/ingest.py
─────────────────────────────────────────────────────────────────────────────
ERA5 climate data download via CDS API, and company/asset data loader.

Design notes:
- ERA5 is requested at monthly frequency then aggregated to annual maxima
  downstream. Requesting monthly (not daily) keeps file sizes manageable
  (~500 MB/variable for a 30-year global run vs ~15 GB daily).
- We write a ~/.cdsapirc from env so users don't need a separate config step.
- Assets are loaded from CSV or GeoJSON; minimal required columns are
  validated with pydantic before any downstream work touches the data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cdsapi
import geopandas as gpd
import pandas as pd
import xarray as xr
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from src.config import settings

# ── CDS API setup ─────────────────────────────────────────────────────────────

def write_cdsapirc() -> None:
    """Write ~/.cdsapirc from environment variables if not already present.

    CDS API expects this file even when using the Python client.
    We create it at runtime so users only need to set env vars.
    """
    rc_path = Path.home() / ".cdsapirc"
    if rc_path.exists():
        logger.debug(".cdsapirc already exists, skipping write.")
        return
    if not settings.cds_api_key:
        raise EnvironmentError(
            "CDS_API_KEY is not set. Add it to your .env file.\n"
            "Format: CDS_API_KEY=YOUR_UID:YOUR_API_KEY\n"
            "Register at: https://cds.climate.copernicus.eu/"
        )
    rc_path.write_text(
        f"url: {settings.cds_api_url}\nkey: {settings.cds_api_key}\nverify: 1\n"
    )
    logger.info(f"Created {rc_path}")


# ── ERA5 variable mapping ──────────────────────────────────────────────────────

# Maps our internal names → CDS API short names
ERA5_VARIABLE_MAP: dict[str, str] = {
    "2m_temperature": "2m_temperature",
    "total_precipitation": "total_precipitation",
    "10m_wind_speed": "10m_u_component_of_wind",  # we combine u+v downstream
}

# Wind requires both u and v components to compute speed
ERA5_WIND_PAIR = ("10m_u_component_of_wind", "10m_v_component_of_wind")


def _build_era5_request(
    variable: str,
    year: int,
    months: list[str] | None = None,
) -> dict:
    """Build a CDS API request dict for a single variable and year.

    Args:
        variable: CDS short name (e.g. '2m_temperature').
        year: Four-digit year.
        months: List of month strings. Defaults to all 12.

    Returns:
        Dict suitable for cdsapi.Client().retrieve().
    """
    if months is None:
        months = [str(m).zfill(2) for m in range(1, 13)]

    return {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": months,
        "day": [str(d).zfill(2) for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "format": "netcdf",
    }


def download_era5_variable(
    variable: str,
    year_start: int | None = None,
    year_end: int | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download ERA5 monthly data for a single variable across a year range.

    Files are saved as ``{output_dir}/{variable}_{year}.nc``.
    Already-downloaded files are skipped unless ``overwrite=True``.

    Args:
        variable: Internal variable name (key in ERA5_VARIABLE_MAP).
        year_start: First year to download (defaults to settings).
        year_end: Last year to download inclusive (defaults to settings).
        output_dir: Directory for downloaded .nc files (defaults to settings).
        overwrite: Re-download even if file already exists.

    Returns:
        List of paths to downloaded (or already-existing) .nc files.
    """
    write_cdsapirc()

    year_start = year_start or settings.era5_year_start
    year_end = year_end or settings.era5_year_end
    output_dir = output_dir or settings.data_raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if variable not in ERA5_VARIABLE_MAP:
        raise ValueError(
            f"Unknown variable '{variable}'. "
            f"Choose from: {list(ERA5_VARIABLE_MAP.keys())}"
        )

    cds_name = ERA5_VARIABLE_MAP[variable]
    client = cdsapi.Client(quiet=True)
    downloaded: list[Path] = []

    # For wind we also need the v-component
    variables_to_fetch = [cds_name]
    if variable == "10m_wind_speed":
        variables_to_fetch = list(ERA5_WIND_PAIR)

    years = range(year_start, year_end + 1)
    for year in tqdm(years, desc=f"Downloading {variable}", unit="year"):
        for cds_var in variables_to_fetch:
            out_path = output_dir / f"{cds_var}_{year}.nc"
            if out_path.exists() and not overwrite:
                logger.debug(f"Already exists, skipping: {out_path.name}")
                downloaded.append(out_path)
                continue

            logger.info(f"Requesting ERA5 {cds_var} for {year}…")
            request = _build_era5_request(cds_var, year)
            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(out_path),
            )
            logger.success(f"Saved → {out_path}")
            downloaded.append(out_path)

    return downloaded


def download_all_era5(
    year_start: int | None = None,
    year_end: int | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Download all configured ERA5 variables.

    Args:
        year_start: First year (defaults to settings).
        year_end: Last year inclusive (defaults to settings).
        output_dir: Directory for files (defaults to settings).
        overwrite: Re-download existing files.

    Returns:
        Dict mapping variable name → list of downloaded file paths.
    """
    results: dict[str, list[Path]] = {}
    for variable in settings.era5_variables:
        results[variable] = download_era5_variable(
            variable=variable,
            year_start=year_start,
            year_end=year_end,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    return results


def load_era5_dataset(
    variable: str,
    raw_dir: Path | None = None,
) -> xr.Dataset:
    """Load all downloaded .nc files for a variable into a single xarray Dataset.

    Uses xarray's lazy multi-file open so large datasets don't blow up RAM.

    Args:
        variable: Internal variable name.
        raw_dir: Directory containing .nc files (defaults to settings).

    Returns:
        xr.Dataset with a 'time' dimension spanning all downloaded years.

    Raises:
        FileNotFoundError: If no .nc files are found for the variable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    cds_name = ERA5_VARIABLE_MAP.get(variable, variable)

    # Wind speed requires u + v
    if variable == "10m_wind_speed":
        patterns = [f"{comp}_*.nc" for comp in ERA5_WIND_PAIR]
    else:
        patterns = [f"{cds_name}_*.nc"]

    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(raw_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found for variable '{variable}' in {raw_dir}. "
            "Run download_era5_variable() first."
        )

    logger.info(f"Loading {len(files)} files for '{variable}'…")
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")

    # Compute wind speed from u/v components
    if variable == "10m_wind_speed" and "u10" in ds and "v10" in ds:
        ds = ds.assign(ws10=lambda d: (d["u10"] ** 2 + d["v10"] ** 2) ** 0.5)
        ds = ds.drop_vars(["u10", "v10"])
        logger.debug("Computed wind speed from u/v components → ws10")

    return ds


# ── Asset / company data loader ───────────────────────────────────────────────

class AssetRecord(BaseModel):
    """Schema for a single financial asset / property record."""

    asset_id: str
    name: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    sector: str
    book_value: float = Field(gt=0, description="USD millions")
    country: str = ""  # filled by geo enrichment

    @field_validator("latitude")
    @classmethod
    def lat_not_zero(cls, v: float) -> float:
        if v == 0.0:
            logger.warning("Latitude is exactly 0.0 — verify this is intentional.")
        return v


REQUIRED_ASSET_COLUMNS: set[str] = {
    "asset_id", "name", "latitude", "longitude", "sector", "book_value"
}


def load_assets(path: Path) -> gpd.GeoDataFrame:
    """Load company/property asset data from CSV or GeoJSON.

    Required columns: asset_id, name, latitude, longitude, sector, book_value.
    Extra columns are preserved as-is.

    Args:
        path: Path to a .csv or .geojson file.

    Returns:
        GeoDataFrame with Point geometry in EPSG:4326.

    Raises:
        ValueError: If required columns are missing or any record fails validation.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".geojson", ".json"):
        df = pd.read_file(path) if hasattr(pd, "read_file") else _load_geojson(path)
    else:
        raise ValueError(f"Unsupported asset file format: {suffix}. Use .csv or .geojson")

    _validate_asset_columns(df, path)

    # Validate each row via pydantic (catches bad coordinates etc.)
    errors: list[str] = []
    for i, row in df.iterrows():
        try:
            AssetRecord(**{col: row[col] for col in REQUIRED_ASSET_COLUMNS if col in df.columns})
        except Exception as e:
            errors.append(f"Row {i} ({row.get('asset_id', '?')}): {e}")
    if errors:
        raise ValueError(
            f"Asset validation failed for {len(errors)} row(s):\n" + "\n".join(errors[:10])
        )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    logger.success(f"Loaded {len(gdf)} assets from {path.name}")
    return gdf


def _load_geojson(path: Path) -> pd.DataFrame:
    """Fallback GeoJSON loader when geopandas.read_file isn't needed."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        props["longitude"] = coords[0]
        props["latitude"] = coords[1]
        rows.append(props)
    return pd.DataFrame(rows)


def _validate_asset_columns(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_ASSET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Asset file {path.name} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def generate_sample_assets(n: int = 50, seed: int = 42) -> gpd.GeoDataFrame:
    """Generate a synthetic asset dataset for development and testing.

    Produces companies with realistic global coordinates weighted toward
    coastal / tropical regions where climate risk is higher.

    Args:
        n: Number of synthetic assets.
        seed: Random seed for reproducibility.

    Returns:
        GeoDataFrame identical in schema to load_assets() output.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    sectors = ["Real Estate", "Energy", "Utilities", "Industrials", "Financials"]
    # Biased toward coastal lat/lon clusters (higher risk regions)
    lat_centres = [22, -23, 51, 35, 1, -34]
    lon_centres = [88, -43, 4, 139, 103, 151]

    lats, lons = [], []
    for _ in range(n):
        idx = rng.integers(len(lat_centres))
        lats.append(float(rng.normal(lat_centres[idx], 8)))
        lons.append(float(rng.normal(lon_centres[idx], 12)))

    df = pd.DataFrame(
        {
            "asset_id": [f"AST{str(i).zfill(4)}" for i in range(n)],
            "name": [f"Asset {i}" for i in range(n)],
            "latitude": np.clip(lats, -85, 85),
            "longitude": np.clip(lons, -180, 180),
            "sector": rng.choice(sectors, n),
            "book_value": rng.lognormal(mean=5.0, sigma=1.2, size=n).round(2),
            "country": "",
        }
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    logger.info(f"Generated {n} synthetic assets.")
    return gdf

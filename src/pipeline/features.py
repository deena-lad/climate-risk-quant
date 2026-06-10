"""
src/pipeline/features.py
─────────────────────────────────────────────────────────────────────────────
Spatial feature engineering: assigns ERA5 climate statistics to each asset
and produces normalised hazard scores (0–100 scale).

Pipeline:
  1. snap_assets_to_grid()  — find nearest ERA5 cell for each asset
  2. extract_hazard_features() — pivot era5 z-scores per asset × year
  3. score_hazards()         — percentile-rank z-scores to 0–100
  4. flag_compound_risk()    — mark assets in top quartile for ≥2 hazards
  5. build_feature_table()   — single entry point combining all steps

The 0–100 scoring follows the same paradigm as MSCI ESG Physical Risk
scores: rank-based normalisation within the asset universe ensures
comparability even when absolute climate values differ across regions.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree

from src.config import settings

# ── Column name constants ─────────────────────────────────────────────────────

HAZARD_ZSCORE_COLS: dict[str, str] = {
    "flood":   "heat_max_c_zscore",   # proxy: heavy precip drives flood risk
    "heat":    "heat_max_c_zscore",
    "cyclone": "wind_max_ms_zscore",
}

# More precise naming when precip data is available
PRECIP_ZSCORE_COL = "precip_max_mm_zscore"
TEMP_ZSCORE_COL   = "heat_max_c_zscore"
WIND_ZSCORE_COL   = "wind_max_ms_zscore"


# ── Step 1: Snap assets to nearest ERA5 grid cell ────────────────────────────

def snap_assets_to_grid(
    assets: gpd.GeoDataFrame,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
) -> gpd.GeoDataFrame:
    """Assign each asset the coordinates of its nearest ERA5 grid cell.

    ERA5 ~0.25° grid has ~28 km resolution. We use a KD-tree for O(n log n)
    nearest-neighbour lookup, which scales to millions of assets.

    Args:
        assets: GeoDataFrame with 'latitude' and 'longitude' columns.
        era5_lats: 1-D array of ERA5 latitude values.
        era5_lons: 1-D array of ERA5 longitude values.

    Returns:
        assets GeoDataFrame with two added columns:
        'era5_lat' and 'era5_lon' — the snapped grid coordinates.
    """
    # Build grid point pairs
    grid_lat, grid_lon = np.meshgrid(era5_lats, era5_lons, indexing="ij")
    grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    # Convert to radians for geodetically-correct distances
    asset_points = np.deg2rad(
        assets[["latitude", "longitude"]].values.astype(float)
    )
    grid_rad = np.deg2rad(grid_points)

    tree = cKDTree(grid_rad)
    _, indices = tree.query(asset_points, k=1, workers=-1)

    assets = assets.copy()
    assets["era5_lat"] = grid_points[indices, 0].round(2)
    assets["era5_lon"] = grid_points[indices, 1].round(2)

    logger.info(f"Snapped {len(assets)} assets to ERA5 grid.")
    return assets


# ── Step 2: Extract per-asset climate features ────────────────────────────────

def extract_hazard_features(
    assets: gpd.GeoDataFrame,
    climate_df: pd.DataFrame,
    era5_lat_col: str = "lat",
    era5_lon_col: str = "lon",
    year: int | None = None,
) -> pd.DataFrame:
    """Join climate z-score data to assets via their snapped grid coordinates.

    If year is specified, returns features for that year only.
    Otherwise returns all years (useful for time-series exposure analysis).

    Args:
        assets: GeoDataFrame with 'era5_lat' and 'era5_lon' columns.
        climate_df: Merged tidy DataFrame from preprocess.merge_climatology().
        era5_lat_col: Latitude column name in climate_df.
        era5_lon_col: Longitude column name in climate_df.
        year: Filter to a specific year. None returns all years.

    Returns:
        DataFrame with asset metadata + all climate feature columns.
    """
    if year is not None:
        climate_df = climate_df[climate_df["year"] == year].copy()

    merged = assets.merge(
        climate_df.rename(
            columns={era5_lat_col: "era5_lat", era5_lon_col: "era5_lon"}
        ),
        on=["era5_lat", "era5_lon"],
        how="left",
    )

    n_missing = merged["year"].isna().sum()
    if n_missing > 0:
        logger.warning(
            f"{n_missing} assets could not be joined to climate data. "
            "Check that ERA5 files cover the asset region."
        )

    logger.debug(f"Extracted features: {merged.shape}")
    return merged.drop(columns=["geometry"], errors="ignore")


# ── Step 3: Normalise z-scores to 0–100 hazard scores ────────────────────────

def score_hazards(
    df: pd.DataFrame,
    precip_zscore_col: str = PRECIP_ZSCORE_COL,
    temp_zscore_col: str = TEMP_ZSCORE_COL,
    wind_zscore_col: str = WIND_ZSCORE_COL,
) -> pd.DataFrame:
    """Convert z-score columns to percentile-rank scores on a 0–100 scale.

    Percentile rank within the universe of assets ensures the score is
    relative (a score of 80 means riskier than 80% of the portfolio),
    which is the industry convention for ESG physical risk screening.

    Missing z-scores (assets where ERA5 data was unavailable) receive
    the median score (50) rather than 0 or NaN to avoid spurious low-risk
    signals for data gaps.

    Args:
        df: DataFrame with z-score columns.
        precip_zscore_col: Column for precipitation anomaly.
        temp_zscore_col: Column for temperature anomaly.
        wind_zscore_col: Column for wind speed anomaly.

    Returns:
        df with added columns: flood_score, heat_score, cyclone_score.
    """
    df = df.copy()

    def _pct_rank(series: pd.Series) -> pd.Series:
        ranked = series.rank(pct=True, na_option="keep") * 100
        return ranked.fillna(50.0).round(1)

    # Flood uses precipitation z-score (fall back to heat z-score if missing)
    flood_source = precip_zscore_col if precip_zscore_col in df.columns else temp_zscore_col
    df["flood_score"]   = _pct_rank(df.get(flood_source, pd.Series(dtype=float)))
    df["heat_score"]    = _pct_rank(df.get(temp_zscore_col, pd.Series(dtype=float)))
    df["cyclone_score"] = _pct_rank(df.get(wind_zscore_col, pd.Series(dtype=float)))

    logger.info("Hazard scores computed (0–100 percentile rank).")
    return df


# ── Step 4: Compound risk flag ─────────────────────────────────────────────────

def flag_compound_risk(
    df: pd.DataFrame,
    threshold: float = 75.0,
    min_hazards: int = 2,
) -> pd.DataFrame:
    """Flag assets exposed to multiple simultaneous high-risk hazards.

    An asset is flagged 'compound' if it scores above ``threshold`` in
    at least ``min_hazards`` of the three hazard types.  Compound events
    are disproportionately damaging (IPCC AR6 Ch. 11) and warrant a
    separate tier in the risk scorecard.

    Args:
        df: DataFrame with flood_score, heat_score, cyclone_score columns.
        threshold: Score above which a hazard counts as 'high risk'.
        min_hazards: Minimum number of high-risk hazards to trigger flag.

    Returns:
        df with added boolean column 'compound_risk'.
    """
    score_cols = ["flood_score", "heat_score", "cyclone_score"]
    present = [c for c in score_cols if c in df.columns]
    if not present:
        logger.warning("No hazard score columns found; compound flag not set.")
        df["compound_risk"] = False
        return df

    high_risk = (df[present] >= threshold).sum(axis=1)
    df["compound_risk"] = (high_risk >= min_hazards).map(bool)
    n_flagged = df["compound_risk"].sum()
    logger.info(
        f"Compound risk: {n_flagged} / {len(df)} assets flagged "
        f"(threshold={threshold}, min_hazards={min_hazards})"
    )
    return df


# ── Step 5: ESG tier assignment ────────────────────────────────────────────────

def assign_esg_tier(composite_score: pd.Series) -> pd.Series:
    """Map composite scores to ESG risk tier labels.

    Tiers mirror MSCI ESG Physical Risk conventions:
        0–25  → Low
        25–50 → Medium
        50–75 → High
        75–   → Critical

    Args:
        composite_score: Series of 0–100 composite risk scores.

    Returns:
        Categorical Series: Low / Medium / High / Critical.
    """
    bins = [0, 25, 50, 75, 100.1]
    labels = ["Low", "Medium", "High", "Critical"]
    return pd.cut(composite_score, bins=bins, labels=labels, right=False)


# ── Master entry point ────────────────────────────────────────────────────────

def build_feature_table(
    assets: gpd.GeoDataFrame,
    climate_dfs: dict[str, pd.DataFrame],
    year: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the full feature table: snap → join → score → flag → save.

    Args:
        assets: GeoDataFrame from ingest.load_assets().
        climate_dfs: Dict mapping variable labels to their preprocessed
                     DataFrames, e.g.:
                     {
                       'heat_max_c':    heat_df,     # has heat_max_c_zscore
                       'precip_max_mm': precip_df,   # has precip_max_mm_zscore
                       'wind_max_ms':   wind_df,     # has wind_max_ms_zscore
                     }
        year: If given, extract features for this year only.
        output_path: Save processed features to this parquet path.

    Returns:
        Feature DataFrame ready for the model layer.
    """
    # 1. Extract unique ERA5 grid coordinates from any available climate df
    first_df = next(iter(climate_dfs.values()))
    era5_lats = np.sort(first_df["lat"].unique())
    era5_lons = np.sort(first_df["lon"].unique())

    # 2. Snap assets to ERA5 grid
    assets_snapped = snap_assets_to_grid(assets, era5_lats, era5_lons)

    # 3. Merge all climate variable DataFrames on (lat, lon, year)
    climate_merged = first_df.copy()
    for label, df in list(climate_dfs.items())[1:]:
        zscore_col = f"{label}_zscore"
        cols_to_add = ["lat", "lon", "year"] + [
            c for c in df.columns if c not in climate_merged.columns or c in ["lat","lon","year"]
        ]
        climate_merged = climate_merged.merge(
            df[cols_to_add] if cols_to_add else df,
            on=["lat", "lon", "year"],
            how="outer",
            suffixes=("", f"_{label}"),
        )

    # 4. Extract per-asset features
    features = extract_hazard_features(
        assets_snapped, climate_merged, year=year
    )

    # 5. Score + flag
    features = score_hazards(features)
    features = flag_compound_risk(features)

    # 6. Composite score (weighted average per settings)
    features["composite_score"] = (
        features["flood_score"]   * settings.weight_flood
        + features["heat_score"]  * settings.weight_heat
        + features["cyclone_score"] * settings.weight_cyclone
    ).round(2)

    # 7. ESG tier
    features["esg_tier"] = assign_esg_tier(features["composite_score"])

    # 8. Save
    if output_path is None:
        year_tag = str(year) if year else "all"
        output_path = settings.data_processed_dir / f"features_{year_tag}.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.success(f"Feature table saved → {output_path} ({len(features):,} rows)")

    return features

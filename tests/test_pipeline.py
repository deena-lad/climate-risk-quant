"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the data pipeline: ingest, preprocess, features, validate.

Tests use entirely synthetic data — no network calls, no files on disk.
Fixtures build minimal but structurally correct objects for each module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import xarray as xr

from src.pipeline.ingest import generate_sample_assets, _validate_asset_columns
from src.pipeline.preprocess import (
    clip_to_bbox,
    resample_annual_max,
    convert_units,
    compute_climatology,
    merge_climatology,
    to_tidy_dataframe,
)
from src.pipeline.features import (
    snap_assets_to_grid,
    score_hazards,
    flag_compound_risk,
    assign_esg_tier,
    build_feature_table,
)
from src.pipeline.validate import (
    check_asset_coordinates,
    check_asset_duplicates,
    check_asset_book_values,
    check_feature_score_range,
    check_feature_join_coverage,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_assets() -> gpd.GeoDataFrame:
    return generate_sample_assets(n=20, seed=0)


@pytest.fixture
def sample_era5_ds() -> xr.Dataset:
    """Minimal synthetic ERA5-like dataset (hourly, 2-year, 4×4 grid)."""
    times = pd.date_range("1991-01-01", periods=24 * 365 * 2, freq="1h")
    lats = np.array([10.0, 20.0, 30.0, 40.0])
    lons = np.array([70.0, 80.0, 90.0, 100.0])
    rng = np.random.default_rng(1)
    data = rng.uniform(280, 310, size=(len(times), len(lats), len(lons)))
    ds = xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds["t2m"].attrs["units"] = "K"
    return ds


@pytest.fixture
def sample_tidy_df() -> pd.DataFrame:
    """Tidy annual-max DataFrame for 4 grid cells over 10 years."""
    rng = np.random.default_rng(2)
    rows = []
    for lat in [10.0, 20.0, 30.0, 40.0]:
        for lon in [70.0, 80.0]:
            for year in range(1991, 2001):
                rows.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "year": year,
                        "heat_max_c": float(rng.uniform(25, 45)),
                        "precip_max_mm": float(rng.uniform(100, 500)),
                        "wind_max_ms": float(rng.uniform(5, 40)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def scored_df(sample_assets, sample_tidy_df) -> pd.DataFrame:
    """Assets with hazard scores applied."""
    df = sample_assets.drop(columns=["geometry"]).copy()
    rng = np.random.default_rng(3)
    n = len(df)
    df["heat_max_c_zscore"]    = rng.uniform(-2, 4, n)
    df["precip_max_mm_zscore"] = rng.uniform(-1, 3, n)
    df["wind_max_ms_zscore"]   = rng.uniform(-2, 5, n)
    df = score_hazards(df)
    return df


# ── Ingest tests ──────────────────────────────────────────────────────────────

class TestGenerateSampleAssets:
    def test_returns_geodataframe(self, sample_assets):
        assert isinstance(sample_assets, gpd.GeoDataFrame)

    def test_correct_columns(self, sample_assets):
        for col in ["asset_id", "latitude", "longitude", "sector", "book_value"]:
            assert col in sample_assets.columns

    def test_coordinates_in_range(self, sample_assets):
        assert sample_assets["latitude"].between(-85, 85).all()
        assert sample_assets["longitude"].between(-180, 180).all()

    def test_book_value_positive(self, sample_assets):
        assert (sample_assets["book_value"] > 0).all()

    def test_reproducible(self):
        a = generate_sample_assets(n=10, seed=99)
        b = generate_sample_assets(n=10, seed=99)
        pd.testing.assert_frame_equal(
            a[["latitude", "longitude"]].reset_index(drop=True),
            b[["latitude", "longitude"]].reset_index(drop=True),
        )


class TestValidateAssetColumns:
    def test_passes_with_required_columns(self):
        df = pd.DataFrame(
            {
                "asset_id": ["A1"],
                "name": ["Test"],
                "latitude": [10.0],
                "longitude": [80.0],
                "sector": ["Energy"],
                "book_value": [100.0],
            }
        )
        _validate_asset_columns(df, path=__file__)  # should not raise

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"asset_id": ["A1"], "latitude": [10.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_asset_columns(df, path=__file__)


# ── Preprocess tests ──────────────────────────────────────────────────────────

class TestClipToBbox:
    def test_clips_latitude(self, sample_era5_ds):
        clipped = clip_to_bbox(sample_era5_ds, lat_min=15, lat_max=35, lon_min=65, lon_max=105)
        assert float(clipped.latitude.min()) >= 15
        assert float(clipped.latitude.max()) <= 35

    def test_clips_longitude(self, sample_era5_ds):
        clipped = clip_to_bbox(sample_era5_ds, lat_min=5, lat_max=45, lon_min=75, lon_max=95)
        assert float(clipped.longitude.min()) >= 75
        assert float(clipped.longitude.max()) <= 95


class TestResampleAnnualMax:
    def test_output_has_year_dim(self, sample_era5_ds):
        annual = resample_annual_max(sample_era5_ds, "t2m")
        assert "year" in annual.dims

    def test_annual_max_gte_monthly_mean(self, sample_era5_ds):
        annual = resample_annual_max(sample_era5_ds, "t2m")
        monthly_mean = float(sample_era5_ds["t2m"].resample(time="1ME").mean().mean())
        assert float(annual.mean()) >= monthly_mean * 0.95  # allow small float tolerance

    def test_output_years_count(self, sample_era5_ds):
        annual = resample_annual_max(sample_era5_ds, "t2m")
        # 2-year dataset → 2 annual values
        assert len(annual.year) == 2


class TestConvertUnits:
    def test_temperature_kelvin_to_celsius(self, sample_era5_ds):
        annual = resample_annual_max(sample_era5_ds, "t2m")
        converted = convert_units(annual, "t2m")
        assert converted.attrs["units"] == "°C"
        # Original ~280–310 K → 6–37 °C after subtraction
        assert float(converted.min()) > -50
        assert float(converted.max()) < 100

    def test_precipitation_metres_to_mm(self):
        da = xr.DataArray([0.01, 0.05, 0.1], dims=["x"])
        converted = convert_units(da, "total_precipitation")
        assert converted.attrs["units"] == "mm"
        np.testing.assert_allclose(converted.values, [10.0, 50.0, 100.0])


class TestClimatology:
    def test_compute_climatology_columns(self, sample_tidy_df):
        clim = compute_climatology(sample_tidy_df, "heat_max_c", 1991, 2000)
        for col in ["heat_max_c_mean", "heat_max_c_std", "heat_max_c_p95"]:
            assert col in clim.columns

    def test_no_zero_std(self, sample_tidy_df):
        clim = compute_climatology(sample_tidy_df, "heat_max_c", 1991, 2000)
        assert (clim["heat_max_c_std"] > 0).all()

    def test_merge_adds_zscore(self, sample_tidy_df):
        clim = compute_climatology(sample_tidy_df, "heat_max_c", 1991, 2000)
        merged = merge_climatology(sample_tidy_df, clim, "heat_max_c")
        assert "heat_max_c_zscore" in merged.columns
        assert merged["heat_max_c_zscore"].notna().any()


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestSnapAssetsToGrid:
    def test_adds_era5_columns(self, sample_assets):
        lats = np.array([10.0, 20.0, 30.0, 40.0])
        lons = np.array([70.0, 80.0, 90.0, 100.0])
        result = snap_assets_to_grid(sample_assets, lats, lons)
        assert "era5_lat" in result.columns
        assert "era5_lon" in result.columns

    def test_snapped_coords_in_grid(self, sample_assets):
        lats = np.array([10.0, 20.0, 30.0, 40.0])
        lons = np.array([70.0, 80.0, 90.0, 100.0])
        result = snap_assets_to_grid(sample_assets, lats, lons)
        assert result["era5_lat"].isin(lats).all()
        assert result["era5_lon"].isin(lons).all()


class TestScoreHazards:
    def test_score_columns_created(self, scored_df):
        for col in ["flood_score", "heat_score", "cyclone_score"]:
            assert col in scored_df.columns

    def test_scores_in_range(self, scored_df):
        for col in ["flood_score", "heat_score", "cyclone_score"]:
            assert scored_df[col].between(0, 100).all(), f"{col} out of [0,100]"

    def test_nan_zscore_gets_median_score(self):
        df = pd.DataFrame({
            "heat_max_c_zscore": [1.0, np.nan, 2.0],
            "precip_max_mm_zscore": [0.5, np.nan, 1.5],
            "wind_max_ms_zscore": [0.2, np.nan, 0.8],
        })
        result = score_hazards(df)
        assert result.loc[1, "heat_score"] == 50.0  # NaN → median (50)


class TestFlagCompoundRisk:
    def test_flag_when_two_hazards_above_threshold(self):
        df = pd.DataFrame({
            "flood_score":   [80.0, 30.0, 80.0],
            "heat_score":    [80.0, 30.0, 20.0],
            "cyclone_score": [10.0, 10.0, 80.0],
        })
        result = flag_compound_risk(df, threshold=75.0, min_hazards=2)
        assert result.loc[0, "compound_risk"] is True   # flood + heat both ≥ 75
        assert result.loc[1, "compound_risk"] is False  # none ≥ 75
        assert result.loc[2, "compound_risk"] is True   # flood + cyclone both ≥ 75

    def test_all_false_when_below_threshold(self):
        df = pd.DataFrame({
            "flood_score":   [50.0, 60.0],
            "heat_score":    [55.0, 65.0],
            "cyclone_score": [40.0, 50.0],
        })
        result = flag_compound_risk(df, threshold=75.0)
        assert not result["compound_risk"].any()


class TestAssignEsgTier:
    @pytest.mark.parametrize("score, expected_tier", [
        (10.0, "Low"),
        (30.0, "Medium"),
        (60.0, "High"),
        (85.0, "Critical"),
        (25.0, "Medium"),  # boundary: 25 maps to Medium (right=False)
    ])
    def test_tier_assignment(self, score, expected_tier):
        result = assign_esg_tier(pd.Series([score]))
        assert str(result.iloc[0]) == expected_tier


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_assets_pass_all_checks(self, sample_assets):
        r1 = check_asset_coordinates(sample_assets)
        r2 = check_asset_duplicates(sample_assets)
        r3 = check_asset_book_values(sample_assets)
        assert r1.passed
        assert r2.passed
        assert r3.passed

    def test_duplicate_asset_fails(self, sample_assets):
        df = pd.concat([sample_assets, sample_assets.iloc[:3]], ignore_index=True)
        result = check_asset_duplicates(df)
        assert not result.passed

    def test_out_of_range_coordinate_fails(self):
        df = generate_sample_assets(n=5, seed=0)
        df.loc[0, "latitude"] = 200.0  # invalid
        result = check_asset_coordinates(df)
        assert not result.passed

    def test_score_range_check_passes(self, scored_df):
        scored_df["composite_score"] = scored_df["flood_score"] * 0.4 + \
            scored_df["heat_score"] * 0.35 + scored_df["cyclone_score"] * 0.25
        result = check_feature_score_range(scored_df)
        assert result.passed

    def test_score_range_check_fails_on_outlier(self, scored_df):
        scored_df = scored_df.copy()
        scored_df.loc[0, "flood_score"] = 150.0  # out of range
        result = check_feature_score_range(scored_df)
        assert not result.passed

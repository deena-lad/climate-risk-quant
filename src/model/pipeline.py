"""
src/model/pipeline.py
─────────────────────────────────────────────────────────────────────────────
End-to-end model training pipeline.

Orchestrates:
  1. Load processed features (or generate synthetic data for dev mode)
  2. Fit HazardScorer on baseline years
  3. Run CompositeRiskModel (sector-weighted scoring)
  4. Run AnomalyDetector (tail-risk flagging)
  5. Apply risk-adjusted valuation (haircut + Climate VaR)
  6. Evaluate (Gini, KS, tail concentration, etc.)
  7. Log to MLflow + save all artifacts

Call run_pipeline() as the single entry point from CLI or notebook.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import settings
from src.model.artifacts import save_run
from src.model.evaluate import alpha_sensitivity, evaluate, log_experiment, weight_sensitivity
from src.model.scorer import AnomalyDetector, CompositeRiskModel, HazardScorer
from src.model.valuation import apply_haircut, climate_var, portfolio_summary, sector_attribution
from src.pipeline.features import assign_esg_tier
from src.pipeline.ingest import generate_sample_assets
from src.pipeline.validate import run_all_checks


# ── Z-score column names expected in the feature table ───────────────────────
ZSCORE_COLS = [
    "precip_max_mm_zscore",
    "heat_max_c_zscore",
    "wind_max_ms_zscore",
]

# Fallback if precip z-score is absent (single-variable datasets)
ZSCORE_COLS_FALLBACK = [
    "heat_max_c_zscore",
    "heat_max_c_zscore",   # duplicate intentional: flood falls back to heat
    "wind_max_ms_zscore",
]


def _load_or_synthesise(
    features_path: Path | None,
    n_synthetic: int,
    seed: int,
) -> pd.DataFrame:
    """Load a real feature parquet or build a synthetic dataset for dev.

    Args:
        features_path: Path to features_*.parquet. None → synthesise.
        n_synthetic: Number of synthetic assets when synthesising.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame ready for the model pipeline.
    """
    import numpy as np

    if features_path is not None and features_path.exists():
        df = pd.read_parquet(features_path)
        logger.info(f"Loaded features from {features_path} ({len(df):,} rows)")
        return df

    logger.warning(
        "No feature parquet found — generating synthetic dataset "
        f"(n={n_synthetic}, seed={seed})."
    )
    rng = np.random.default_rng(seed)
    assets = generate_sample_assets(n=n_synthetic, seed=seed)
    df = assets.drop(columns=["geometry"]).copy()

    # Simulate z-scores correlated with latitude (higher near equator)
    lat_abs = df["latitude"].abs()
    df["heat_max_c_zscore"]    = rng.normal(2.0 - lat_abs * 0.04, 1.0, n_synthetic)
    df["precip_max_mm_zscore"] = rng.normal(1.5 - lat_abs * 0.03, 1.2, n_synthetic)
    df["wind_max_ms_zscore"]   = rng.normal(1.0 - lat_abs * 0.02, 1.5, n_synthetic)

    return df


def _resolve_zscore_cols(df: pd.DataFrame) -> list[str]:
    """Return whichever z-score column set is present in df."""
    if all(c in df.columns for c in ZSCORE_COLS):
        return ZSCORE_COLS
    if all(c in df.columns for c in ZSCORE_COLS_FALLBACK):
        logger.warning(
            "precipitation z-score column not found; "
            "flood score will use heat z-score as proxy."
        )
        return ZSCORE_COLS_FALLBACK
    present = [c for c in df.columns if c.endswith("_zscore")]
    if len(present) < 3:
        raise ValueError(
            f"Need at least 3 z-score columns, found: {present}. "
            "Re-run the preprocessing pipeline."
        )
    return present[:3]


def run_pipeline(
    features_path: Path | None = None,
    output_dir: Path | None = None,
    run_name: str | None = None,
    haircut_method: str = "linear",
    alpha: float | None = None,
    anomaly_contamination: float = 0.05,
    baseline_year_start: int | None = None,
    baseline_year_end: int | None = None,
    n_synthetic: int = 200,
    seed: int = 42,
    run_sensitivity: bool = True,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Run the full climate risk model pipeline end-to-end.

    Args:
        features_path: Path to processed features parquet. If None, synthetic
                       data is generated for development/demo use.
        output_dir: Where to save artifacts. Defaults to a timestamped
                    subdirectory under data/processed/runs/.
        run_name: MLflow run display name. Auto-generated if None.
        haircut_method: 'linear' or 'convex' haircut schedule.
        alpha: Risk haircut coefficient. Defaults to settings value.
        anomaly_contamination: Expected tail-risk fraction for IsolationForest.
        baseline_year_start: Start of climatological baseline for scorer fit.
        baseline_year_end: End of climatological baseline for scorer fit.
        n_synthetic: Number of synthetic assets if no real data available.
        seed: Random seed for reproducibility.
        run_sensitivity: Whether to compute alpha and weight sensitivity tables.
        log_mlflow: Whether to log results to MLflow.

    Returns:
        Dict with keys:
            scored_df       — fully scored + valued DataFrame
            metrics         — evaluation metrics dict
            portfolio        — portfolio summary dict
            climate_var      — Climate VaR dict
            sector_attr      — sector attribution DataFrame
            alpha_sensitivity — sensitivity table (if run_sensitivity=True)
            weight_sensitivity — weight sensitivity table (if run_sensitivity=True)
            run_id           — MLflow run ID (empty string if not logged)
            output_dir       — Path where artifacts were saved
    """
    alpha = alpha if alpha is not None else settings.risk_haircut_alpha
    baseline_year_start = baseline_year_start or settings.era5_year_start
    baseline_year_end   = baseline_year_end   or min(
        settings.era5_year_end, baseline_year_start + 30
    )
    run_name = run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_dir or (
        settings.data_processed_dir / "runs" / run_name
    )

    logger.info(f"{'='*60}")
    logger.info(f"Climate Risk Pipeline: {run_name}")
    logger.info(
        f"  alpha={alpha}, method={haircut_method}, "
        f"baseline={baseline_year_start}–{baseline_year_end}"
    )
    logger.info(f"{'='*60}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = _load_or_synthesise(features_path, n_synthetic, seed)

    # ── 2. Data validation ────────────────────────────────────────────────────
    logger.info("Step 1/7 — Data validation")
    import geopandas as gpd
    from shapely.geometry import Point
    if "geometry" not in df.columns and "latitude" in df.columns:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
    else:
        gdf = df  # type: ignore[assignment]

    validation_results = run_all_checks(
        assets=gdf if isinstance(gdf, gpd.GeoDataFrame) else None,
        feature_df=df if "composite_score" in df.columns else None,
    )
    n_failed = sum(not r.passed for r in validation_results)
    if n_failed > 0:
        logger.warning(f"{n_failed} validation check(s) failed — proceeding with caution.")

    # ── 3. Fit HazardScorer ───────────────────────────────────────────────────
    logger.info("Step 2/7 — Fitting HazardScorer")
    zscore_cols = _resolve_zscore_cols(df)

    # Filter to baseline years if 'year' column is present
    baseline_df = (
        df[df["year"].between(baseline_year_start, baseline_year_end)]
        if "year" in df.columns
        else df
    )
    hazard_scorer = HazardScorer(clip_zscore=5.0)
    df = hazard_scorer.fit_transform(df, zscore_cols)

    # ── 4. CompositeRiskModel ─────────────────────────────────────────────────
    logger.info("Step 3/7 — Computing composite risk scores")
    composite_model = CompositeRiskModel()

    # Ensure sector column exists
    if "sector" not in df.columns:
        df["sector"] = "Unknown"

    df = composite_model.score(df)

    # PCA diagnostic
    pca_diagnostics = composite_model.fit_pca(df)
    logger.info(
        f"  PCA: PC1 explains {pca_diagnostics['pc1_variance_pct']}% "
        f"of hazard variance"
    )

    # ── 5. Anomaly detection ──────────────────────────────────────────────────
    logger.info("Step 4/7 — Anomaly / tail-risk detection")
    anomaly_detector = AnomalyDetector(contamination=anomaly_contamination)
    df = anomaly_detector.fit_predict(df)

    # ── 6. Valuation ──────────────────────────────────────────────────────────
    logger.info("Step 5/7 — Risk-adjusted valuation")
    if "book_value" not in df.columns:
        import numpy as np
        rng_bv = __import__("numpy").random.default_rng(seed + 1)
        df["book_value"] = rng_bv.lognormal(5.0, 1.2, len(df)).round(2)

    df = apply_haircut(df, method=haircut_method, alpha=alpha)
    port_summary   = portfolio_summary(df)
    cvar           = climate_var(df)
    sector_attr    = sector_attribution(df)

    logger.info(
        f"  NAV: {port_summary['total_risk_adjusted_nav_usd_m']:.1f} USD M | "
        f"Haircut: {port_summary['portfolio_haircut_pct']:.1f}% | "
        f"95% Climate VaR: {cvar['var_usd_m']:.1f} USD M"
    )

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    logger.info("Step 6/7 — Evaluation")
    metrics = evaluate(df, portfolio_summary=port_summary, climate_var=cvar)
    metrics["pca"] = pca_diagnostics
    metrics["validation_failures"] = n_failed

    # Optional sensitivity analyses
    alpha_sens = weight_sens = None
    if run_sensitivity:
        logger.info("  Running sensitivity analyses…")
        alpha_sens  = alpha_sensitivity(df)
        weight_sens = weight_sensitivity(df)
        logger.info("  Sensitivity analyses complete.")

    # ── 8. Log + save ─────────────────────────────────────────────────────────
    logger.info("Step 7/7 — Logging and saving artifacts")
    params: dict[str, Any] = {
        "alpha":                 alpha,
        "haircut_method":        haircut_method,
        "anomaly_contamination": anomaly_contamination,
        "baseline_year_start":   baseline_year_start,
        "baseline_year_end":     baseline_year_end,
        "n_assets":              len(df),
        "weight_flood":          settings.weight_flood,
        "weight_heat":           settings.weight_heat,
        "weight_cyclone":        settings.weight_cyclone,
    }

    run_id = ""
    if log_mlflow:
        try:
            run_id = log_experiment(
                metrics=metrics,
                params=params,
                artifacts_dir=output_dir,
                run_name=run_name,
            )
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    save_run(
        output_dir=output_dir,
        hazard_scorer=hazard_scorer,
        composite_model=composite_model,
        anomaly_detector=anomaly_detector,
        scored_df=df,
        metrics=metrics,
        extra_metadata={"params": params, "mlflow_run_id": run_id},
    )

    logger.info(f"{'='*60}")
    logger.success(f"Pipeline complete. Artifacts → {output_dir}")
    logger.info(f"{'='*60}")

    result: dict[str, Any] = {
        "scored_df":          df,
        "metrics":            metrics,
        "portfolio":          port_summary,
        "climate_var":        cvar,
        "sector_attr":        sector_attr,
        "run_id":             run_id,
        "output_dir":         output_dir,
    }
    if run_sensitivity:
        result["alpha_sensitivity"]  = alpha_sens
        result["weight_sensitivity"] = weight_sens

    return result

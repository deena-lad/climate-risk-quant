"""
src/model/valuation.py
─────────────────────────────────────────────────────────────────────────────
Risk-adjusted valuation: applies climate haircuts to book values and
computes portfolio-level climate Value-at-Risk (Climate VaR).

Valuation model:
    risk_adjusted_value = book_value × (1 − α × composite_score / 100)

Where α (risk_haircut_alpha) is the maximum proportional haircut at a
composite score of 100.  This linear specification is the simplest
defensible form; the module also provides a convex haircut variant
(α × (score/100)^β) for sensitivity analysis.

Climate VaR is computed as the weighted-average haircut at a given
confidence level (e.g. 95th-percentile worst-case exposure), consistent
with ECB climate stress-test methodology (ECB, 2022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings


# ── Core haircut functions ────────────────────────────────────────────────────

def linear_haircut(
    composite_score: pd.Series,
    alpha: float | None = None,
) -> pd.Series:
    """Compute a linear climate haircut fraction for each asset.

    haircut_fraction = α × (composite_score / 100)

    Args:
        composite_score: Series of 0–100 composite risk scores.
        alpha: Maximum haircut coefficient. Defaults to settings value.

    Returns:
        Series of haircut fractions in [0, α].
    """
    alpha = alpha if alpha is not None else settings.risk_haircut_alpha
    return (alpha * composite_score / 100.0).clip(0.0, alpha)


def convex_haircut(
    composite_score: pd.Series,
    alpha: float | None = None,
    beta: float = 2.0,
) -> pd.Series:
    """Compute a convex (accelerating) climate haircut.

    haircut_fraction = α × (composite_score / 100)^β

    A beta > 1 creates a convex schedule: assets near score=50 get
    relatively modest haircuts, but scores above ~75 see rapidly
    increasing adjustments — better reflecting non-linear climate tail risk.

    Args:
        composite_score: Series of 0–100 composite risk scores.
        alpha: Maximum haircut. Defaults to settings value.
        beta: Convexity parameter (1.0 = linear, 2.0 = quadratic).

    Returns:
        Series of haircut fractions in [0, α].
    """
    alpha = alpha if alpha is not None else settings.risk_haircut_alpha
    return (alpha * (composite_score / 100.0) ** beta).clip(0.0, alpha)


def apply_haircut(
    df: pd.DataFrame,
    method: str = "linear",
    alpha: float | None = None,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Apply a climate haircut to each asset's book value.

    Adds columns:
        haircut_fraction  — the fractional reduction applied
        haircut_usd       — absolute value reduction (USD M)
        risk_adjusted_value — book_value × (1 − haircut_fraction)

    Args:
        df: DataFrame with composite_score and book_value columns.
        method: 'linear' or 'convex'.
        alpha: Maximum haircut coefficient.
        beta: Convexity parameter (only used when method='convex').

    Returns:
        df with three added columns.

    Raises:
        ValueError: If required columns are missing or method is unknown.
    """
    for col in ("composite_score", "book_value"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    df = df.copy()

    if method == "linear":
        df["haircut_fraction"] = linear_haircut(df["composite_score"], alpha)
    elif method == "convex":
        df["haircut_fraction"] = convex_haircut(df["composite_score"], alpha, beta)
    else:
        raise ValueError(f"Unknown haircut method '{method}'. Choose 'linear' or 'convex'.")

    df["haircut_usd"] = (df["book_value"] * df["haircut_fraction"]).round(4)
    df["risk_adjusted_value"] = (df["book_value"] - df["haircut_usd"]).round(4)

    logger.info(
        f"Haircut applied ({method}, α={alpha or settings.risk_haircut_alpha}). "
        f"Total haircut: {df['haircut_usd'].sum():.1f} USD M "
        f"({df['haircut_usd'].sum() / df['book_value'].sum():.1%} of NAV)"
    )
    return df


# ── Portfolio-level metrics ───────────────────────────────────────────────────

def portfolio_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate portfolio climate risk statistics.

    Args:
        df: DataFrame after apply_haircut().

    Returns:
        Dict with keys:
            total_book_value_usd_m
            total_risk_adjusted_nav_usd_m
            total_haircut_usd_m
            portfolio_haircut_pct
            weighted_avg_composite_score
            n_assets
            n_high_risk          (esg_tier in High/Critical)
            high_risk_nav_pct
    """
    required = ["book_value", "risk_adjusted_value", "composite_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' required for portfolio_summary().")

    total_bv   = float(df["book_value"].sum())
    total_nav  = float(df["risk_adjusted_value"].sum())
    total_cut  = total_bv - total_nav
    wavg_score = float(
        (df["composite_score"] * df["book_value"]).sum() / total_bv
    )

    high_risk_mask = (
        df["esg_tier"].isin(["High", "Critical"])
        if "esg_tier" in df.columns
        else pd.Series(False, index=df.index)
    )
    high_risk_nav  = float(df.loc[high_risk_mask, "risk_adjusted_value"].sum())

    summary = {
        "total_book_value_usd_m":       round(total_bv,   2),
        "total_risk_adjusted_nav_usd_m": round(total_nav,  2),
        "total_haircut_usd_m":          round(total_cut,  2),
        "portfolio_haircut_pct":        round(total_cut / total_bv * 100, 2),
        "weighted_avg_composite_score": round(wavg_score, 2),
        "n_assets":                     int(len(df)),
        "n_high_risk":                  int(high_risk_mask.sum()),
        "high_risk_nav_pct":            round(high_risk_nav / total_nav * 100, 2),
    }
    logger.info(
        f"Portfolio summary: NAV={total_nav:.1f} USD M, "
        f"haircut={summary['portfolio_haircut_pct']:.1f}%, "
        f"weighted score={wavg_score:.1f}"
    )
    return summary


def climate_var(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate portfolio Climate Value-at-Risk via Monte Carlo simulation.

    Methodology (aligned with ECB 2022 climate stress test):
      - Sample composite_score uncertainty using ±15% multiplicative noise
        to represent model/scenario uncertainty.
      - For each simulation, recompute haircuts and portfolio NAV.
      - Climate VaR = difference between baseline NAV and the
        (1 − confidence_level) percentile of simulated NAVs.

    Args:
        df: DataFrame after apply_haircut() with book_value and
            composite_score columns.
        confidence_level: VaR confidence level (default 0.95 = 95th pctl).
        n_simulations: Number of Monte Carlo paths.
        seed: Random seed.

    Returns:
        Dict with keys:
            baseline_nav_usd_m
            var_usd_m          — Climate VaR at given confidence level
            var_pct_nav        — VaR as % of baseline NAV
            es_usd_m           — Expected Shortfall (CVaR)
            confidence_level
    """
    rng = np.random.default_rng(seed)
    scores = df["composite_score"].values          # (n_assets,)
    book_values = df["book_value"].values          # (n_assets,)
    alpha = settings.risk_haircut_alpha

    baseline_nav = float(
        (book_values * (1 - alpha * scores / 100)).sum()
    )

    # Monte Carlo: perturb scores with multiplicative noise
    noise = rng.normal(loc=1.0, scale=0.15, size=(n_simulations, len(scores)))
    sim_scores = np.clip(scores[np.newaxis, :] * noise, 0, 100)
    sim_haircuts = alpha * sim_scores / 100
    sim_navs = (book_values[np.newaxis, :] * (1 - sim_haircuts)).sum(axis=1)

    var_threshold = np.percentile(sim_navs, (1 - confidence_level) * 100)
    var_usd = baseline_nav - var_threshold
    es_usd  = baseline_nav - sim_navs[sim_navs <= var_threshold].mean()

    result = {
        "baseline_nav_usd_m": round(baseline_nav, 2),
        "var_usd_m":          round(float(var_usd), 2),
        "var_pct_nav":        round(float(var_usd) / baseline_nav * 100, 2),
        "es_usd_m":           round(float(es_usd), 2),
        "confidence_level":   confidence_level,
    }
    logger.info(
        f"Climate VaR ({confidence_level:.0%}): {var_usd:.1f} USD M "
        f"({result['var_pct_nav']:.1f}% of NAV), ES: {es_usd:.1f} USD M"
    )
    return result


def sector_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Break down total climate haircut by sector.

    Args:
        df: DataFrame after apply_haircut() with sector column.

    Returns:
        DataFrame indexed by sector with columns:
            n_assets, total_book_value, total_haircut_usd,
            avg_composite_score, haircut_pct, share_of_total_haircut.
    """
    if "sector" not in df.columns:
        raise ValueError("'sector' column required for sector_attribution().")

    agg = df.groupby("sector").agg(
        n_assets=("asset_id", "count") if "asset_id" in df.columns
                 else ("book_value", "count"),
        total_book_value=("book_value", "sum"),
        total_haircut_usd=("haircut_usd", "sum"),
        avg_composite_score=("composite_score", "mean"),
    ).reset_index()

    agg["haircut_pct"] = (
        agg["total_haircut_usd"] / agg["total_book_value"] * 100
    ).round(2)
    agg["share_of_total_haircut"] = (
        agg["total_haircut_usd"] / agg["total_haircut_usd"].sum() * 100
    ).round(2)
    agg = agg.sort_values("total_haircut_usd", ascending=False).reset_index(drop=True)

    return agg

"""
src/model/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation suite and MLflow experiment tracking.

Metrics computed:
  Score distribution quality:
    - Gini coefficient: measures differentiation across the portfolio.
      A high Gini (→1) means scores spread well; Gini near 0 means all
      assets get similar scores (useless model).
    - KS statistic: compares score distribution to a reference uniform.
      A well-calibrated risk model should deviate from uniform (has signal).
    - Tail concentration ratio: share of total book value in top-quartile
      risk assets. Key metric for systemic risk assessment.

  Compound risk quality:
    - Compound detection rate: what fraction of top-composite-score assets
      are also flagged as compound risk?

  Anomaly detection:
    - Contamination realised: actual fraction flagged vs intended.

  Valuation:
    - Portfolio haircut % and Climate VaR % are the headline numbers
      a risk officer would report to a board.

All metrics, parameters, and the scored DataFrame are logged to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.config import settings


# ── Individual metric functions ───────────────────────────────────────────────

def gini_coefficient(scores: pd.Series) -> float:
    """Compute the Gini coefficient of a score distribution.

    Gini = 0   → all assets have identical scores (no differentiation)
    Gini = 1   → one asset has all the risk (maximum concentration)
    Target for a useful risk model: Gini > 0.30.

    Args:
        scores: Series of non-negative risk scores.

    Returns:
        Gini coefficient in [0, 1].
    """
    arr = np.asarray(scores.dropna(), dtype=float)

    if len(arr) == 0:
        return float("nan")

    if np.all(arr == 0):
        return 0.0

    # Ensure non-negative values
    arr = np.clip(arr, 0, None)

    # Sort ascending
    arr = np.sort(arr)

    n = len(arr)
    index = np.arange(1, n + 1)

    gini = np.sum((2 * index - n - 1) * arr) / (n * arr.sum())
    gini = max(0.0, min(1.0, gini))

    return float(gini)


def ks_uniformity(scores: pd.Series) -> dict[str, float]:
    """KS test comparing score distribution to Uniform[0,100].

    A model with signal should deviate significantly from uniform —
    some regions/assets should cluster at high scores.

    Args:
        scores: Series of 0–100 risk scores.

    Returns:
        Dict with ks_statistic and ks_pvalue.
    """
    normed = scores.dropna() / 100.0
    ks_stat, ks_pval = stats.kstest(normed, "uniform")
    return {"ks_statistic": round(float(ks_stat), 4), "ks_pvalue": round(float(ks_pval), 4)}


def tail_concentration_ratio(
    df: pd.DataFrame,
    score_col: str = "composite_score",
    value_col: str = "book_value",
    top_quantile: float = 0.75,
) -> float:
    """Fraction of total portfolio book value in the top-risk quantile.

    Args:
        df: Feature DataFrame.
        score_col: Column of risk scores.
        value_col: Column of asset book values.
        top_quantile: Threshold percentile (0.75 = top quartile).

    Returns:
        Ratio in [0, 1].
    """
    threshold = df[score_col].quantile(top_quantile)
    top_value  = df.loc[df[score_col] >= threshold, value_col].sum()
    total      = df[value_col].sum()
    return round(float(top_value / total), 4) if total > 0 else 0.0


def compound_detection_rate(
    df: pd.DataFrame,
    top_quantile: float = 0.75,
) -> float:
    """Fraction of high-composite-score assets also flagged as compound.

    Args:
        df: DataFrame with composite_score and compound_risk columns.
        top_quantile: Top-score threshold defining 'high risk'.

    Returns:
        Detection rate in [0, 1], or nan if columns missing.
    """
    if "compound_risk" not in df.columns or "composite_score" not in df.columns:
        return float("nan")
    threshold = df["composite_score"].quantile(top_quantile)
    high_risk = df[df["composite_score"] >= threshold]
    if len(high_risk) == 0:
        return 0.0
    return round(float(high_risk["compound_risk"].mean()), 4)


def score_percentile_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute p25/p50/p75/p95 for all hazard and composite scores.

    Args:
        df: Feature DataFrame with score columns.

    Returns:
        Nested dict: {score_col: {p25, p50, p75, p95}}.
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    summary: dict[str, Any] = {}
    for col in score_cols:
        q = df[col].quantile([0.25, 0.50, 0.75, 0.95]).round(1).to_dict()
        summary[col] = {f"p{int(k*100)}": v for k, v in q.items()}
    return summary


# ── Full evaluation suite ─────────────────────────────────────────────────────

def evaluate(
    df: pd.DataFrame,
    portfolio_summary: dict[str, float] | None = None,
    climate_var: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation suite and return all metrics.

    Args:
        df: Fully scored + valued DataFrame.
        portfolio_summary: Output of valuation.portfolio_summary().
        climate_var: Output of valuation.climate_var().

    Returns:
        Dict of all computed metrics.
    """
    metrics: dict[str, Any] = {}

    # Score distribution quality
    if "composite_score" in df.columns:
        metrics["gini_composite"]         = gini_coefficient(df["composite_score"])
        ks = ks_uniformity(df["composite_score"])
        metrics.update(ks)
        metrics["tail_concentration_75"]  = tail_concentration_ratio(df)
        metrics["compound_detection_rate"]= compound_detection_rate(df)
        metrics["percentile_summary"]     = score_percentile_summary(df)

    # Per-hazard Gini
    for hazard in ["flood_score", "heat_score", "cyclone_score"]:
        if hazard in df.columns:
            metrics[f"gini_{hazard.replace('_score','')}"] = gini_coefficient(df[hazard])

    # Anomaly detection realised contamination
    if "tail_risk_flag" in df.columns:
        metrics["anomaly_contamination_realised"] = round(
            float(df["tail_risk_flag"].mean()), 4
        )

    # Valuation metrics (passed in from valuation module)
    if portfolio_summary:
        metrics.update({f"port_{k}": v for k, v in portfolio_summary.items()})
    if climate_var:
        metrics.update({f"cvar_{k}": v for k, v in climate_var.items()})

    logger.info(
        f"Evaluation complete. "
        f"Gini={metrics.get('gini_composite', 'n/a'):.3f}, "
        f"Tail-conc={metrics.get('tail_concentration_75', 'n/a'):.2%}, "
        f"Compound-detect={metrics.get('compound_detection_rate', 'n/a'):.2%}"
    )
    return metrics


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_experiment(
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifacts_dir: Path | None = None,
    run_name: str = "climate_risk_run",
    experiment_name: str = "climate-risk-quant",
) -> str:
    """Log parameters, metrics, and artifacts to MLflow.

    Args:
        metrics: Dict from evaluate().
        params: Model hyperparameters / config to log.
        artifacts_dir: Directory of files to log as MLflow artifacts.
        run_name: Display name for this run.
        experiment_name: MLflow experiment to log under.

    Returns:
        MLflow run_id string.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Parameters
        flat_params = _flatten_dict(params)
        mlflow.log_params(flat_params)

        # Metrics (scalars only; skip nested dicts)
        flat_metrics = {
            k: v for k, v in _flatten_dict(metrics).items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(flat_metrics)

        # Log full metrics JSON as artifact
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = artifacts_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
            mlflow.log_artifact(str(metrics_path))

            # Log any parquet files in the artifacts dir
            for parquet_file in artifacts_dir.glob("*.parquet"):
                mlflow.log_artifact(str(parquet_file))

        run_id = run.info.run_id
        logger.success(f"MLflow run logged: {run_id} (experiment='{experiment_name}')")
        return run_id


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict for MLflow logging."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def alpha_sensitivity(
    df: pd.DataFrame,
    alpha_range: list[float] | None = None,
) -> pd.DataFrame:
    """Compute portfolio haircut across a range of alpha values.

    Useful for showing how sensitive the NAV is to the haircut assumption —
    a key parameter when presenting to a risk committee.

    Args:
        df: DataFrame with composite_score and book_value.
        alpha_range: List of alpha values to test. Defaults to 0.05–0.50.

    Returns:
        DataFrame with columns: alpha, total_haircut_usd, portfolio_haircut_pct,
        risk_adjusted_nav.
    """
    from src.model.valuation import apply_haircut, portfolio_summary

    if alpha_range is None:
        alpha_range = [round(x, 2) for x in np.arange(0.05, 0.55, 0.05)]

    rows = []
    for alpha in alpha_range:
        valued = apply_haircut(df.copy(), method="linear", alpha=alpha)
        summary = portfolio_summary(valued)
        rows.append(
            {
                "alpha": alpha,
                "total_haircut_usd_m": summary["total_haircut_usd_m"],
                "portfolio_haircut_pct": summary["portfolio_haircut_pct"],
                "risk_adjusted_nav_usd_m": summary["total_risk_adjusted_nav_usd_m"],
            }
        )
    result = pd.DataFrame(rows)
    logger.info(f"Alpha sensitivity computed for {len(alpha_range)} values.")
    return result


def weight_sensitivity(
    df: pd.DataFrame,
    n_samples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte Carlo sensitivity: sample random hazard weights, measure Gini.

    Shows how robust the composite score differentiation is to weight
    uncertainty — a common question in model validation reviews.

    Args:
        df: DataFrame with flood_score, heat_score, cyclone_score.
        n_samples: Number of random weight combinations to test.
        seed: Random seed.

    Returns:
        DataFrame with columns: w_flood, w_heat, w_cyclone, gini.
    """
    from src.model.scorer import SCORE_COLS

    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_samples):
        raw = rng.dirichlet(np.ones(3))  # random weights summing to 1
        w_flood, w_heat, w_cyclone = raw
        scores_matrix = df[SCORE_COLS].fillna(50.0).values
        composite = scores_matrix @ raw
        gini = gini_coefficient(pd.Series(composite))
        results.append(
            {
                "w_flood":   round(float(w_flood), 4),
                "w_heat":    round(float(w_heat), 4),
                "w_cyclone": round(float(w_cyclone), 4),
                "gini":      round(gini, 4),
            }
        )
    return pd.DataFrame(results)

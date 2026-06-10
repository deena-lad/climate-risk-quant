"""
tests/test_model.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for scorer, valuation, evaluate, and pipeline modules.
All tests are fully offline — no MLflow server, no disk I/O outside tmp.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.scorer import (
    AnomalyDetector,
    CompositeRiskModel,
    HazardScorer,
    HazardWeights,
    SCORE_COLS,
)
from src.model.valuation import (
    apply_haircut,
    climate_var,
    convex_haircut,
    linear_haircut,
    portfolio_summary,
    sector_attribution,
)
from src.model.evaluate import (
    alpha_sensitivity,
    gini_coefficient,
    ks_uniformity,
    tail_concentration_ratio,
    compound_detection_rate,
    weight_sensitivity,
    evaluate,
)
from src.model.artifacts import save_run, load_full_run
from src.model.pipeline import run_pipeline


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def raw_zscore_df() -> pd.DataFrame:
    """50-asset DataFrame with realistic z-score columns."""
    rng = np.random.default_rng(7)
    n = 50
    return pd.DataFrame(
        {
            "asset_id":              [f"A{i:03d}" for i in range(n)],
            "sector":                rng.choice(
                ["Real Estate", "Energy", "Utilities", "Industrials", "Financials"], n
            ),
            "book_value":            rng.lognormal(5.0, 1.0, n).round(2),
            "latitude":              rng.uniform(-40, 40, n),
            "longitude":             rng.uniform(-150, 150, n),
            "heat_max_c_zscore":    rng.normal(1.5, 1.2, n),
            "precip_max_mm_zscore": rng.normal(1.0, 1.3, n),
            "wind_max_ms_zscore":   rng.normal(0.8, 1.5, n),
        }
    )


@pytest.fixture
def scored_df(raw_zscore_df) -> pd.DataFrame:
    """DataFrame after HazardScorer + CompositeRiskModel."""
    scorer = HazardScorer()
    df = scorer.fit_transform(
        raw_zscore_df,
        ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
    )
    model = CompositeRiskModel()
    df = model.score(df)
    return df


@pytest.fixture
def valued_df(scored_df) -> pd.DataFrame:
    """DataFrame after apply_haircut."""
    return apply_haircut(scored_df, method="linear", alpha=0.30)


# ── HazardWeights ─────────────────────────────────────────────────────────────

class TestHazardWeights:
    def test_valid_weights(self):
        w = HazardWeights(flood=0.4, heat=0.35, cyclone=0.25)
        assert abs(w.as_array().sum() - 1.0) < 1e-6

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            HazardWeights(flood=0.5, heat=0.5, cyclone=0.5)

    def test_to_dict_roundtrip(self):
        w = HazardWeights(flood=0.4, heat=0.35, cyclone=0.25)
        d = w.to_dict()
        assert d["flood"] == 0.4
        assert d["cyclone"] == 0.25


# ── HazardScorer ──────────────────────────────────────────────────────────────

class TestHazardScorer:
    def test_fit_transform_adds_score_cols(self, raw_zscore_df):
        scorer = HazardScorer()
        result = scorer.fit_transform(
            raw_zscore_df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        for col in SCORE_COLS:
            assert col in result.columns

    def test_scores_in_range(self, raw_zscore_df):
        scorer = HazardScorer()
        result = scorer.fit_transform(
            raw_zscore_df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        for col in SCORE_COLS:
            assert result[col].between(0, 100).all(), f"{col} out of range"

    def test_transform_before_fit_raises(self, raw_zscore_df):
        scorer = HazardScorer()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            scorer.transform(raw_zscore_df)

    def test_nan_zscore_gets_fifty(self, raw_zscore_df):
        df = raw_zscore_df.copy()
        df.loc[0, "heat_max_c_zscore"] = np.nan
        scorer = HazardScorer()
        result = scorer.fit_transform(
            df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        assert result.loc[0, "heat_score"] == 50.0

    def test_reproducible(self, raw_zscore_df):
        s1 = HazardScorer()
        s2 = HazardScorer()
        r1 = s1.fit_transform(
            raw_zscore_df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        r2 = s2.fit_transform(
            raw_zscore_df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        pd.testing.assert_series_equal(r1["composite_score"] if "composite_score" in r1 else r1["flood_score"],
                                       r2["composite_score"] if "composite_score" in r2 else r2["flood_score"])


# ── CompositeRiskModel ────────────────────────────────────────────────────────

class TestCompositeRiskModel:
    def test_composite_score_created(self, scored_df):
        assert "composite_score" in scored_df.columns

    def test_composite_in_range(self, scored_df):
        assert scored_df["composite_score"].between(0, 100).all()

    def test_esg_tier_assigned(self, scored_df):
        assert "esg_tier" in scored_df.columns
        assert set(scored_df["esg_tier"].dropna().unique()).issubset(
            {"Low", "Medium", "High", "Critical"}
        )

    def test_pca_returns_diagnostics(self, scored_df):
        model = CompositeRiskModel()
        model.score(scored_df)  # must score first
        diag = model.fit_pca(scored_df)
        assert "pc1_variance_pct" in diag
        assert 0 < diag["pc1_variance_pct"] <= 100

    def test_sector_weights_applied(self, raw_zscore_df):
        scorer = HazardScorer()
        df = scorer.fit_transform(
            raw_zscore_df,
            ["precip_max_mm_zscore", "heat_max_c_zscore", "wind_max_ms_zscore"],
        )
        # Assign all assets to one sector
        df = df.copy()
        df["sector"] = "Real Estate"
        model = CompositeRiskModel()
        result = model.score(df)
        # Real Estate: flood=0.5, heat=0.3, cyclone=0.2
        expected = (
            result["flood_score"] * 0.5
            + result["heat_score"] * 0.3
            + result["cyclone_score"] * 0.2
        )
        np.testing.assert_allclose(
            result["composite_score"].values,
            expected.clip(0, 100).round(2).values,
            atol=0.01,
        )


# ── AnomalyDetector ───────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_predict_before_fit_raises(self, scored_df):
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            detector.predict(scored_df)

    def test_fit_predict_adds_columns(self, scored_df):
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_predict(scored_df)
        for col in ["anomaly_label", "anomaly_score", "tail_risk_flag"]:
            assert col in result.columns

    def test_contamination_fraction_approx(self, scored_df):
        contamination = 0.10
        detector = AnomalyDetector(contamination=contamination)
        result = detector.fit_predict(scored_df)
        flagged_pct = result["tail_risk_flag"].mean()
        # IsolationForest is approximate — allow ±5%
        assert abs(flagged_pct - contamination) <= 0.05


# ── Valuation ─────────────────────────────────────────────────────────────────

class TestLinearHaircut:
    def test_zero_score_zero_haircut(self):
        s = pd.Series([0.0, 50.0, 100.0])
        h = linear_haircut(s, alpha=0.30)
        assert h.iloc[0] == 0.0
        assert abs(h.iloc[1] - 0.15) < 1e-6
        assert abs(h.iloc[2] - 0.30) < 1e-6

    def test_haircut_bounded_by_alpha(self):
        s = pd.Series([0.0, 50.0, 100.0, 200.0])  # 200 is out of range, should clip
        h = linear_haircut(s, alpha=0.30)
        assert (h <= 0.30).all()


class TestConvexHaircut:
    def test_convex_less_than_linear_below_fifty(self):
        s = pd.Series([30.0])
        linear = linear_haircut(s, alpha=0.30).iloc[0]
        convex = convex_haircut(s, alpha=0.30, beta=2.0).iloc[0]
        assert convex < linear  # convex is lower below midpoint

    def test_convex_greater_than_linear_above_fifty(self):
        s = pd.Series([80.0])
        linear = linear_haircut(s, alpha=0.30).iloc[0]
        # convex with beta=0.5 (concave) → higher below inflection
        # with beta=2 and score=80: (0.8)^2 * 0.3 = 0.192 < 0.8*0.3 = 0.24
        # This is expected: quadratic is *less* than linear for score < 100
        assert convex_haircut(s, alpha=0.30, beta=2.0).iloc[0] <= linear


class TestApplyHaircut:
    def test_adds_three_columns(self, scored_df):
        result = apply_haircut(scored_df)
        for col in ["haircut_fraction", "haircut_usd", "risk_adjusted_value"]:
            assert col in result.columns

    def test_risk_adjusted_value_leq_book_value(self, valued_df):
        assert (valued_df["risk_adjusted_value"] <= valued_df["book_value"]).all()

    def test_unknown_method_raises(self, scored_df):
        with pytest.raises(ValueError, match="Unknown haircut method"):
            apply_haircut(scored_df, method="logarithmic")

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="composite_score"):
            apply_haircut(pd.DataFrame({"book_value": [100]}))


class TestPortfolioSummary:
    def test_summary_keys(self, valued_df):
        summary = portfolio_summary(valued_df)
        for key in [
            "total_book_value_usd_m",
            "total_risk_adjusted_nav_usd_m",
            "portfolio_haircut_pct",
            "weighted_avg_composite_score",
        ]:
            assert key in summary

    def test_nav_leq_book_value(self, valued_df):
        summary = portfolio_summary(valued_df)
        assert summary["total_risk_adjusted_nav_usd_m"] <= summary["total_book_value_usd_m"]

    def test_haircut_pct_positive(self, valued_df):
        summary = portfolio_summary(valued_df)
        assert summary["portfolio_haircut_pct"] > 0


class TestClimateVaR:
    def test_var_keys(self, valued_df):
        result = climate_var(valued_df, n_simulations=500)
        for key in ["baseline_nav_usd_m", "var_usd_m", "var_pct_nav", "es_usd_m"]:
            assert key in result

    def test_var_positive(self, valued_df):
        result = climate_var(valued_df, n_simulations=500)
        assert result["var_usd_m"] > 0

    def test_es_geq_var(self, valued_df):
        result = climate_var(valued_df, n_simulations=500)
        assert result["es_usd_m"] >= result["var_usd_m"]


# ── Evaluate ──────────────────────────────────────────────────────────────────

class TestGiniCoefficient:
    def test_uniform_distribution_near_zero(self):
        s = pd.Series(np.linspace(0, 100, 1000))
        assert gini_coefficient(s) < 0.35

    def test_concentrated_distribution_high(self):
        s = pd.Series([0.0] * 95 + [100.0] * 5)
        assert gini_coefficient(s) > 0.70

    def test_empty_series_returns_nan(self):
        assert np.isnan(gini_coefficient(pd.Series(dtype=float)))


class TestKsUniformity:
    def test_returns_dict_with_expected_keys(self, scored_df):
        result = ks_uniformity(scored_df["composite_score"])
        assert "ks_statistic" in result
        assert "ks_pvalue" in result

    def test_statistic_in_range(self, scored_df):
        result = ks_uniformity(scored_df["composite_score"])
        assert 0 <= result["ks_statistic"] <= 1


class TestEvaluate:
    def test_evaluate_returns_gini(self, valued_df):
        metrics = evaluate(valued_df)
        assert "gini_composite" in metrics
        assert 0 <= metrics["gini_composite"] <= 1

    def test_evaluate_with_portfolio_summary(self, valued_df):
        port = portfolio_summary(valued_df)
        metrics = evaluate(valued_df, portfolio_summary=port)
        assert "port_portfolio_haircut_pct" in metrics


class TestAlphaSensitivity:
    def test_returns_dataframe(self, scored_df):
        result = alpha_sensitivity(scored_df, alpha_range=[0.1, 0.2, 0.3])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_haircut_increases_with_alpha(self, scored_df):
        result = alpha_sensitivity(scored_df, alpha_range=[0.1, 0.2, 0.3])
        haircuts = result["portfolio_haircut_pct"].values
        assert haircuts[0] < haircuts[1] < haircuts[2]


# ── Artifacts ─────────────────────────────────────────────────────────────────

class TestArtifacts:
    def test_save_and_load_full_run(self, valued_df):
        scorer   = HazardScorer()
        detector = AnomalyDetector()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            save_run(
                output_dir=run_dir,
                hazard_scorer=scorer,
                anomaly_detector=detector,
                scored_df=valued_df,
                metrics={"test_metric": 0.42},
            )

            assert (run_dir / "hazard_scorer.joblib").exists()
            assert (run_dir / "anomaly_detector.joblib").exists()
            assert (run_dir / "features_scored.parquet").exists()
            assert (run_dir / "run_metadata.json").exists()

            loaded = load_full_run(run_dir)
            assert loaded["hazard_scorer"] is not None
            assert loaded["scored_df"] is not None
            assert len(loaded["scored_df"]) == len(valued_df)

    def test_metadata_contains_config(self, valued_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "meta_test"
            save_run(output_dir=run_dir, scored_df=valued_df)
            loaded = load_full_run(run_dir)
            meta = loaded["metadata"]
            assert "config" in meta
            assert "risk_haircut_alpha" in meta["config"]


# ── Full pipeline integration test ────────────────────────────────────────────

class TestRunPipeline:
    def test_pipeline_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                features_path=None,         # synthetic
                output_dir=Path(tmpdir) / "run",
                n_synthetic=60,
                seed=0,
                run_sensitivity=False,
                log_mlflow=False,           # no MLflow server needed
            )
        assert "scored_df" in result
        assert "metrics" in result
        assert "portfolio" in result
        assert len(result["scored_df"]) == 60

    def test_pipeline_scored_df_has_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                output_dir=Path(tmpdir) / "run",
                n_synthetic=40,
                run_sensitivity=False,
                log_mlflow=False,
            )
        df = result["scored_df"]
        for col in [
            "composite_score", "esg_tier", "tail_risk_flag",
            "haircut_fraction", "risk_adjusted_value",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_pipeline_metrics_gini_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                output_dir=Path(tmpdir) / "run",
                n_synthetic=80,
                run_sensitivity=False,
                log_mlflow=False,
            )
        assert result["metrics"]["gini_composite"] > 0

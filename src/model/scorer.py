"""
src/model/scorer.py
─────────────────────────────────────────────────────────────────────────────
HazardScorer and CompositeRiskModel: the core model objects.

Architecture decision — why not a regression model?
    Physical climate risk has no labelled training targets (there's no
    "ground truth" damage label per asset per year at global scale). The
    defensible and auditable approach used by MSCI, S&P Trucost, and central
    bank stress tests is:
      1. Normalise each hazard variable to a percentile score.
      2. Combine with sector-specific weights via a weighted average.
      3. Apply PCA to verify the composite is not dominated by one hazard.
      4. Expose all weights as interpretable parameters, not learned weights.

    IsolationForest is added as an anomaly detector to flag assets whose
    multi-hazard risk profile is structurally unlike the rest of the portfolio
    — a useful signal for tail-risk concentration that the linear composite
    can miss.

Classes:
    HazardScorer     — normalises raw climate z-scores to 0–100
    CompositeRiskModel — weights + combines + PCA-validates the composite
    AnomalyDetector  — IsolationForest wrapper for tail-risk flagging
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.config import settings


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class HazardWeights:
    """Sector-specific hazard weights. All rows must sum to 1.0."""

    flood: float = field(default_factory=lambda: settings.weight_flood)
    heat: float = field(default_factory=lambda: settings.weight_heat)
    cyclone: float = field(default_factory=lambda: settings.weight_cyclone)

    def __post_init__(self) -> None:
        total = self.flood + self.heat + self.cyclone
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"HazardWeights must sum to 1.0, got {total:.4f}"
            )

    def as_array(self) -> np.ndarray:
        return np.array([self.flood, self.heat, self.cyclone])

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# Default sector weight overrides (empirically derived from IPCC AR6 sector
# vulnerability assessments; real-world would come from proprietary research).
SECTOR_WEIGHTS: dict[str, HazardWeights] = {
    "Real Estate":  HazardWeights(flood=0.50, heat=0.30, cyclone=0.20),
    "Energy":       HazardWeights(flood=0.30, heat=0.25, cyclone=0.45),
    "Utilities":    HazardWeights(flood=0.40, heat=0.35, cyclone=0.25),
    "Industrials":  HazardWeights(flood=0.35, heat=0.35, cyclone=0.30),
    "Financials":   HazardWeights(flood=0.40, heat=0.35, cyclone=0.25),
}

SCORE_COLS = ["flood_score", "heat_score", "cyclone_score"]


# ── HazardScorer ──────────────────────────────────────────────────────────────

class HazardScorer:
    """Normalises raw ERA5 z-scores into 0–100 percentile hazard scores.

    Wraps a RobustScaler (resistant to outlier grid cells) followed by
    a rank-based mapping to [0, 100]. The scaler is fit on training data
    (historical baseline) and applied to all years including future
    scenarios.

    Args:
        clip_zscore: Absolute z-score values above this are clipped before
                     scaling. Prevents extreme single-year events from
                     compressing the rest of the distribution.
    """

    def __init__(self, clip_zscore: float = 5.0) -> None:
        self.clip_zscore = clip_zscore
        self._scalers: dict[str, RobustScaler] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, zscore_cols: list[str]) -> "HazardScorer":
        """Fit one RobustScaler per z-score column on historical data.

        Fitting per-column (rather than jointly) means each hazard's
        score distribution is independently normalised, preventing a
        high-variance hazard from dominating the rank mapping.

        Args:
            df: DataFrame containing z-score columns.
            zscore_cols: Column names of the z-score features to fit on.

        Returns:
            self (fluent interface).
        """
        self._zscore_cols = zscore_cols
        self._scalers: dict[str, RobustScaler] = {}
        for col in zscore_cols:
            scaler = RobustScaler()
            values = df[col].clip(-self.clip_zscore, self.clip_zscore).fillna(0.0)
            scaler.fit(values.values.reshape(-1, 1))
            self._scalers[col] = scaler
        self._is_fitted = True
        logger.info(
            f"HazardScorer fitted on {len(df)} samples, "
            f"cols={zscore_cols}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform z-scores to 0–100 percentile scores.

        Missing z-scores (NaN) receive a score of 50 (median), ensuring
        data gaps don't artificially suppress risk signals.

        Args:
            df: DataFrame with the same z-score columns used in fit().

        Returns:
            Copy of df with added columns: flood_score, heat_score,
            cyclone_score.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        hazard_names = ["flood", "heat", "cyclone"]

        for zscore_col, hazard in zip(self._zscore_cols, hazard_names):
            raw = df[zscore_col].clip(-self.clip_zscore, self.clip_zscore)
            filled = raw.fillna(raw.median())
            scaled = pd.Series(
                self._scalers[zscore_col]
                .transform(filled.values.reshape(-1, 1))
                .ravel(),
                index=df.index,
            )
            # Rank to percentile then scale to 0-100; NaN positions get 50
            score = scaled.rank(pct=True, na_option="keep") * 100
            result = score.copy()
            result[raw.isna()] = 50.0
            df[f"{hazard}_score"] = result.round(1)

        return df

    def fit_transform(self, df: pd.DataFrame, zscore_cols: list[str]) -> pd.DataFrame:
        return self.fit(df, zscore_cols).transform(df)


# ── CompositeRiskModel ────────────────────────────────────────────────────────

class CompositeRiskModel:
    """Combines individual hazard scores into a sector-weighted composite.

    Also runs a PCA diagnostic to quantify how much variance is captured
    by the first principal component (a well-calibrated composite should
    have PC1 explaining > 50% of hazard score variance).

    Args:
        sector_weights: Dict mapping sector name → HazardWeights.
                        Falls back to global settings weights for unknown
                        sectors.
        default_weights: HazardWeights for sectors not in sector_weights.
    """

    def __init__(
        self,
        sector_weights: dict[str, HazardWeights] | None = None,
        default_weights: HazardWeights | None = None,
    ) -> None:
        self.sector_weights = sector_weights or SECTOR_WEIGHTS
        self.default_weights = default_weights or HazardWeights()
        self._pca = PCA(n_components=3)
        self._pca_fitted = False

    def _get_weights(self, sector: str) -> HazardWeights:
        return self.sector_weights.get(sector, self.default_weights)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sector-aware composite score for each asset.

        Args:
            df: DataFrame with flood_score, heat_score, cyclone_score,
                and sector columns.

        Returns:
            df with added columns: composite_score, esg_tier.
        """
        missing = [c for c in SCORE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing hazard score columns: {missing}")

        df = df.copy()
        composite = np.zeros(len(df))

        for sector, group_idx in df.groupby("sector").groups.items():
            w = self._get_weights(sector)
            group_scores = df.loc[group_idx, SCORE_COLS].values  # (n, 3)
            composite[group_idx] = group_scores @ w.as_array()

        df["composite_score"] = np.clip(composite, 0, 100).round(2)

        # ESG tier
        from src.pipeline.features import assign_esg_tier
        df["esg_tier"] = assign_esg_tier(df["composite_score"])

        return df

    def fit_pca(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit PCA on hazard scores and return diagnostic statistics.

        Args:
            df: DataFrame with flood_score, heat_score, cyclone_score.

        Returns:
            Dict with keys: explained_variance_ratio, pc1_loadings,
            pc1_variance_pct.
        """
        X = df[SCORE_COLS].dropna().values
        self._pca.fit(X)
        self._pca_fitted = True

        evr = self._pca.explained_variance_ratio_
        loadings = dict(zip(SCORE_COLS, self._pca.components_[0].round(4)))
        diagnostics = {
            "explained_variance_ratio": evr.tolist(),
            "pc1_loadings": loadings,
            "pc1_variance_pct": round(float(evr[0]) * 100, 1),
        }
        logger.info(
            f"PCA diagnostic: PC1 explains {diagnostics['pc1_variance_pct']}% "
            f"of hazard score variance. Loadings: {loadings}"
        )
        return diagnostics

    def get_pca_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Project assets onto PCA components (for scatter viz).

        Args:
            df: DataFrame with hazard score columns.

        Returns:
            Array of shape (n_assets, 3) with PCA coordinates.
        """
        if not self._pca_fitted:
            self.fit_pca(df)
        X = df[SCORE_COLS].fillna(50.0).values
        return self._pca.transform(X)


# ── AnomalyDetector ───────────────────────────────────────────────────────────

class AnomalyDetector:
    """IsolationForest wrapper for multi-hazard tail-risk detection.

    Flags assets whose combination of hazard scores is structurally
    unusual relative to the portfolio — catches non-linear compound
    exposures that a linear composite score misses.

    Args:
        contamination: Expected fraction of anomalous assets.
                       0.05 = expect ~5% of portfolio to be extreme outliers.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> "AnomalyDetector":
        """Fit the isolation forest.

        Args:
            df: Feature DataFrame.
            feature_cols: Columns to use. Defaults to SCORE_COLS.

        Returns:
            self.
        """
        feature_cols = feature_cols or SCORE_COLS
        X = df[feature_cols].fillna(50.0).values
        self._model.fit(X)
        self._feature_cols = feature_cols
        self._is_fitted = True
        logger.info(
            f"AnomalyDetector fitted on {len(df)} assets, "
            f"contamination={self.contamination}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label each asset as normal (1) or anomalous (-1).

        Also adds 'anomaly_score' (more negative = more anomalous).

        Args:
            df: DataFrame with the same feature columns used in fit().

        Returns:
            df with added columns: anomaly_label (-1/1), anomaly_score,
            tail_risk_flag (bool).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        df = df.copy()
        X = df[self._feature_cols].fillna(50.0).values
        df["anomaly_label"] = self._model.predict(X)
        df["anomaly_score"] = self._model.score_samples(X).round(4)
        df["tail_risk_flag"] = df["anomaly_label"] == -1

        n_flagged = df["tail_risk_flag"].sum()
        logger.info(
            f"Anomaly detection: {n_flagged}/{len(df)} assets flagged as tail risk "
            f"({n_flagged/len(df):.1%})"
        )
        return df

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, feature_cols).predict(df)

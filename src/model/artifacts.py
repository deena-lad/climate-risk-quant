"""
src/model/artifacts.py
─────────────────────────────────────────────────────────────────────────────
Save and load all model artifacts: scorer, composite model, anomaly detector,
scored DataFrames, and run metadata.

Artifact layout under a given output directory:
    {run_dir}/
    ├── hazard_scorer.joblib
    ├── composite_model.joblib
    ├── anomaly_detector.joblib
    ├── features_scored.parquet
    ├── run_metadata.json
    └── metrics.json          (written by evaluate.log_experiment)

Using joblib for sklearn objects and parquet for DataFrames gives a good
balance of: human-inspectable output, fast load times, and portability
across Python versions without pickle version mismatch issues.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from src.config import settings
from src.model.scorer import AnomalyDetector, CompositeRiskModel, HazardScorer


# ── Save ──────────────────────────────────────────────────────────────────────

def save_run(
    output_dir: Path,
    hazard_scorer: HazardScorer | None = None,
    composite_model: CompositeRiskModel | None = None,
    anomaly_detector: AnomalyDetector | None = None,
    scored_df: pd.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist all model objects and outputs from a training run.

    Args:
        output_dir: Directory to write artifacts into (created if absent).
        hazard_scorer: Fitted HazardScorer instance.
        composite_model: Fitted CompositeRiskModel instance.
        anomaly_detector: Fitted AnomalyDetector instance.
        scored_df: Final scored + valued DataFrame.
        metrics: Evaluation metrics dict.
        extra_metadata: Any additional key/value pairs for the metadata file.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if hazard_scorer is not None:
        p = output_dir / "hazard_scorer.joblib"
        joblib.dump(hazard_scorer, p, compress=3)
        saved.append(p.name)

    if composite_model is not None:
        p = output_dir / "composite_model.joblib"
        joblib.dump(composite_model, p, compress=3)
        saved.append(p.name)

    if anomaly_detector is not None:
        p = output_dir / "anomaly_detector.joblib"
        joblib.dump(anomaly_detector, p, compress=3)
        saved.append(p.name)

    if scored_df is not None:
        p = output_dir / "features_scored.parquet"
        scored_df.to_parquet(p, index=False)
        saved.append(p.name)

    if metrics is not None:
        p = output_dir / "metrics.json"
        p.write_text(json.dumps(metrics, indent=2, default=str))
        saved.append(p.name)

    # Run metadata
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_saved": saved,
        "config": {
            "risk_haircut_alpha": settings.risk_haircut_alpha,
            "weight_flood":      settings.weight_flood,
            "weight_heat":       settings.weight_heat,
            "weight_cyclone":    settings.weight_cyclone,
            "era5_year_start":   settings.era5_year_start,
            "era5_year_end":     settings.era5_year_end,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.success(
        f"Run artifacts saved to {output_dir}/ "
        f"[{', '.join(saved + ['run_metadata.json'])}]"
    )
    return output_dir


# ── Load ──────────────────────────────────────────────────────────────────────

def load_hazard_scorer(run_dir: Path) -> HazardScorer:
    """Load a fitted HazardScorer from a run directory.

    Args:
        run_dir: Directory containing hazard_scorer.joblib.

    Returns:
        Fitted HazardScorer.

    Raises:
        FileNotFoundError: If the artifact file doesn't exist.
    """
    path = run_dir / "hazard_scorer.joblib"
    _assert_exists(path)
    scorer: HazardScorer = joblib.load(path)
    logger.info(f"Loaded HazardScorer from {path}")
    return scorer


def load_composite_model(run_dir: Path) -> CompositeRiskModel:
    """Load a CompositeRiskModel from a run directory.

    Args:
        run_dir: Directory containing composite_model.joblib.

    Returns:
        CompositeRiskModel instance.
    """
    path = run_dir / "composite_model.joblib"
    _assert_exists(path)
    model: CompositeRiskModel = joblib.load(path)
    logger.info(f"Loaded CompositeRiskModel from {path}")
    return model


def load_anomaly_detector(run_dir: Path) -> AnomalyDetector:
    """Load a fitted AnomalyDetector from a run directory.

    Args:
        run_dir: Directory containing anomaly_detector.joblib.

    Returns:
        Fitted AnomalyDetector.
    """
    path = run_dir / "anomaly_detector.joblib"
    _assert_exists(path)
    detector: AnomalyDetector = joblib.load(path)
    logger.info(f"Loaded AnomalyDetector from {path}")
    return detector


def load_scored_df(run_dir: Path) -> pd.DataFrame:
    """Load the scored + valued feature DataFrame from a run directory.

    Args:
        run_dir: Directory containing features_scored.parquet.

    Returns:
        DataFrame.
    """
    path = run_dir / "features_scored.parquet"
    _assert_exists(path)
    df = pd.read_parquet(path)
    logger.info(f"Loaded scored DataFrame ({len(df)} rows) from {path}")
    return df


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Load run metadata JSON from a run directory.

    Args:
        run_dir: Directory containing run_metadata.json.

    Returns:
        Metadata dict.
    """
    path = run_dir / "run_metadata.json"
    _assert_exists(path)
    return json.loads(path.read_text())


def load_full_run(run_dir: Path) -> dict[str, Any]:
    """Load all available artifacts from a run directory.

    Silently skips any artifact not present (partial runs are OK).

    Args:
        run_dir: Path to run directory.

    Returns:
        Dict with keys: hazard_scorer, composite_model, anomaly_detector,
        scored_df, metadata. Values are None if artifact not found.
    """
    result: dict[str, Any] = {
        "hazard_scorer":     None,
        "composite_model":   None,
        "anomaly_detector":  None,
        "scored_df":         None,
        "metadata":          None,
    }

    loaders = {
        "hazard_scorer":    (run_dir / "hazard_scorer.joblib",     lambda p: joblib.load(p)),
        "composite_model":  (run_dir / "composite_model.joblib",   lambda p: joblib.load(p)),
        "anomaly_detector": (run_dir / "anomaly_detector.joblib",  lambda p: joblib.load(p)),
        "scored_df":        (run_dir / "features_scored.parquet",  lambda p: pd.read_parquet(p)),
        "metadata":         (run_dir / "run_metadata.json",        lambda p: json.loads(p.read_text())),
    }

    for key, (path, loader) in loaders.items():
        if path.exists():
            try:
                result[key] = loader(path)
                logger.debug(f"Loaded {key} from {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
        else:
            logger.debug(f"{path.name} not found in {run_dir}, skipping.")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run the training pipeline first, or check the run directory path."
        )


def latest_run_dir(base_dir: Path | None = None) -> Path:
    """Return the most recently created run directory under base_dir.

    Args:
        base_dir: Parent directory to search. Defaults to data/processed/runs/.

    Returns:
        Path to the most recent run subdirectory.

    Raises:
        FileNotFoundError: If no run directories are found.
    """
    base_dir = base_dir or (settings.data_processed_dir / "runs")
    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {base_dir}. "
            "Run the training pipeline first."
        )
    logger.info(f"Latest run: {run_dirs[0]}")
    return run_dirs[0]

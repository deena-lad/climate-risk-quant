"""
app/components/sidebar.py
─────────────────────────────────────────────────────────────────────────────
Sidebar controls for the Streamlit dashboard.

Returns a typed SidebarConfig dataclass so the main app can pass settings
to the pipeline without coupling the UI to the model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st


@dataclass
class SidebarConfig:
    """All user-controllable parameters captured from the sidebar."""

    # Data source
    use_synthetic: bool
    uploaded_file_path: Path | None
    n_synthetic: int

    # Model parameters
    alpha: float
    haircut_method: str
    anomaly_contamination: float

    # Hazard weights
    weight_flood: float
    weight_heat: float
    weight_cyclone: float

    # Filters
    selected_sectors: list[str]
    min_book_value: float
    esg_tier_filter: list[str]

    # View options
    top_n_table: int
    show_compound_only: bool


def render_sidebar(available_sectors: list[str] | None = None) -> SidebarConfig:
    """Render all sidebar widgets and return the collected config.

    Args:
        available_sectors: List of sector names to populate the
                           multi-select. If None, uses a default list.

    Returns:
        SidebarConfig with all current widget values.
    """
    if available_sectors is None:
        available_sectors = [
            "Real Estate", "Energy", "Utilities", "Industrials", "Financials"
        ]

    st.sidebar.title("⚙️ Controls")

    # ── Data source ───────────────────────────────────────────────────────────
    st.sidebar.header("📂 Data Source")
    use_synthetic = st.sidebar.radio(
        "Asset data",
        options=["Synthetic (demo)", "Upload CSV"],
        index=0,
        help="Use built-in synthetic assets or upload your own CSV.",
    ) == "Synthetic (demo)"

    uploaded_file_path: Path | None = None
    n_synthetic = 200

    if use_synthetic:
        n_synthetic = st.sidebar.slider(
            "Number of synthetic assets",
            min_value=50, max_value=500, value=200, step=50,
            help="More assets = more stable score distributions.",
        )
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload asset CSV",
            type=["csv"],
            help=(
                "Required columns: asset_id, name, latitude, longitude, "
                "sector, book_value"
            ),
        )
        if uploaded is not None:
            import tempfile, shutil
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv"
            )
            shutil.copyfileobj(uploaded, tmp)
            tmp.close()
            uploaded_file_path = Path(tmp.name)
            st.sidebar.success(f"Loaded: {uploaded.name}")

    st.sidebar.divider()

    # ── Valuation parameters ──────────────────────────────────────────────────
    st.sidebar.header("💰 Valuation")
    alpha = st.sidebar.slider(
        "Haircut coefficient α",
        min_value=0.05, max_value=0.60, value=0.30, step=0.05,
        format="%.2f",
        help=(
            "Maximum fractional haircut applied to an asset with "
            "composite score = 100.  α = 0.30 → 30% max reduction."
        ),
    )
    haircut_method = st.sidebar.selectbox(
        "Haircut schedule",
        options=["linear", "convex"],
        index=0,
        help=(
            "linear: haircut grows proportionally with score.  "
            "convex: haircut accelerates above score 50 (heavier tail penalty)."
        ),
    )

    st.sidebar.divider()

    # ── Hazard weights ────────────────────────────────────────────────────────
    st.sidebar.header("⚖️ Hazard Weights")
    st.sidebar.caption("Weights must sum to 1.0 (auto-normalised).")

    raw_flood   = st.sidebar.slider("Flood weight",   0.05, 0.70, 0.40, 0.05)
    raw_heat    = st.sidebar.slider("Heat weight",    0.05, 0.70, 0.35, 0.05)
    raw_cyclone = st.sidebar.slider("Cyclone weight", 0.05, 0.70, 0.25, 0.05)

    # Auto-normalise
    total = raw_flood + raw_heat + raw_cyclone
    w_flood   = round(raw_flood   / total, 4)
    w_heat    = round(raw_heat    / total, 4)
    w_cyclone = round(1.0 - w_flood - w_heat, 4)

    st.sidebar.caption(
        f"Normalised → flood: {w_flood:.2f} | "
        f"heat: {w_heat:.2f} | cyclone: {w_cyclone:.2f}"
    )

    st.sidebar.divider()

    # ── Anomaly detection ─────────────────────────────────────────────────────
    st.sidebar.header("🔍 Anomaly Detection")
    anomaly_contamination = st.sidebar.slider(
        "Expected tail-risk fraction",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        format="%.2f",
        help=(
            "Fraction of assets IsolationForest treats as structural outliers. "
            "0.05 = 5% of portfolio expected to be tail-risk anomalies."
        ),
    )

    st.sidebar.divider()

    # ── Portfolio filters ─────────────────────────────────────────────────────
    st.sidebar.header("🔎 Filters")
    selected_sectors = st.sidebar.multiselect(
        "Sectors",
        options=available_sectors,
        default=available_sectors,
        help="Show only assets from selected sectors.",
    )
    esg_tiers = ["Low", "Medium", "High", "Critical"]
    esg_tier_filter = st.sidebar.multiselect(
        "ESG risk tiers",
        options=esg_tiers,
        default=esg_tiers,
        help="Filter the scorecard table and charts by ESG tier.",
    )
    min_book_value = st.sidebar.number_input(
        "Minimum book value (USD M)",
        min_value=0.0, max_value=10_000.0, value=0.0, step=10.0,
        help="Exclude assets below this book value threshold.",
    )

    st.sidebar.divider()

    # ── View options ──────────────────────────────────────────────────────────
    st.sidebar.header("🖥️ View Options")
    top_n_table = st.sidebar.slider(
        "Assets in scorecard",
        min_value=5, max_value=50, value=15, step=5,
    )
    show_compound_only = st.sidebar.checkbox(
        "Show compound-risk assets only",
        value=False,
        help="Filter all charts to assets flagged for multi-hazard exposure.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Climate Risk Quantification v0.1.0")
    st.sidebar.caption("Data: ERA5 (Copernicus CDS)")

    return SidebarConfig(
        use_synthetic=use_synthetic,
        uploaded_file_path=uploaded_file_path,
        n_synthetic=n_synthetic,
        alpha=alpha,
        haircut_method=haircut_method,
        anomaly_contamination=anomaly_contamination,
        weight_flood=w_flood,
        weight_heat=w_heat,
        weight_cyclone=w_cyclone,
        selected_sectors=selected_sectors,
        min_book_value=min_book_value,
        esg_tier_filter=esg_tier_filter,
        top_n_table=top_n_table,
        show_compound_only=show_compound_only,
    )

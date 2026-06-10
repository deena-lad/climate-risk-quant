"""
app/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
Main Streamlit application for the Climate Risk Quantification dashboard.

Run locally:
    streamlit run app/streamlit_app.py

Pages (implemented as tabs):
    1. 🗺️  Risk Map          — geographic heatmap of composite scores
    2. 📊  Score Analysis    — distributions, correlations, compound risk
    3. 💰  Valuation         — NAV waterfall, sector attribution, sensitivity
    4. 📋  Asset Scorecard   — filterable ranked table of all assets
    5. 🔬  Model Diagnostics — PCA, anomaly detection, weight sensitivity

Interactive elements (≥2 required by spec):
    ✓ Sidebar sliders: alpha, hazard weights, n_synthetic, top_n, contamination
    ✓ Sidebar selects: haircut method, sector filter, ESG tier filter
    ✓ File upload: custom asset CSV
    ✓ Checkbox: compound-risk-only filter
    ✓ Tab navigation: 5 views
    ✓ Score histogram: dropdown to pick which hazard to display
    ✓ Download button: export scored DataFrame as CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from app.components.metrics_row import render_metrics_row
from app.components.sidebar import render_sidebar
from src.model.evaluate import alpha_sensitivity, evaluate, weight_sensitivity
from src.model.pipeline import run_pipeline
from src.model.scorer import AnomalyDetector, CompositeRiskModel, HazardScorer
from src.model.valuation import (
    apply_haircut,
    climate_var,
    portfolio_summary,
    sector_attribution,
)
from src.viz.charts import (
    alpha_sensitivity_chart,
    compound_scatter,
    geo_risk_map,
    hazard_violin,
    nav_waterfall,
    score_histogram,
    scorecard_table,
    sector_attribution_chart,
    sector_tier_bar,
    weight_sensitivity_heatmap,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Climate Risk Quantification",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Remove top padding */
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
    /* KPI card styling */
    [data-testid="stMetric"] {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        border-left: 4px solid #1565C0;
    }
    /* Tab font */
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    /* Divider spacing */
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def _run_pipeline_cached(
    use_synthetic: bool,
    uploaded_path_str: str | None,
    n_synthetic: int,
    alpha: float,
    haircut_method: str,
    anomaly_contamination: float,
    weight_flood: float,
    weight_heat: float,
    weight_cyclone: float,
    seed: int = 42,
) -> dict:
    """Cached pipeline run.  Re-runs only when inputs change.

    Cache key is the full tuple of parameters, so changing any slider
    triggers a fresh run while identical configs reuse cached results.
    """
    import os, tempfile

    # Patch settings weights in-process (simplest approach for demo)
    from src import config as cfg_mod
    cfg_mod.settings.weight_flood   = weight_flood
    cfg_mod.settings.weight_heat    = weight_heat
    cfg_mod.settings.weight_cyclone = weight_cyclone
    cfg_mod.settings.risk_haircut_alpha = alpha

    features_path = Path(uploaded_path_str) if uploaded_path_str else None

    with tempfile.TemporaryDirectory() as tmp:
        result = run_pipeline(
            features_path=features_path,
            output_dir=Path(tmp) / "run",
            alpha=alpha,
            haircut_method=haircut_method,
            anomaly_contamination=anomaly_contamination,
            n_synthetic=n_synthetic,
            seed=seed,
            run_sensitivity=True,
            log_mlflow=False,
        )

    # Convert Path to str so Streamlit cache can serialise it
    result.pop("output_dir", None)
    return result


def _apply_filters(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply sidebar filters to the scored DataFrame."""
    if cfg.selected_sectors and "sector" in df.columns:
        df = df[df["sector"].isin(cfg.selected_sectors)]
    if cfg.esg_tier_filter and "esg_tier" in df.columns:
        df = df[df["esg_tier"].isin(cfg.esg_tier_filter)]
    if cfg.min_book_value > 0 and "book_value" in df.columns:
        df = df[df["book_value"] >= cfg.min_book_value]
    if cfg.show_compound_only and "compound_risk" in df.columns:
        df = df[df["compound_risk"]]
    return df.reset_index(drop=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌍 Climate Risk Quantification for Financial Assets")
st.caption(
    "Physical climate hazard (flood · heat · cyclone) mapped to "
    "asset-level exposure and risk-adjusted portfolio valuation. "
    "Data: ERA5 reanalysis · Methodology: TCFD / ECB stress-test aligned."
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

cfg = render_sidebar()


# ── Run pipeline ──────────────────────────────────────────────────────────────

with st.spinner("Running climate risk model…"):
    try:
        result = _run_pipeline_cached(
            use_synthetic=cfg.use_synthetic,
            uploaded_path_str=(
                str(cfg.uploaded_file_path) if cfg.uploaded_file_path else None
            ),
            n_synthetic=cfg.n_synthetic,
            alpha=cfg.alpha,
            haircut_method=cfg.haircut_method,
            anomaly_contamination=cfg.anomaly_contamination,
            weight_flood=cfg.weight_flood,
            weight_heat=cfg.weight_heat,
            weight_cyclone=cfg.weight_cyclone,
        )
        pipeline_error = None
    except Exception as e:
        pipeline_error = e
        result = None

if pipeline_error:
    st.error(f"Pipeline error: {pipeline_error}")
    st.stop()

df_full: pd.DataFrame = result["scored_df"]
port: dict             = result["portfolio"]
cvar: dict             = result["climate_var"]
metrics: dict          = result["metrics"]
sector_attr: pd.DataFrame = result["sector_attr"]
alpha_sens: pd.DataFrame  = result.get("alpha_sensitivity", pd.DataFrame())
weight_sens: pd.DataFrame = result.get("weight_sensitivity", pd.DataFrame())

# Filtered view for charts
df = _apply_filters(df_full, cfg)

if len(df) == 0:
    st.warning("No assets match the current filters. Adjust the sidebar filters.")
    st.stop()

# Recompute portfolio summary on filtered data
try:
    port_filtered = portfolio_summary(df)
    cvar_filtered = climate_var(df, n_simulations=2000)  # fast for UI
except Exception:
    port_filtered = port
    cvar_filtered = cvar


# ── KPI row ───────────────────────────────────────────────────────────────────

render_metrics_row(
    port_summary=port_filtered,
    climate_var=cvar_filtered,
    eval_metrics=metrics,
)
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Risk Map",
    "📊 Score Analysis",
    "💰 Valuation",
    "📋 Asset Scorecard",
    "🔬 Model Diagnostics",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Map
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Global Portfolio Risk Map")
    st.caption(
        "Circle size = book value (USD M).  "
        "Colour = composite climate risk score (green→red).  "
        "Diamond markers = compound risk (multi-hazard exposure)."
    )

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        if "latitude" in df.columns:
            fig_map = geo_risk_map(df)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Latitude/longitude columns not found in dataset.")

    with col_stats:
        st.markdown("**Tier breakdown**")
        tier_counts = df["esg_tier"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        )
        tier_colours = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            st.markdown(f"{tier_colours[tier]} **{tier}**: {count} ({pct:.0f}%)")

        st.divider()
        st.markdown("**Compound risk**")
        if "compound_risk" in df.columns:
            n_comp = df["compound_risk"].sum()
            comp_val = df.loc[df["compound_risk"], "book_value"].sum()
            st.metric("Flagged assets", n_comp)
            st.metric("Book value at compound risk", f"${comp_val:,.0f}M")
        else:
            st.info("Compound risk column not available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Score Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score Distribution Analysis")

    row1_l, row1_r = st.columns(2)

    with row1_l:
        st.plotly_chart(hazard_violin(df), use_container_width=True)
        st.caption(
            "Box shows IQR; violins show full distribution. "
            "Dashed red line = high-risk threshold (score ≥ 75)."
        )

    with row1_r:
        # Interactive: user picks which score to histogram
        score_choice = st.selectbox(
            "Select score to inspect",
            options=["composite_score", "flood_score", "heat_score", "cyclone_score"],
            format_func=lambda x: x.replace("_score", "").title() + " score",
            key="histogram_select",
        )
        if score_choice in df.columns:
            st.plotly_chart(score_histogram(df, score_choice), use_container_width=True)

    st.divider()

    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.plotly_chart(compound_scatter(df), use_container_width=True)
        st.caption(
            "Top-right quadrant (flood ≥ 75 AND heat ≥ 75) = highest compound risk. "
            "Size = book value; colour = cyclone score."
        )

    with row2_r:
        st.plotly_chart(sector_tier_bar(df), use_container_width=True)
        st.caption(
            "Stacked bars show tier composition per sector. "
            "Real Estate and Utilities typically carry the most fixed-location risk."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Valuation
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Risk-Adjusted Valuation")

    # Merge VaR into port summary for waterfall
    port_for_waterfall = {**port_filtered, "cvar_var_usd_m": cvar_filtered.get("var_usd_m")}

    row1_l, row1_r = st.columns(2)
    with row1_l:
        st.plotly_chart(nav_waterfall(port_for_waterfall), use_container_width=True)
        st.caption(
            "Waterfall: book value reduced by climate haircut → "
            "risk-adjusted NAV, then further stressed by 95% Climate VaR."
        )
    with row1_r:
        st.plotly_chart(
            sector_attribution_chart(sector_attribution(df)),
            use_container_width=True,
        )
        st.caption(
            "Left: absolute haircut (USD M) by sector.  "
            "Right: haircut as % of sector book value — normalises for sector size."
        )

    st.divider()
    st.subheader("Alpha (α) Sensitivity Analysis")
    st.caption(
        "How does portfolio haircut and NAV change as the haircut "
        "coefficient varies?  Current α is highlighted by your sidebar selection."
    )

    if not alpha_sens.empty:
        st.plotly_chart(alpha_sensitivity_chart(alpha_sens), use_container_width=True)
    else:
        st.info("Run with run_sensitivity=True to see this chart.")

    # Tabular breakdown
    with st.expander("📄 Full alpha sensitivity table"):
        if not alpha_sens.empty:
            st.dataframe(
                alpha_sens.style.format({
                    "alpha":                   "{:.2f}",
                    "total_haircut_usd_m":     "{:,.1f}",
                    "portfolio_haircut_pct":   "{:.1f}%",
                    "risk_adjusted_nav_usd_m": "{:,.1f}",
                }),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Asset Scorecard
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Asset-Level Climate Risk Scorecard")

    # Search / filter row
    search_col, sort_col, _ = st.columns([2, 2, 3])
    with search_col:
        search_term = st.text_input(
            "Search asset name / ID",
            placeholder="e.g. AST0042",
            key="scorecard_search",
        )
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=["composite_score", "flood_score", "heat_score",
                     "cyclone_score", "book_value", "risk_adjusted_value"],
            key="scorecard_sort",
        )

    df_card = df.copy()
    if search_term:
        mask = (
            df_card.get("name", pd.Series("", index=df_card.index))
                .str.contains(search_term, case=False, na=False)
            | df_card.get("asset_id", pd.Series("", index=df_card.index))
                .str.contains(search_term, case=False, na=False)
        )
        df_card = df_card[mask]

    if sort_by in df_card.columns:
        df_card = df_card.sort_values(sort_by, ascending=False)

    st.plotly_chart(
        scorecard_table(df_card, top_n=cfg.top_n_table),
        use_container_width=True,
    )

    # Download button (interactive element 2)
    csv_bytes = df_card.drop(columns=["geometry"], errors="ignore").to_csv(index=False)
    st.download_button(
        label="⬇️ Download scored assets (CSV)",
        data=csv_bytes,
        file_name="climate_risk_scored_assets.csv",
        mime="text/csv",
        help="Download the full scored + valued asset table.",
    )

    # Summary stats below table
    with st.expander("📊 Summary statistics"):
        summary_cols = [c for c in [
            "composite_score", "flood_score", "heat_score",
            "cyclone_score", "book_value", "risk_adjusted_value",
            "haircut_fraction",
        ] if c in df_card.columns]
        st.dataframe(
            df_card[summary_cols].describe().round(2),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Diagnostics
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model Diagnostics")

    # PCA card
    pca_diag = metrics.get("pca", {})
    diag_l, diag_r = st.columns(2)

    with diag_l:
        st.markdown("**PCA: hazard score variance explained**")
        if pca_diag:
            evr = pca_diag.get("explained_variance_ratio", [])
            pc1 = pca_diag.get("pc1_variance_pct", 0)
            pc_labels = [f"PC{i+1}" for i in range(len(evr))]
            import plotly.express as px_local
            fig_pca = px_local.bar(
                x=pc_labels,
                y=[round(v * 100, 1) for v in evr],
                labels={"x": "Principal component", "y": "Variance explained (%)"},
                color_discrete_sequence=["#1565C0"],
                text=[f"{v*100:.1f}%" for v in evr],
            )
            fig_pca.update_layout(
                title=f"PCA Variance Explained (PC1 = {pc1}%)",
                showlegend=False, height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(
                f"PC1 loadings: "
                + ", ".join(
                    f"{k.replace('_score','')}: {v:+.3f}"
                    for k, v in pca_diag.get("pc1_loadings", {}).items()
                )
            )
        else:
            st.info("PCA diagnostics not available.")

    with diag_r:
        st.markdown("**Evaluation metrics**")
        eval_display = {
            "Gini (composite)":        f"{abs(metrics.get('gini_composite', 0)):.3f}",
            "KS statistic":            f"{metrics.get('ks_statistic', 0):.3f}",
            "KS p-value":              f"{metrics.get('ks_pvalue', 0):.4f}",
            "Tail conc. (top 25%)":    f"{metrics.get('tail_concentration_75', 0):.1%}",
            "Compound detect rate":    f"{metrics.get('compound_detection_rate', 0) or 0:.1%}",
            "Anomaly contamination":   f"{metrics.get('anomaly_contamination_realised', 0):.1%}",
            "Validation failures":     str(metrics.get("validation_failures", 0)),
        }
        for label, value in eval_display.items():
            col_k, col_v = st.columns([3, 1])
            col_k.markdown(f"**{label}**")
            col_v.markdown(f"`{value}`")

    st.divider()

    # Weight sensitivity heatmap
    st.markdown("**Composite Score Sensitivity to Hazard Weights**")
    st.caption(
        "Each cell = average Gini coefficient over 200 random weight combinations "
        "with flood weight (x) and heat weight (y). Brighter = better risk differentiation."
    )
    if not weight_sens.empty:
        st.plotly_chart(weight_sensitivity_heatmap(weight_sens), use_container_width=True)
    else:
        st.info("Weight sensitivity data not available.")

    # Tail-risk assets
    st.divider()
    st.markdown("**Tail-Risk Flagged Assets (IsolationForest anomalies)**")
    if "tail_risk_flag" in df.columns:
        tail_df = df[df["tail_risk_flag"]].copy()
        if len(tail_df):
            show_cols = [c for c in [
                "asset_id", "name", "sector", "composite_score",
                "flood_score", "heat_score", "cyclone_score",
                "anomaly_score", "book_value",
            ] if c in tail_df.columns]
            st.dataframe(
                tail_df[show_cols].sort_values("anomaly_score").style.format(
                    {c: "{:.2f}" for c in show_cols if tail_df[c].dtype == float}
                ),
                use_container_width=True,
            )
            st.caption(
                "Anomaly score: more negative = more structurally unusual "
                "multi-hazard profile relative to the portfolio."
            )
        else:
            st.info("No tail-risk assets in current filtered view.")
    else:
        st.info("Tail risk column not available.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Climate Risk Quantification · Built with ERA5, GeoPandas, scikit-learn, "
    "MLflow, Plotly, Streamlit · "
    "Methodology: TCFD Physical Risk Framework · ECB 2022 Climate Stress Test"
)

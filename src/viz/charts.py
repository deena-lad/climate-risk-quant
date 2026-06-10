"""
src/viz/charts.py
─────────────────────────────────────────────────────────────────────────────
All Plotly figure builders used by the Streamlit dashboard.

Each function returns a go.Figure that Streamlit renders via
st.plotly_chart(fig, use_container_width=True).

Design rules applied throughout:
  - Every chart has a title, axis labels, and a consistent colour palette.
  - Figures are self-contained: pass a DataFrame in, get a Figure out.
  - No global state — safe to call from multiple Streamlit pages / threads.
  - All colours come from PALETTE so the dashboard is visually coherent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared palette ────────────────────────────────────────────────────────────

PALETTE = {
    "flood":     "#2196F3",
    "heat":      "#F44336",
    "cyclone":   "#9C27B0",
    "composite": "#FF9800",
    "Low":       "#4CAF50",
    "Medium":    "#FFC107",
    "High":      "#FF5722",
    "Critical":  "#B71C1C",
    "bg":        "rgba(0,0,0,0)",   # transparent background
    "grid":      "#E0E0E0",
}

TIER_ORDER = ["Low", "Medium", "High", "Critical"]

_LAYOUT_BASE: dict[str, Any] = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#212121")),
                      **_LAYOUT_BASE)
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


# ── 1. Global risk heatmap (scatter_geo) ──────────────────────────────────────

def geo_risk_map(df: pd.DataFrame) -> go.Figure:
    """Scatter globe showing each asset coloured by composite risk score.

    Size encodes book value so high-value assets are immediately visible.
    Compound-risk assets are shown as diamonds (symbol='diamond').

    Args:
        df: Scored DataFrame with latitude, longitude, composite_score,
            book_value, esg_tier, compound_risk, name, sector columns.

    Returns:
        Plotly Figure (scatter_geo).
    """
    df = df.copy()
    df["symbol"] = df["compound_risk"].map({True: "diamond", False: "circle"}) \
        if "compound_risk" in df.columns else "circle"
    df["size_scaled"] = (
        (df["book_value"] - df["book_value"].min())
        / (df["book_value"].max() - df["book_value"].min() + 1e-9) * 18 + 4
    ).round(1)

    hover_cols = {
        "composite_score": ":.1f",
        "book_value":      ":.1f",
        "esg_tier":        True,
    }
    if "compound_risk" in df.columns:
        hover_cols["compound_risk"] = True
    if "sector" in df.columns:
        hover_cols["sector"] = True

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="composite_score",
        size="size_scaled",
        hover_name="name" if "name" in df.columns else "asset_id",
        hover_data=hover_cols,
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        size_max=22,
        projection="natural earth",
        labels={
            "composite_score": "Risk score",
            "book_value":      "Book value (USD M)",
            "size_scaled":     "Scaled size",
        },
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        font=dict(family="Inter, sans-serif", size=12),
        hoverlabel=dict(bgcolor="white", font_size=12),
        title=dict(text="Portfolio Climate Risk Map", font=dict(size=14)),
        coloraxis_colorbar=dict(
            title="Composite<br>risk score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0 (Low)", "25", "50", "75", "100 (Critical)"],
            len=0.7,
        ),
        geo=dict(
            showland=True, landcolor="#F5F5F5",
            showocean=True, oceancolor="#E3F2FD",
            showcoastlines=True, coastlinecolor="#BDBDBD",
            showframe=False,
            projection_type="natural earth",
        ),
        height=440,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── 2. Hazard score violin ────────────────────────────────────────────────────

def hazard_violin(df: pd.DataFrame) -> go.Figure:
    """Violin + box plot of the four score distributions.

    Args:
        df: DataFrame with flood_score, heat_score, cyclone_score,
            composite_score columns.

    Returns:
        Plotly Figure.
    """
    score_map = {
        "flood_score":    ("Flood",     PALETTE["flood"]),
        "heat_score":     ("Heat",      PALETTE["heat"]),
        "cyclone_score":  ("Cyclone",   PALETTE["cyclone"]),
        "composite_score":("Composite", PALETTE["composite"]),
    }

    fig = go.Figure()
    for col, (label, colour) in score_map.items():
        if col not in df.columns:
            continue
        fig.add_trace(go.Violin(
            y=df[col].dropna(),
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colour,
            opacity=0.6,
            line_color=colour,
            points=False,
        ))

    fig.add_hline(
        y=75, line_dash="dash", line_color="red",
        annotation_text="High-risk threshold (75)",
        annotation_position="top right",
        opacity=0.6,
    )
    _apply_base(fig, "Hazard Score Distributions Across Portfolio")
    fig.update_layout(
        yaxis=dict(title="Score (0–100 percentile)", range=[0, 105]),
        xaxis_title="Hazard type",
        showlegend=False,
        height=380,
    )
    return fig


# ── 3. ESG tier stacked bar by sector ─────────────────────────────────────────

def sector_tier_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart: number of assets per ESG tier, grouped by sector.

    Args:
        df: DataFrame with sector and esg_tier columns.

    Returns:
        Plotly Figure.
    """
    if "sector" not in df.columns or "esg_tier" not in df.columns:
        return go.Figure()

    pivot = (
        df.groupby(["sector", "esg_tier"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TIER_ORDER, fill_value=0)
        .reset_index()
    )

    fig = go.Figure()
    for tier in TIER_ORDER:
        if tier in pivot.columns:
            fig.add_trace(go.Bar(
                name=tier,
                x=pivot["sector"],
                y=pivot[tier],
                marker_color=PALETTE[tier],
            ))

    _apply_base(fig, "ESG Risk Tier Composition by Sector")
    fig.update_layout(
        barmode="stack",
        xaxis_title="Sector",
        yaxis_title="Number of assets",
        legend=dict(title="Risk tier", traceorder="normal"),
        height=360,
    )
    return fig


# ── 4. Compound risk scatter ───────────────────────────────────────────────────

def compound_scatter(df: pd.DataFrame) -> go.Figure:
    """Flood vs heat scatter; cyclone encoded as colour, book value as size.

    Args:
        df: Scored DataFrame.

    Returns:
        Plotly Figure.
    """
    needed = {"flood_score", "heat_score", "cyclone_score", "book_value"}
    if not needed.issubset(df.columns):
        return go.Figure()

    hover_name = "name" if "name" in df.columns else "asset_id"
    hover_data: dict[str, Any] = {
        "composite_score": ":.1f",
        "sector":          True,
    }
    if "compound_risk" in df.columns:
        hover_data["compound_risk"] = True

    fig = px.scatter(
        df,
        x="flood_score",
        y="heat_score",
        size="book_value",
        color="cyclone_score",
        color_continuous_scale="Purples",
        range_color=[0, 100],
        hover_name=hover_name,
        hover_data=hover_data,
        size_max=24,
        opacity=0.75,
        labels={
            "flood_score":   "Flood score (0–100)",
            "heat_score":    "Heat score (0–100)",
            "cyclone_score": "Cyclone score",
            "book_value":    "Book value (USD M)",
        },
    )
    fig.add_hline(y=75, line_dash="dash", line_color=PALETTE["heat"],
                  opacity=0.5, annotation_text="High heat (75)")
    fig.add_vline(x=75, line_dash="dash", line_color=PALETTE["flood"],
                  opacity=0.5, annotation_text="High flood (75)")

    _apply_base(fig, "Compound Risk: Flood × Heat (size = book value, colour = cyclone)")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Cyclone<br>score"),
        height=420,
    )
    return fig


# ── 5. Alpha sensitivity line chart ───────────────────────────────────────────

def alpha_sensitivity_chart(alpha_df: pd.DataFrame) -> go.Figure:
    """Line chart showing portfolio haircut % vs haircut coefficient alpha.

    Args:
        alpha_df: Output of evaluate.alpha_sensitivity().

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=alpha_df["alpha"],
        y=alpha_df["portfolio_haircut_pct"],
        name="Portfolio haircut (%)",
        line=dict(color=PALETTE["composite"], width=2.5),
        mode="lines+markers",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=alpha_df["alpha"],
        y=alpha_df["risk_adjusted_nav_usd_m"],
        name="Risk-adjusted NAV (USD M)",
        line=dict(color=PALETTE["flood"], width=2, dash="dot"),
        mode="lines+markers",
    ), secondary_y=True)

    fig.update_xaxes(title_text="Haircut coefficient α", showgrid=True,
                     gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="Portfolio haircut (%)", secondary_y=False,
                     showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="Risk-adjusted NAV (USD M)", secondary_y=True,
                     showgrid=False)
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="NAV Sensitivity to Haircut Coefficient α",
                   font=dict(size=14)),
        legend=dict(x=0.01, y=0.99),
        height=360,
    )
    return fig


# ── 6. Portfolio NAV waterfall ─────────────────────────────────────────────────

def nav_waterfall(port_summary: dict) -> go.Figure:
    """Waterfall chart decomposing book value → risk-adjusted NAV.

    Args:
        port_summary: Output of valuation.portfolio_summary().

    Returns:
        Plotly Figure.
    """
    book   = port_summary.get("total_book_value_usd_m", 0)
    haircut = port_summary.get("total_haircut_usd_m", 0)
    nav    = port_summary.get("total_risk_adjusted_nav_usd_m", 0)
    cvar   = port_summary.get("cvar_var_usd_m", None)

    measures = ["absolute", "relative", "total"]
    x_labels = ["Book value", "Climate haircut", "Risk-adj. NAV"]
    y_values = [book, -haircut, nav]
    colours  = ["#42A5F5", "#EF5350", "#66BB6A"]

    if cvar is not None:
        measures += ["relative", "total"]
        x_labels += ["Climate VaR (95%)", "Stressed NAV"]
        y_values += [-cvar, nav - cvar]
        colours  += ["#AB47BC", "#26A69A"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=[f"{v:,.0f}" for v in y_values],
        textposition="outside",
        connector=dict(line=dict(color="#9E9E9E", width=1)),
        increasing=dict(marker_color="#66BB6A"),
        decreasing=dict(marker_color="#EF5350"),
        totals=dict(marker_color="#42A5F5"),
    ))

    _apply_base(fig, "Portfolio Value Waterfall (USD M)")
    fig.update_layout(
        yaxis_title="USD Millions",
        showlegend=False,
        height=380,
    )
    return fig


# ── 7. Sector attribution bar ─────────────────────────────────────────────────

def sector_attribution_chart(sector_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total climate haircut and haircut % per sector.

    Args:
        sector_df: Output of valuation.sector_attribution().

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total haircut (USD M)", "Haircut as % of sector book value"),
        horizontal_spacing=0.12,
    )

    sector_sorted = sector_df.sort_values("total_haircut_usd", ascending=True)

    fig.add_trace(go.Bar(
        y=sector_sorted["sector"],
        x=sector_sorted["total_haircut_usd"],
        orientation="h",
        marker_color=PALETTE["composite"],
        name="Haircut (USD M)",
        text=sector_sorted["total_haircut_usd"].round(0).astype(int),
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=sector_sorted["sector"],
        x=sector_sorted["haircut_pct"],
        orientation="h",
        marker_color=PALETTE["heat"],
        name="Haircut (%)",
        text=sector_sorted["haircut_pct"].round(1).astype(str) + "%",
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Climate Haircut Attribution by Sector", font=dict(size=14)),
        showlegend=False,
        height=360,
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"])
    return fig


# ── 8. Score distribution histogram ───────────────────────────────────────────

def score_histogram(df: pd.DataFrame, score_col: str = "composite_score") -> go.Figure:
    """Histogram of a single score column with ESG tier colour bands.

    Args:
        df: Scored DataFrame.
        score_col: Column to histogram (default 'composite_score').

    Returns:
        Plotly Figure.
    """
    label_map = {
        "composite_score": "Composite risk score",
        "flood_score":     "Flood score",
        "heat_score":      "Heat score",
        "cyclone_score":   "Cyclone score",
    }
    label = label_map.get(score_col, score_col)

    fig = go.Figure()

    # Tier background bands
    bands = [(0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 100, "Critical")]
    for lo, hi, tier in bands:
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=PALETTE[tier], opacity=0.08,
            line_width=0,
            annotation_text=tier,
            annotation_position="top",
            annotation=dict(font_size=10, font_color=PALETTE[tier]),
        )

    fig.add_trace(go.Histogram(
        x=df[score_col].dropna(),
        nbinsx=30,
        marker_color=PALETTE.get(score_col.replace("_score", ""), PALETTE["composite"]),
        opacity=0.8,
        name=label,
    ))

    # Median line
    med = df[score_col].median()
    fig.add_vline(
        x=med, line_dash="dash", line_color="#212121", opacity=0.7,
        annotation_text=f"Median: {med:.1f}",
        annotation_position="top right",
    )

    _apply_base(fig, f"Distribution of {label}")
    fig.update_layout(
        xaxis=dict(title=label, range=[0, 100]),
        yaxis_title="Number of assets",
        showlegend=False,
        height=320,
    )
    return fig


# ── 9. Weight sensitivity heatmap ─────────────────────────────────────────────

def weight_sensitivity_heatmap(weight_df: pd.DataFrame) -> go.Figure:
    """2-D density heatmap: flood weight vs heat weight, coloured by Gini.

    Args:
        weight_df: Output of evaluate.weight_sensitivity().

    Returns:
        Plotly Figure.
    """
    fig = px.density_heatmap(
        weight_df,
        x="w_flood",
        y="w_heat",
        z="gini",
        histfunc="avg",
        nbinsx=15,
        nbinsy=15,
        color_continuous_scale="Viridis",
        labels={
            "w_flood": "Flood weight",
            "w_heat":  "Heat weight",
            "gini":    "Avg Gini",
        },
    )
    _apply_base(fig, "Gini Coefficient Sensitivity to Hazard Weights")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Avg Gini"),
        height=360,
    )
    return fig


# ── 10. Asset scorecard table ─────────────────────────────────────────────────

def scorecard_table(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Colour-coded table of the top-N riskiest assets.

    Args:
        df: Scored + valued DataFrame.
        top_n: Number of rows to show (sorted by composite_score desc).

    Returns:
        Plotly Figure (go.Table).
    """
    display_cols = [
        "asset_id", "name", "sector",
        "flood_score", "heat_score", "cyclone_score",
        "composite_score", "esg_tier", "book_value",
        "risk_adjusted_value", "compound_risk",
    ]
    present = [c for c in display_cols if c in df.columns]
    sub = df.sort_values("composite_score", ascending=False).head(top_n)[present]

    # Colour-code composite_score cells
    def score_color(score: float) -> str:
        if score >= 75:   return PALETTE["Critical"]
        if score >= 50:   return PALETTE["High"]
        if score >= 25:   return PALETTE["Medium"]
        return PALETTE["Low"]

    composite_colors = [score_color(s) for s in sub.get("composite_score", [])]

    header_vals = [c.replace("_", " ").title() for c in present]
    cell_vals   = [sub[c].round(1).tolist() if sub[c].dtype in [float] else sub[c].tolist()
                   for c in present]

    # Apply colour only to the composite_score column
    comp_idx = present.index("composite_score") if "composite_score" in present else -1
    cell_colors = []
    for i, col in enumerate(present):
        if i == comp_idx:
            cell_colors.append(composite_colors)
        else:
            cell_colors.append(["white"] * len(sub))

    fig = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color="#1565C0",
            font=dict(color="white", size=11),
            align="center",
            height=28,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colors,
            font=dict(size=11),
            align=["left" if sub[c].dtype == object else "right" for c in present],
            height=24,
        ),
    ))
    _apply_base(fig, f"Top {top_n} Assets by Composite Risk Score")
    fig.update_layout(height=min(120 + top_n * 26, 600), margin=dict(l=0, r=0, t=40, b=0))
    return fig

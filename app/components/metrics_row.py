"""
app/components/metrics_row.py
─────────────────────────────────────────────────────────────────────────────
Renders the headline KPI cards at the top of the dashboard.
Uses st.metric() with delta arrows to show how the current config
compares to a reference baseline (alpha=0.30, equal weights).
"""

from __future__ import annotations

import streamlit as st


def render_metrics_row(
    port_summary: dict,
    climate_var: dict,
    eval_metrics: dict,
    ref_haircut_pct: float = 0.0,
) -> None:
    """Render a row of 6 KPI metric cards.

    Args:
        port_summary: Output of valuation.portfolio_summary().
        climate_var: Output of valuation.climate_var().
        eval_metrics: Output of evaluate.evaluate().
        ref_haircut_pct: Reference portfolio haircut % for delta display.
                         Pass 0.0 to suppress deltas on first load.
    """
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    nav = port_summary.get("total_risk_adjusted_nav_usd_m", 0)
    haircut_pct = port_summary.get("portfolio_haircut_pct", 0)
    haircut_usd = port_summary.get("total_haircut_usd_m", 0)
    var_usd = climate_var.get("var_usd_m", 0)
    var_pct = climate_var.get("var_pct_nav", 0)
    gini = eval_metrics.get("gini_composite", 0)
    n_critical = port_summary.get("n_high_risk", 0)
    wavg_score = port_summary.get("weighted_avg_composite_score", 0)

    delta_haircut = (
        f"{haircut_pct - ref_haircut_pct:+.1f}pp"
        if ref_haircut_pct > 0
        else None
    )

    with c1:
        st.metric(
            label="📊 Risk-Adj. NAV",
            value=f"${nav:,.0f}M",
            help="Total portfolio net asset value after climate haircuts.",
        )
    with c2:
        st.metric(
            label="✂️ Portfolio Haircut",
            value=f"{haircut_pct:.1f}%",
            delta=delta_haircut,
            delta_color="inverse",
            help=f"Total haircut: ${haircut_usd:,.0f}M USD",
        )
    with c3:
        st.metric(
            label="⚠️ Climate VaR (95%)",
            value=f"${var_usd:,.0f}M",
            delta=f"{var_pct:.1f}% of NAV",
            delta_color="off",
            help="Monte Carlo 95th-percentile downside vs baseline NAV.",
        )
    with c4:
        st.metric(
            label="📈 Score Gini",
            value=f"{abs(gini):.3f}",
            help=(
                "Gini coefficient of composite score distribution. "
                "Higher = better risk differentiation across portfolio. "
                "Target > 0.30."
            ),
        )
    with c5:
        st.metric(
            label="🔴 High/Critical Assets",
            value=f"{n_critical}",
            help="Number of assets in High or Critical ESG tier.",
        )
    with c6:
        st.metric(
            label="🌡️ Wtd. Avg. Score",
            value=f"{wavg_score:.1f}",
            help="Book-value-weighted average composite risk score.",
        )

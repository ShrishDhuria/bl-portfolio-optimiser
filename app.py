"""
app.py
======
Phase 6 — Streamlit dashboard for the Black-Litterman portfolio optimiser.

Run with:
    streamlit run app.py

Design:
  - Sidebar: views (Q + confidence sliders), concentration cap, τ
  - Tab 1: Portfolio weights, active bets, KPI strip
  - Tab 2: Efficient frontier with three portfolios marked
  - Tab 3: Π vs μ_BL diagnostics, view errors, risk contributions

Expensive computations are cached via @st.cache_data so slider drags only
re-run the cheap stages (views, posterior, optimisation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from market_data import build_market_data, UNIVERSE
from equilibrium import build_equilibrium
from views import View, make_relative_view, build_P_Q_Omega
from bl_model import black_litterman_posterior
from optimiser import (
    max_sharpe, min_variance, efficient_frontier,
    portfolio_stats, risk_contributions,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BL Portfolio Optimiser — SX5E",
    page_icon="📈",
    layout="wide",
)
st.title("Black-Litterman Portfolio Optimiser")
st.caption("SX5E Top 15 · ESSEC MIF project · Methodology: He-Litterman (1999), Idzorek (2005)")


# ---------------------------------------------------------------------------
# Cached pipeline stages
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching market data (3 years)...")
def cached_market_data():
    """Run once at startup, cached for the session."""
    return build_market_data(years=3)


@st.cache_data(show_spinner="Computing equilibrium prior...")
def cached_equilibrium(_Sigma, _w_mkt, _returns, r_f, erp):
    # Leading underscore on DataFrame args = exclude from hash (DataFrames not hashable)
    return build_equilibrium(_Sigma, _w_mkt, _returns, r_f, erp=erp)


@st.cache_data(show_spinner="Computing efficient frontier...")
def cached_frontier(_mu, _Sigma, r_f, n_points, max_weight):
    return efficient_frontier(_mu.values, _Sigma.values,
                              r_f=r_f, n_points=n_points)


# ---------------------------------------------------------------------------
# Load market data (cached)
# ---------------------------------------------------------------------------
md = cached_market_data()
tickers = list(md["Sigma"].index)
ticker_to_name = {tk: UNIVERSE[tk] for tk in tickers}
name_to_ticker = {v: k for k, v in ticker_to_name.items()}


# ---------------------------------------------------------------------------
# Sidebar — views and constraints
# ---------------------------------------------------------------------------
st.sidebar.header("Views")
st.sidebar.caption("Add or modify investor views. Q is annualised return.")

# --- View 1: Absolute (one asset) ---
with st.sidebar.expander("View 1 — Absolute", expanded=True):
    v1_asset_name = st.selectbox("Asset", list(ticker_to_name.values()),
                                  index=0, key="v1_asset")
    v1_Q = st.slider("Q (annual return)", -20.0, 30.0, 12.0, 0.5, key="v1_Q",
                     format="%.1f%%") / 100.0
    v1_conf = st.slider("Confidence", 5, 99, 60, 5, key="v1_conf",
                        format="%d%%") / 100.0
    v1_active = st.checkbox("Include view 1", value=True, key="v1_on")

# --- View 2: Simple relative (two assets) ---
with st.sidebar.expander("View 2 — Simple relative"):
    v2_long_name = st.selectbox("Long",  list(ticker_to_name.values()),
                                 index=list(ticker_to_name.values()).index("BNP Paribas"),
                                 key="v2_long")
    v2_short_name = st.selectbox("Short", list(ticker_to_name.values()),
                                  index=list(ticker_to_name.values()).index("LVMH"),
                                  key="v2_short")
    v2_Q = st.slider("Q (spread)", -15.0, 15.0, 4.0, 0.5, key="v2_Q",
                     format="%.1f%%") / 100.0
    v2_conf = st.slider("Confidence", 5, 99, 40, 5, key="v2_conf",
                        format="%d%%") / 100.0
    v2_active = st.checkbox("Include view 2", value=True, key="v2_on")

# --- View 3: Basket relative ---
with st.sidebar.expander("View 3 — Basket relative"):
    v3_long_names = st.multiselect(
        "Long basket",
        list(ticker_to_name.values()),
        default=["BNP Paribas", "Banco Santander"],
        key="v3_long",
    )
    v3_short_names = st.multiselect(
        "Short basket",
        list(ticker_to_name.values()),
        default=["TotalEnergies", "Sanofi"],
        key="v3_short",
    )
    v3_Q = st.slider("Q (basket spread)", -15.0, 15.0, 3.0, 0.5, key="v3_Q",
                     format="%.1f%%") / 100.0
    v3_conf = st.slider("Confidence", 5, 99, 50, 5, key="v3_conf",
                        format="%d%%") / 100.0
    v3_active = st.checkbox("Include view 3", value=True, key="v3_on")


# --- Constraints + parameters ---
st.sidebar.header("Constraints & parameters")
max_w = st.sidebar.slider("Per-asset cap", 5, 100, 20, 5,
                          format="%d%%",
                          help="Max weight per asset. UCITS typical 10%; we default 20%.") / 100.0
tau   = st.sidebar.slider("τ (prior uncertainty)", 0.005, 0.10, 0.025, 0.005,
                          format="%.3f",
                          help="He-Litterman default = 0.025. Higher = views move posterior more.")
erp   = st.sidebar.slider("Assumed ERP", 2.0, 10.0, 5.0, 0.5,
                          format="%.1f%%",
                          help="Long-run equity risk premium used to compute λ.") / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown(f"**€STR risk-free:** {md['r_f']:.2%}")
st.sidebar.markdown(f"**Data window:** {md['prices'].index[0].date()} → {md['prices'].index[-1].date()}")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
eq = cached_equilibrium(md["Sigma"], md["w_mkt"], md["returns"], md["r_f"], erp)

# Build the views list from sidebar state
views: list[View] = []
if v1_active:
    views.append(View(
        name=f"{v1_asset_name} returns {v1_Q:.1%}",
        P={name_to_ticker[v1_asset_name]: 1.0}, Q=v1_Q, confidence=v1_conf,
    ))
if v2_active and v2_long_name != v2_short_name:
    views.append(View(
        name=f"{v2_long_name} beats {v2_short_name} by {v2_Q:.1%}",
        P={name_to_ticker[v2_long_name]: 1.0,
           name_to_ticker[v2_short_name]: -1.0},
        Q=v2_Q, confidence=v2_conf,
    ))
if v3_active and v3_long_names and v3_short_names:
    overlap = set(v3_long_names) & set(v3_short_names)
    if overlap:
        st.sidebar.error(f"Basket overlap: {overlap}. Remove from one side.")
    else:
        views.append(make_relative_view(
            outperformers=[name_to_ticker[n] for n in v3_long_names],
            underperformers=[name_to_ticker[n] for n in v3_short_names],
            spread=v3_Q, confidence=v3_conf, w_mkt=md["w_mkt"],
            name=f"{v3_long_names} beat {v3_short_names} by {v3_Q:.1%}",
        ))

if not views:
    st.warning("No views active — posterior equals prior. Toggle on at least one view.")
    P_mat, Q_vec, Omega = np.zeros((0, len(tickers))), np.zeros(0), np.zeros((0, 0))
    mu_BL = eq["Pi"]
    Sigma_BL = md["Sigma"]
    view_error = np.zeros(0)
else:
    P_mat, Q_vec, Omega = build_P_Q_Omega(views, tickers, md["Sigma"], tau=tau)
    bl = black_litterman_posterior(eq["Pi"], md["Sigma"], P_mat, Q_vec, Omega, tau=tau)
    mu_BL    = bl["mu_BL"]
    Sigma_BL = bl["Sigma_BL"]
    view_error = bl["view_error"]

# Optimisation
p_sharpe = max_sharpe(mu_BL.values, Sigma_BL.values, md["r_f"], max_weight=max_w)
p_minvar = min_variance(Sigma_BL.values, max_weight=max_w)
w_mkt    = md["w_mkt"].reindex(tickers).fillna(0.0).values
w_equal  = np.ones(len(tickers)) / len(tickers)

stats_sharpe = portfolio_stats(p_sharpe["weights"], mu_BL.values, Sigma_BL.values, md["r_f"])
stats_minvar = portfolio_stats(p_minvar["weights"], mu_BL.values, Sigma_BL.values, md["r_f"])
stats_mkt    = portfolio_stats(w_mkt,               mu_BL.values, Sigma_BL.values, md["r_f"])
stats_equal  = portfolio_stats(w_equal,             mu_BL.values, Sigma_BL.values, md["r_f"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Portfolio", "📈 Efficient Frontier", "🔍 Diagnostics"])


# ====== TAB 1 — PORTFOLIO ======
with tab1:
    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BL Sharpe",
              f"{stats_sharpe['sharpe']:.3f}",
              f"{stats_sharpe['sharpe'] - stats_mkt['sharpe']:+.3f} vs benchmark")
    c2.metric("BL E[r]",         f"{stats_sharpe['return']:.2%}",
              f"{stats_sharpe['return'] - stats_mkt['return']:+.2%}")
    c3.metric("BL σ",            f"{stats_sharpe['vol']:.2%}",
              f"{stats_sharpe['vol'] - stats_mkt['vol']:+.2%}")
    c4.metric("# names",
              int((p_sharpe["weights"] > 1e-4).sum()),
              f"vs {int((w_mkt > 1e-4).sum())} in benchmark")

    st.markdown("### Portfolio weights")
    weights_df = pd.DataFrame({
        "Asset":     [ticker_to_name[t] for t in tickers],
        "BL Sharpe": p_sharpe["weights"],
        "Market":    w_mkt,
        "Equal":     w_equal,
    }).set_index("Asset")
    st.bar_chart(weights_df, height=380)

    st.markdown("### Active bets — BL vs market-cap benchmark")
    active = pd.Series(p_sharpe["weights"] - w_mkt,
                       index=[ticker_to_name[t] for t in tickers])
    active = active.sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in active.values]
    ax.barh(active.index, active.values * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Active weight (percentage points)")
    ax.set_title("Overweight (green) / Underweight (red) vs benchmark")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig, use_container_width=True)


# ====== TAB 2 — EFFICIENT FRONTIER ======
with tab2:
    st.markdown("### Efficient frontier on the BL posterior")
    frontier = cached_frontier(mu_BL, Sigma_BL, md["r_f"], 40, max_w)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Frontier curve
    ax.plot(frontier["vol"] * 100, frontier["target_return"] * 100,
            color="#34495e", linewidth=2, label="Efficient frontier (BL, capped)")

    # Three portfolios as scatter points
    ax.scatter(stats_sharpe["vol"] * 100, stats_sharpe["return"] * 100,
               s=200, color="#27ae60", marker="*", zorder=5,
               label=f"BL max Sharpe ({stats_sharpe['sharpe']:.3f})")
    ax.scatter(stats_minvar["vol"] * 100, stats_minvar["return"] * 100,
               s=120, color="#2980b9", marker="o", zorder=5,
               label=f"BL min variance ({stats_minvar['sharpe']:.3f})")
    ax.scatter(stats_mkt["vol"] * 100, stats_mkt["return"] * 100,
               s=120, color="#e67e22", marker="s", zorder=5,
               label=f"Market cap ({stats_mkt['sharpe']:.3f})")
    ax.scatter(stats_equal["vol"] * 100, stats_equal["return"] * 100,
               s=120, color="#8e44ad", marker="D", zorder=5,
               label=f"Equal weight ({stats_equal['sharpe']:.3f})")

    # Capital market line through max-Sharpe
    x_cml = np.linspace(0, stats_sharpe["vol"] * 100 * 1.4, 50)
    y_cml = md["r_f"] * 100 + stats_sharpe["sharpe"] * x_cml
    ax.plot(x_cml, y_cml, "--", color="#27ae60", alpha=0.5,
            label=f"Capital market line (r_f = {md['r_f']:.2%})")

    ax.set_xlabel("Volatility σ (%)")
    ax.set_ylabel("Expected return E[r] (%)")
    ax.set_title(f"Efficient frontier — per-asset cap = {max_w:.0%}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "The capital market line is the locus of all combinations of the risk-free "
        "asset and the max-Sharpe portfolio. Its slope equals the max Sharpe ratio."
    )


# ====== TAB 3 — DIAGNOSTICS ======
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Prior Π vs Posterior μ_BL")
        diag_df = pd.DataFrame({
            "Asset":     [ticker_to_name[t] for t in tickers],
            "Π (prior)": eq["Pi"].values,
            "μ_BL":      mu_BL.values,
        }).set_index("Asset")
        diag_df["Δ (bps)"] = (diag_df["μ_BL"] - diag_df["Π (prior)"]) * 10_000

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(diag_df["Π (prior)"] * 100, diag_df["μ_BL"] * 100,
                   s=80, alpha=0.7, color="#2980b9")
        lim = max(diag_df["Π (prior)"].max(), diag_df["μ_BL"].max()) * 100 * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="No tilt (μ_BL = Π)")
        for name, row in diag_df.iterrows():
            ax.annotate(name, (row["Π (prior)"] * 100, row["μ_BL"] * 100),
                        fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Prior Π (%)")
        ax.set_ylabel("Posterior μ_BL (%)")
        ax.set_title("Prior-to-posterior shift per asset")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown("### Risk contributions — BL max Sharpe")
        rc = risk_contributions(p_sharpe["weights"], Sigma_BL.values)
        rc_pct = rc / stats_sharpe["vol"] * 100
        rc_df = pd.DataFrame({
            "Asset":           [ticker_to_name[t] for t in tickers],
            "% of total risk": rc_pct,
        })
        rc_df = rc_df[rc_df["% of total risk"] > 0.1].sort_values(
            "% of total risk", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(rc_df["Asset"], rc_df["% of total risk"], color="#34495e")
        ax.set_xlabel("% of total portfolio σ")
        ax.set_title("Per-asset risk contribution")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.markdown("### View errors (Q − PΠ) — how surprising each view is vs the prior")
    if len(views) > 0:
        ve_rows = []
        for i, v in enumerate(views):
            ve_rows.append({
                "View":           v.name,
                "Q":              f"{Q_vec[i]:+.2%}",
                "P·Π (prior)":    f"{(Q_vec[i] - view_error[i]):+.2%}",
                "Error":          f"{view_error[i]:+.2%}",
                "Confidence":     f"{v.confidence:.0%}",
                "ω":              f"{Omega[i, i]:.6f}",
            })
        st.dataframe(pd.DataFrame(ve_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Small errors mean the view is already implied by equilibrium. "
            "Big errors mean the view is materially informative."
        )
    else:
        st.info("No active views.")

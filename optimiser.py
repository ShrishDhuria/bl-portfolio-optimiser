"""
optimiser.py
============
Phase 5 — Mean-variance optimisation on the Black-Litterman posterior.

Three optimisations:
  - Max Sharpe   : maximise (w'μ - r_f) / sqrt(w'Σw)
  - Min Variance : minimise w'Σw
  - Efficient Frontier : sweep target returns, min variance at each

All long-only (w >= 0) with full investment (sum(w) = 1).

The long-only constraint is appropriate for buy-side AM mandates and has
the practical benefit of regularising the optimiser — without it, MV
optimisation produces extreme positions on the most-mispriced names.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------
def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                    r_f: float = 0.0) -> dict:
    """Annualised expected return, vol, Sharpe given weights."""
    w = np.asarray(w)
    r = float(w @ mu)
    var = float(w @ Sigma @ w)
    vol = float(np.sqrt(var))
    sharpe = (r - r_f) / vol if vol > 0 else 0.0
    return {"return": r, "vol": vol, "sharpe": sharpe}


def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Marginal risk contribution per asset.

    RC_i = w_i · (Σ w)_i / σ_p     (sums to σ_p)

    This decomposes total portfolio risk into per-asset contributions,
    accounting for both an asset's own variance and its covariances with
    everything else.  Standard practitioner metric for diagnosing
    portfolio concentration.
    """
    w = np.asarray(w)
    Sigma = np.asarray(Sigma)
    var = float(w @ Sigma @ w)
    if var <= 0:
        return np.zeros_like(w)
    return w * (Sigma @ w) / np.sqrt(var)


# ---------------------------------------------------------------------------
# Max Sharpe
# ---------------------------------------------------------------------------
def max_sharpe(mu: np.ndarray, Sigma: np.ndarray, r_f: float = 0.0,
               long_only: bool = True, max_weight: float = 1.0) -> dict:
    """Maximise Sharpe ratio under sum=1 (and optional long-only).

    max_weight : per-asset cap, e.g. 0.10 for a 10% UCITS-style limit.
                 Set to 1.0 (default) for no cap.
    """
    n = len(mu)
    mu_v = np.asarray(mu)
    Sigma_v = np.asarray(Sigma)

    def neg_sharpe(w):
        r = w @ mu_v
        vol = np.sqrt(w @ Sigma_v @ w)
        return -(r - r_f) / vol if vol > 0 else 1e6

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    lo = 0.0 if long_only else -1.0
    bounds = [(lo, max_weight)] * n

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        print(f"[warn] max_sharpe did not converge: {result.message}")

    w_opt = result.x
    return {"weights": w_opt, **portfolio_stats(w_opt, mu_v, Sigma_v, r_f)}


# ---------------------------------------------------------------------------
# Min Variance
# ---------------------------------------------------------------------------
def min_variance(Sigma: np.ndarray, long_only: bool = True,
                 max_weight: float = 1.0) -> dict:
    """Minimum-variance portfolio under sum=1 (and optional long-only).

    max_weight : per-asset cap (default 1.0 = no cap).
    """
    n = Sigma.shape[0]
    Sigma_v = np.asarray(Sigma)

    def variance(w):
        return w @ Sigma_v @ w

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    lo = 0.0 if long_only else -1.0
    bounds = [(lo, max_weight)] * n

    result = minimize(
        variance, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        print(f"[warn] min_variance did not converge: {result.message}")

    w_opt = result.x
    return {"weights": w_opt, "vol": float(np.sqrt(result.fun))}


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def efficient_frontier(mu: np.ndarray, Sigma: np.ndarray,
                       r_f: float = 0.0, n_points: int = 50,
                       long_only: bool = True) -> pd.DataFrame:
    """Sweep target returns and solve min variance at each.

    Returns a DataFrame with one row per frontier point, columns:
      target_return, vol, sharpe, weights (as np.ndarray)
    """
    n = len(mu)
    mu_v = np.asarray(mu)
    Sigma_v = np.asarray(Sigma)

    # Range: from min-variance return up to the max single-asset return
    mv = min_variance(Sigma_v, long_only=long_only)
    r_min = float(mv["weights"] @ mu_v)
    r_max = float(mu_v.max()) * 0.999  # buffer from corner
    targets = np.linspace(r_min, r_max, n_points)

    rows = []
    w_prev = mv["weights"]  # warm-start at min variance, then walk along curve
    for r_t in targets:
        def variance(w):
            return w @ Sigma_v @ w

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=r_t: w @ mu_v - t},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
        result = minimize(
            variance, w_prev, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.success:
            vol = float(np.sqrt(result.fun))
            sharpe = (r_t - r_f) / vol if vol > 0 else 0.0
            rows.append({
                "target_return": r_t, "vol": vol, "sharpe": sharpe,
                "weights": result.x,
            })
            w_prev = result.x

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def weights_table(portfolios: dict[str, np.ndarray], tickers: list[str],
                  names: dict[str, str]) -> pd.DataFrame:
    """Side-by-side weights for a dict of named portfolios."""
    rows = []
    for i, tk in enumerate(tickers):
        row = {"Asset": names.get(tk, tk)}
        for pname, w in portfolios.items():
            row[pname] = float(w[i])
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_weights_table(df: pd.DataFrame, pct: bool = True) -> str:
    """Pretty-print weights as percentages."""
    fmt = (lambda x: f"{x*100:6.2f}%") if pct else (lambda x: f"{x:.4f}")
    out = df.copy()
    for col in out.columns:
        if col != "Asset":
            out[col] = out[col].map(fmt)
    return out.to_string(index=False)


# ---------------------------------------------------------------------------
# CLI sanity run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import build_market_data, UNIVERSE
    from equilibrium import build_equilibrium
    from views import View, make_relative_view, build_P_Q_Omega
    from bl_model import black_litterman_posterior

    md = build_market_data(years=3)
    eq = build_equilibrium(md["Sigma"], md["w_mkt"], md["returns"], md["r_f"])

    views = [
        View(name="ASML returns 12%",
             P={"ASML.AS": 1.0}, Q=0.12, confidence=0.6),
        View(name="BNP beats LVMH by 4%",
             P={"BNP.PA": 1.0, "MC.PA": -1.0}, Q=0.04, confidence=0.4),
        make_relative_view(
            outperformers=["BNP.PA", "SAN.MC"],
            underperformers=["TTE.PA", "SAN.PA"],
            spread=0.03, confidence=0.5, w_mkt=md["w_mkt"],
            name="Banks beat Energy+Pharma by 3%",
        ),
    ]

    tickers = list(md["Sigma"].index)
    P, Q, Omega = build_P_Q_Omega(views, tickers, md["Sigma"], tau=0.025)
    bl = black_litterman_posterior(
        eq["Pi"], md["Sigma"], P, Q, Omega, tau=0.025
    )

    mu_BL = bl["mu_BL"].values
    Sigma_BL = bl["Sigma_BL"].values
    r_f = md["r_f"]
    w_mkt = md["w_mkt"].reindex(tickers).fillna(0.0).values

    # ---- Three portfolios on the BL posterior ----
    p_sharpe       = max_sharpe(mu_BL, Sigma_BL, r_f)                       # uncapped
    p_sharpe_20    = max_sharpe(mu_BL, Sigma_BL, r_f, max_weight=0.20)      # UCITS-ish 20%
    p_sharpe_10    = max_sharpe(mu_BL, Sigma_BL, r_f, max_weight=0.10)      # tight 10%
    p_minvar       = min_variance(Sigma_BL)
    w_equal        = np.ones(len(tickers)) / len(tickers)

    # ---- Statistics for the comparison portfolios (under BL posterior) ----
    stats_mkt    = portfolio_stats(w_mkt, mu_BL, Sigma_BL, r_f)
    stats_eq     = portfolio_stats(w_equal, mu_BL, Sigma_BL, r_f)
    stats_mv     = portfolio_stats(p_minvar["weights"], mu_BL, Sigma_BL, r_f)

    # ---- Efficient frontier ----
    frontier = efficient_frontier(mu_BL, Sigma_BL, r_f=r_f, n_points=40)

    print("=" * 84)
    print("BL Phase 5 — Mean-variance optimisation on the posterior")
    print("=" * 84)

    print("\nPortfolio statistics (all evaluated under BL posterior):")
    print(f"{'Portfolio':<30}{'E[r]':>10}{'σ':>10}{'Sharpe':>10}{'# names':>10}")
    print("-" * 70)
    n_nz = lambda w: int((np.asarray(w) > 1e-4).sum())
    print(f"{'BL max Sharpe (uncapped)':<30}{p_sharpe['return']*100:>9.2f}%{p_sharpe['vol']*100:>9.2f}%{p_sharpe['sharpe']:>10.3f}{n_nz(p_sharpe['weights']):>10}")
    print(f"{'BL max Sharpe (cap 20%)':<30}{p_sharpe_20['return']*100:>9.2f}%{p_sharpe_20['vol']*100:>9.2f}%{p_sharpe_20['sharpe']:>10.3f}{n_nz(p_sharpe_20['weights']):>10}")
    print(f"{'BL max Sharpe (cap 10%)':<30}{p_sharpe_10['return']*100:>9.2f}%{p_sharpe_10['vol']*100:>9.2f}%{p_sharpe_10['sharpe']:>10.3f}{n_nz(p_sharpe_10['weights']):>10}")
    print(f"{'BL min variance':<30}{stats_mv['return']*100:>9.2f}%{stats_mv['vol']*100:>9.2f}%{stats_mv['sharpe']:>10.3f}{n_nz(p_minvar['weights']):>10}")
    print(f"{'Market-cap (benchmark)':<30}{stats_mkt['return']*100:>9.2f}%{stats_mkt['vol']*100:>9.2f}%{stats_mkt['sharpe']:>10.3f}{n_nz(w_mkt):>10}")
    print(f"{'Equal-weight (1/N)':<30}{stats_eq['return']*100:>9.2f}%{stats_eq['vol']*100:>9.2f}%{stats_eq['sharpe']:>10.3f}{n_nz(w_equal):>10}")

    # ---- Weights table ----
    print("\nPortfolio weights comparison:")
    wt = weights_table({
        "Uncapped":   p_sharpe["weights"],
        "Cap 20%":    p_sharpe_20["weights"],
        "Cap 10%":    p_sharpe_10["weights"],
        "MinVar":     p_minvar["weights"],
        "Market":     w_mkt,
    }, tickers, UNIVERSE)
    print(fmt_weights_table(wt))

    # ---- Active bets BL-capped-20% vs benchmark (more interview-presentable) ----
    print("\nActive bets — BL max Sharpe (20% cap) minus market-cap benchmark (sorted):")
    active = pd.Series(p_sharpe_20["weights"] - w_mkt, index=tickers)
    active_sorted = active.sort_values(ascending=False)
    for tk, a in active_sorted.items():
        if abs(a) > 1e-4:
            sign = "OW" if a > 0 else "UW"
            print(f"  {sign}  {UNIVERSE[tk]:<22}  {a*100:+7.2f} pp")

    # ---- Risk contributions of the BL portfolio (20% cap version) ----
    print("\nRisk contributions in BL max Sharpe (20% cap) portfolio:")
    rc = risk_contributions(p_sharpe_20["weights"], Sigma_BL)
    rc_sorted = pd.Series(rc, index=tickers).sort_values(ascending=False)
    for tk, r in rc_sorted.items():
        if abs(r) > 1e-6:
            pct = r / p_sharpe_20["vol"] * 100
            print(f"  {UNIVERSE[tk]:<22}  σ contrib = {r*100:6.2f}%   ({pct:5.1f}% of total)")

    # ---- Efficient frontier summary ----
    print(f"\nEfficient frontier: {len(frontier)} points computed.")
    print(f"  Min vol point:   E[r] = {frontier['target_return'].iloc[0]*100:.2f}%,   σ = {frontier['vol'].iloc[0]*100:.2f}%")
    print(f"  Max Sharpe pt:   E[r] = {p_sharpe['return']*100:.2f}%,   σ = {p_sharpe['vol']*100:.2f}%,   Sharpe = {p_sharpe['sharpe']:.3f}")
    print(f"  Top of curve:    E[r] = {frontier['target_return'].iloc[-1]*100:.2f}%,   σ = {frontier['vol'].iloc[-1]*100:.2f}%")

    # Export frontier for Phase 6 plotting
    frontier[["target_return", "vol", "sharpe"]].to_csv(
        "data_efficient_frontier.csv", index=False
    )
    print("\nFrontier exported to data_efficient_frontier.csv for Phase 6 plotting.")

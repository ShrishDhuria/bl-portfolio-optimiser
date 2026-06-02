"""
equilibrium.py
==============
Phase 2 — equilibrium implied returns (the Black-Litterman prior).

Computes:
  - Risk-aversion coefficient λ (two estimators, for comparison)
  - Market portfolio variance σ²_m = w' Σ w
  - Implied equilibrium return vector  Π = λ Σ w_mkt
  - Sanity check: the unconstrained optimal portfolio at (Π, Σ) must
    recover w_mkt exactly (up to numerical error from inverting Σ)

This is the *prior* in the Bayesian framing of BL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Risk aversion λ
# ---------------------------------------------------------------------------
def market_portfolio_returns(returns: pd.DataFrame, w_mkt: pd.Series) -> pd.Series:
    """Daily return series of the cap-weighted market portfolio."""
    # Ensure alignment by reindexing weights to the returns columns
    w = w_mkt.reindex(returns.columns).fillna(0.0)
    return returns @ w


def risk_aversion_empirical(returns: pd.DataFrame, w_mkt: pd.Series,
                            r_f: float, trading_days: int = 252) -> float:
    """λ from realised market mean and variance.

    λ = (E[r_m] - r_f) / Var(r_m)

    Numerator is the realised equity risk premium — noisy by construction.
    Use for comparison only.
    """
    r_m = market_portfolio_returns(returns, w_mkt)
    er_m = float(r_m.mean()) * trading_days
    var_m = float(r_m.var(ddof=1)) * trading_days
    return (er_m - r_f) / var_m


def risk_aversion_from_erp(Sigma: pd.DataFrame, w_mkt: pd.Series,
                           erp: float = 0.05) -> float:
    """λ from a fixed equity-risk-premium assumption (textbook BL).

    λ = ERP / σ²_m   where   σ²_m = w_mkt' Σ w_mkt

    He & Litterman (1999) use ERP ≈ 5%.  Result is far more stable than
    the empirical estimator and is what practitioners actually use.
    """
    w = w_mkt.reindex(Sigma.index).fillna(0.0).values
    var_m = float(w @ Sigma.values @ w)
    return erp / var_m


def market_variance(Sigma: pd.DataFrame, w_mkt: pd.Series) -> float:
    """Model-implied annualised market variance σ²_m = w' Σ w."""
    w = w_mkt.reindex(Sigma.index).fillna(0.0).values
    return float(w @ Sigma.values @ w)


# ---------------------------------------------------------------------------
# Implied equilibrium returns Π
# ---------------------------------------------------------------------------
def implied_equilibrium_returns(lam: float, Sigma: pd.DataFrame,
                                w_mkt: pd.Series) -> pd.Series:
    """Π = λ Σ w_mkt — the BL prior on expected returns."""
    w = w_mkt.reindex(Sigma.index).fillna(0.0).values
    pi = lam * (Sigma.values @ w)
    return pd.Series(pi, index=Sigma.index, name="Pi")


# ---------------------------------------------------------------------------
# Sanity check — round trip Π through unconstrained MV optimiser
# ---------------------------------------------------------------------------
def equilibrium_sanity_check(Pi: pd.Series, Sigma: pd.DataFrame,
                             lam: float, w_mkt: pd.Series) -> pd.DataFrame:
    """Verify w* = (1/λ) Σ⁻¹ Π recovers w_mkt.

    Algebraically this is an identity: Π = λΣw  ⇒  (1/λ)Σ⁻¹Π = w.
    Numerically it is a useful health check on Σ — if max abs diff
    is more than a few basis points, Σ is poorly conditioned.
    """
    w_mkt_aligned = w_mkt.reindex(Sigma.index).fillna(0.0)
    Sigma_inv = np.linalg.inv(Sigma.values)
    w_implied = (1.0 / lam) * (Sigma_inv @ Pi.values)
    return pd.DataFrame({
        "w_market":  w_mkt_aligned.values,
        "w_implied": w_implied,
        "diff_bps":  (w_implied - w_mkt_aligned.values) * 10_000,
    }, index=Sigma.index)


# ---------------------------------------------------------------------------
# Bundle Phase 2 outputs
# ---------------------------------------------------------------------------
def build_equilibrium(Sigma: pd.DataFrame, w_mkt: pd.Series, returns: pd.DataFrame,
                      r_f: float, erp: float = 0.05) -> dict:
    """Run the full Phase 2 pipeline."""
    lam_emp = risk_aversion_empirical(returns, w_mkt, r_f)
    lam_erp = risk_aversion_from_erp(Sigma, w_mkt, erp=erp)
    lam = lam_erp  # use ERP-anchored downstream

    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)
    var_m = market_variance(Sigma, w_mkt)

    return {
        "lambda":      lam,
        "lambda_emp":  lam_emp,
        "lambda_erp":  lam_erp,
        "Pi":          Pi,
        "sigma_m":     np.sqrt(var_m),
        "var_m":       var_m,
        "erp_assumed": erp,
    }


# ---------------------------------------------------------------------------
# CLI sanity print
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import build_market_data, UNIVERSE

    md = build_market_data(years=3)
    Sigma  = md["Sigma"]     # Ledoit-Wolf shrunk
    w_mkt  = md["w_mkt"]
    rets   = md["returns"]
    r_f    = md["r_f"]

    eq = build_equilibrium(Sigma, w_mkt, rets, r_f, erp=0.05)

    print("=" * 64)
    print("BL Phase 2 — Equilibrium implied returns")
    print("=" * 64)
    print(f"Assumed ERP:            {eq['erp_assumed']:.2%}")
    print(f"Market portfolio σ:     {eq['sigma_m']:.2%}")
    print(f"Market portfolio σ²:    {eq['var_m']:.4f}")
    print()
    print(f"λ (empirical):          {eq['lambda_emp']:.3f}")
    print(f"λ (ERP-anchored, used): {eq['lambda_erp']:.3f}")
    print()

    print("Implied equilibrium returns Π (annualised):")
    Pi_sorted = eq["Pi"].sort_values(ascending=False)
    for tk, val in Pi_sorted.items():
        print(f"  {UNIVERSE[tk]:<22} {val:6.2%}")

    print("\nSanity check — (1/λ) Σ⁻¹ Π should equal w_mkt:")
    check = equilibrium_sanity_check(eq["Pi"], Sigma, eq["lambda"], w_mkt)
    print(check.round(6).to_string())
    print(f"\nMax |w_implied - w_mkt|: {check['diff_bps'].abs().max():.2f} bps")
    print(f"This should be < 1 bps; larger values indicate a poorly conditioned Σ.")

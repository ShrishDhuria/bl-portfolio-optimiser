"""
bl_model.py
===========
Phase 4 — Black-Litterman posterior expected returns and covariance.

Combines:
  - Prior     :  μ ~ N(Π, τΣ)            (from equilibrium.py)
  - Likelihood:  Q = Pμ + ε, ε ~ N(0,Ω)  (from views.py)
  - Posterior :  μ_BL  and  Σ_BL

The Theil (alternative) form is used as primary:

    μ_BL = Π + τΣ P' (P τΣ P' + Ω)^(-1) (Q - P Π)
    M    = τΣ - τΣ P' (P τΣ P' + Ω)^(-1) P τΣ
    Σ_BL = Σ + M

This form:
  - avoids inverting Ω (entries get tiny at high confidence)
  - inverts only the k×k matrix (P τΣ P' + Ω), not the N×N original
  - reads naturally as "prior + tilt"

The original BL form is implemented separately for numerical cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primary computation — Theil form
# ---------------------------------------------------------------------------
def black_litterman_posterior(
    Pi: pd.Series,
    Sigma: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.025,
) -> dict:
    """Compute the Black-Litterman posterior using the Theil form.

    Returns a dict with mu_BL, Sigma_BL, M, view_error, tilt, and the
    intermediate gain matrix K.
    """
    Pi_v = Pi.values
    Sigma_v = Sigma.values
    tau_Sigma = tau * Sigma_v

    # k×k matrix to invert — this is what makes the Theil form efficient
    A = P @ tau_Sigma @ P.T + Omega
    A_inv = np.linalg.inv(A)

    # Gain matrix K = τΣ P' A^(-1)  — shape (N, k)
    K = tau_Sigma @ P.T @ A_inv

    # How much your views surprise the prior
    view_error = Q - P @ Pi_v                # shape (k,)

    # Posterior mean = prior + tilt
    tilt = K @ view_error                    # shape (N,)
    mu_BL = Pi_v + tilt

    # Posterior covariance of the estimated mean
    M = tau_Sigma - K @ P @ tau_Sigma

    # Posterior covariance for downstream optimisation
    Sigma_BL = Sigma_v + M

    return {
        "mu_BL":      pd.Series(mu_BL, index=Pi.index, name="mu_BL"),
        "Sigma_BL":   pd.DataFrame(Sigma_BL, index=Pi.index, columns=Pi.index),
        "M":          pd.DataFrame(M,        index=Pi.index, columns=Pi.index),
        "tilt":       pd.Series(tilt, index=Pi.index, name="tilt"),
        "view_error": view_error,
        "K":          K,
    }


# ---------------------------------------------------------------------------
# Original BL form — used only for cross-checking the Theil form
# ---------------------------------------------------------------------------
def posterior_original_form(
    Pi: pd.Series,
    Sigma: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.025,
) -> dict:
    """μ_BL via the original BL formula.

    μ_BL = [(τΣ)^(-1) + P' Ω^(-1) P]^(-1) [(τΣ)^(-1) Π + P' Ω^(-1) Q]
    M    = [(τΣ)^(-1) + P' Ω^(-1) P]^(-1)

    Inverts Ω directly and an N×N matrix — kept here purely for verification.
    """
    Pi_v = Pi.values
    Sigma_v = Sigma.values
    tau_Sigma_inv = np.linalg.inv(tau * Sigma_v)
    Omega_inv = np.linalg.inv(Omega)

    M = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
    mu_BL = M @ (tau_Sigma_inv @ Pi_v + P.T @ Omega_inv @ Q)
    Sigma_BL = Sigma_v + M

    return {
        "mu_BL":    pd.Series(mu_BL, index=Pi.index),
        "Sigma_BL": pd.DataFrame(Sigma_BL, index=Pi.index, columns=Pi.index),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def posterior_summary(Pi: pd.Series, mu_BL: pd.Series,
                      Sigma: pd.DataFrame, Sigma_BL: pd.DataFrame,
                      names: dict[str, str]) -> pd.DataFrame:
    """Prior-vs-posterior side-by-side for expected returns and vols."""
    rows = []
    for tk in Pi.index:
        prior_sigma = np.sqrt(Sigma.loc[tk, tk])
        post_sigma  = np.sqrt(Sigma_BL.loc[tk, tk])
        rows.append({
            "Asset":     names.get(tk, tk),
            "Π":         f"{Pi[tk]*100:6.2f}%",
            "μ_BL":      f"{mu_BL[tk]*100:6.2f}%",
            "Δμ (bps)":  f"{(mu_BL[tk] - Pi[tk])*10_000:+8.1f}",
            "σ_prior":   f"{prior_sigma*100:5.2f}%",
            "σ_post":    f"{post_sigma*100:5.2f}%",
            "Δσ (bps)":  f"{(post_sigma - prior_sigma)*10_000:+5.1f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI sanity print
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import build_market_data, UNIVERSE
    from equilibrium import build_equilibrium
    from views import View, make_relative_view, build_P_Q_Omega

    # --- Inputs ---
    md = build_market_data(years=3)
    eq = build_equilibrium(md["Sigma"], md["w_mkt"], md["returns"], md["r_f"])

    views = [
        View(
            name="ASML returns 12% (semiconductor cycle recovery)",
            P={"ASML.AS": 1.0}, Q=0.12, confidence=0.6,
        ),
        View(
            name="BNP beats LVMH by 4% (banks attractive vs luxury)",
            P={"BNP.PA": 1.0, "MC.PA": -1.0}, Q=0.04, confidence=0.4,
        ),
        make_relative_view(
            outperformers=["BNP.PA", "SAN.MC"],
            underperformers=["TTE.PA", "SAN.PA"],
            spread=0.03, confidence=0.5,
            w_mkt=md["w_mkt"],
            name="Banks beat Energy+Pharma by 3%",
        ),
    ]

    tickers = list(md["Sigma"].index)
    P, Q, Omega = build_P_Q_Omega(views, tickers, md["Sigma"], tau=0.025)

    # --- Posterior ---
    bl = black_litterman_posterior(
        Pi=eq["Pi"], Sigma=md["Sigma"], P=P, Q=Q, Omega=Omega, tau=0.025
    )

    print("=" * 80)
    print("BL Phase 4 — Posterior expected returns and covariance")
    print("=" * 80)

    print("\nView error (Q - PΠ): how surprising each view is vs the prior")
    for i, ve in enumerate(bl["view_error"]):
        prior_implied = Q[i] - ve
        print(f"  V{i+1}: Q={Q[i]:+.2%}   prior implies P·Π={prior_implied:+.2%}   error={ve:+.2%}")

    print("\nPrior vs Posterior — expected returns and asset volatilities:")
    summary = posterior_summary(eq["Pi"], bl["mu_BL"],
                                md["Sigma"], bl["Sigma_BL"], UNIVERSE)
    print(summary.to_string(index=False))

    print("\nLargest tilts |μ_BL - Π|, ranked:")
    tilt_sorted = bl["tilt"].abs().sort_values(ascending=False)
    for tk in tilt_sorted.index:
        sign = "+" if bl["tilt"][tk] > 0 else "-"
        print(f"  {UNIVERSE[tk]:<22} {bl['tilt'][tk]*10_000:+8.1f} bps")

    # --- Numerical cross-check ---
    print("\nNumerical cross-check (Theil vs original form):")
    bl_orig = posterior_original_form(eq["Pi"], md["Sigma"], P, Q, Omega, tau=0.025)
    max_mu_diff = float(np.max(np.abs(bl["mu_BL"].values - bl_orig["mu_BL"].values)))
    max_sigma_diff = float(np.max(np.abs(bl["Sigma_BL"].values - bl_orig["Sigma_BL"].values)))
    print(f"  Max |μ_BL_Theil - μ_BL_original|       = {max_mu_diff:.2e}")
    print(f"  Max |Σ_BL_Theil - Σ_BL_original|       = {max_sigma_diff:.2e}")
    print(f"  Both should be < 1e-10 — confirms the matrix algebra.")

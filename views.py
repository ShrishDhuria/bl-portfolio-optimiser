"""
views.py
========
Phase 3 — investor views and confidence (Idzorek method).

Supports three view types:
  1. Absolute   : "ASML returns 12% annually"
  2. Relative   : "BNP outperforms LVMH by 4%"
  3. Basket     : "Banks outperform Energy+Pharma by 3%"   (cap-weighted)

Confidence is expressed as a percentage (0-100%) and mapped to Ω entries
via the Walters-Idzorek formula:

        ω_k = ((1 - c_k) / c_k) · P_k τ Σ P_k'

      c=1   → ω=0    (view treated as certain)
      c=0.5 → ω=PτΣP' (He-Litterman default)
      c→0   → ω→∞    (view ignored, posterior = prior)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# View representation
# ---------------------------------------------------------------------------
@dataclass
class View:
    """A single investor view.

    P : dict mapping {ticker: coefficient}.  Tickers not listed have coef 0.
        Absolute "ASML returns 12%":       {"ASML.AS": 1.0}, Q=0.12
        Simple relative "BNP beats LVMH":  {"BNP.PA": 1.0, "MC.PA": -1.0}, Q=0.04
    Q : expected return for the view (annualised, decimal)
    confidence : in (0, 1]; 0.5 = He-Litterman default
    name : optional human-readable label
    """
    P: dict[str, float]
    Q: float
    confidence: float
    name: str = ""

    def __post_init__(self):
        if not (0 < self.confidence <= 1):
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if not self.P:
            raise ValueError("View needs at least one nonzero pick.")


# ---------------------------------------------------------------------------
# Helper — basket relative view with cap-weighting
# ---------------------------------------------------------------------------
def make_relative_view(
    outperformers: Iterable[str],
    underperformers: Iterable[str],
    spread: float,
    confidence: float,
    w_mkt: pd.Series | None = None,
    name: str = "",
) -> View:
    """Construct a basket-vs-basket relative view.

    Each side is cap-weighted within itself (He-Litterman convention) and
    normalised so the sum of positive weights is +1 and the sum of negative
    weights is -1.  This makes Q directly interpretable as the spread
    between the two baskets.

    If w_mkt is None, falls back to equal-weighting within each basket.
    """
    out_list = list(outperformers)
    und_list = list(underperformers)

    def basket_weights(tickers: list[str], sign: float) -> dict[str, float]:
        if w_mkt is not None:
            sub = w_mkt.reindex(tickers).fillna(0.0)
            if sub.sum() == 0:
                raise ValueError(f"No market-cap data for basket: {tickers}")
            w = sub / sub.sum()
        else:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        return {tk: sign * float(v) for tk, v in w.items()}

    P: dict[str, float] = {}
    for tk, v in basket_weights(out_list, +1.0).items():
        P[tk] = P.get(tk, 0.0) + v
    for tk, v in basket_weights(und_list, -1.0).items():
        P[tk] = P.get(tk, 0.0) + v

    return View(
        P=P, Q=spread, confidence=confidence,
        name=name or f"{out_list} beat {und_list} by {spread:.1%}",
    )


# ---------------------------------------------------------------------------
# Assembly — P, Q, Ω matrices
# ---------------------------------------------------------------------------
def build_P_Q_Omega(
    views: list[View],
    tickers: list[str],
    Sigma: pd.DataFrame,
    tau: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble P (k×N), Q (k×1), Ω (k×k diagonal) from a list of views.

    Ω entries are computed via the Walters-Idzorek confidence formula.
    """
    k = len(views)
    n = len(tickers)
    idx = {tk: i for i, tk in enumerate(tickers)}

    P = np.zeros((k, n))
    Q = np.zeros(k)
    omegas = np.zeros(k)

    Sigma_v = Sigma.values

    for i, view in enumerate(views):
        for tk, coef in view.P.items():
            if tk not in idx:
                raise KeyError(
                    f"View '{view.name or i+1}' references unknown ticker {tk}"
                )
            P[i, idx[tk]] = coef
        Q[i] = view.Q

        # Idzorek confidence mapping
        p_row = P[i, :]
        var_view = float(p_row @ Sigma_v @ p_row.T) * tau
        omegas[i] = ((1.0 - view.confidence) / view.confidence) * var_view

    Omega = np.diag(omegas)
    return P, Q, Omega


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
def views_summary(views: list[View], P: np.ndarray, Q: np.ndarray,
                  Omega: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """Return a tabular summary suitable for printing."""
    rows = []
    for i, v in enumerate(views):
        omega_i = Omega[i, i]
        rows.append({
            "name":              v.name or f"View {i+1}",
            "Q":                 f"{v.Q:.2%}",
            "confidence":        f"{v.confidence:.0%}",
            "ω":                 f"{omega_i:.6f}",
            "1σ band on view":   f"±{np.sqrt(omega_i):.2%}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI sanity print
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_data import build_market_data, UNIVERSE
    from equilibrium import build_equilibrium

    md = build_market_data(years=3)
    eq = build_equilibrium(md["Sigma"], md["w_mkt"], md["returns"], md["r_f"])

    # ---- Demo views — illustrative, not investment advice ----
    views = [
        # 1) Absolute view
        View(
            name="ASML returns 12% (semiconductor cycle recovery)",
            P={"ASML.AS": 1.0},
            Q=0.12,
            confidence=0.6,
        ),
        # 2) Simple relative view
        View(
            name="BNP beats LVMH by 4% (banks attractive vs luxury)",
            P={"BNP.PA": 1.0, "MC.PA": -1.0},
            Q=0.04,
            confidence=0.4,
        ),
        # 3) Basket relative view (cap-weighted within each basket)
        make_relative_view(
            outperformers=["BNP.PA", "SAN.MC"],          # banks
            underperformers=["TTE.PA", "SAN.PA"],        # energy + pharma
            spread=0.03,
            confidence=0.5,
            w_mkt=md["w_mkt"],
            name="Banks beat Energy+Pharma by 3%",
        ),
    ]

    tickers = list(md["Sigma"].index)
    P, Q, Omega = build_P_Q_Omega(views, tickers, md["Sigma"], tau=0.025)

    print("=" * 64)
    print("BL Phase 3 — Views and Idzorek confidence mapping")
    print("=" * 64)
    print(f"\nNumber of views (k):  {len(views)}")
    print(f"τ (prior scaling):    0.025")
    print()

    print("View summary:")
    print(views_summary(views, P, Q, Omega, tickers).to_string(index=False))
    print()

    print("P matrix (k × N) — view picks:")
    P_df = pd.DataFrame(
        P,
        index=[f"V{i+1}" for i in range(len(views))],
        columns=[UNIVERSE[t] for t in tickers],
    )
    print(P_df.round(3).to_string())
    print()

    print("Q vector (view expected returns):")
    for i, q in enumerate(Q):
        print(f"  V{i+1}: {q:.2%}")
    print()

    print("Ω diagonal (view uncertainties):")
    for i, omega in enumerate(np.diag(Omega)):
        print(f"  V{i+1}: ω = {omega:.6f}    1σ = ±{np.sqrt(omega):.2%}")

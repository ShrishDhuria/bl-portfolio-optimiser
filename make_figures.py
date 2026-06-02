"""Regenerate the README results figure from the committed real outputs.

Offline: reads the saved EURO STOXX top-15 covariance (Ledoit-Wolf), market
weights, and efficient frontier (the same artefacts the pipeline produced),
applies one illustrative pair of views, and renders prior-vs-posterior returns
alongside the frontier.

    pip install matplotlib
    python make_figures.py        # writes docs/bl_results.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium import implied_equilibrium_returns, risk_aversion_from_erp
from views import View, build_P_Q_Omega
from bl_model import black_litterman_posterior

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)


def _short(t):
    return t.split(".")[0]


def main():
    Sigma = pd.read_csv(os.path.join(HERE, "data_sigma_lw.csv"), index_col=0)
    Sigma.columns = Sigma.index  # ensure square alignment
    w_mkt = pd.read_csv(os.path.join(HERE, "data_w_mkt.csv"), index_col=0)["weight"]
    frontier = pd.read_csv(os.path.join(HERE, "data_efficient_frontier.csv"))

    tickers = list(Sigma.index)
    lam = risk_aversion_from_erp(Sigma, w_mkt, erp=0.05)
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    # One illustrative pair of views (see README caption): a moderate-confidence
    # tilt toward a name and a low-confidence absolute view on another.
    views = [
        View(P={tickers[0]: 1.0, tickers[1]: -1.0}, Q=0.04, confidence=0.6,
             name=f"{_short(tickers[0])} > {_short(tickers[1])} by 4%"),
        View(P={tickers[2]: 1.0}, Q=0.10, confidence=0.4,
             name=f"{_short(tickers[2])} ~ 10%"),
    ]
    P, Q, Omega = build_P_Q_Omega(views, tickers, Sigma, tau=0.025)
    mu_BL = black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=0.025)["mu_BL"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A — efficient frontier (real, from the saved sweep)
    ax1.plot(frontier["vol"] * 100, frontier["target_return"] * 100,
             color="#1f3b57", lw=2)
    imax = frontier["sharpe"].idxmax()
    ax1.scatter(frontier.loc[imax, "vol"] * 100,
                frontier.loc[imax, "target_return"] * 100,
                color="#c44e34", zorder=5, s=55, label="Max-Sharpe")
    ax1.set_xlabel("Annualised volatility (%)")
    ax1.set_ylabel("Expected return (%)")
    ax1.set_title("Efficient frontier — EURO STOXX top 15")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    # Panel B — implied equilibrium vs BL posterior expected returns
    x = np.arange(len(tickers))
    ax2.bar(x - 0.2, Pi.values * 100, width=0.4, label="Implied Π",
            color="#9bb4c7")
    ax2.bar(x + 0.2, mu_BL.values * 100, width=0.4, label="BL posterior μ",
            color="#1f3b57")
    ax2.set_xticks(x)
    ax2.set_xticklabels([_short(t) for t in tickers], rotation=60, ha="right",
                        fontsize=8)
    ax2.set_ylabel("Annualised return (%)")
    ax2.set_title("Views tilt the prior only where you hold them")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(DOCS, "bl_results.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

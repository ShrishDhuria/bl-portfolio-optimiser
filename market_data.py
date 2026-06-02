"""
market_data.py
==============
Phase 1 data layer for the Black-Litterman portfolio optimiser.

Responsibilities:
  - Pull daily prices for the SX5E universe (Yahoo Finance)
  - Compute log returns and the annualised covariance matrix
  - Apply Ledoit-Wolf shrinkage to stabilise the covariance estimate
  - Fetch market cap weights for the equilibrium prior
  - Fetch the €STR risk-free rate from the ECB Data Portal

Run as a script to print a sanity-check summary of the bundle.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Universe — SX5E top 15 constituents
# ---------------------------------------------------------------------------
# Yahoo Finance ticker -> display name.
# SX5E composition rebalances; verify constituents at refresh time.
UNIVERSE: Dict[str, str] = {
    "ASML.AS":  "ASML",
    "MC.PA":    "LVMH",
    "TTE.PA":   "TotalEnergies",
    "SAP.DE":   "SAP",
    "SAN.PA":   "Sanofi",
    "SIE.DE":   "Siemens",
    "AIR.PA":   "Airbus",
    "SU.PA":    "Schneider Electric",
    "OR.PA":    "L'Oreal",
    "BNP.PA":   "BNP Paribas",
    "SAN.MC":   "Banco Santander",
    "IBE.MC":   "Iberdrola",
    "ITX.MC":   "Inditex",
    "STLAM.MI": "Stellantis",
    "ALV.DE":   "Allianz",   # Allianz substituted for Unilever (no longer SX5E)
}

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Price + return data
# ---------------------------------------------------------------------------
def fetch_prices(years: int = 3, end: datetime | None = None) -> pd.DataFrame:
    """Download daily adjusted close prices for the universe.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=int(365.25 * years) + 10)  # buffer for weekends
    tickers = list(UNIVERSE.keys())

    raw = yf.download(
        tickers=tickers,
        start=start.date(),
        end=end.date(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # When downloading multiple tickers, yfinance returns multi-level columns.
    # We extract each ticker's "Close" column into a flat frame.
    if isinstance(raw.columns, pd.MultiIndex):
        available = [t for t in tickers if t in raw.columns.get_level_values(0)]
        prices = pd.DataFrame({t: raw[t]["Close"] for t in available})
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop fully empty rows, forward-fill gaps (holidays differ across exchanges),
    # then drop any remaining NaNs at the boundaries.
    return prices.dropna(how="all").ffill().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
def covariance_matrix(returns: pd.DataFrame, shrinkage: bool = True) -> pd.DataFrame:
    """Annualised covariance matrix of returns.

    Sample covariance has high estimation error in off-diagonal entries
    (the number of pairwise covariances grows as N^2/2 while the sample
    size grows linearly).  Ledoit-Wolf shrinks the estimate toward a
    structured target (typically a constant-correlation matrix), trading
    a small bias for a large reduction in variance.  This is the
    practitioner standard for portfolio-optimisation inputs.
    """
    if shrinkage:
        lw = LedoitWolf().fit(returns.values)
        cov_daily = lw.covariance_
    else:
        cov_daily = returns.cov().values

    cov_annual = cov_daily * TRADING_DAYS_PER_YEAR
    return pd.DataFrame(cov_annual, index=returns.columns, columns=returns.columns)


# ---------------------------------------------------------------------------
# Market cap weights
# ---------------------------------------------------------------------------
def market_cap_weights(tickers: list[str] | None = None) -> pd.Series:
    """Fetch market caps via yfinance and convert to weights summing to 1."""
    tickers = tickers or list(UNIVERSE.keys())
    caps: Dict[str, float] = {}
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            cap = t.info.get("marketCap")
            if not cap:  # fall back to fast_info which is more reliable
                cap = getattr(t, "fast_info", {}).get("market_cap")
            caps[tk] = float(cap) if cap else np.nan
        except Exception as e:
            print(f"[warn] market cap fetch failed for {tk}: {e}")
            caps[tk] = np.nan

    s = pd.Series(caps, dtype=float).dropna()
    return s / s.sum()


# ---------------------------------------------------------------------------
# Risk-free rate — €STR via ECB Data Portal
# ---------------------------------------------------------------------------
ECB_ESTR_URL = (
    "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT"
    "?lastNObservations=1&format=csvdata"
)

def fetch_estr() -> float:
    """Latest €STR rate as a decimal (e.g. 0.0325 for 3.25%)."""
    try:
        r = requests.get(ECB_ESTR_URL, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        rate_pct = float(df["OBS_VALUE"].iloc[-1])  # ECB reports % p.a.
        return rate_pct / 100.0
    except Exception as e:
        print(f"[warn] €STR fetch failed ({e}); falling back to cached 1.93%")
        return 0.0193


# ---------------------------------------------------------------------------
# Bundle the full Phase 1 output
# ---------------------------------------------------------------------------
def build_market_data(years: int = 3) -> Dict[str, object]:
    """Run the full data pipeline and return all artefacts in one dict."""
    prices = fetch_prices(years=years)
    rets = log_returns(prices)
    Sigma = covariance_matrix(rets, shrinkage=True)
    Sigma_sample = covariance_matrix(rets, shrinkage=False)

    w_mkt = market_cap_weights(list(prices.columns)).reindex(prices.columns).dropna()
    w_mkt = w_mkt / w_mkt.sum()  # re-normalise after any drops

    r_f = fetch_estr()

    return {
        "prices": prices,
        "returns": rets,
        "Sigma": Sigma,                # Ledoit-Wolf shrunk, annualised
        "Sigma_sample": Sigma_sample,  # sample, annualised (for comparison)
        "w_mkt": w_mkt,                # market-cap weights, sum = 1
        "r_f": r_f,                    # €STR as a decimal
        "tickers": list(prices.columns),
        "names": [UNIVERSE[t] for t in prices.columns],
    }


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building BL market-data bundle...")
    md = build_market_data(years=3)

    print(f"\nUniverse:        {len(md['tickers'])} stocks")
    print(f"Date range:      {md['prices'].index[0].date()} -> {md['prices'].index[-1].date()}")
    print(f"Observations:    {len(md['returns'])} daily returns")
    print(f"€STR risk-free:  {md['r_f']:.4%}")

    print("\nMarket cap weights:")
    for tk, w in md["w_mkt"].sort_values(ascending=False).items():
        print(f"  {UNIVERSE[tk]:<22} {w:6.2%}")

    print("\nDiagonal of annualised Σ (volatility):")
    for tk in md["tickers"]:
        vol_lw = np.sqrt(md["Sigma"].loc[tk, tk])
        vol_sm = np.sqrt(md["Sigma_sample"].loc[tk, tk])
        print(f"  {UNIVERSE[tk]:<22} LW: {vol_lw:6.2%}   Sample: {vol_sm:6.2%}")

    # Export everything to CSV for the Excel parallel build.
    md["prices"].to_csv("data_prices.csv")
    md["returns"].to_csv("data_returns.csv")
    md["Sigma"].to_csv("data_sigma_lw.csv")
    md["Sigma_sample"].to_csv("data_sigma_sample.csv")
    md["w_mkt"].to_csv("data_w_mkt.csv", header=["weight"])
    print("\nCSVs written for Excel import: data_prices.csv, data_returns.csv, "
          "data_sigma_lw.csv, data_sigma_sample.csv, data_w_mkt.csv")

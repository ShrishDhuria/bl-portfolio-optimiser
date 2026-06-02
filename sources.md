# Data Sources

| Data | Source | Method | Notes |
|---|---|---|---|
| Daily adjusted close prices | Yahoo Finance | `yfinance.download(...)` | 3y rolling window, `auto_adjust=True` (splits + divs) |
| Market cap weights | Yahoo Finance | `Ticker.info["marketCap"]` with `fast_info` fallback | Spot caps; for production use month-end snapshots |
| Risk-free rate (€STR) | ECB Data Portal | REST API, dataset `EST.B.EU000A2X2A25.WT` | Reported as % p.a.; converted to decimal |
| Covariance estimator | Ledoit-Wolf shrinkage | `sklearn.covariance.LedoitWolf` | Sample covariance kept in parallel for comparison |
| BL methodology | He & Litterman (1999); Idzorek (2005) | Reference papers | Confidence mapping for Ω follows Idzorek |

## Universe — SX5E top 15 constituents
ASML, LVMH, TotalEnergies, SAP, Sanofi, Siemens, Airbus, Schneider Electric, L'Oréal, BNP Paribas, Banco Santander, Iberdrola, Inditex, Stellantis, Allianz.

Note: SX5E composition rebalances; verify constituents on each refresh.

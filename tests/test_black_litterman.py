"""Property tests for the Black-Litterman optimiser.

These assert the *mathematical* facts a reviewer would probe in an interview,
not that functions merely return floats:

  - no views  -> posterior mean collapses onto the equilibrium prior;
  - confidence -> 0 (omega -> inf) -> posterior approaches the prior;
  - the equilibrium round-trip identity (1/lambda) Sigma^-1 Pi = w_mkt;
  - the Theil and original BL forms agree to machine precision;
  - a 100%-confidence view is enforced exactly (P mu_BL = Q);
  - Euler risk contributions sum to portfolio volatility;
  - the posterior covariance stays symmetric and PSD;
  - optimiser weights respect the budget and the per-name cap.

Floating-point comparisons use ``pytest.approx`` (scalars) or
``np.testing.assert_allclose`` (arrays); ``==`` is never used on floats.
"""
import numpy as np
import pytest

from bl_model import black_litterman_posterior, posterior_original_form
from equilibrium import implied_equilibrium_returns, equilibrium_sanity_check
from views import View, build_P_Q_Omega
from optimiser import portfolio_stats, risk_contributions, min_variance, max_sharpe

TAU = 0.025


# --------------------------------------------------------------------------
# Equilibrium prior
# --------------------------------------------------------------------------
def test_equilibrium_round_trip_identity(market):
    """(1/lambda) Sigma^-1 Pi must recover the market weights exactly."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    chk = equilibrium_sanity_check(Pi, Sigma, lam, w_mkt)
    # The engine reports the discrepancy in basis points; it should be ~0.
    assert np.max(np.abs(chk["diff_bps"].values)) < 1e-6

    # ...and the same identity stated directly.
    w_implied = (1.0 / lam) * (np.linalg.inv(Sigma.values) @ Pi.values)
    np.testing.assert_allclose(w_implied, w_mkt.values, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# Posterior: limiting behaviour
# --------------------------------------------------------------------------
def test_no_views_recovers_prior(market):
    """With an empty view set, the posterior mean equals the prior Pi."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)
    n = len(Pi)

    res = black_litterman_posterior(
        Pi, Sigma,
        P=np.empty((0, n)), Q=np.empty(0), Omega=np.empty((0, 0)),
        tau=TAU,
    )
    np.testing.assert_allclose(res["mu_BL"].values, Pi.values, rtol=1e-10, atol=1e-12)


def test_low_confidence_view_approaches_prior(market):
    """As confidence -> 0 (omega -> inf), the view is ignored and mu_BL -> Pi."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    tickers = market["tickers"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    v = View(P={tickers[0]: 1.0}, Q=0.25, confidence=1e-6, name="almost ignored")
    P, Q, Omega = build_P_Q_Omega([v], tickers, Sigma, tau=TAU)
    res = black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=TAU)

    # Within a basis point of the prior despite a wildly different view.
    np.testing.assert_allclose(res["mu_BL"].values, Pi.values, atol=1e-4)


def test_full_confidence_view_is_enforced(market):
    """A 100%-confidence view (omega = 0) is honoured exactly: P mu_BL = Q."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    tickers = market["tickers"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    v = View(P={tickers[0]: 1.0}, Q=0.20, confidence=1.0, name="certain view")
    P, Q, Omega = build_P_Q_Omega([v], tickers, Sigma, tau=TAU)
    assert Omega[0, 0] == pytest.approx(0.0, abs=1e-15)

    res = black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=TAU)
    np.testing.assert_allclose(P @ res["mu_BL"].values, Q, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------
# Posterior: cross-check and well-posedness
# --------------------------------------------------------------------------
def test_theil_equals_original_form(market):
    """The Theil (primary) and original BL forms must agree to ~machine eps."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    tickers = market["tickers"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    views = [
        View(P={tickers[0]: 1.0, tickers[1]: -1.0}, Q=0.03, confidence=0.5, name="A>B"),
        View(P={tickers[2]: 1.0}, Q=0.11, confidence=0.7, name="C absolute"),
    ]
    P, Q, Omega = build_P_Q_Omega(views, tickers, Sigma, tau=TAU)

    theil = black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=TAU)
    orig = posterior_original_form(Pi, Sigma, P, Q, Omega, tau=TAU)

    np.testing.assert_allclose(theil["mu_BL"].values, orig["mu_BL"].values,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(theil["Sigma_BL"].values, orig["Sigma_BL"].values,
                               rtol=1e-10, atol=1e-12)


def test_posterior_covariance_symmetric_and_psd(market):
    """Sigma_BL must be symmetric and positive semi-definite."""
    Sigma, w_mkt, lam = market["Sigma"], market["w_mkt"], market["lam"]
    tickers = market["tickers"]
    Pi = implied_equilibrium_returns(lam, Sigma, w_mkt)

    v = View(P={tickers[1]: 1.0}, Q=0.13, confidence=0.6, name="B absolute")
    P, Q, Omega = build_P_Q_Omega([v], tickers, Sigma, tau=TAU)
    Sig_bl = black_litterman_posterior(Pi, Sigma, P, Q, Omega, tau=TAU)["Sigma_BL"].values

    np.testing.assert_allclose(Sig_bl, Sig_bl.T, rtol=1e-12, atol=1e-14)
    assert np.linalg.eigvalsh(Sig_bl).min() > -1e-10


# --------------------------------------------------------------------------
# Optimiser invariants
# --------------------------------------------------------------------------
def test_risk_contributions_sum_to_volatility(market):
    """Euler decomposition: sum_i RC_i = portfolio sigma."""
    Sigma, mu = market["Sigma"].values, market["mu"].values
    w = np.array([0.4, 0.3, 0.2, 0.1])

    rc = risk_contributions(w, Sigma)
    vol = portfolio_stats(w, mu, Sigma)["vol"]
    assert rc.sum() == pytest.approx(vol, rel=1e-10)


def test_min_variance_respects_budget_and_cap(market):
    Sigma = market["Sigma"].values
    res = min_variance(Sigma, long_only=True, max_weight=0.5)
    w = res["weights"]
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.min() >= -1e-9               # long-only
    assert w.max() <= 0.5 + 1e-9          # per-name cap


def test_max_sharpe_cap_binds(market):
    Sigma, mu = market["Sigma"].values, market["mu"].values
    res = max_sharpe(mu, Sigma, r_f=0.0, long_only=True, max_weight=0.4)
    w = res["weights"]
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.max() <= 0.4 + 1e-9

"""Shared fixtures for the Black-Litterman test-suite.

The project modules (``bl_model``, ``equilibrium``, ``views``, ``optimiser``)
live at the project root, so we put that root on ``sys.path`` here rather
than relying on an editable install.

The ``market`` fixture is a small, fully synthetic 4-asset market: a positive
-definite annualised covariance built from fixed vols and a fixed correlation
matrix, plus cap weights and a risk-aversion coefficient. Nothing here touches
the network — these tests exercise the *maths*, not the data layer.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# --- make the project root importable -------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope="session")
def market():
    """A small synthetic but well-conditioned market.

    Returns a dict with:
      tickers : list[str]
      Sigma   : pd.DataFrame  annualised covariance (PSD, invertible)
      w_mkt   : pd.Series     cap weights, sum to 1
      lam     : float         risk-aversion coefficient
      mu      : pd.Series     an arbitrary expected-return vector for optimiser tests
    """
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    vols = np.array([0.20, 0.25, 0.30, 0.18])
    corr = np.array([
        [1.00, 0.30, 0.20, 0.10],
        [0.30, 1.00, 0.40, 0.20],
        [0.20, 0.40, 1.00, 0.30],
        [0.10, 0.20, 0.30, 1.00],
    ])
    cov = np.outer(vols, vols) * corr
    Sigma = pd.DataFrame(cov, index=tickers, columns=tickers)

    w_mkt = pd.Series([0.40, 0.30, 0.20, 0.10], index=tickers)
    lam = 2.5  # any positive value works; the round-trip identity is lambda-free
    mu = pd.Series([0.08, 0.10, 0.12, 0.06], index=tickers)

    return {"tickers": tickers, "Sigma": Sigma, "w_mkt": w_mkt, "lam": lam, "mu": mu}

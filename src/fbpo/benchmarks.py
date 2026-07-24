"""Expected-return estimators and the benchmark roster.

Ten strategies, differing only in what they feed the optimizer:

SRHist    -- max-Sharpe on historical mean asset returns
`SRF1        -- max-Sharpe on factor-implied means, market premium from
                     the prevailing historical mean
SRML        -- factor-implied means, market premium from the ML forecast
SRML_shrunk -- as SRML, with Vasicek-shrunk betas
BayesStein``  -- max-Sharpe on Bayes-Stein shrunk historical means
GMV  `MDRP ``InvVol``, ``EW`` -- no return estimate at all
DJIA        -- the index itself

The factor strategies are the paper's contribution. Instead of estimating N
expected returns from noisy sample means, estimate *one* market premium and
project it through betas:

    mu_i = rf + beta_i * (market premium)

That replaces N noisy parameters with one forecast plus N betas, and betas are
estimated far more precisely than means. The comparison SRHist vs SRF1 isolates
the value of the factor structure; SRF1 vs SRML isolates the value of the ML
forecast.

UNITS. Everything here is daily: daily mu, daily rf, daily covariance. A
monthly premium forecast is divided by 21. Max-Sharpe weights are invariant to
independent positive rescaling of mu and Sigma, but *not* to a mismatch
between mu and rf -- so the one rule that matters is that they share units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21

BENCHMARK_STRATEGIES = ("GMVP", "MDRP", "InvVol", "EW", "DJIA")
FACTOR_STRATEGIES = ("SRF1", "SRML", "SRML_shrunk")
MEAN_STRATEGIES = ("SRHist", "BayesStein")
ALL_STRATEGIES = MEAN_STRATEGIES + FACTOR_STRATEGIES + BENCHMARK_STRATEGIES


def historical_mean_mu(window: pd.DataFrame) -> np.ndarray:
    """Sample mean of daily returns over the estimation window.

    The naive estimator, and the reason this literature exists: with T=252 the
    standard error of a mean daily return is roughly the daily volatility over
    16, which is the same order as the mean itself.
    """
    return window.mean().to_numpy()


def bayes_stein_mu(window: pd.DataFrame, covariance: np.ndarray) -> np.ndarray:
    """Jorion (1986) Bayes-Stein shrinkage toward the minimum-variance mean.

        w = (N + 2) / ((N + 2) + T * (mu - mu_0)' Sigma^-1 (mu - mu_0))
        mu_bs = (1 - w) mu + w mu_0

    The shrinkage target mu_0 is the return on the minimum-variance portfolio,
    not a scalar constant: Jorion's argument is that the grand mean is itself
    an arbitrary choice, while the MV portfolio's return is the point the
    cross-section has least information about.
    """
    mu = window.mean().to_numpy()
    n, t = len(mu), len(window)

    precision = np.linalg.pinv(covariance)
    ones = np.ones(n)
    mu_zero = float((ones @ precision @ mu) / (ones @ precision @ ones))

    deviation = mu - mu_zero
    quadratic = float(deviation @ precision @ deviation)
    if quadratic <= 0:
        return mu

    weight = (n + 2) / ((n + 2) + t * quadratic)
    weight = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - weight) * mu + weight * mu_zero


def factor_mu(
    beta: np.ndarray, monthly_premium: float, daily_risk_free: float
) -> np.ndarray:
    """Project a market premium forecast through betas.

    mu_i = rf + beta_i * premium. The premium arrives as a monthly figure from
    the forecast engine and is converted to daily units here, once, so the
    conversion cannot be applied twice by accident.
    """
    daily_premium = monthly_premium / TRADING_DAYS_PER_MONTH
    return daily_risk_free + np.asarray(beta, dtype=float).ravel() * daily_premium


def fetch_djia(cfg, *, refresh: bool = False) -> pd.Series:
    """Daily returns of the Dow Jones Industrial Average itself.

    The index is price-weighted and includes WBA for the whole sample, so it is
    not reproducible from the 29-name panel. Downloading it directly is the
    honest option: it is the benchmark a reader will compare against.
    """
    from pathlib import Path

    import yfinance as yf

    from fbpo.data import RAW_DIR, _resolve_end

    cache = Path(RAW_DIR) / "djia_index.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["djia"]

    frame = yf.download(
        "^DJI",
        start=cfg.data.price_start,
        end=_resolve_end(cfg.data.price_end),
        auto_adjust=True,
        progress=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("could not download ^DJI; the DJIA benchmark is unavailable")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()

    returns = close.pct_change().dropna().rename("djia")
    cache.parent.mkdir(parents=True, exist_ok=True)
    returns.to_frame().to_parquet(cache)
    return returns


def expected_returns(
    strategy: str,
    window: pd.DataFrame,
    covariance: np.ndarray,
    beta: np.ndarray | None = None,
    shrunk_beta: np.ndarray | None = None,
    premium_hist: float | None = None,
    premium_ml: float | None = None,
    daily_risk_free: float = 0.0,
) -> np.ndarray | None:
    """The expected-return vector a strategy feeds to the optimizer.

    Returns ``None`` for strategies that use no return estimate, which is the
    signal that the optimizer should not be asked for max-Sharpe weights.
    """
    if strategy in BENCHMARK_STRATEGIES:
        return None
    if strategy == "SRHist":
        return historical_mean_mu(window)
    if strategy == "BayesStein":
        return bayes_stein_mu(window, covariance)
    if strategy == "SRF1":
        return factor_mu(beta, premium_hist, daily_risk_free)
    if strategy == "SRML":
        return factor_mu(beta, premium_ml, daily_risk_free)
    if strategy == "SRML_shrunk":
        return factor_mu(shrunk_beta, premium_ml, daily_risk_free)
    raise ValueError(f"unknown strategy {strategy!r}")


def objective_for(strategy: str) -> str:
    """Which optimizer each strategy calls."""
    if strategy in MEAN_STRATEGIES or strategy in FACTOR_STRATEGIES:
        return "max_sharpe"
    return {"GMVP": "gmvp", "MDRP": "mdrp", "InvVol": "inverse_vol", "EW": "equal_weight"}[
        strategy
    ]

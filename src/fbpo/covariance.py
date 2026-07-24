"""Covariance estimators and their conditioning diagnostics.

With N=29 assets and T=252 days the sample covariance is estimable, but the
maximum-Sharpe solution is most sensitive to its *smallest* eigenvalues --
which is exactly where estimation error concentrates. Inverting a poorly
conditioned sample covariance turns noise into confident-looking weights.

Five estimators, in increasing order of structure imposed:

* ``sample``       -- no structure, maximum estimation error
* ``ledoit_wolf``  -- shrinks toward a scaled identity, optimal shrinkage
                      intensity chosen analytically
* ``oas``          -- oracle-approximating shrinkage, better when returns are
                      close to Gaussian
* ``ewma``         -- exponential weighting, so recent observations dominate
* ``single_index`` -- imposes the market factor structure outright:
                      Sigma = s2_m b b' + diag(s2_e)

and ``yang_zhang``, which is not a covariance estimator but a volatility one:
it keeps the correlation matrix from close-to-close returns and rescales the
diagonal with a range-based volatility that uses the high and low. Range
estimators are several times more efficient per observation than close-to-close,
which matters when the window is short.

Every estimator returns a symmetric matrix; ``nearest_psd`` is applied at the
boundary so the optimizer never receives an indefinite matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_EIGENVALUE = 1e-10


def nearest_psd(matrix: np.ndarray, epsilon: float = MIN_EIGENVALUE) -> np.ndarray:
    """Clip negative eigenvalues to `epsilon`, preserving eigenvectors.

    Sample covariances of masked or partially-missing data can come back with
    tiny negative eigenvalues from floating-point error. A solver handed an
    indefinite matrix either fails or returns something meaningless, so the
    projection happens here rather than being discovered downstream.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    if values.min() >= epsilon:
        return symmetric
    clipped = np.clip(values, epsilon, None)
    return vectors @ np.diag(clipped) @ vectors.T


def sample_covariance(window: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance, ddof=1."""
    return nearest_psd(window.cov(ddof=1).to_numpy())


def ledoit_wolf_covariance(window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity.

    The shrinkage intensity is chosen to minimise expected squared error, not
    tuned. That is the appeal: no free parameter to overfit.
    """
    from sklearn.covariance import ledoit_wolf

    estimate, _ = ledoit_wolf(window.to_numpy(), assume_centered=False)
    return nearest_psd(estimate)


def oas_covariance(window: pd.DataFrame) -> np.ndarray:
    """Oracle-approximating shrinkage (Chen et al. 2010).

    Converges to the true covariance faster than Ledoit-Wolf when returns are
    close to Gaussian, and slightly worse when they are not -- which for daily
    equity returns is a real caveat, hence both are offered.
    """
    from sklearn.covariance import oas

    estimate, _ = oas(window.to_numpy(), assume_centered=False)
    return nearest_psd(estimate)


def ewma_covariance(window: pd.DataFrame, halflife: int = 63) -> np.ndarray:
    """Exponentially weighted covariance.

    Recent observations carry more weight, so the estimate adapts to a
    volatility regime change instead of averaging across it. The cost is a
    smaller effective sample: a 63-day halflife over 252 days uses roughly 90
    observations' worth of information.
    """
    decay = 0.5 ** (1.0 / halflife)
    n = len(window)
    weights = decay ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()

    values = window.to_numpy()
    mean = np.average(values, axis=0, weights=weights)
    centred = values - mean
    covariance = (centred * weights[:, None]).T @ centred
    covariance = covariance / (1.0 - np.sum(weights**2))
    return nearest_psd(covariance)
def single_index_covariance(
    beta: np.ndarray,
    market_variance: float,
    residual_variance: np.ndarray,
    delta: float = 1.0,
    sample: np.ndarray | None = None,
) -> np.ndarray:
    """Sharpe's single-index model: Sigma = s2_m b b' + diag(s2_e).

    Every pairwise covariance is forced through the market factor, which
    reduces the number of estimated parameters from N(N+1)/2 to 2N+1. With
    N=29 that is 435 parameters down to 59 -- a large reduction in estimation
    error, bought with the assumption that residuals are uncorrelated.

    `delta` blends toward the sample covariance: delta=1 is the pure factor
    model, delta=0 is pure sample. The configured 0.5 is Ledoit-Wolf's
    structured-shrinkage idea with the market as the structure.
    """
    factor = market_variance * np.outer(beta, beta) + np.diag(residual_variance)
    if delta >= 1.0 or sample is None:
        return nearest_psd(factor)
    blended = delta * factor + (1.0 - delta) * sample
    return nearest_psd(blended)


def yang_zhang_volatility(
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Yang-Zhang range volatility over the supplied window, annualised.

    Combines three components: overnight (close-to-open) variance, open-to-close
    variance, and the drift-independent Rogers-Satchell estimator built from the
    high and low. Range estimators extract far more information per day than a
    close-to-close estimate, because the path within the day is observed rather
    than discarded.

    This is why ``auto_adjust=True`` is enforced in the data layer: with
    ``auto_adjust=False`` only Close is adjusted, and unadjusted High/Low make
    every Rogers-Satchell term wrong on any day following a split.
    """
    n = len(closes)
    if n < 3:
        raise ValueError(f"Yang-Zhang needs at least 3 observations, got {n}")

    log_ho = np.log(highs / opens)
    log_lo = np.log(lows / opens)
    log_co = np.log(closes / opens)
    log_oc = np.log(opens / closes.shift(1))

    rogers_satchell = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).mean()
    overnight = log_oc.var(ddof=1)
    open_to_close = log_co.var(ddof=1)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = overnight + k * open_to_close + (1.0 - k) * rogers_satchell
    return np.sqrt(variance.clip(lower=0.0) * trading_days_per_year)


def rescale_with_volatility(
    covariance: np.ndarray, target_volatility: np.ndarray, source_volatility: np.ndarray
) -> np.ndarray:
    """Keep the correlation structure, replace the diagonal: Sigma = D R D.

    Correlations from close-to-close returns are reasonably well estimated;
    volatilities are not. This swaps in a better volatility estimate without
    disturbing the correlation matrix.
    """
    safe_source = np.where(source_volatility > 0, source_volatility, np.nan)
    correlation = covariance / np.outer(safe_source, safe_source)
    np.fill_diagonal(correlation, 1.0)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    return nearest_psd(correlation * np.outer(target_volatility, target_volatility))


def covariance_diagnostics(covariance: np.ndarray) -> dict[str, float]:
    """Conditioning summary. Cheap to compute, and the explanation for odd results.

    SPEC S5 expects a condition number between 20 and 400. Above 1e5 means
    duplicate columns or T < N. PC1 variance share typically sits at 0.35-0.55
    and rises to 0.65-0.80 in March 2020, when everything correlated at once --
    a diagnostic that doubles as a regime indicator.
    """
    values = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    values = np.clip(values, 0.0, None)
    total = values.sum()

    shares = values / total if total > 0 else np.zeros_like(values)
    nonzero = shares[shares > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0

    return {
        "min_eigenvalue": float(values.min()),
        "max_eigenvalue": float(values.max()),
        "condition_number": float(values.max() / values.min()) if values.min() > 0 else np.inf,
        "effective_rank": float(np.exp(entropy)),
        "pc1_share": float(values.max() / total) if total > 0 else np.nan,
    }


def estimate_covariance(window: pd.DataFrame, kind: str = "sample", **kwargs) -> np.ndarray:
    """Dispatch on the configured estimator.

    Factor and range estimators need inputs the return window alone does not
    carry (betas, OHLC), so they are called directly by the backtester rather
    than through this helper.
    """
    if kind == "sample":
        return sample_covariance(window)
    if kind == "ledoit_wolf":
        return ledoit_wolf_covariance(window)
    if kind == "oas":
        return oas_covariance(window)
    if kind == "ewma":
        return ewma_covariance(window, halflife=kwargs.get("halflife", 63))
    if kind in {"single_index", "yang_zhang"}:
        raise ValueError(
            f"{kind!r} requires auxiliary inputs; call its constructor directly"
        )
    raise ValueError(f"unknown covariance estimator {kind!r}")

"""Inference on backtest results.

A ranking is not a finding. The strategies here hold overlapping positions and
their daily returns correlate above 0.9, which cuts both ways: the standard
error of any single Sharpe ratio is large (roughly 0.3 over twelve years), but
the standard error of the *difference* between two of them is far smaller,
because the common market component cancels. Paired tests are therefore much
more powerful than comparing confidence intervals, and comparing intervals is
the mistake this module exists to prevent.

Three tests, each answering a different question:

* :func:`paired_sharpe_test` -- is strategy A's Sharpe different from B's?
* :func:`deflated_sharpe_ratio` -- is the *winner's* Sharpe still impressive
  once you account for how many strategies were tried?
* :func:`clark_west` -- does the larger forecasting model beat the nested
  smaller one? Standard Diebold-Mariano is biased against the larger model
  when models are nested, which is exactly our SVR-vs-prevailing-mean case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series | None = None, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of a return series."""
    excess = returns if risk_free is None else returns - risk_free.reindex(returns.index).fillna(0.0)
    excess = excess.dropna()
    if len(excess) < 2 or excess.std(ddof=1) == 0:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))


def newey_west_variance(x: np.ndarray, lags: int | None = None) -> float:
    """HAC variance of the sample mean of `x`.

    Daily strategy returns are mildly autocorrelated and strongly
    heteroskedastic. An i.i.d. standard error understates uncertainty; the
    Bartlett-weighted sum of autocovariances corrects for both.

    Default lag length follows Newey-West's rule of thumb, 4(T/100)^(2/9).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (t / 100.0) ** (2.0 / 9.0)))

    centred = x - x.mean()
    variance = float(centred @ centred) / t
    for lag in range(1, min(lags, t - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(centred[lag:] @ centred[:-lag]) / t
        variance += 2.0 * weight * covariance
    return variance / t


def stationary_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index path.

    Blocks have geometrically distributed length with mean `block_length`,
    which preserves the series' autocorrelation structure while keeping the
    resampled series stationary -- unlike fixed-length blocks, which introduce
    artificial periodicity at the block boundary.
    """
    p = 1.0 / block_length
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def paired_sharpe_test(
    a: pd.Series,
    b: pd.Series,
    reps: int = 10_000,
    block_length: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Test whether two strategies have different Sharpe ratios.

    Reports both the analytic Jobson-Korkie test with the Memmel correction
    and a stationary-bootstrap p-value. They usually agree; when they do not,
    trust the bootstrap, since JKM assumes i.i.d. normal returns and daily
    equity returns are neither.

    The test is paired: it resamples the two series *together*, preserving
    their contemporaneous correlation. Treating them as independent samples
    would inflate the standard error several-fold and hide real differences.
    """
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return {"n": len(joined)}

    x = joined.iloc[:, 0].to_numpy()
    y = joined.iloc[:, 1].to_numpy()
    n = len(x)

    mu_x, mu_y = x.mean(), y.mean()
    sd_x, sd_y = x.std(ddof=1), y.std(ddof=1)
    cov = float(np.cov(x, y, ddof=1)[0, 1])

    sharpe_x = mu_x / sd_x * np.sqrt(TRADING_DAYS)
    sharpe_y = mu_y / sd_y * np.sqrt(TRADING_DAYS)

    # Jobson-Korkie (1981) with Memmel's (2003) correction.
    var_x, var_y = sd_x**2, sd_y**2
    theta = (
        2.0 * var_x * var_y
        - 2.0 * sd_x * sd_y * cov
        + 0.5 * mu_x**2 * var_y
        + 0.5 * mu_y**2 * var_x
        - (mu_x * mu_y / (sd_x * sd_y)) * cov**2
    ) / n
    jkm_z = (sd_y * mu_x - sd_x * mu_y) / np.sqrt(theta) if theta > 0 else np.nan
    jkm_p = float(2.0 * (1.0 - sps.norm.cdf(abs(jkm_z)))) if np.isfinite(jkm_z) else np.nan

    # Paired stationary bootstrap under the null of equal Sharpe ratios.
    rng = np.random.default_rng(seed)
    observed = sharpe_x - sharpe_y
    centred_x = x - mu_x
    centred_y = y - mu_y

    count = 0
    for _ in range(reps):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        bx, by = centred_x[idx], centred_y[idx]
        sx, sy = bx.std(ddof=1), by.std(ddof=1)
        if sx <= 0 or sy <= 0:
            continue
        difference = (bx.mean() / sx - by.mean() / sy) * np.sqrt(TRADING_DAYS)
        if abs(difference) >= abs(observed):
            count += 1

    return {
        "sharpe_a": float(sharpe_x),
        "sharpe_b": float(sharpe_y),
        "difference": float(observed),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "jkm_z": float(jkm_z),
        "jkm_p": jkm_p,
        "bootstrap_p": float(count / reps),
        "n": int(n),
    }
def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Probability that the true Sharpe exceeds `benchmark_sharpe`.

    Corrects for skewness and kurtosis, which matter: negatively skewed,
    fat-tailed returns make a given Sharpe less trustworthy than the normal
    approximation implies, and equity strategies are reliably both.
    """
    x = returns.dropna().to_numpy()
    n = len(x)
    if n < 30 or x.std(ddof=1) == 0:
        return np.nan

    sr = x.mean() / x.std(ddof=1)
    sr_benchmark = benchmark_sharpe / np.sqrt(periods)
    skew = float(sps.skew(x))
    kurtosis = float(sps.kurtosis(x, fisher=False))

    denominator = np.sqrt(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2)
    if denominator <= 0:
        return np.nan
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denominator))


def expected_maximum_sharpe(trial_sharpes: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Expected maximum Sharpe from N independent trials with zero true skill.

    Bailey and Lopez de Prado (2014). The intuition: run twenty strategies with
    no edge and the best one still looks good. This is how good, in annualised
    units -- the bar the winner has to clear before it means anything.
    """
    sharpes = np.asarray(trial_sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    n = len(sharpes)
    if n < 2:
        return np.nan

    variance = float(np.var(sharpes, ddof=1)) / periods
    if variance <= 0:
        return 0.0

    z_one = sps.norm.ppf(1.0 - 1.0 / n)
    z_two = sps.norm.ppf(1.0 - 1.0 / (n * np.e))
    expected = np.sqrt(variance) * ((1.0 - EULER_MASCHERONI) * z_one + EULER_MASCHERONI * z_two)
    return float(expected * np.sqrt(periods))


def deflated_sharpe_ratio(
    returns: pd.Series, trial_sharpes: np.ndarray, periods: int = TRADING_DAYS
) -> dict[str, float]:
    """Probability the winner's Sharpe is real, given how many were tried.

    The maximum of N correlated Sharpe ratios is upward-biased by construction.
    DSR is the probabilistic Sharpe ratio computed against that inflated
    benchmark rather than against zero. Below 0.95 means the result is not
    distinguishable from the best of N lucky draws.

    `trial_sharpes` must be an honest count. Understating it is the single
    easiest way to make a backtest look better than it is.
    """
    threshold = expected_maximum_sharpe(trial_sharpes, periods)
    return {
        "sharpe": sharpe_ratio(returns, periods=periods),
        "n_trials": int(np.isfinite(np.asarray(trial_sharpes, dtype=float)).sum()),
        "expected_max_sharpe": threshold,
        "deflated_sharpe": probabilistic_sharpe_ratio(returns, threshold, periods),
    }


def clark_west(
    actual: pd.Series, restricted: pd.Series, unrestricted: pd.Series, lags: int | None = None
) -> dict[str, float]:
    """Clark-West test for nested out-of-sample forecast comparison.

    When the smaller model is nested in the larger one, the larger model's MSE
    is biased upward under the null, because it estimates parameters that are
    truly zero. Diebold-Mariano therefore under-rejects. Clark and West (2007)
    add back the estimation-noise term:

        f_t = (y - f1)^2 - [(y - f2)^2 - (f1 - f2)^2]

    A positive mean of f_t favours the larger model. One-sided by construction:
    the alternative is that the larger model helps, never that it hurts.
    """
    index = actual.index.intersection(restricted.index).intersection(unrestricted.index)
    y = actual.loc[index].to_numpy()
    f1 = restricted.loc[index].to_numpy()
    f2 = unrestricted.loc[index].to_numpy()

    adjusted = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    variance = newey_west_variance(adjusted, lags)
    if not np.isfinite(variance) or variance <= 0:
        return {"cw_stat": np.nan, "p_value": np.nan, "n": len(index)}

    statistic = float(adjusted.mean() / np.sqrt(variance))
    return {
        "cw_stat": statistic,
        "p_value": float(1.0 - sps.norm.cdf(statistic)),
        "mean_adjusted": float(adjusted.mean()),
        "n": int(len(index)),
    }


def pairwise_sharpe_table(
    returns: pd.DataFrame,
    reference: str,
    risk_free: pd.Series | None = None,
    reps: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Every strategy tested against one reference, sorted by difference.

    The reference should be the benchmark the claim is made against -- usually
    ``EW``, since DeMiguel, Garlappi and Uppal (2009) is the null hypothesis
    this whole literature has to beat.

    `risk_free` must be supplied to make these Sharpe ratios comparable with
    those from :func:`fbpo.backtest.summarise`, which reports excess-return
    Sharpes. Omitting it inflates every figure by rf/volatility.
    """
    frame = returns if risk_free is None else returns.sub(
        risk_free.reindex(returns.index).fillna(0.0), axis=0
    )
    rows = {}
    for column in frame.columns:
        if column == reference:
            continue
        rows[column] = paired_sharpe_test(
            frame[column].dropna(), frame[reference].dropna(), reps=reps, seed=seed
        )
    return pd.DataFrame(rows).T.sort_values("difference", ascending=False)

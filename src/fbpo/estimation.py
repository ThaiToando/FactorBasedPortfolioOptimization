"""Rolling market betas with empirical-Bayes shrinkage.

Three estimators, in increasing order of how much they trust the data:

* Raw OLS -- the 252-day trailing regression of asset return on market return.
* Blume -- the fixed practitioner rule (2/3)b + 1/3, shrinking every asset by
  the same amount regardless of how well its beta was estimated.
* Vasicek -- empirical Bayes. Each asset is pulled toward the cross-sectional
  mean by a weight that depends on its own estimation error, so a precisely
  measured beta keeps its value while a noisy one is pulled toward the crowd.

Why shrinkage is not optional here: a maximum-Sharpe optimizer treats its
inputs as truth. Feed it raw betas and it allocates to whichever asset's
estimation error happened to look most attractive. Shrinkage is the cheapest
defence against that, and Vasicek is the version that adapts to how noisy each
estimate actually is.

Everything is computed by array operations on rolling moments rather than by
looping over regressions: 29 assets x 144 rebalances is 4,176 regressions, and
the closed form makes them a handful of vectorised passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_VARIANCE = 1e-12


def rolling_beta(
    returns: pd.DataFrame,
    market: pd.Series,
    window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing OLS beta and its squared standard error, per asset per date.

    The standard error uses the closed form

        se^2(b) = (Var(r_i) - b^2 Var(r_m)) / ((T - 2) Var(r_m))

    which is the residual variance of the regression divided by T-2 and by the
    market variance -- identical to what a per-asset OLS fit would report, at a
    fraction of the cost.

    Returns ``(beta, se_squared)``, both aligned to `returns`. Rows before a
    full window are NaN: `min_periods=window` is deliberate, since a beta
    estimated on a partial window is exactly the silent error this project
    guards against.
    """
    if not returns.index.equals(market.index):
        raise ValueError("returns and market must share an index; align before estimating")

    market = market.astype(float)
    roll = returns.rolling(window=window, min_periods=window)

    cov_im = roll.cov(market)
    var_i = roll.var()
    var_m = market.rolling(window=window, min_periods=window).var()

    safe_var_m = var_m.where(var_m > MIN_VARIANCE)

    beta = cov_im.div(safe_var_m, axis=0)
    residual_var = var_i.sub(beta.pow(2).mul(safe_var_m, axis=0))
    residual_var = residual_var.clip(lower=0.0)
    se_squared = residual_var.div(safe_var_m * (window - 2), axis=0)

    return beta, se_squared


def market_variance(market: pd.Series, window: int = 252) -> pd.Series:
    """Trailing market variance -- reused by the single-index covariance model."""
    return market.rolling(window=window, min_periods=window).var()


def residual_variance(
    returns: pd.DataFrame,
    market: pd.Series,
    beta: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Idiosyncratic variance implied by the single-index decomposition.

    Var(r_i) = b^2 Var(r_m) + Var(e_i), so the residual is what the market
    factor does not explain. Phase 3's factor covariance needs this on its
    diagonal.
    """
    var_i = returns.rolling(window=window, min_periods=window).var()
    var_m = market.rolling(window=window, min_periods=window).var()
    implied = beta.pow(2).mul(var_m, axis=0)
    return var_i.sub(implied).clip(lower=0.0)
def blume_adjust(beta: pd.DataFrame, weight: float = 2.0 / 3.0) -> pd.DataFrame:
    """Blume (1971): b_adj = (2/3) b + 1/3.

    A fixed rule -- every asset shrinks by the same amount toward 1.0 whether
    its beta was measured on clean data or noise. Included as the benchmark
    that Vasicek has to beat.
    """
    return beta * weight + (1.0 - weight)


def vasicek_shrink(
    beta: pd.DataFrame,
    se_squared: pd.DataFrame,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Empirical-Bayes shrinkage toward the contemporaneous cross-sectional mean.

        w_i = s2_cross / (s2_cross + se^2(b_i))
        b_shrunk = w_i * b_i + (1 - w_i) * mean(b)

    The prior is estimated from the cross-section at each date, not from a
    constant: in a calm month betas cluster tightly, s2_cross is small, and
    shrinkage is aggressive; in a dispersed month the data is allowed to speak.

    `mask` restricts the cross-section to investable assets. Including an
    ineligible asset would contaminate the prior with a beta estimated on a
    partial window.
    """
    eligible = beta.notna() & se_squared.notna()
    if mask is not None:
        eligible &= mask.reindex(index=beta.index, columns=beta.columns).fillna(False)

    valid_beta = beta.where(eligible)

    cross_mean = valid_beta.mean(axis=1)
    cross_var = valid_beta.var(axis=1, ddof=1)

    denominator = cross_var.to_numpy()[:, None] + se_squared.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(denominator > 0, cross_var.to_numpy()[:, None] / denominator, np.nan)

    weight_frame = pd.DataFrame(weights, index=beta.index, columns=beta.columns)
    shrunk = weight_frame * valid_beta + (1.0 - weight_frame) * cross_mean.to_numpy()[:, None]
    return shrunk.where(eligible)


def apply_shrinkage(
    beta: pd.DataFrame,
    se_squared: pd.DataFrame,
    kind: str = "none",
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dispatch on the configured shrinkage estimator."""
    if kind == "none":
        return beta
    if kind == "blume":
        return blume_adjust(beta)
    if kind == "vasicek":
        return vasicek_shrink(beta, se_squared, mask=mask)
    raise ValueError(f"unknown beta_shrinkage {kind!r}; expected none, blume or vasicek")


def beta_diagnostics(beta: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional summary per date -- the sanity check for the market proxy.

    SPEC S5: the cross-sectional mean of beta should sit in [0.90, 1.10] and
    the cross-sectional standard deviation in [0.20, 0.45]. A mean far from 1
    means the market series is wrong or misaligned; a standard deviation below
    0.05 means the market was regressed on itself.
    """
    valid = beta if mask is None else beta.where(mask.reindex_like(beta).fillna(False))
    return pd.DataFrame(
        {
            "n": valid.notna().sum(axis=1),
            "mean": valid.mean(axis=1),
            "std": valid.std(axis=1, ddof=1),
            "min": valid.min(axis=1),
            "max": valid.max(axis=1),
        }
    )


def estimate_betas(cfg, returns=None, market=None):
    """Convenience wrapper: load, estimate, shrink, and mask in one call.

    Returns ``(beta, se_squared, mask)`` where `beta` already reflects the
    configured shrinkage and is NaN wherever an asset is not investable.
    """
    from fbpo.data import investable_mask, load_market, load_returns

    if returns is None:
        returns = load_returns(cfg)
    if market is None:
        market = load_market(cfg)["mkt"]

    window = cfg.estimation.window
    beta, se_squared = rolling_beta(returns, market, window)
    mask = investable_mask(returns, cfg.estimation.min_obs)
    shrunk = apply_shrinkage(beta, se_squared, cfg.estimation.beta_shrinkage, mask=mask)
    return shrunk.where(mask), se_squared.where(mask), mask

"""Phase 2 gate: rolling betas and shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpo.estimation import (
    apply_shrinkage,
    beta_diagnostics,
    blume_adjust,
    rolling_beta,
    vasicek_shrink,
)


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.Series]:
    """Known betas: construct returns from a market with beta 0.5, 1.0, 1.5."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", periods=600, freq="B")
    market = pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index, name="mkt")
    noise = rng.normal(0.0, 0.005, (len(index), 3))
    returns = pd.DataFrame(
        {
            "LOW": 0.5 * market.to_numpy() + noise[:, 0],
            "MID": 1.0 * market.to_numpy() + noise[:, 1],
            "HIGH": 1.5 * market.to_numpy() + noise[:, 2],
        },
        index=index,
    )
    return returns, market


def test_recovers_known_betas(synthetic) -> None:
    """The estimator must return the coefficients it was built from."""
    returns, market = synthetic
    beta, _ = rolling_beta(returns, market, window=252)
    final = beta.dropna().iloc[-1]
    assert final["LOW"] == pytest.approx(0.5, abs=0.05)
    assert final["MID"] == pytest.approx(1.0, abs=0.05)
    assert final["HIGH"] == pytest.approx(1.5, abs=0.05)


def test_matches_ols_on_the_same_window(synthetic) -> None:
    """Closed form must agree with an explicit least-squares fit."""
    returns, market = synthetic
    beta, _ = rolling_beta(returns, market, window=252)
    window_end = beta.dropna().index[-1]
    sample = returns.loc[:window_end].iloc[-252:]
    mkt = market.loc[sample.index]
    design = np.column_stack([np.ones(len(mkt)), mkt.to_numpy()])
    for column in returns.columns:
        ols = np.linalg.lstsq(design, sample[column].to_numpy(), rcond=None)[0][1]
        assert beta.loc[window_end, column] == pytest.approx(ols, rel=1e-9)


def test_partial_window_is_nan(synthetic) -> None:
    """A beta on a partial window is the silent error this project guards against."""
    returns, market = synthetic
    beta, _ = rolling_beta(returns, market, window=252)
    assert beta.iloc[:251].isna().all().all()
    assert beta.iloc[251].notna().all()


def test_standard_errors_are_non_negative(synthetic) -> None:
    returns, market = synthetic
    _, se_squared = rolling_beta(returns, market, window=252)
    assert (se_squared.dropna() >= 0).all().all()


def test_blume_is_exact() -> None:
    beta = pd.DataFrame({"A": [0.0, 1.0, 2.0]})
    adjusted = blume_adjust(beta)
    assert adjusted["A"].tolist() == pytest.approx([1 / 3, 1.0, 5 / 3])


def test_vasicek_shrinks_toward_the_cross_sectional_mean(synthetic) -> None:
    """Every shrunk beta lies between its raw value and the cross-sectional mean."""
    returns, market = synthetic
    beta, se_squared = rolling_beta(returns, market, window=252)
    shrunk = vasicek_shrink(beta, se_squared)

    row = beta.dropna().index[-1]
    mean = beta.loc[row].mean()
    for column in beta.columns:
        raw, adj = beta.loc[row, column], shrunk.loc[row, column]
        assert min(raw, mean) - 1e-12 <= adj <= max(raw, mean) + 1e-12


def test_vasicek_reduces_dispersion(synthetic) -> None:
    """Shrinkage must compress the cross-section, never expand it."""
    returns, market = synthetic
    beta, se_squared = rolling_beta(returns, market, window=252)
    shrunk = vasicek_shrink(beta, se_squared)
    raw_std = beta.std(axis=1, ddof=1).dropna()
    shrunk_std = shrunk.std(axis=1, ddof=1).dropna()
    assert (shrunk_std <= raw_std + 1e-12).all()


def test_unknown_shrinkage_is_rejected(synthetic) -> None:
    returns, market = synthetic
    beta, se_squared = rolling_beta(returns, market, window=252)
    with pytest.raises(ValueError, match="unknown beta_shrinkage"):
        apply_shrinkage(beta, se_squared, "bayes-stein-ish")


@pytest.mark.network
def test_cross_sectional_beta_is_plausible(cfg, returns_daily, market_daily) -> None:
    """SPEC S5. Mean near 1 confirms the market proxy; std confirms it is not degenerate.

    The equal-weighted mean of 29 mega-caps against the CRSP value-weighted
    total market sits slightly below 1 by construction, since the index
    includes higher-beta small caps that the Dow does not.
    """
    beta, _ = rolling_beta(returns_daily, market_daily["mkt"], cfg.estimation.window)
    stats = beta_diagnostics(beta).dropna()
    assert 0.85 <= stats["mean"].mean() <= 1.10
    assert 0.20 <= stats["std"].mean() <= 0.45
    assert beta.min().min() > -0.5
    assert beta.max().max() < 3.0

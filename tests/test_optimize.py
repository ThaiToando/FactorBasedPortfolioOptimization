"""Phase 3 gate: convex portfolio construction."""

from __future__ import annotations

import numpy as np
import pytest

from fbpo.optimize import (
    OptimizationError,
    equal_weight,
    gmvp,
    herfindahl,
    inverse_volatility,
    max_sharpe,
    mdrp,
    optimize_weights,
    portfolio_variance,
    robust_max_sharpe,
)


@pytest.fixture(scope="module")
def problem() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(300, 6))
    covariance = np.cov(sample.T)
    mu = np.array([0.02, 0.05, 0.03, 0.08, 0.01, 0.06])
    return mu, covariance


def _sharpe(w, mu, covariance) -> float:
    return float((mu @ w) / np.sqrt(w @ covariance @ w))


def test_max_sharpe_beats_random_long_only_portfolios(problem) -> None:
    """The Schaible transform must find the global optimum, not a local one."""
    mu, covariance = problem
    w = max_sharpe(mu, covariance)
    rng = np.random.default_rng(11)
    best_random = max(
        _sharpe(v, mu, covariance) for v in rng.dirichlet(np.ones(len(mu)), 200_000)
    )
    assert _sharpe(w, mu, covariance) >= best_random - 1e-9


def test_weights_are_a_portfolio(problem) -> None:
    mu, covariance = problem
    for w in (max_sharpe(mu, covariance), gmvp(covariance), mdrp(covariance)):
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= -1e-9).all()


def test_weight_cap_is_respected(problem) -> None:
    """The cap is linear in the transformed space, so it must bind exactly."""
    mu, covariance = problem
    w = max_sharpe(mu, covariance, weight_cap=0.30)
    assert w.max() <= 0.30 + 1e-6


def test_gmvp_has_the_lowest_variance(problem) -> None:
    """Definitional: no long-only portfolio may have lower variance."""
    _, covariance = problem
    g = gmvp(covariance)
    rng = np.random.default_rng(3)
    for candidate in rng.dirichlet(np.ones(covariance.shape[0]), 20_000):
        assert portfolio_variance(g, covariance) <= portfolio_variance(candidate, covariance) + 1e-12


def test_max_sharpe_is_infeasible_without_positive_excess(problem) -> None:
    _, covariance = problem
    with pytest.raises(OptimizationError, match="positive expected excess"):
        max_sharpe(np.full(6, -0.01), covariance)


def test_fallback_to_gmvp_is_reported(problem) -> None:
    """The backtester counts fallbacks; a silent substitution would hide a regime."""
    _, covariance = problem
    w, used = optimize_weights("max_sharpe", covariance, mu=np.full(6, -0.01))
    assert used == "gmvp_fallback"
    assert np.allclose(w, gmvp(covariance), atol=1e-6)


def test_robust_at_zero_kappa_equals_max_sharpe(problem) -> None:
    mu, covariance = problem
    assert np.allclose(robust_max_sharpe(mu, covariance, 0.0), max_sharpe(mu, covariance))


def test_robust_sharpe_is_at_most_the_nominal_optimum(problem) -> None:
    """Plain max-Sharpe is the global optimum under the nominal mu, so any
    robust portfolio must score at or below it on that same objective. That
    gap is the price paid for protection against mu being wrong."""
    mu, covariance = problem
    plain = max_sharpe(mu, covariance)
    robust = robust_max_sharpe(mu, covariance, kappa=0.5, n_obs=252)
    assert _sharpe(robust, mu, covariance) <= _sharpe(plain, mu, covariance) + 1e-9


def test_robust_differs_from_plain_once_kappa_bites(problem) -> None:
    """kappa=0 is exactly max-Sharpe; a positive kappa must move the weights."""
    mu, covariance = problem
    plain = max_sharpe(mu, covariance)
    robust = robust_max_sharpe(mu, covariance, kappa=1.0, n_obs=252)
    assert not np.allclose(plain, robust, atol=1e-6)


def test_robust_is_infeasible_when_the_uncertainty_set_swallows_mu(problem) -> None:
    """Beyond a threshold no portfolio has a positive worst-case premium.
    Failing with a diagnostic beats returning something arbitrary."""
    mu, covariance = problem
    with pytest.raises(OptimizationError, match="kappa"):
        robust_max_sharpe(mu, covariance, kappa=50.0, n_obs=252)


def test_mdrp_maximises_the_diversification_ratio(problem) -> None:
    _, covariance = problem
    vol = np.sqrt(np.diag(covariance))

    def ratio(w):
        return float((vol @ w) / np.sqrt(w @ covariance @ w))

    w = mdrp(covariance)
    assert ratio(w) >= ratio(equal_weight(len(vol))) - 1e-9


def test_herfindahl_of_equal_weight(problem) -> None:
    """SPEC S5 anchor: HHI equal to 1/N is how a constant-mu bug announces itself."""
    assert herfindahl(equal_weight(29)) == pytest.approx(1 / 29)


def test_inverse_volatility_ignores_correlations(problem) -> None:
    _, covariance = problem
    w = inverse_volatility(covariance)
    vol = np.sqrt(np.diag(covariance))
    assert np.allclose(w, (1 / vol) / (1 / vol).sum())

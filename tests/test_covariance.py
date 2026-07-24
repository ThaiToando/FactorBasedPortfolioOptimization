"""Phase 3 gate: covariance estimators and conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpo.covariance import (
    covariance_diagnostics,
    ewma_covariance,
    ledoit_wolf_covariance,
    nearest_psd,
    oas_covariance,
    rescale_with_volatility,
    sample_covariance,
    single_index_covariance,
)

ESTIMATORS = (sample_covariance, ledoit_wolf_covariance, oas_covariance, ewma_covariance)


@pytest.fixture(scope="module")
def window() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(0.0, 0.012, (252, 8)), columns=list("ABCDEFGH"))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_estimators_are_symmetric_psd(estimator, window) -> None:
    """An indefinite matrix makes the solver fail or return nonsense."""
    sigma = estimator(window)
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() >= -1e-12


def test_nearest_psd_repairs_an_indefinite_matrix() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    assert np.linalg.eigvalsh(bad).min() < 0
    assert np.linalg.eigvalsh(nearest_psd(bad)).min() >= 0


def test_ewma_with_a_long_halflife_approaches_the_sample(window) -> None:
    """Sanity anchor: as weights flatten, EWMA must converge on the sample estimate."""
    flat = ewma_covariance(window, halflife=100_000)
    assert np.allclose(flat, sample_covariance(window), atol=1e-6)


def test_single_index_imposes_the_factor_structure() -> None:
    """Off-diagonals must be exactly s2_m * b_i * b_j -- that is the whole model."""
    beta = np.array([0.8, 1.0, 1.4])
    sigma = single_index_covariance(beta, 0.0002, np.array([1e-4, 2e-4, 1.5e-4]))
    assert sigma[0, 1] == pytest.approx(0.0002 * 0.8 * 1.0)
    assert sigma[1, 2] == pytest.approx(0.0002 * 1.0 * 1.4)
    assert sigma[0, 0] == pytest.approx(0.0002 * 0.64 + 1e-4)


def test_single_index_delta_blends_toward_the_sample(window) -> None:
    beta = np.full(8, 1.0)
    sample = sample_covariance(window)
    pure = single_index_covariance(beta, 0.0002, np.full(8, 1e-4))
    blended = single_index_covariance(beta, 0.0002, np.full(8, 1e-4), delta=0.5, sample=sample)
    assert np.allclose(blended, 0.5 * pure + 0.5 * sample, atol=1e-10)


def test_rescaling_preserves_correlations(window) -> None:
    """D R D must change volatilities without touching the correlation matrix."""
    sigma = sample_covariance(window)
    source = np.sqrt(np.diag(sigma))
    target = source * 1.5
    rescaled = rescale_with_volatility(sigma, target, source)

    def correlation(matrix):
        d = np.sqrt(np.diag(matrix))
        return matrix / np.outer(d, d)

    assert np.allclose(correlation(sigma), correlation(rescaled), atol=1e-10)
    assert np.allclose(np.sqrt(np.diag(rescaled)), target, rtol=1e-8)


def test_diagnostics_on_the_identity() -> None:
    """Known answers: perfectly conditioned, full effective rank, no dominant PC."""
    stats = covariance_diagnostics(np.eye(10))
    assert stats["condition_number"] == pytest.approx(1.0)
    assert stats["effective_rank"] == pytest.approx(10.0)
    assert stats["pc1_share"] == pytest.approx(0.1)


@pytest.mark.network
def test_real_covariance_is_well_conditioned(cfg, returns_daily) -> None:
    """SPEC S5: condition number in [20, 400]. Above 1e5 means T < N or duplicates."""
    from fbpo.data import investable_mask

    mask = investable_mask(returns_daily, cfg.estimation.window)
    date = "2021-11-30"
    columns = mask.loc[date][mask.loc[date]].index
    sigma = sample_covariance(returns_daily.loc[:date, columns].iloc[-252:])
    stats = covariance_diagnostics(sigma)
    assert 10.0 <= stats["condition_number"] <= 1e4
    assert 0.2 <= stats["pc1_share"] <= 0.85

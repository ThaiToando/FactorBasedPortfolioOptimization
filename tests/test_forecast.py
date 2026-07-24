"""Phase 4 gate: the forecast engine and the epsilon-tube guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbpo.config import Config
from fbpo.forecast import (
    MIN_FORECAST_STD,
    align_design,
    assert_forecast_is_not_flat,
    build_svr_pipeline,
    campbell_thompson,
    combination_forecast,
    monthly_market_excess,
    out_of_sample_r2,
    prevailing_mean_forecast,
    shrink_to_prevailing_mean,
)


@pytest.fixture(scope="module")
def toy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(5)
    index = pd.date_range("2000-01-31", periods=180, freq="ME")
    signals = pd.DataFrame(rng.normal(size=(180, 4)), index=index, columns=list("abcd"))
    target = pd.Series(rng.normal(0.006, 0.04, 180), index=index, name="mkt_excess")
    return signals, target


def test_monthly_compounding_is_not_summation() -> None:
    """Summing daily returns misstates the month; the difference compounds."""
    index = pd.date_range("2020-01-01", periods=21, freq="B")
    market = pd.DataFrame({"mkt": np.full(21, 0.01), "rf": np.zeros(21)}, index=index)
    monthly = monthly_market_excess(market)
    assert monthly.iloc[0] == pytest.approx(1.01**21 - 1)
    assert monthly.iloc[0] != pytest.approx(0.21)


def test_design_matrix_predicts_the_following_month(toy) -> None:
    """The look-ahead guard: X at month t must pair with y realised at t+1."""
    signals, target = toy
    X, y = align_design(signals, target)
    for date in X.index[:20]:
        following = target.index[target.index.get_loc(date) + 1]
        assert y.loc[date] == pytest.approx(target.loc[following])


def test_prevailing_mean_uses_only_the_past(toy) -> None:
    _, target = toy
    forecasts = prevailing_mean_forecast(target, min_train=60)
    for date in forecasts.index[:10]:
        position = target.index.get_loc(date)
        assert forecasts.loc[date] == pytest.approx(target.iloc[:position].mean())


def test_first_forecast_respects_min_train(toy) -> None:
    _, target = toy
    assert len(prevailing_mean_forecast(target, min_train=60)) == len(target) - 60


def test_combination_averages_univariate_forecasts(toy) -> None:
    """RSZ is the mean of K one-variable regressions, not one K-variable fit."""
    signals, target = toy
    X, y = align_design(signals, target)
    combination = combination_forecast(X, y, min_train=60)

    date = combination.index[0]
    position = y.index.get_loc(date)
    manual = []
    for column in X.columns:
        slope, intercept = np.polyfit(X[column].iloc[:position], y.iloc[:position], 1)
        manual.append(intercept + slope * X[column].iloc[position])
    assert combination.loc[date] == pytest.approx(np.mean(manual))


def test_campbell_thompson_clips_only_negatives() -> None:
    series = pd.Series([-0.05, 0.0, 0.03])
    assert campbell_thompson(series).tolist() == [0.0, 0.0, 0.03]


def test_oos_r2_is_zero_against_itself(toy) -> None:
    _, target = toy
    benchmark = prevailing_mean_forecast(target, 60)
    assert out_of_sample_r2(target, benchmark, benchmark) == pytest.approx(0.0)


def test_oos_r2_is_positive_for_a_better_forecast(toy) -> None:
    _, target = toy
    benchmark = prevailing_mean_forecast(target, 60)
    index = benchmark.index
    cheating = target.loc[index] * 0.5 + benchmark * 0.5
    assert out_of_sample_r2(target.loc[index], cheating, benchmark) > 0


def test_flat_forecast_assertion_fires() -> None:
    constant = pd.Series(np.full(144, 0.006))
    with pytest.raises(ValueError, match="standard deviation"):
        assert_forecast_is_not_flat(constant)


def test_flat_forecast_assertion_passes_a_live_forecast() -> None:
    rng = np.random.default_rng(1)
    live = pd.Series(rng.normal(0.006, 0.01, 144))
    assert assert_forecast_is_not_flat(live) > MIN_FORECAST_STD


def test_unscaled_target_collapses_to_a_constant(toy) -> None:
    """The epsilon-tube trap, demonstrated rather than asserted.

    With standardize_y off, epsilon is measured in raw return units against a
    target whose own standard deviation is ~0.04. At the top of the configured
    grid (epsilon=0.2) every training point falls inside the tube, all dual
    coefficients go to zero, and the model returns its intercept -- a constant
    by construction, not by degree.

    Scaling y makes epsilon mean "a fraction of a standard deviation", under
    which the same grid is entirely sane. This test exists so the failure mode
    stays visible in the repository: it is the reason standardize_y is not a
    tunable preference.
    """
    signals, target = toy
    X, y = align_design(signals, target)
    train_X, train_y = X.to_numpy()[:120], y.to_numpy()[:120]
    test_X = X.to_numpy()[120:]

    def dispersion(cfg, epsilon: float, prefix: str) -> float:
        model = build_svr_pipeline(cfg)
        model.set_params(**{f"{prefix}epsilon": epsilon})
        model.fit(train_X, train_y)
        return float(np.std(model.predict(test_X)))

    unscaled_cfg = Config(svr={"standardize_y": False})

    assert dispersion(unscaled_cfg, 0.2, "svr__") < MIN_FORECAST_STD
    assert dispersion(Config(), 0.2, "regressor__svr__") > MIN_FORECAST_STD


def test_shrinkage_blends_toward_the_benchmark(toy) -> None:
    _, target = toy
    benchmark = prevailing_mean_forecast(target, 60)
    model = benchmark + 0.02
    blended = shrink_to_prevailing_mean(model, benchmark, omega=0.5)
    assert np.allclose(blended.to_numpy(), benchmark.to_numpy() + 0.01)

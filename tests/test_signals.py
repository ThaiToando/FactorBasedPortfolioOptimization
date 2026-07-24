"""Phase 1 gate: the signal panel and its causality contract."""

from __future__ import annotations

import numpy as np
import pytest

from fbpo.signals import (
    AUTO_SIGNALS,
    PUBLICATION_LAG_MONTHS,
    apply_publication_lags,
    ffd_weights,
    frac_diff,
)

pytestmark = pytest.mark.network


def test_panel_shape(signals_monthly) -> None:
    """228 months, 2003-01 through 2021-12, 13 auto signals."""
    assert signals_monthly.shape == (228, 13)
    assert str(signals_monthly.index.min().date()) == "2003-01-31"
    assert str(signals_monthly.index.max().date()) == "2021-12-31"


def test_every_signal_is_populated(signals_monthly) -> None:
    """A column that is entirely NaN means a transform failed silently."""
    counts = signals_monthly.notna().sum()
    assert counts.min() > 0, f"empty signals: {counts[counts == 0].index.tolist()}"


def test_columns_are_exactly_the_auto_set(signals_monthly) -> None:
    assert tuple(signals_monthly.columns) == AUTO_SIGNALS


@pytest.mark.parametrize("signal", AUTO_SIGNALS)
def test_no_signal_uses_unpublished_data(signal, signals_unlagged) -> None:
    """SPEC S2.4. The value stamped at month M equals the raw value at M - lag.

    Catches a missing lag and a doubled lag with the same assertion.
    """
    lag = PUBLICATION_LAG_MONTHS[signal]
    lagged = apply_publication_lags(signals_unlagged)[signal].dropna()
    expected = signals_unlagged[signal].shift(lag).reindex(lagged.index)
    assert lagged.equals(expected)


def test_ffd_weights_start_at_one_and_decay() -> None:
    """Weights are returned oldest-first, so the last entry is the current obs."""
    w = ffd_weights(0.5)
    assert w[-1] == 1.0
    assert abs(w[0]) < abs(w[-1])


def test_frac_diff_at_d_one_is_a_first_difference() -> None:
    """d=1 must reproduce .diff() exactly -- the boundary case of the transform."""
    import pandas as pd

    s = pd.Series(np.arange(50.0), index=pd.date_range("2020-01-01", periods=50, freq="D"))
    assert np.allclose(frac_diff(s, 1.0).values, s.diff().dropna().values)


def test_frac_diff_window_is_capped(signals_monthly) -> None:
    """FFD is fixed-width: an uncapped window silently eats the sample."""
    for name in ("DGS10", "DTB3"):
        assert signals_monthly[name].notna().sum() >= 200


def test_market_momentum_skips_the_recent_month(signals_monthly) -> None:
    """12m-1m momentum must be plausible in magnitude, not a cumulative index."""
    mom = signals_monthly["mkt_mom"].dropna()
    assert mom.min() > -0.7
    assert mom.max() < 1.5


def test_realised_volatility_is_annualised(signals_monthly) -> None:
    """SPEC S5: annualised market vol sits well inside [0.03, 1.2]."""
    rvol = signals_monthly["mkt_rvol"].dropna()
    assert rvol.min() > 0.03, "too small -- forgot the sqrt(252)"
    assert rvol.max() < 1.2, "too large -- multiplied by 252 instead"

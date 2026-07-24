"""Phase 1 gate: the data layer.

Written before src/fbpo/data.py exists. Every test here is expected to fail on
first run; they are the executable definition of what the loader must satisfy.

The universe-size test is the important one. DOW first traded 2019-03-20, so
with a 252-day window it is not eligible until ~2020-03-20 and the first
rebalance that may include it is 2020-03-31. A loader that returns 30 assets
throughout does not crash -- it silently estimates DOW's beta on 40
observations and lets the optimizer allocate to it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fbpo.data", reason="src/fbpo/data.py arrives in Phase 1 step 2")

from fbpo.data import load_market, load_prices, load_returns, investable_mask  # noqa: E402


@pytest.mark.network
def test_returns_frame_shape(returns_daily) -> None:
    rows, cols = returns_daily.shape
    assert 3414 <= rows <= 3424, f"expected ~3,419 daily rows, got {rows}"
    assert cols == 29


@pytest.mark.network
def test_every_column_is_complete_except_dow(returns_daily) -> None:
    counts = returns_daily.notna().sum()
    others = counts.drop("DOW")
    assert others.nunique() == 1, f"ragged history outside DOW: {counts.to_dict()}"
    assert counts["DOW"] < others.iloc[0]


@pytest.mark.network
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2010-01-29", 28),
        ("2020-02-28", 28),
        ("2020-03-31", 29),
        ("2021-11-30", 29),
    ],
)
def test_universe_size_by_date(investable, date: str, expected: int) -> None:
    """SPEC S1.4. The 28 -> 29 transition at 2020-03-31 is the DOW eligibility rule."""
    assert int(investable.loc[date].sum()) == expected


@pytest.mark.network
def test_risk_free_is_a_daily_decimal_rate(market_daily) -> None:
    """SPEC S5: rf in [0, 0.0001]. A value near 0.02 means /100 applied twice or never."""
    rf = market_daily["rf"]
    assert rf.min() >= 0.0
    assert rf.max() <= 0.0001, f"rf max {rf.max():.6f} -- check the /100 conversion"


@pytest.mark.network
def test_daily_returns_are_plausible(returns_daily) -> None:
    """|r| > 0.6 means an unadjusted split slipped through auto_adjust."""
    stacked = returns_daily.stack()
    assert stacked.min() >= -0.6
    assert stacked.max() <= 0.6


@pytest.mark.network
def test_no_adj_close_column_survives(raw_prices) -> None:
    """auto_adjust=True must fold adjustment into OHLC; an 'Adj Close' means it did not."""
    assert "Adj Close" not in raw_prices.columns.get_level_values(0)


@pytest.mark.network
def test_calendar_is_the_intersection_of_prices_and_factors(returns_daily, market_daily) -> None:
    assert returns_daily.index.equals(market_daily.index)
    assert returns_daily.index.is_monotonic_increasing
    assert not returns_daily.index.has_duplicates

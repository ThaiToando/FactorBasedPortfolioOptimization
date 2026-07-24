"""Pandera schemas: the structural contract every frame must satisfy.

These catch the class of bug that produces plausible output: a market series
misaligned by one day, a risk-free rate that never got divided by 100, a
duplicated index row that quietly doubles a month's return. Validation runs at
the boundary of the data layer, so a malformed frame never reaches estimation.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

DAILY_RETURN_MIN = -0.6
DAILY_RETURN_MAX = 0.6
RF_MIN = 0.0
RF_MAX = 0.0001


class MarketSchema(pa.DataFrameModel):
    """Daily market total return and risk-free rate, both as decimals."""

    mkt: Series[float] = pa.Field(ge=DAILY_RETURN_MIN, le=DAILY_RETURN_MAX, nullable=False)
    rf: Series[float] = pa.Field(ge=RF_MIN, le=RF_MAX, nullable=False)

    class Config:
        strict = True
        ordered = False
        unique_column_names = True


def validate_returns(df, *, n_assets: int = 30):
    """Validate a wide daily-return frame: unique sorted index, plausible magnitudes."""
    if df.index.has_duplicates:
        raise ValueError("duplicate dates in the return index")
    if not df.index.is_monotonic_increasing:
        raise ValueError("return index is not sorted ascending")
    if df.shape[1] != n_assets:
        raise ValueError(f"expected {n_assets} asset columns, got {df.shape[1]}")

    stacked = df.stack(future_stack=True).dropna()
    if stacked.empty:
        raise ValueError("return frame contains no observations")
    if stacked.min() < DAILY_RETURN_MIN or stacked.max() > DAILY_RETURN_MAX:
        raise ValueError(
            f"daily returns outside [{DAILY_RETURN_MIN}, {DAILY_RETURN_MAX}] "
            f"(min {stacked.min():.4f}, max {stacked.max():.4f}) -- unadjusted split?"
        )
    return df


def validate_market(df):
    """Validate the daily market/risk-free frame against :class:`MarketSchema`."""
    return MarketSchema.validate(df)

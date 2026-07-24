"""Monthly signal panel: macro and market predictors of the equity premium.

Causality contract (SPEC S2.4). The signal vector stamped at month M is the
information a forecaster could actually have used at the close of month M:

1. Each series is resampled to month-end, last observation.
2. Each is then shifted forward by its publication lag, so a series released
   mid-February cannot inform a forecast made at the end of January.

The lag dictionary below is that contract. A parametrised test asserts, for
every signal, that ``panel[sig] == unlagged[sig].shift(lag)`` -- which catches
a missing lag and a doubled one with the same assertion.

This module builds ``signals_auto``: the 13 signals obtainable with zero manual
downloads (11 FRED series plus 2 derived from the market return). The nine
workbook signals of ``signals_full`` are purely additive; nothing downstream
changes shape when they are added.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fbpo.config import Config, resolve_date

PROCESSED_DIR = Path("data/processed")
SIGNALS_PARQUET = PROCESSED_DIR / "signals_monthly.parquet"

# FRED series -> transform. Levels that are non-stationary get fractionally
# differenced rather than first-differenced, which keeps the memory that made
# them predictive (Addendum II, III.5.3).
FRED_TRANSFORMS: dict[str, str] = {
    "T10Y3M": "level",  # term spread, already a spread in %
    "BAA10Y": "level",  # default spread
    "DGS10": "ffd",  # 10Y Treasury yield, non-stationary in level
    "DTB3": "ffd",  # 3M T-bill
    "VIXCLS": "log",  # implied volatility
    "NFCI": "level",  # Chicago Fed financial conditions
    "UMCSENT": "pct_change_12",  # consumer sentiment
    "INDPRO": "pct_change_12",  # industrial production
    "UNRATE": "diff_12",  # unemployment rate
    "CPIAUCSL": "pct_change_12",  # inflation
    "M2REAL": "pct_change_12",  # real money growth
}

# Months of publication delay beyond month-end alignment. Zero means the value
# is observable at the month-end it is stamped on (daily market data); one
# means it is released during the following month.
PUBLICATION_LAG_MONTHS: dict[str, int] = {
    "T10Y3M": 0,
    "BAA10Y": 0,
    "DGS10": 0,
    "DTB3": 0,
    "VIXCLS": 0,
    "NFCI": 0,
    "UMCSENT": 1,
    "INDPRO": 1,
    "UNRATE": 1,
    "CPIAUCSL": 1,
    "M2REAL": 2,
    "mkt_mom": 0,
    "mkt_rvol": 0,
}

AUTO_SIGNALS: tuple[str, ...] = (*FRED_TRANSFORMS, "mkt_mom", "mkt_rvol")


def ffd_weights(d: float, thresh: float = 1e-5) -> np.ndarray:
    """Fixed-width fractional-difference weights, truncated at `thresh`."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-5, max_width: int = 252) -> pd.Series:
    """Fractionally difference a series by order `d`, fixed-width window."""
    if d == 0:
        return series.copy()
    w = ffd_weights(d, thresh)
    if len(w) - 1 > max_width:
        w = w[-(max_width + 1) :]
    width = len(w) - 1
    out: dict[pd.Timestamp, float] = {}
    for i in range(width, len(series)):
        window = series.iloc[i - width : i + 1]
        if window.notna().all():
            out[series.index[i]] = float(np.dot(w, window))
    return pd.Series(out, name=series.name)


def min_stationary_d(
    series: pd.Series, grid: np.ndarray | None = None, pval: float = 0.05
) -> float:
    """Smallest d on the grid whose fractional difference passes an ADF test.

    Memory and stationarity trade off: d=0 keeps all memory but fails ADF,
    d=1 is a plain first difference and throws the memory away.
    """
    from statsmodels.tsa.stattools import adfuller

    if grid is None:
        grid = np.arange(0.0, 1.01, 0.05)
    clean = series.dropna()
    for d in grid:
        differenced = frac_diff(clean, float(d)).dropna()
        if len(differenced) < 30:
            continue
        if adfuller(differenced, maxlag=1, regression="c")[1] < pval:
            return float(d)
    return 1.0


def fetch_fred(start: str, end: str) -> pd.DataFrame:
    """Download the 11 FRED series at their native frequency."""
    from pandas_datareader import data as pdr

    frames: dict[str, pd.Series] = {}
    for series_id in FRED_TRANSFORMS:
        try:
            table = pdr.DataReader(series_id, "fred", start, end)
        except Exception as exc:  # noqa: BLE001 - name the series that failed
            raise RuntimeError(f"FRED download failed for {series_id!r}: {exc}") from exc
        frames[series_id] = table[series_id]

    raw = pd.DataFrame(frames)
    raw.index = pd.DatetimeIndex(raw.index).normalize()
    return raw.sort_index()


def to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample to month-end, last observation. Monthly series are unaffected."""
    return frame.resample("ME").last()


def apply_transform(series: pd.Series, kind: str) -> pd.Series:
    """Apply the configured stationarity transform to one series."""
    if kind == "level":
        return series
    if kind == "log":
        positive = series.where(series > 0)
        return np.log(positive)
    if kind == "pct_change_12":
        return series.pct_change(12)
    if kind == "diff_12":
        return series.diff(12)
    if kind == "ffd":
        d = min_stationary_d(series)
        return frac_diff(series, d).reindex(series.index)
    raise ValueError(f"unknown transform {kind!r} for {series.name!r}")


def build_price_signals(market: pd.DataFrame, trading_days_per_year: int = 252) -> pd.DataFrame:
    """The two signals derived from the market return itself.

    ``mkt_mom`` is cumulative market return from t-12m to t-1m, skipping the
    most recent month. The skip is not decoration: the one-month reversal
    effect is well documented, and including month t contaminates a momentum
    signal with it.

    ``mkt_rvol`` is the annualised standard deviation of daily market returns
    over the trailing 21 trading days, sampled at month-end.
    """
    mkt = market["mkt"]

    monthly = (1.0 + mkt).resample("ME").prod() - 1.0
    cumulative = np.log1p(monthly).rolling(12).sum().shift(1)
    mkt_mom = np.expm1(cumulative)

    daily_vol = mkt.rolling(21).std() * np.sqrt(trading_days_per_year)
    mkt_rvol = daily_vol.resample("ME").last()

    return pd.DataFrame({"mkt_mom": mkt_mom, "mkt_rvol": mkt_rvol})
def _fetch_start(cfg: Config) -> str:
    """Two years before the signal start, so 12-month transforms and the
    fractional-difference window have data to consume without eating into
    the panel itself."""
    return (pd.Timestamp(cfg.data.signal_start) - pd.DateOffset(years=2)).date().isoformat()


def _signal_market(cfg: Config) -> pd.DataFrame:
    """Daily market return covering the full signal window.

    The price-derived signals need market history back to `signal_start`
    (2003), while `price_start` is 2008 because the stock panel only needs
    252 days before the first rebalance. The French factors reach back to
    1926, so they are fetched separately here rather than truncating the
    signal panel to the stock panel's start.
    """
    from pandas_datareader import data as pdr

    from fbpo.data import FF_DAILY_DATASET, build_market

    end = (resolve_date(cfg.data.price_end) - pd.Timedelta(days=1)).isoformat()
    table = pdr.DataReader(FF_DAILY_DATASET, "famafrench", _fetch_start(cfg), end)[0]
    table = table.astype(float) / 100.0
    table.index = pd.DatetimeIndex(table.index.to_timestamp()).normalize()
    return build_market(table.sort_index())


def build_unlagged_panel(cfg: Config) -> pd.DataFrame:
    """Month-end panel with transforms applied but publication lags NOT applied.

    Exists so the causality test has something to compare against: the lagged
    panel must equal this frame shifted by each signal's lag, exactly.

    Fractional differencing is applied at the series' native daily frequency
    and only then resampled. Applied to monthly data the fixed-width window
    consumes a large fraction of a 228-month panel.
    """
    start = _fetch_start(cfg)
    end = (resolve_date(cfg.data.price_end) - pd.Timedelta(days=1)).isoformat()

    daily = fetch_fred(start, end)
    columns: dict[str, pd.Series] = {}
    for name, kind in FRED_TRANSFORMS.items():
        if kind == "ffd":
            native = daily[name].dropna()
            d = min_stationary_d(native)
            columns[name] = frac_diff(native, d).resample("ME").last()
        else:
            columns[name] = apply_transform(to_month_end(daily[[name]])[name], kind)

    transformed = pd.DataFrame(columns)
    price_signals = build_price_signals(_signal_market(cfg), cfg.data.trading_days_per_year)
    panel = transformed.join(price_signals, how="outer")
    return panel[list(AUTO_SIGNALS)].sort_index()


def apply_publication_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift each column forward by its publication lag. The causality step."""
    return pd.DataFrame(
        {name: panel[name].shift(PUBLICATION_LAG_MONTHS[name]) for name in panel.columns}
    )


def build_signal_panel(cfg: Config, *, refresh: bool = False) -> pd.DataFrame:
    """The causal monthly signal panel, cached to parquet."""
    if SIGNALS_PARQUET.exists() and not refresh:
        return pd.read_parquet(SIGNALS_PARQUET)

    panel = apply_publication_lags(build_unlagged_panel(cfg))

    first = pd.Timestamp(cfg.data.signal_start)
    last = pd.Timestamp(resolve_date(cfg.data.price_end)) - pd.Timedelta(days=1)
    panel = panel.loc[(panel.index >= first) & (panel.index <= last)]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(SIGNALS_PARQUET)
    return panel


def load_signals(cfg: Config | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Public accessor for the signal panel."""
    return build_signal_panel(cfg or Config(), refresh=refresh)

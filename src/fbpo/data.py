"""Data layer: prices, market factors, trading calendar, eligibility, manifest.

Causality contract for this module:

* Prices are adjusted with ``auto_adjust=True`` so that OHLC *and* Close are
  split- and dividend-adjusted. With ``auto_adjust=False`` only ``Adj Close``
  is adjusted, leaving High/Low raw -- which silently corrupts the range-based
  volatility estimators used later. We assert the flag took effect.
* The trading calendar is the *intersection* of price dates and French factor
  dates. Nothing downstream may invent a date.
* An asset is investable at date t only if it has ``window`` consecutive
  non-null daily returns ending at t. This single rule is what produces the
  29 -> 30 universe transition at 2020-03-31 (DOW first traded 2019-03-20).

Returns convention (SPEC S2.1): log returns for estimation, simple returns for
portfolio compounding. Mixing them costs ~2 bps/month at these magnitudes and
is a correctness error, so both are produced explicitly and named.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fbpo.config import Config, resolve_date

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/manifest.json")
UNIVERSE_DIR = Path("configs/universes")

PRICES_PARQUET = RAW_DIR / "prices_ohlcv.parquet"
FACTORS_PARQUET = RAW_DIR / "ff_factors_daily.parquet"
RETURNS_PARQUET = PROCESSED_DIR / "returns_daily.parquet"
MARKET_PARQUET = PROCESSED_DIR / "market_daily.parquet"

FF_DAILY_DATASET = "F-F_Research_Data_Factors_daily"


def load_universe(name: str, root: Path = UNIVERSE_DIR) -> list[str]:
    """Read a committed universe file. Never scraped at runtime."""
    path = root / f"{name}.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    tickers = [str(t).strip().upper() for t in spec["tickers"]]
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{path}: duplicate tickers")
    return sorted(tickers)


def fetch_prices(cfg: Config, *, refresh: bool = False) -> pd.DataFrame:
    """Download daily OHLCV for the configured universe, or read the cache.

    Returns a frame with a two-level column index ``(field, ticker)``.
    """
    if PRICES_PARQUET.exists() and not refresh:
        return pd.read_parquet(PRICES_PARQUET)

    import yfinance as yf

    tickers = load_universe(cfg.data.universe)
    raw = yf.download(
        tickers,
        start=cfg.data.price_start,
        end=_resolve_end(cfg.data.price_end),
        auto_adjust=cfg.data.auto_adjust,
        actions=False,
        group_by="column",
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check connectivity and the ticker list")

    fields = raw.columns.get_level_values(0).unique().tolist()
    if "Adj Close" in fields:
        raise RuntimeError(
            "'Adj Close' present after download: auto_adjust did not take effect, so "
            "High/Low are unadjusted and range-based volatility would be corrupted"
        )
    missing = set(tickers) - set(raw.columns.get_level_values(1))
    if missing:
        raise RuntimeError(f"yfinance returned no columns for: {sorted(missing)}")

    raw = raw.sort_index()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    PRICES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(PRICES_PARQUET)
    return raw


def _resolve_end(value: str) -> str:
    """yfinance `end` is exclusive; the sentinel resolves to tomorrow."""
    if value == "today":
        return (dt.date.today() + dt.timedelta(days=1)).isoformat()
    return value


def fetch_factors(cfg: Config, *, refresh: bool = False) -> pd.DataFrame:
    """Download the daily Fama-French factor table, or read the cache.

    Units: the library serves percent. Everything is divided by 100 exactly once.
    """
    if FACTORS_PARQUET.exists() and not refresh:
        return pd.read_parquet(FACTORS_PARQUET)

    from pandas_datareader import data as pdr

    end = resolve_date(cfg.data.price_end) - dt.timedelta(days=1)
    table = pdr.DataReader(
        FF_DAILY_DATASET,
        "famafrench",
        cfg.data.price_start,
        end.isoformat(),
    )[0]

    table = table.astype(float) / 100.0
    table.index = pd.DatetimeIndex(table.index.to_timestamp()).normalize()
    table = table.sort_index()

    FACTORS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FACTORS_PARQUET)
    return table


def build_market(factors: pd.DataFrame) -> pd.DataFrame:
    """Market *total* return and risk-free rate.

    Beta is estimated against the total market return ``Mkt-RF + RF``, not the
    excess return. One convention, asserted, used everywhere.
    """
    market = pd.DataFrame(
        {
            "mkt": factors["Mkt-RF"] + factors["RF"],
            "rf": factors["RF"],
        }
    )
    return market.dropna()


def trading_calendar(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DatetimeIndex:
    """The intersection of price dates and factor dates defines the calendar."""
    index = prices.index.intersection(market.index)
    return pd.DatetimeIndex(sorted(index))


def build_returns(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex, *, kind: str = "log"
) -> pd.DataFrame:
    """Daily returns from adjusted Close, restricted to the trading calendar."""
    close = prices["Close"].reindex(calendar).sort_index()
    if kind == "log":
        returns = np.log(close).diff()
    elif kind == "simple":
        returns = close.pct_change()
    else:
        raise ValueError(f"unknown return kind {kind!r}; expected 'log' or 'simple'")
    return returns.iloc[1:].sort_index(axis=1)

def investable_mask(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Boolean frame: True where an asset has `window` consecutive non-null returns.

    This is the eligibility rule, stated once. DOW (first trade 2019-03-20)
    enters the universe at 2020-03-31 and not before.
    """
    observed = returns.notna()
    complete = observed.rolling(window=window, min_periods=window).sum() == window
    return complete.fillna(False)


def load_prices(cfg: Config | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Raw OHLCV frame (two-level columns)."""
    return fetch_prices(cfg or Config(), refresh=refresh)


def load_market(cfg: Config | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Daily market total return and risk-free rate, on the trading calendar."""
    cfg = cfg or Config()
    prices = fetch_prices(cfg, refresh=refresh)
    market = build_market(fetch_factors(cfg, refresh=refresh))
    calendar = trading_calendar(prices, market)
    return market.reindex(calendar).iloc[1:]


def load_returns(
    cfg: Config | None = None, *, kind: str = "log", refresh: bool = False
) -> pd.DataFrame:
    """Daily returns for the configured universe, on the trading calendar."""
    cfg = cfg or Config()
    prices = fetch_prices(cfg, refresh=refresh)
    market = build_market(fetch_factors(cfg, refresh=refresh))
    calendar = trading_calendar(prices, market)
    return build_returns(prices, calendar, kind=kind)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    return {
        "sha256": _sha256(path),
        "rows": int(frame.shape[0]),
        "cols": int(frame.shape[1]),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "bytes": path.stat().st_size,
    }


def write_manifest(cfg: Config, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Record a checksum per cached artifact. Committed; the data is not."""
    artifacts = {
        p.name: _describe(p)
        for p in (PRICES_PARQUET, FACTORS_PARQUET, RETURNS_PARQUET, MARKET_PARQUET)
        if p.exists()
    }
    manifest = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config_hash": cfg.hash,
        "universe": cfg.data.universe,
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(path: Path = MANIFEST_PATH) -> list[str]:
    """Re-hash every artifact; return a list of mismatch descriptions (empty = clean)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `fbpo fetch-data` first")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    known = (PRICES_PARQUET, FACTORS_PARQUET, RETURNS_PARQUET, MARKET_PARQUET)
    for name, recorded in manifest["artifacts"].items():
        candidate = next((p for p in known if p.name == name), None)
        if candidate is None or not candidate.exists():
            problems.append(f"{name}: missing")
            continue
        actual = _sha256(candidate)
        if actual != recorded["sha256"]:
            problems.append(f"{name}: sha256 {actual[:12]} != recorded {recorded['sha256'][:12]}")
    return problems


def fetch_all(cfg: Config, *, refresh: bool = False) -> dict[str, Any]:
    """Fetch, derive, cache, and checksum every input the backtest needs."""
    prices = fetch_prices(cfg, refresh=refresh)
    market_full = build_market(fetch_factors(cfg, refresh=refresh))
    calendar = trading_calendar(prices, market_full)

    returns = build_returns(prices, calendar, kind=cfg.backtest.return_type_estimation)
    market = market_full.reindex(calendar).iloc[1:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(RETURNS_PARQUET)
    market.to_parquet(MARKET_PARQUET)

    mask = investable_mask(returns, cfg.estimation.window)
    return {
        "manifest": write_manifest(cfg),
        "trading_days": int(len(calendar)),
        "assets": int(returns.shape[1]),
        "first_full_universe": str(mask.sum(axis=1).idxmax()),
    }

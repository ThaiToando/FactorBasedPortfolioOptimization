"""Shared fixtures. Determinism is pinned at import time, before any test runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from fbpo.determinism import set_deterministic

REPO_ROOT = Path(__file__).resolve().parents[1]

set_deterministic(42)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"

@pytest.fixture(scope="session")
def cfg():
    from fbpo.config import Config

    return Config()


@pytest.fixture(scope="session")
def raw_prices(cfg):
    from fbpo.data import fetch_prices

    return fetch_prices(cfg)


@pytest.fixture(scope="session")
def returns_daily(cfg):
    from fbpo.data import load_returns

    return load_returns(cfg)


@pytest.fixture(scope="session")
def market_daily(cfg):
    from fbpo.data import load_market

    return load_market(cfg)


@pytest.fixture(scope="session")
def investable(cfg, returns_daily):
    from fbpo.data import investable_mask

    return investable_mask(returns_daily, cfg.estimation.window)
@pytest.fixture(scope="session")
def signals_monthly(cfg):
    from fbpo.signals import load_signals

    return load_signals(cfg)


@pytest.fixture(scope="session")
def signals_unlagged(cfg):
    from fbpo.signals import build_unlagged_panel

    return build_unlagged_panel(cfg)

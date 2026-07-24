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

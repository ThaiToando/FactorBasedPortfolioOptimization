"""Phase 0 gate: workspace scaffold and the frozen universe file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REQUIRED_FILES = [
    "pyproject.toml",
    "README.md",
    "Makefile",
    "Dockerfile",
    ".python-version",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".devcontainer/devcontainer.json",
    ".vscode/settings.json",
    "configs/base.yaml",
    "configs/live.yaml",
    "configs/simulation.yaml",
    "configs/universes/dow30_static_2021.yaml",
    "src/fbpo/__init__.py",
    "src/fbpo/config.py",
    "src/fbpo/cli.py",
    "src/fbpo/determinism.py",
]

EXPECTED_DOW30 = {
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
}


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(repo_root: Path, relative: str) -> None:
    assert (repo_root / relative).is_file(), f"missing {relative}"


def test_universe_is_exactly_the_2021_djia(configs_dir: Path) -> None:
    spec = yaml.safe_load(
        (configs_dir / "universes" / "dow30_static_2021.yaml").read_text(encoding="utf-8")
    )
    tickers = spec["tickers"]
    assert len(tickers) == 30
    assert len(set(tickers)) == 30
    assert all(t == t.upper() and t.isalpha() for t in tickers)
    assert set(tickers) == EXPECTED_DOW30
    assert spec["as_of"] == "2021-12-31"

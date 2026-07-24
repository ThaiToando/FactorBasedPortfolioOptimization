"""Phase 0 gate: the CLI surface."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from fbpo.cli import app
from fbpo.config import Config

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "dumb", "PYTHONHASHSEED": "0"}

CORE_COMMANDS = ("fetch-data", "backtest", "simulate", "figures", "all")


def test_help_lists_the_five_core_commands() -> None:
    result = runner.invoke(app, ["--help"], env=ENV)
    assert result.exit_code == 0
    for command in CORE_COMMANDS:
        assert command in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"], env=ENV)
    assert result.exit_code == 0
    assert "fbpo" in result.stdout


def test_config_show_reports_the_hash(repo_root: Path) -> None:
    path = repo_root / "configs" / "base.yaml"
    result = runner.invoke(app, ["config-show", "--config", str(path)], env=ENV)
    assert result.exit_code == 0
    assert Config.from_yaml(path).hash in result.stdout


def test_config_show_json_is_the_canonical_form(repo_root: Path) -> None:
    path = repo_root / "configs" / "base.yaml"
    result = runner.invoke(app, ["config-show", "--config", str(path), "--json"], env=ENV)
    assert result.exit_code == 0
    assert result.stdout.strip() == Config.from_yaml(path).canonical_json()


def test_unimplemented_command_exits_with_code_two(repo_root: Path) -> None:
    path = repo_root / "configs" / "base.yaml"
    result = runner.invoke(app, ["backtest", "--config", str(path)], env=ENV)
    assert result.exit_code == 2


def test_missing_config_file_is_rejected() -> None:
    result = runner.invoke(app, ["config-show", "--config", "configs/does_not_exist.yaml"], env=ENV)
    assert result.exit_code != 0

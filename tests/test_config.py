"""Phase 0 gate: the configuration engine."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fbpo.config import Config


def test_defaults_equal_base_yaml(configs_dir: Path) -> None:
    """Load-bearing invariant: override-only configs inherit exactly base.yaml."""
    assert Config.from_yaml(configs_dir / "base.yaml") == Config()


def test_hash_is_eight_hex_characters() -> None:
    digest = Config().hash
    assert len(digest) == 8
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_hash_is_stable_across_loads(configs_dir: Path) -> None:
    a = Config.from_yaml(configs_dir / "base.yaml")
    b = Config.from_yaml(configs_dir / "base.yaml")
    assert a.hash == b.hash


def test_hash_is_invariant_to_yaml_key_order(configs_dir: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((configs_dir / "base.yaml").read_text(encoding="utf-8"))
    shuffled = dict(reversed(list(raw.items())))
    scrambled = tmp_path / "scrambled.yaml"
    scrambled.write_text(yaml.safe_dump(shuffled, sort_keys=False), encoding="utf-8")
    assert Config.from_yaml(scrambled).hash == Config.from_yaml(configs_dir / "base.yaml").hash


def test_hash_changes_when_a_parameter_changes() -> None:
    base = Config()
    variant = base.model_copy(
        update={"estimation": base.estimation.model_copy(update={"window": 126})}
    )
    assert variant.hash != base.hash


def test_canonical_json_is_the_hash_preimage() -> None:
    cfg = Config()
    expected = hashlib.sha256(cfg.canonical_json().encode("utf-8")).hexdigest()[:8]
    assert cfg.hash == expected


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "typo.yaml"
    bad.write_text("estimation:\n  windwo: 252\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.from_yaml(bad)


def test_negative_window_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(estimation={"window": -5})


def test_min_obs_below_window_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(estimation={"window": 252, "min_obs": 63})


def test_reversed_timeline_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(backtest={"first_rebalance": "2021-11-30", "last_rebalance": "2009-12-31"})


def test_malformed_date_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(data={"price_start": "06/02/2008"})


def test_today_sentinel_is_accepted() -> None:
    cfg = Config(data={"price_end": "today"}, backtest={"last_rebalance": "today"})
    assert cfg.data.price_end == "today"


def test_config_is_immutable() -> None:
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize("name", ["base.yaml", "live.yaml", "simulation.yaml"])
def test_shipped_configs_load(configs_dir: Path, name: str) -> None:
    cfg = Config.from_yaml(configs_dir / name)
    assert len(cfg.hash) == 8


def test_base_config_matches_spec_timeline(configs_dir: Path) -> None:
    cfg = Config.from_yaml(configs_dir / "base.yaml")
    assert cfg.backtest.expected_rebalances == 144
    assert cfg.backtest.first_rebalance == "2009-12-31"
    assert cfg.backtest.last_rebalance == "2021-11-30"
    assert cfg.data.price_end == "2022-01-04"
    assert cfg.estimation.window == 252
    assert cfg.forecast.min_train_months == 60
    assert cfg.svr.standardize_y is True


def test_roundtrip_through_yaml_preserves_the_hash(tmp_path: Path) -> None:
    cfg = Config()
    written = cfg.to_yaml(tmp_path / "roundtrip.yaml")
    assert Config.from_yaml(written).hash == cfg.hash

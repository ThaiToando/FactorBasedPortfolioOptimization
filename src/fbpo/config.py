"""Typed, validated, hashed configuration.

Every constant in the project lives here or in a YAML file that validates
against these models (SPEC S4). Two invariants are load-bearing:

1. ``Config()`` with no arguments is byte-identical to ``configs/base.yaml``.
   That is what makes ``live.yaml`` and ``simulation.yaml`` safe as
   override-only files instead of full copies that silently drift.
2. ``Config.hash`` is a stable 8-character fingerprint over the canonical JSON
   form, so ``reports/results_<hash>.parquet`` is self-identifying and the
   sensitivity grid is a loop over YAML files.

Validation is strict: unknown keys are rejected (a typo'd key must not be
silently ignored) and models are frozen (a config cannot mutate mid-backtest).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

TODAY_SENTINEL = "today"


def _validate_date(value: str) -> str:
    """Accept an ISO ``YYYY-MM-DD`` date or the literal sentinel ``today``."""
    if value == TODAY_SENTINEL:
        return value
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"expected an ISO date 'YYYY-MM-DD' or the sentinel {TODAY_SENTINEL!r}, got {value!r}"
        ) from exc
    return value


ISODate = Annotated[str, AfterValidator(_validate_date)]


def resolve_date(value: str) -> dt.date:
    """Resolve a configured date string, expanding the ``today`` sentinel."""
    if value == TODAY_SENTINEL:
        return dt.date.today()
    return dt.date.fromisoformat(value)


class _Base(BaseModel):
    """Strict base: unknown keys rejected, instances immutable, defaults validated."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DataCfg(_Base):
    """Universe, sample window, and data sources (SPEC S1-S3)."""

    universe: str = "dow30_static_2021"
    price_start: ISODate = "2008-06-02"
    price_end: ISODate = "2022-01-04"  # yfinance `end` is EXCLUSIVE -> through 2021-12-31
    signal_start: ISODate = "2003-01-31"
    signal_set: Literal["signals_auto", "signals_full", "signals_stable"] = "signals_full"
    market_source: Literal["french"] = "french"
    auto_adjust: bool = True
    trading_days_per_year: int = Field(252, ge=1, le=366)


class BacktestCfg(_Base):
    """Rebalance timeline (SPEC S3). A rebalance date is the last NYSE trading day of a month."""

    first_rebalance: ISODate = "2009-12-31"
    last_rebalance: ISODate = "2021-11-30"
    frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    expected_rebalances: int | None = Field(144, ge=1)
    intramonth_rebalancing: bool = False
    return_type_estimation: Literal["log", "simple"] = "log"
    return_type_compounding: Literal["log", "simple"] = "simple"


class EstimationCfg(_Base):
    """Beta and covariance estimation (Phases 2-3)."""

    window: int = Field(252, ge=21, le=1260)
    min_obs: int = Field(252, ge=21, le=1260)
    beta_shrinkage: Literal["none", "blume", "vasicek"] = "none"
    covariance: Literal[
        "sample", "ledoit_wolf", "oas", "single_index", "ewma", "yang_zhang"
    ] = "sample"
    ewma_halflife: int = Field(63, ge=1)
    single_index_delta: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _min_obs_covers_window(self) -> EstimationCfg:
        if self.min_obs < self.window:
            raise ValueError(
                f"min_obs ({self.min_obs}) < window ({self.window}): an asset would be declared "
                "investable before it has a full estimation window, producing short-window betas"
            )
        return self


class ForecastCfg(_Base):
    """Market-premium forecaster (Phase 4)."""

    model: Literal[
        "prevailing_mean", "combination", "ridge", "pca_ols", "pls", "svr", "gp", "gbm"
    ] = "svr"
    min_train_months: int = Field(60, ge=12)
    tune: bool = True
    refit_every: int = Field(12, ge=1)
    cv_splits: int = Field(4, ge=2)
    embargo_months: int = Field(1, ge=0)
    campbell_thompson: bool = True
    shrink_omega: float = Field(1.0, ge=0.0, le=1.0)
    target: Literal["market_excess_monthly"] = "market_excess_monthly"


class SvrGrid(_Base):
    """GridSearchCV search space. Field names mirror scikit-learn parameter names."""

    C: list[float] = [0.1, 1.0, 10.0, 100.0]
    gamma: list[Literal["scale", "auto"] | float] = ["scale", 0.001, 0.01, 0.1]
    epsilon: list[float] = [0.01, 0.05, 0.1, 0.2]

    @model_validator(mode="after")
    def _grids_non_empty(self) -> SvrGrid:
        for name in ("C", "gamma", "epsilon"):
            if not getattr(self, name):
                raise ValueError(f"svr.grid.{name} must contain at least one candidate value")
        if any(c <= 0 for c in self.C):
            raise ValueError("svr.grid.C must be strictly positive")
        if any(e < 0 for e in self.epsilon):
            raise ValueError("svr.grid.epsilon must be non-negative")
        return self


class SvrCfg(_Base):
    """SVR settings.

    ``standardize_y`` is the guard against the epsilon-tube flat-forecast trap:
    a monthly excess-return series has std ~0.04, so epsilon=0.1 on the raw
    target exceeds the entire scale of y and every prediction collapses to a
    constant. Phase 4 asserts forecast std >= 0.0020 as the executable check.
    """

    kernel: Literal["rbf", "linear", "poly", "sigmoid"] = "rbf"
    standardize_X: bool = True
    standardize_y: bool = True
    grid: SvrGrid = SvrGrid()


class OptimizeCfg(_Base):
    """Portfolio construction (Phase 3)."""

    objective: Literal[
        "max_sharpe", "gmvp", "mdrp", "min_cvar", "equal_weight", "inverse_vol"
    ] = "max_sharpe"
    solver: Literal["clarabel", "osqp", "slsqp"] = "clarabel"
    long_only: bool = True
    fully_invested: bool = True
    weight_cap: float = Field(1.0, gt=0.0, le=1.0)
    fallback_when_no_positive_mu: Literal["gmvp", "equal_weight"] = "gmvp"
    turnover_penalty: float = Field(0.0, ge=0.0)
    robust_kappa: float = Field(0.0, ge=0.0)


class CostsCfg(_Base):
    """Transaction costs. Headline results are gross; the sweep is reported alongside."""

    bps_per_side: float = Field(0.0, ge=0.0, le=100.0)
    cost_sweep_bps: list[float] = [0.0, 5.0, 10.0, 20.0]


class SimulationCfg(_Base):
    """Monte Carlo estimation-noise study (Phase 6)."""

    n_paths: int = Field(1000, ge=1)
    block_length_days: int = Field(21, ge=1)
    bootstrap: Literal["stationary", "circular", "iid"] = "stationary"
    mean_perturbation_sigma: list[float] = [0.0, 0.01, 0.02, 0.05]


class SensitivityGridCfg(_Base):
    """4 windows x 3 frequencies x 8 universes x 5 subperiods = 480 cells (SPEC S9)."""

    window: list[int] = [63, 126, 252, 504]
    frequency: list[Literal["weekly", "monthly", "quarterly"]] = [
        "weekly",
        "monthly",
        "quarterly",
    ]
    universe: list[str] = [
        "dow30_static_2021",
        "dow30_pit",
        "sp100_static_2021",
        "random30_seed0",
        "random30_seed1",
        "random30_seed2",
        "random30_seed3",
        "random30_seed4",
    ]
    subperiod: list[str] = [
        "2010-2013",
        "2014-2017",
        "2018-2021",
        "2010-2019",
        "2020-2021",
    ]


class StatsCfg(_Base):
    """Inference, resampling, and overfitting diagnostics (Phase 6)."""

    bootstrap_reps: int = Field(10_000, ge=100)
    block_length_days: int = Field(21, ge=1)
    newey_west_lags: int = Field(5, ge=0)
    cvar_alpha: float = Field(0.95, gt=0.5, lt=1.0)
    conformal_alpha: float = Field(0.10, gt=0.0, lt=0.5)
    aci_gamma: float = Field(0.01, gt=0.0, le=1.0)
    cpcv_groups: int = Field(6, ge=3)
    cpcv_test_groups: int = Field(2, ge=1)
    cpcv_embargo_days: int = Field(5, ge=0)
    pbo_blocks: int = Field(16, ge=2)
    vol_target_annual: float = Field(0.10, gt=0.0, le=1.0)
    vol_target_window: int = Field(21, ge=1)
    vol_target_cap: float = Field(1.0, gt=0.0)

    @model_validator(mode="after")
    def _cpcv_split_is_feasible(self) -> StatsCfg:
        if self.cpcv_test_groups >= self.cpcv_groups:
            raise ValueError(
                f"cpcv_test_groups ({self.cpcv_test_groups}) must be strictly less than "
                f"cpcv_groups ({self.cpcv_groups}); otherwise no training data remains"
            )
        return self


class Config(_Base):
    """Root configuration. Defaults are identical to ``configs/base.yaml``."""

    seed: int = 42
    data: DataCfg = DataCfg()
    backtest: BacktestCfg = BacktestCfg()
    estimation: EstimationCfg = EstimationCfg()
    forecast: ForecastCfg = ForecastCfg()
    svr: SvrCfg = SvrCfg()
    optimize: OptimizeCfg = OptimizeCfg()
    costs: CostsCfg = CostsCfg()
    simulation: SimulationCfg = SimulationCfg()
    sensitivity_grid: SensitivityGridCfg = SensitivityGridCfg()
    stats: StatsCfg = StatsCfg()

    @model_validator(mode="after")
    def _timeline_is_ordered(self) -> Config:
        price_start = resolve_date(self.data.price_start)
        price_end = resolve_date(self.data.price_end)
        signal_start = resolve_date(self.data.signal_start)
        first = resolve_date(self.backtest.first_rebalance)
        last = resolve_date(self.backtest.last_rebalance)

        if price_start >= price_end:
            raise ValueError(
                f"data.price_start ({self.data.price_start}) must precede "
                f"data.price_end ({self.data.price_end})"
            )
        if first > last:
            raise ValueError(
                f"backtest.first_rebalance ({self.backtest.first_rebalance}) must not follow "
                f"backtest.last_rebalance ({self.backtest.last_rebalance})"
            )
        if price_start > first:
            raise ValueError(
                "data.price_start must precede backtest.first_rebalance by at least "
                "estimation.window trading days"
            )
        if signal_start > first:
            raise ValueError(
                "data.signal_start must precede backtest.first_rebalance by at least "
                "forecast.min_train_months months"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load and validate a YAML configuration file."""
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"{p}: the top level of a config file must be a mapping, got {type(raw).__name__}"
            )
        try:
            return cls(**raw)
        except ValidationError as exc:
            exc.add_note(f"while loading configuration from {p}")
            raise

    def to_yaml(self, path: str | Path) -> Path:
        """Write the fully resolved configuration, including defaults."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return p

    def canonical_json(self) -> str:
        """Key-sorted, whitespace-free JSON: the exact preimage of :attr:`hash`."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def hash(self) -> str:
        """Stable 8-character fingerprint used in output filenames and the experiment log."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:8]
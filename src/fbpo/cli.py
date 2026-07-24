"""Command-line interface.

``fbpo <command>`` is the only supported entry point; the Makefile and every CI
workflow call through it. Commands whose numerical module has not been built yet
exit with code 2 and name the phase that delivers them, so a premature
``make all`` fails loudly instead of producing an empty artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fbpo import __version__
from fbpo.config import Config
from fbpo.determinism import hashseed_is_pinned, set_deterministic, write_env_fingerprint

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Factor-based portfolio optimization (Auh & Cho, 2023) - reproduction and extension.",
)

DEFAULT_CONFIG = Path("configs/base.yaml")

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML configuration file.",
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fbpo {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Runs before every command."""


def _load(config: Path) -> Config:
    """Load a config and immediately pin determinism controls."""
    cfg = Config.from_yaml(config)
    set_deterministic(cfg.seed)
    if not hashseed_is_pinned():
        typer.secho(
            "warning: PYTHONHASHSEED is not '0'. Results remain correct but are not "
            "bit-reproducible. Launch via `make` or the devcontainer to pin it.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    return cfg


def _pending(phase: int, modules: str) -> None:
    """Fail loudly for a command whose numerical module has not been built yet."""
    typer.secho(
        f"not implemented yet: this command arrives in PHASE {phase} ({modules}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("config-show")
def config_show(
    config: ConfigOption = DEFAULT_CONFIG,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the canonical JSON that the hash is computed over.")
    ] = False,
) -> None:
    """Validate a configuration file and print its hash and headline parameters."""
    cfg = _load(config)

    if as_json:
        typer.echo(cfg.canonical_json())
        return

    rows: list[tuple[str, str]] = [
        ("config file", str(config)),
        ("config hash", cfg.hash),
        ("seed", str(cfg.seed)),
        ("universe", cfg.data.universe),
        ("signal set", cfg.data.signal_set),
        ("price window", f"{cfg.data.price_start} -> {cfg.data.price_end} (end exclusive)"),
        ("signal start", cfg.data.signal_start),
        (
            "rebalances",
            f"{cfg.backtest.first_rebalance} -> {cfg.backtest.last_rebalance} "
            f"({cfg.backtest.expected_rebalances} expected, {cfg.backtest.frequency})",
        ),
        (
            "estimation",
            f"window={cfg.estimation.window} beta_shrinkage={cfg.estimation.beta_shrinkage} "
            f"cov={cfg.estimation.covariance}",
        ),
        (
            "forecast",
            f"model={cfg.forecast.model} min_train={cfg.forecast.min_train_months}m "
            f"tune={cfg.forecast.tune} refit_every={cfg.forecast.refit_every} "
            f"campbell_thompson={cfg.forecast.campbell_thompson}",
        ),
        (
            "optimize",
            f"objective={cfg.optimize.objective} solver={cfg.optimize.solver} "
            f"long_only={cfg.optimize.long_only} weight_cap={cfg.optimize.weight_cap}",
        ),
        ("costs", f"{cfg.costs.bps_per_side} bps/side, sweep {cfg.costs.cost_sweep_bps}"),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        typer.echo(f"{key:<{width}}  {value}")


@app.command("env-fingerprint")
def env_fingerprint_cmd(
    out: Annotated[Path, typer.Option("--out", help="Destination path.")] = Path(
        "reports/env_fingerprint.json"
    ),
) -> None:
    """Record platform, package versions, BLAS vendor and thread counts."""
    path = write_env_fingerprint(out)
    typer.echo(f"wrote {path}")


@app.command("fetch-data")
def fetch_data(
    config: ConfigOption = DEFAULT_CONFIG,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore the cache and re-download.")
    ] = False,
    verify: Annotated[
        bool, typer.Option("--verify", help="Check the cache against data/manifest.json.")
    ] = False,
) -> None:
    """Download and cache prices, the French factors, and the monthly signal panel."""
    cfg = _load(config)
    typer.echo(
        f"[{cfg.hash}] universe={cfg.data.universe} signals={cfg.data.signal_set} "
        f"{cfg.data.price_start} -> {cfg.data.price_end} refresh={refresh} verify={verify}"
    )
    _pending(1, "src/fbpo/data.py, src/fbpo/signals.py, src/fbpo/schemas.py")


@app.command("backtest")
def backtest(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run the walk-forward backtest and write reports/results_<hash>.parquet."""
    cfg = _load(config)
    typer.echo(f"[{cfg.hash}] would write reports/results_{cfg.hash}.parquet")
    _pending(5, "src/fbpo/backtest.py, src/fbpo/benchmarks.py")


@app.command("simulate")
def simulate(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Monte Carlo estimation-noise study."""
    cfg = _load(config)
    typer.echo(f"[{cfg.hash}] {cfg.simulation.n_paths} paths, {cfg.simulation.bootstrap} bootstrap")
    _pending(6, "src/fbpo/uncertainty.py")


@app.command("figures")
def figures(
    results: Annotated[
        Path | None, typer.Option("--results", help="Results parquet; defaults to the latest.")
    ] = None,
) -> None:
    """Regenerate every figure from cached results."""
    typer.echo(f"results={results or 'latest'}")
    _pending(6, "src/fbpo/reporting.py")


@app.command("all")
def run_all(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """fetch-data -> backtest -> simulate -> figures."""
    cfg = _load(config)
    typer.echo(f"[{cfg.hash}] full pipeline")
    _pending(5, "the full pipeline lands once Phases 1-5 are complete")
"""Walk-forward backtester.

Three specification details that are easy to get subtly wrong and expensive:

DRIFT. ``intramonth_rebalancing: false`` means weights are set at month-end and
then left alone. Daily P&L is sum_i w_it r_it where w_it evolves by each
asset's own return -- not reset to the target each day. Resetting daily is an
implicit rebalancing strategy that harvests volatility and inflates the Sharpe
ratio by a tenth or more. It is the most common error in this literature.

UNITS. Betas and covariances come from log returns; portfolio P&L compounds
simple returns. Mixing them costs a few basis points a month, which across 144
months is not a rounding error.

TURNOVER. Cost applies to the distance from the *drifted* weights to the new
target, not from last month's target. Drift does part of the rebalancing for
free, and charging for it overstates turnover by roughly a third.

Every rebalance also records which objective actually ran. A max-Sharpe
strategy that fell back to GMVP in 30 of 144 months is a different strategy
from one that never did, and the summary reports the count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fbpo.benchmarks import ALL_STRATEGIES, expected_returns, objective_for
from fbpo.covariance import estimate_covariance, sample_covariance
from fbpo.estimation import rolling_beta, vasicek_shrink
from fbpo.optimize import OptimizationError, herfindahl, optimize_weights

TRADING_DAYS_PER_MONTH = 21


def rebalance_dates(calendar: pd.DatetimeIndex, first: str, last: str) -> pd.DatetimeIndex:
    """Month-end trading days between `first` and `last`, inclusive.

    A rebalance date is the last date in the trading calendar within its month,
    never a nominal calendar month-end: 2021-10-31 was a Sunday.
    """
    frame = pd.Series(calendar, index=calendar)
    monthly = frame.resample("ME").last().dropna()
    dates = pd.DatetimeIndex(monthly.to_numpy())
    return dates[(dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last))]


def drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Evolve weights by one day of returns, renormalising to sum to 1.

    w_next = w (1 + r) / sum(w (1 + r)). This is what "no intramonth
    rebalancing" means mechanically: the portfolio value shifts toward whatever
    went up.
    """
    grown = weights * (1.0 + returns)
    total = grown.sum()
    if total <= 0:
        return weights
    return grown / total


def compound_month(
    weights: np.ndarray, month_returns: pd.DataFrame
) -> tuple[pd.Series, np.ndarray]:
    """Daily portfolio returns for one holding period, with weights drifting.

    Returns ``(daily_returns, terminal_weights)``. The terminal weights are the
    starting point for the next rebalance's turnover calculation.
    """
    current = np.asarray(weights, dtype=float).copy()
    daily: dict[pd.Timestamp, float] = {}

    for date, row in month_returns.iterrows():
        asset_returns = row.to_numpy()
        daily[date] = float(current @ asset_returns)
        current = drift_weights(current, asset_returns)

    return pd.Series(daily, name="ret"), current


def turnover(target: np.ndarray, drifted: np.ndarray) -> float:
    """One-way turnover: half the sum of absolute weight changes.

    Halving makes it "fraction of the portfolio traded" rather than double
    counting each trade's buy and sell side.
    """
    return float(0.5 * np.abs(np.asarray(target) - np.asarray(drifted)).sum())


def annualised_metrics(daily: pd.Series, risk_free: pd.Series | None = None) -> dict[str, float]:
    """Standard performance summary from a daily return series."""
    n = len(daily)
    if n == 0:
        return {}

    total_growth = float((1.0 + daily).prod())
    years = n / 252.0
    cagr = total_growth ** (1.0 / years) - 1.0
    volatility = float(daily.std(ddof=1) * np.sqrt(252))

    if risk_free is not None:
        excess = daily - risk_free.reindex(daily.index).fillna(0.0)
    else:
        excess = daily
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else np.nan

    downside = excess[excess < 0].std(ddof=1)
    sortino = float(excess.mean() / downside * np.sqrt(252)) if downside > 0 else np.nan

    curve = (1.0 + daily).cumprod()
    drawdown = float((curve / curve.cummax() - 1.0).min())

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": float(cagr / abs(drawdown)) if drawdown < 0 else np.nan,
        "n_days": n,
    }
def _prepare(cfg):
    """Load every input the backtest needs, once."""
    from fbpo.benchmarks import fetch_djia
    from fbpo.data import investable_mask, load_market, load_returns
    from fbpo.forecast import generate_forecasts

    log_returns = load_returns(cfg, kind="log")
    simple_returns = load_returns(cfg, kind="simple")
    market = load_market(cfg)
    mask = investable_mask(log_returns, cfg.estimation.min_obs)
    beta, se_squared = rolling_beta(log_returns, market["mkt"], cfg.estimation.window)
    shrunk = vasicek_shrink(beta, se_squared, mask=mask)
    forecasts = generate_forecasts(cfg)
    djia = fetch_djia(cfg)
    return log_returns, simple_returns, market, mask, beta, shrunk, forecasts, djia


def run_backtest(cfg, strategies: tuple[str, ...] = ALL_STRATEGIES) -> dict:
    """Walk forward across every rebalance date, for every strategy.

    Returns a dict with ``returns`` (daily, one column per strategy),
    ``weights`` (per rebalance), and ``diagnostics`` (turnover, HHI, the
    objective actually used, and the number of investable assets).
    """
    log_returns, simple_returns, market, mask, beta, shrunk, forecasts, djia = _prepare(cfg)

    calendar = log_returns.index
    dates = rebalance_dates(calendar, cfg.backtest.first_rebalance, cfg.backtest.last_rebalance)

    if cfg.backtest.expected_rebalances is not None:
        assert len(dates) == cfg.backtest.expected_rebalances, (
            f"expected {cfg.backtest.expected_rebalances} rebalances, built {len(dates)}"
        )

    daily: dict[str, dict] = {s: {} for s in strategies}
    records: list[dict] = []
    previous: dict[str, tuple[np.ndarray, list[str]]] = {}

    for i, date in enumerate(dates):
        eligible = mask.loc[date]
        columns = list(eligible[eligible].index)
        if len(columns) < 2:
            continue

        window = log_returns.loc[:date, columns].iloc[-cfg.estimation.window :]
        covariance = estimate_covariance(window, cfg.estimation.covariance)
        risk_free = float(market.loc[date, "rf"])

        premium_hist = float(forecasts["prevailing_mean"].asof(date))
        premium_ml = float(forecasts["model"].asof(date))

        end = dates[i + 1] if i + 1 < len(dates) else calendar[-1]
        holding = simple_returns.loc[
            (simple_returns.index > date) & (simple_returns.index <= end), columns
        ]
        if holding.empty:
            continue

        for strategy in strategies:
            if strategy == "DJIA":
                continue

            mu = expected_returns(
                strategy,
                window,
                covariance,
                beta=beta.loc[date, columns].to_numpy(),
                shrunk_beta=shrunk.loc[date, columns].to_numpy(),
                premium_hist=premium_hist,
                premium_ml=premium_ml,
                daily_risk_free=risk_free,
            )

            try:
                weights, used = optimize_weights(
                    objective_for(strategy),
                    covariance,
                    mu=mu,
                    risk_free=risk_free,
                    weight_cap=cfg.optimize.weight_cap,
                    solver=cfg.optimize.solver,
                    robust_kappa=cfg.optimize.robust_kappa,
                    fallback=cfg.optimize.fallback_when_no_positive_mu,
                    n_obs=cfg.estimation.window,
                )
            except OptimizationError:
                weights, used = np.full(len(columns), 1.0 / len(columns)), "failed_equal_weight"

            drifted = _align_previous(previous.get(strategy), columns)
            traded = turnover(weights, drifted) if drifted is not None else 1.0
            cost = traded * 2.0 * cfg.costs.bps_per_side / 10_000.0

            month_returns, terminal = compound_month(weights, holding)
            month_returns.iloc[0] -= cost
            daily[strategy].update(month_returns.to_dict())
            previous[strategy] = (terminal, columns)

            records.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "objective_used": used,
                    "n_assets": len(columns),
                    "turnover": traded,
                    "hhi": herfindahl(weights),
                    "max_weight": float(weights.max()),
                    "premium_ml": premium_ml,
                    "premium_hist": premium_hist,
                }
            )

    frame = pd.DataFrame(daily)
    if "DJIA" in strategies:
        frame["DJIA"] = djia.reindex(frame.index)

    return {
        "returns": frame.dropna(how="all"),
        "diagnostics": pd.DataFrame(records),
        "rebalance_dates": dates,
        "risk_free": market["rf"],
    }


def _align_previous(
    state: tuple[np.ndarray, list[str]] | None, columns: list[str]
) -> np.ndarray | None:
    """Map last month's drifted weights onto this month's investable set.

    An asset that became eligible this month starts at zero weight, so entering
    it is charged as turnover -- which is correct, since it has to be bought.
    """
    if state is None:
        return None
    weights, previous_columns = state
    lookup = dict(zip(previous_columns, weights, strict=True))
    return np.array([lookup.get(c, 0.0) for c in columns])


def summarise(result: dict) -> pd.DataFrame:
    """Performance table, one row per strategy."""
    returns, risk_free = result["returns"], result["risk_free"]
    diagnostics = result["diagnostics"]

    rows = {}
    for strategy in returns.columns:
        series = returns[strategy].dropna()
        metrics = annualised_metrics(series, risk_free)
        subset = diagnostics[diagnostics["strategy"] == strategy]
        if not subset.empty:
            metrics["avg_turnover"] = float(subset["turnover"].mean())
            metrics["avg_hhi"] = float(subset["hhi"].mean())
            metrics["fallbacks"] = int((subset["objective_used"].str.contains("fallback")).sum())
        rows[strategy] = metrics

    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)

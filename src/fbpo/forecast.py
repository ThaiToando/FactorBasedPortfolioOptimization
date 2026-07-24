"""Out-of-sample forecasts of the monthly equity premium.

The target is the market excess return, Mkt-RF, at monthly frequency. Every
forecast for month t+1 is produced using only data observable at the close of
month t: the signal panel is already publication-lagged, and the training
window is strictly expanding, never centred.

THE FAILURE MODE THIS MODULE IS BUILT AROUND
--------------------------------------------
Support vector regression ignores errors smaller than epsilon. Monthly excess
returns have a standard deviation of roughly 0.04, and the configured grid
searches epsilon up to 0.2 -- five times the entire scale of the target. Fit on
raw y, an SVR discovers that predicting a constant satisfies every constraint,
and returns identical forecasts at every step.

Nothing about this looks wrong. There is no error and no warning. The optimizer
receives a constant mu across assets and quietly returns 1/N, producing a
plausible equity curve with a Sharpe near 0.9. The tell is a Herfindahl index
of exactly 1/N.

Two defences, neither optional:

StandardScaler on y as well as X, so epsilon operates on a unit-variance
  target where 0.1 means a tenth of a standard deviation, not 2.5 of them.
:func: assert_forecast_is_not_flat, which fails loudly if the dispersion of
  forecasts across the walk-forward path collapses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_FORECAST_STD = 0.0020


def monthly_market_excess(market: pd.DataFrame) -> pd.Series:
    """Compound daily market excess returns to a monthly series.

    Compounding (1+r) and subtracting 1 is deliberate: summing daily excess
    returns would misstate the monthly figure by a few basis points, and those
    accumulate across 144 months.
    """
    excess = market["mkt"] - market["rf"]
    monthly = (1.0 + excess).resample("ME").prod() - 1.0
    return monthly.rename("mkt_excess")


def align_design(signals: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align signals at month t with the target realised in month t+1.

    This shift is where look-ahead bias would enter if it entered anywhere:
    X.loc[t] must predict y.loc[t+1], never y.loc[t].
    """
    y = target.shift(-1).rename("target")
    frame = signals.join(y, how="inner").dropna()
    return frame.drop(columns=["target"]), frame["target"]


def prevailing_mean_forecast(y: pd.Series, min_train: int = 60) -> pd.Series:
    """Expanding-window historical mean.

    Goyal and Welch (2008) showed this is close to unbeatable out of sample.
    It is the benchmark that matters: a model that cannot beat the running
    average of the thing it predicts has found nothing.
    """
    forecasts: dict[pd.Timestamp, float] = {}
    for i in range(min_train, len(y)):
        forecasts[y.index[i]] = float(y.iloc[:i].mean())
    return pd.Series(forecasts, name="prevailing_mean")


def combination_forecast(X: pd.DataFrame, y: pd.Series, min_train: int = 60) -> pd.Series:
    """Rapach, Strauss and Zhou (2010): the mean of univariate OLS forecasts.

    Each signal is regressed on the target alone, each produces a forecast, and
    the forecasts are averaged with equal weights. Averaging predictions rather
    than fitting one multivariate model is what tames the estimation error --
    the strongest simple competitor in this literature, and a much harder
    benchmark than the prevailing mean.
    """
    forecasts: dict[pd.Timestamp, float] = {}
    values = X.to_numpy()
    target = y.to_numpy()

    for i in range(min_train, len(y)):
        history_X = values[:i]
        history_y = target[:i]
        current = values[i]

        predictions = []
        for j in range(values.shape[1]):
            column = history_X[:, j]
            if np.std(column) < 1e-12:
                continue
            slope, intercept = np.polyfit(column, history_y, 1)
            predictions.append(intercept + slope * current[j])

        forecasts[y.index[i]] = (
            float(np.mean(predictions)) if predictions else float(history_y.mean())
        )

    return pd.Series(forecasts, name="combination")


def campbell_thompson(forecasts: pd.Series) -> pd.Series:
    """Clip negative forecasts to zero.

    Campbell and Thompson (2008): an investor would never act on a forecast of
    a negative equity premium, so allowing the model to emit one gives it
    credit for a prediction nobody would trade. The constraint is a restriction
    on the model, and it reliably improves out-of-sample R-squared.
    """
    return forecasts.clip(lower=0.0)


def out_of_sample_r2(actual: pd.Series, forecast: pd.Series, benchmark: pd.Series) -> float:
    """Campbell-Thompson OOS R-squared against a benchmark forecast.

    Positive means the model beat the benchmark in mean squared error. Values
    above 0.005 are considered economically meaningful in this literature;
    anything above 0.20 on monthly data means look-ahead bias, not skill.
    """
    index = actual.index.intersection(forecast.index).intersection(benchmark.index)
    a, f, b = actual.loc[index], forecast.loc[index], benchmark.loc[index]
    sse_model = float(((a - f) ** 2).sum())
    sse_benchmark = float(((a - b) ** 2).sum())
    if sse_benchmark <= 0:
        return np.nan
    return 1.0 - sse_model / sse_benchmark
def build_svr_pipeline(cfg):
    """StandardScaler -> SVR, wrapped so that y is scaled too.

    ``TransformedTargetRegressor`` is what makes epsilon meaningful: it scales
    y before fitting and inverts the scaling on predict, so the tube width is
    expressed in standard deviations of the target rather than in raw return
    units. Without it, the configured epsilon grid silently produces constants.
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    inner = Pipeline(
        [
            ("scale", StandardScaler() if cfg.svr.standardize_X else "passthrough"),
            ("svr", SVR(kernel=cfg.svr.kernel)),
        ]
    )
    if not cfg.svr.standardize_y:
        return inner
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def _grid(cfg) -> dict[str, list]:
    """Hyperparameter grid, with prefixes matching the pipeline nesting."""
    prefix = "regressor__svr__" if cfg.svr.standardize_y else "svr__"
    return {
        f"{prefix}C": list(cfg.svr.grid.C),
        f"{prefix}gamma": list(cfg.svr.grid.gamma),
        f"{prefix}epsilon": list(cfg.svr.grid.epsilon),
    }


def _select_hyperparameters(cfg, X: np.ndarray, y: np.ndarray) -> dict:
    """Inner TimeSeriesSplit search on the training window only.

    An ordinary K-fold would train on future months to predict past ones. The
    split must respect time, which is the whole reason for TimeSeriesSplit.
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    search = GridSearchCV(
        build_svr_pipeline(cfg),
        _grid(cfg),
        cv=TimeSeriesSplit(n_splits=cfg.forecast.cv_splits),
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_params_


def svr_walk_forward(cfg, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window SVR forecasts, re-tuned every ``refit_every`` months.

    Re-running the grid search at every step would mean 144 x 4 folds x 64 grid
    points -- roughly 37,000 fits -- and would let hyperparameters chase month-
    to-month noise. Annual re-selection is both faster and closer to what a
    practitioner could actually do.

    Returns ``(forecasts, chosen_parameters)``; the second frame records which
    hyperparameters were live at each step, which is what lets you check
    afterwards whether epsilon drifted to the top of its grid.
    """
    from threadpoolctl import threadpool_limits

    values = X.to_numpy()
    target = y.to_numpy()
    min_train = cfg.forecast.min_train_months

    forecasts: dict[pd.Timestamp, float] = {}
    chosen: dict[pd.Timestamp, dict] = {}
    params: dict | None = None

    with threadpool_limits(limits=1):
        for i in range(min_train, len(y)):
            train_X, train_y = values[:i], target[:i]

            needs_tuning = params is None or (i - min_train) % cfg.forecast.refit_every == 0
            if cfg.forecast.tune and needs_tuning:
                params = _select_hyperparameters(cfg, train_X, train_y)

            model = build_svr_pipeline(cfg)
            if params:
                model.set_params(**params)
            model.fit(train_X, train_y)

            forecasts[y.index[i]] = float(model.predict(values[i : i + 1])[0])
            chosen[y.index[i]] = dict(params or {})

    return pd.Series(forecasts, name="svr"), pd.DataFrame(chosen).T


def assert_forecast_is_not_flat(forecasts: pd.Series, minimum: float = MIN_FORECAST_STD) -> float:
    """Fail loudly when the forecast path has collapsed to a constant.

    A flat forecast is not a bad model -- it is a broken one, and it will not
    announce itself anywhere downstream except as an unexpectedly diversified
    portfolio.
    """
    dispersion = float(forecasts.std(ddof=1))
    if not np.isfinite(dispersion) or dispersion < minimum:
        raise ValueError(
            f"forecast standard deviation {dispersion:.6f} is below {minimum}: the model is "
            "emitting a near-constant. Check that standardize_y is true -- an epsilon of 0.1 "
            "on an unscaled target whose own std is ~0.04 makes a constant optimal."
        )
    return dispersion


def shrink_to_prevailing_mean(model: pd.Series, benchmark: pd.Series, omega: float) -> pd.Series:
    """Blend a model forecast toward the prevailing mean.

    omega=1 is the pure model, omega=0 the pure benchmark. Shrinking a noisy
    forecast toward a robust one is the same idea as Vasicek shrinkage applied
    to betas, one level up.
    """
    index = model.index.intersection(benchmark.index)
    return (omega * model.loc[index] + (1.0 - omega) * benchmark.loc[index]).rename(model.name)


def generate_forecasts(cfg, signals: pd.DataFrame | None = None, market: pd.DataFrame | None = None):
    """Produce every forecast series the backtest needs, in one pass.

    Returns a frame with columns ``actual``, ``prevailing_mean``,
    ``combination`` and ``model``, all on the same index.
    """
    from fbpo.signals import _signal_market, load_signals

    if signals is None:
        signals = load_signals(cfg)
    if market is None:
        # The target needs market history back to signal_start (2003), not
        # price_start (2008): 60 months of training must precede the first
        # rebalance at 2009-12. The French factors reach back to 1926, so
        # there is no reason to inherit the stock panel's start date here.
        market = _signal_market(cfg)

    target = monthly_market_excess(market)
    X, y = align_design(signals, target)

    baseline = prevailing_mean_forecast(y, cfg.forecast.min_train_months)
    combination = combination_forecast(X, y, cfg.forecast.min_train_months)

    if cfg.forecast.model == "svr":
        model, _ = svr_walk_forward(cfg, X, y)
    elif cfg.forecast.model == "prevailing_mean":
        model = baseline.rename("model")
    elif cfg.forecast.model == "combination":
        model = combination.rename("model")
    else:
        raise ValueError(f"forecast model {cfg.forecast.model!r} is not implemented")

    if cfg.forecast.shrink_omega < 1.0:
        model = shrink_to_prevailing_mean(model, baseline, cfg.forecast.shrink_omega)

    assert_forecast_is_not_flat(model)

    if cfg.forecast.campbell_thompson:
        model = campbell_thompson(model)
        baseline = campbell_thompson(baseline)
        combination = campbell_thompson(combination)

    return pd.DataFrame(
        {
            "actual": y,
            "prevailing_mean": baseline,
            "combination": combination,
            "model": model,
        }
    ).dropna()

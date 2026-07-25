# fbpo , Factor-Based Portfolio Optimization

Independent reproduction and extension of **Auh, J.K. & Cho, W. (2023),
"Factor-based portfolio optimization," *Economics Letters* 228, 111137**
([doi](https://doi.org/10.1016/j.econlet.2023.111137)).

Research and educational purposes only. **Not investment advice.**

---

## What this is

The paper's thesis: estimating a portfolio's expected returns from raw
historical means injects idiosyncratic noise that drives mean–variance
optimizers into corner solutions. A parsimonious single-factor structure
(μᵢ = r_f + βᵢ · premium) estimates *one* market premium and projects it
through betas replacing N noisy parameters with one forecast plus N
precisely-estimated loadings. A machine-learning forecast (SVR) of the premium then adds forward-looking information.

This repository reproduces that result from public data on a 2009–2021 sample of
DJIA constituents and extends it with formal inference the original omits:
paired Sharpe bootstrap tests, deflated Sharpe ratios, Clark–West nested
forecast tests, and FF5+MOM factor attribution.

## Headline result

The factor-structure claim **reproduces cleanly**: it cuts portfolio
concentration to one-fifth that of naive estimation (matching the paper), earns
~3% annualised alpha against six factors (t ≈ 3) where naive estimation earns
none, and makes SRHist the worst out-of-sample performer despite the most
machinery whic the paper's key finding.

The ML extension **diverges, for a documented reason**: with the 13 FRED-
automatable signals used here (versus the paper's 22, which include licensed
options and sentiment data), the SVR forecast adds nothing beyond the historical
mean. The factor structure is the robust contribution; the ML uplift depends on
the richer signal set. See [`RESULTS.md`](RESULTS.md) for the full comparison.

An unexpected finding: **Vasicek beta shrinkage (1973) contributes more to
Sharpe than the SVR does.**

## Reproduce it

```bash
uv sync --all-extras
uv run fbpo config-show --config configs/base.yaml   # prints hash 43c9ce2e
uv run fbpo fetch-data  --config configs/base.yaml   # ~2 min, hits network
uv run fbpo backtest    --config configs/base.yaml   # ~8 min, 144 rebalances
uv run pytest -q                                     # 130+ tests
```

Every result artifact lands in `reports/`, named by the config hash so it is
self-identifying. Bit-reproducible on a fixed BLAS thread count; see
`reports/env_fingerprint.json`.


## Results at a glance

### Cumulative performance

![Equity curves](docs/01_equity.png)

### Out-of-sample return distributions

![Return distributions](docs/02_distributions.png)

### Risk and return

![Risk-return](docs/03_risk_return.png)

### Factor-model alpha

![Attribution](docs/04_attribution.png)

### Concentration and cost

![Concentration and cost](docs/06_concentration_cost.png)

### Alpha per unit of trading

![Turnover vs alpha](docs/07_turnover_alpha.png)

### The 13-signal panel

![Signal panel](docs/09_signal_panel.png)

### Vasicek beta shrinkage

![Beta shrinkage](docs/10_beta_shrinkage.png)

### The premium forecast

![Forecast series](docs/11_forecast_series.png)

## Interactive simulations

- [Estimation error and the corner solution](docs/sim1_estimation_error.html)
- [Factor structure versus direct estimation](docs/sim2_factor_vs_direct.html)

*(Simulations run in-browser. Download and open, or enable GitHub Pages to make them live.)*

## Layout

| Module             | Role                                                     |
| ------------------ | -------------------------------------------------------- |
| `config.py`      | Typed, validated, hashed configuration (pydantic)        |
| `data.py`        | Prices, factors, calendar, the DOW eligibility rule      |
| `signals.py`     | 13-signal causal panel with publication lags             |
| `estimation.py`  | Rolling betas, Vasicek / Blume shrinkage                 |
| `covariance.py`  | Five covariance estimators + conditioning diagnostics    |
| `optimize.py`    | Convex max-Sharpe (Schaible), GMVP, MDRP, robust variant |
| `forecast.py`    | SVR walk-forward, RSZ combination, ε-tube guard         |
| `benchmarks.py`  | Expected-return estimators, 10-strategy roster           |
| `backtest.py`    | Walk-forward engine (weight drift, turnover, costs)      |
| `stats.py`       | Paired Sharpe bootstrap, DSR, Clark–West                |
| `attribution.py` | FF5 + MOM regression, Newey–West errors                 |

## Extensions beyond the paper

Vasicek/Blume beta shrinkage · Bayes–Stein means · Ledoit–Wolf / OAS / EWMA /
single-index / Yang–Zhang covariance · robust (ellipsoidal) max-Sharpe · RSZ
combination forecast · paired stationary-bootstrap Sharpe tests · deflated
Sharpe · Clark–West · FF5+MOM attribution · transaction-cost sweep · full test
suite and determinism controls.

## Deviations

WBA excluded (delisted 2025-08, no free adjusted history → 29-asset universe);
13 FRED signals instead of 22 (licensed data unavailable); full-sample Sharpe
instead of the paper's rolling-mean estimator. All detailed in
[`RESULTS.md`](RESULTS.md) §8 and `docs/deviations.md`.

## Citation

> Auh, J.K., & Cho, W. (2023). Factor-based portfolio optimization.
> *Economics Letters*, 228, 111137. https://doi.org/10.1016/j.econlet.2023.111137

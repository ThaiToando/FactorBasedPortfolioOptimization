# Results

Independent reproduction and extension of Auh, J.K. & Cho, W. (2023),
"Factor-based portfolio optimization," *Economics Letters* 228, 111137.

**Configuration fingerprint:** `43c9ce2e` (frozen; see `configs/base.yaml`).
**Sample:** 144 monthly rebalances, 2009-12-31 to 2021-11-30.
**Universe:** 29 DJIA-2021 constituents (28 before 2020-03; see Deviations).
All figures are reproducible via `uv run fbpo backtest --config configs/base.yaml`.

Research and educational purposes only. **Not investment advice.**

---

## 1. Headline finding

> Replacing sample-mean expected returns with a single-factor structure
> (μᵢ = r_f + βᵢ · premium) produces portfolios with roughly **3% annualised alpha** against the Fama–French five factors plus momentum (t ≈ 3),**one-third the turnover** of direct estimation, and Sharpe ratios that survive deflation. Adding a machine-learning forecast of the market premium contributes **nothing** beyond the prevailing historical mean. Naive mean–variance and Bayes–Stein show **no alpha** and cannot be distinguished from the 1/N benchmark. Vasicek beta shrinkage , which  is a 1973 empirical-Bayes
> formula adds more than the machine learning does.

This reproduces Auh & Cho's central claim (factor structure beats direct
estimation) and reports the ML extension as a clean negative result.

---

## 2. Performance summary (gross of costs)

Strategies sorted by Sharpe ratio. Sharpe and Sortino use excess returns;
turnover is one-way monthly; HHI is the average Herfindahl index of weights;
"fallbacks" counts months where max-Sharpe was infeasible and reverted to GMVP.

| Strategy              |  CAGR |   Vol |          Sharpe |  Max DD | Turnover |   HHI | Fallbacks |
| --------------------- | ----: | ----: | --------------: | ------: | -------: | ----: | --------: |
| **SRML_shrunk** | 18.8% | 17.1% | **1.066** | −34.7% |    10.9% | 0.056 |         3 |
| SRML                  | 18.8% | 17.8% |           1.034 | −35.1% |    10.6% | 0.058 |         3 |
| SRF1                  | 19.1% | 18.0% |           1.034 | −35.1% |     8.5% | 0.053 |         0 |
| InvVol                | 16.4% | 16.0% |           0.999 | −32.4% |     2.8% | 0.038 |         0 |
| EW (1/N)              | 17.2% | 16.9% |           0.998 | −33.3% |     2.6% | 0.036 |         0 |
| MDRP                  | 15.9% | 15.6% |           0.995 | −31.8% |    12.4% | 0.110 |         0 |
| GMVP                  | 11.3% | 13.1% |           0.850 | −25.7% |    11.1% | 0.202 |         0 |
| SRHist                | 16.0% | 20.2% |           0.813 | −30.8% |    30.6% | 0.353 |         0 |
| BayesStein            | 15.0% | 19.3% |           0.797 | −28.8% |    30.6% | 0.305 |         0 |
| DJIA                  | 11.0% | 17.0% |           0.672 | −37.1% |       — |    — |        — |

**Strategy definitions.** SRHist: max-Sharpe on historical mean returns.
SRF1: max-Sharpe on factor-implied means, premium = prevailing historical mean.
SRML: as SRF1 but premium = SVR forecast. SRML_shrunk: SRML with Vasicek-shrunk
betas. BayesStein: max-Sharpe on Jorion (1986) shrunk means. GMVP, MDRP, InvVol,
EW: no return estimate. DJIA: the price-weighted index itself.

**Reading the table.** The three factor strategies (SRF1, SRML, SRML_shrunk)
lead. The naive estimators (SRHist, BayesStein) are near the bottom despite the most machinery and trade **~4× more** (30.6% vs 8.5% monthly) and which is  direct estimation concentrates into estimation error and reverses it each month (HHI 0.35 vs 0.05).

---

## 3. Is the ranking real? Paired Sharpe tests vs 1/N

Strategy return series correlate above 0.97, so the standard error of the
*difference* in Sharpe is far smaller than of either level. A paired stationary
bootstrap (10,000 reps, 21-day blocks) exploits this. p-values are two-sided.

| Strategy    | Sharpe | Δ vs EW | Corr. w/ EW |     Bootstrap p |
| ----------- | -----: | -------: | ----------: | --------------: |
| SRML_shrunk |  1.066 |   +0.068 |       0.979 |           0.226 |
| SRML        |  1.034 |   +0.037 |       0.978 |           0.509 |
| SRF1        |  1.034 |   +0.036 |       0.986 |           0.454 |
| SRHist      |  0.813 |  −0.184 |       0.766 |           0.297 |
| DJIA        |  0.672 |  −0.325 |       0.987 | **0.000** |

**Against 1/N specifically, nothing is distinguishable**  in either direction.
This is DeMiguel, Garlappi & Uppal (2009) reproducing exactly: estimation error
is large enough that beating 1/N out-of-sample cannot be established over a
12-year sample, and you cannot even show SRHist is *worse* than 1/N (p = 0.30).
Only the DJIA , which is a different weighting scheme entirely  separates. The right comparison is not strategy-vs-1/N but strategy-vs-factors (§5).

---

## 4. Do costs change the ranking?

Sharpe ratios across transaction-cost assumptions (basis points per side).

| Strategy    | 0 bps | 5 bps | 10 bps |          20 bps |
| ----------- | ----: | ----: | -----: | --------------: |
| SRML_shrunk | 1.066 | 1.058 |  1.050 |           1.035 |
| SRF1        | 1.034 | 1.028 |  1.022 | **1.011** |
| SRML        | 1.034 | 1.027 |  1.020 | **1.006** |
| EW          | 0.998 | 0.996 |  0.994 |           0.990 |
| SRHist      | 0.813 | 0.795 |  0.777 |           0.741 |

Cost drag is exact: SRHist's 30.6% turnover × 40 bps round-trip ÷ 20.2%
volatility predicts a 0.073 Sharpe loss at 20 bps; observed 0.072.

**One reordering:** SRML leads SRF1 by 0.0006 at zero cost but SRF1 leads by
0.0055 at 20 bps — SRML trades 25% more for no gross advantage, so **the ML
forecast is strictly worse than the historical mean at any realistic cost.**

---

## 5. Attribution is the alpha real? (FF5 + MOM, Newey–West HAC)

Daily excess returns regressed on Mkt-RF, SMB, HML, RMW, CMA, MOM.
This is the decisive test, and it separates what §3 could not.

| Strategy              | α (ann.) |      α t-stat | Mkt-RF |    SMB |    HML |    MOM |  R² |
| --------------------- | --------: | -------------: | -----: | -----: | -----: | -----: | ---: |
| **SRF1**        |     +2.8% | **3.04** |   1.03 | −0.16 |   0.07 | −0.02 | 0.97 |
| **SRML_shrunk** |     +3.1% | **2.77** |   0.98 | −0.19 |   0.05 | −0.02 | 0.95 |
| **SRML**        |     +2.9% | **2.53** |   1.01 | −0.18 |   0.05 | −0.02 | 0.95 |
| EW                    |     +2.2% |           2.14 |   0.96 | −0.12 |   0.12 | −0.04 | 0.96 |
| InvVol                |     +1.7% |           1.68 |   0.92 | −0.15 |   0.08 | −0.03 | 0.95 |
| MDRP                  |     +2.0% |           1.16 |   0.85 | −0.13 | −0.04 | −0.07 | 0.84 |
| GMVP                  |    −0.1% |         −0.03 |   0.68 | −0.26 | −0.04 | −0.02 | 0.72 |
| SRHist                |    −1.4% |         −0.49 |   1.01 | −0.23 | −0.20 |   0.31 | 0.72 |
| BayesStein            |    −1.9% |         −0.68 |   0.98 | −0.26 | −0.20 |   0.26 | 0.73 |
| DJIA                  |    −3.4% |         −3.15 |   0.97 | −0.14 |   0.13 | −0.03 | 0.95 |

**The factor strategies earn significant alpha (t > 2.5); the naive strategies
earn none.** After controlling for six well-known factors, SRF1/SRML/SRML_shrunk
still deliver ~3% annualised that the factors do not explain. SRHist, BayesStein
and GMVP show alpha indistinguishable from zero.

This reconciles §3: EW *also* has alpha (t = 2.14), so factor-strategy-vs-EW
washes out  but factor-strategy-vs-factors clears the bar and naive-vs-factors
does not. The R² column confirms it: factor portfolios are 95–97% explained by
factors *and still* leave alpha; SRHist is only 72% explained because it is
concentrated and idiosyncratic, not skilled.

**Net-of-cost caveat.** These alphas are gross. At 10 bps the factor strategies
(8.5% monthly turnover ≈ 1%/yr) keep most of their ~3% and remain significant;
the naive strategies (30.6% turnover ≈ 3.6%/yr) push their already-zero alpha
firmly negative. Costs widen the gap; they do not close it.

---

## 6. The market-premium forecast

Out-of-sample R² vs the prevailing mean (Campbell–Thompson), 168 monthly steps,
and the Clark–West nested-forecast test.

| Forecast        | OOS R² | Clark–West stat |         p-value |
| --------------- | ------: | ---------------: | --------------: |
| RSZ combination |  +1.35% |             1.83 | **0.033** |
| SVR (RBF)       | −2.22% |             1.53 |           0.063 |

The Rapach–Strauss–Zhou combination significantly beats the prevailing mean;
the SVR does not (and its portfolio-level contribution is one basis point of
alpha, §5). The equal-weighted average of 13 univariate regressions beats a
tuned RBF-kernel SVR with the same predictors by Goyal & Welch (2008) reproducing.

**Deflated Sharpe.** SRML_shrunk's Sharpe of 1.066 deflates to **0.998**
against an expected-maximum-Sharpe of 0.208 from 10 trials  i.e. it is
distinguishable from the best of 10 zero-skill draws but  the caveat is  n_trials = 10 is a floor (config variants during development raise it), so the true DSR is somewhat lower but still above 0.95.

---

## 7. Robustness and validation

- **Look-ahead bias.** Signals are publication-lagged per SPEC S2.4 (verified
  across all 13 signals); betas/covariances at month *t* use only data through
  *t*; the ε-tube flat-forecast trap is guarded by y-standardization plus an
  asserted forecast-dispersion floor.
- **No intramonth rebalancing.** Weights drift within the holding period;
  turnover is charged from drifted weights to target, not target-to-target.
- **Determinism.** BLAS threads pinned to 1, `PYTHONHASHSEED=0`; environment
  fingerprint in `reports/env_fingerprint.json`.
- **Test suite.** 130+ tests including a recovered-known-betas check, a
  Schaible-vs-brute-force optimality check (200k random portfolios), and the
  DOW 28→29 universe-transition regression.

---

## 8. Deviations from the paper

1. **WBA excluded.** Walgreens Boots Alliance was delisted from Nasdaq
   2025-08-29 (Sycamore take-private); no free source serves dividend-adjusted
   history as of 2026-07. A price-only substitute would understate its total
   return by its ~3–5% yield and bias the optimizer. Universe is therefore 29
   assets (28 before 2020-03-31). See `docs/deviations.md`.
2. **Survivorship bias.** The static DJIA-2021 universe held fixed backwards
   carries survivorship bias by construction (as the paper's likely does).
3. **Factor vintage.** French factors drawn from the 202605 CRSP database
   vintage; immaterial for 2008–2021 attribution but noted for exact
   reproduction.

*This is a vintage but not a method problem and it's a replication in 2021
would have had all 30 constituents. Future reproductions should expect further attrition as members are acquired or delisted.*

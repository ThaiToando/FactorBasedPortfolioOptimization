"""Factor attribution: is the alpha real, or is it factor exposure?

The paired Sharpe tests showed the top strategies are statistically
indistinguishable from equal-weight, and correlate with it above 0.97. This
module explains why. Regress each strategy's daily excess return on the
Fama-French five factors plus momentum:

    r_strategy - rf = alpha + b_mkt (Mkt-RF) + b_smb SMB + b_hml HML
                            + b_rmw RMW + b_cma CMA + b_mom MOM + e

Two readings matter. If alpha is near zero and insignificant, the strategy's
average return is fully explained by its factor exposures -- there is no skill
beyond loading on known premia. And if the factor loadings are near-identical
across strategies, the strategies are the same bet wearing different labels,
which is what a 0.97 correlation means made concrete.

Standard errors are Newey-West (HAC): daily strategy returns are autocorrelated
and heteroskedastic, so OLS standard errors would overstate the significance of
every coefficient, including alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")


def _fetch_momentum_daily(start: str, end: str) -> pd.Series:
    """Daily momentum factor, read directly from Ken French's CSV zip.

    pandas_datareader's parser mishandles the daily momentum file's header
    text on this version, leaving a string in the index. Reading the zip
    ourselves and slicing the numeric block is immune to that. Data rows look
    like ``19361114,-0.76,`` -- note the trailing comma, which is why the
    field count is checked as >= 2 rather than == 2.
    """
    import io
    import urllib.request
    import zipfile

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_daily_CSV.zip"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    raw = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()

    rows = []
    for line in raw:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
            try:
                rows.append((parts[0], float(parts[1])))
            except ValueError:
                continue

    frame = pd.DataFrame(rows, columns=["date", "MOM"])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    series = frame.set_index("date")["MOM"] / 100.0
    return series.loc[start:end]


def load_factor_panel(cfg, *, refresh: bool = False) -> pd.DataFrame:
    """Daily FF5 + momentum factors, as decimals, on the trading calendar.

    The five-factor table comes through pandas_datareader (its format parses
    cleanly); the daily momentum factor is fetched directly because the reader
    chokes on its header. Both are divided by 100 exactly once.
    """
    from pathlib import Path

    from pandas_datareader import data as pdr

    from fbpo.config import resolve_date
    from fbpo.data import RAW_DIR

    cache = Path(RAW_DIR) / "ff5_mom_daily.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    start = cfg.data.price_start
    end = (resolve_date(cfg.data.price_end) - pd.Timedelta(days=1)).isoformat()

    ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0]
    ff5 = ff5.astype(float) / 100.0
    ff5.columns = [c.strip() for c in ff5.columns]
    ff5.index = pd.DatetimeIndex(ff5.index.to_timestamp()).normalize()

    mom = _fetch_momentum_daily(start, end)
    mom.index = pd.DatetimeIndex(mom.index).normalize()

    panel = ff5.join(mom, how="inner")

    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_index().to_parquet(cache)
    return panel.sort_index()


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 5) -> dict[str, np.ndarray]:
    """OLS with Newey-West HAC standard errors.

    Returns coefficients, HAC standard errors, t-statistics and R-squared. The
    design matrix X must already include an intercept column; alpha is its
    coefficient.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta

    # Newey-West meat matrix: S = sum of Bartlett-weighted autocovariances of X'e.
    s = (X * residuals[:, None]).T @ (X * residuals[:, None])
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = (X[lag:] * residuals[lag:, None]).T @ (X[:-lag] * residuals[:-lag, None])
        s += weight * (gamma + gamma.T)

    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))

    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": beta,
        "se": se,
        "t": np.where(se > 0, beta / se, np.nan),
        "r_squared": r_squared,
        "n": n,
    }


def attribute(strategy_excess: pd.Series, factors: pd.DataFrame, lags: int = 5) -> dict[str, float]:
    """Regress one strategy's excess return on FF5 + MOM.

    Alpha is reported annualised (multiplied by 252) so it reads in the same
    units as the CAGR figures. Its t-statistic is not annualised -- scaling a
    coefficient and its standard error by the same constant leaves t unchanged.
    """
    aligned = pd.concat(
        [strategy_excess.rename("y"), factors[list(FACTOR_COLUMNS)]], axis=1
    ).dropna()
    if len(aligned) < 60:
        return {"n": len(aligned)}

    y = aligned["y"].to_numpy()
    X = np.column_stack([np.ones(len(aligned)), aligned[list(FACTOR_COLUMNS)].to_numpy()])
    fit = newey_west_ols(y, X, lags=lags)

    names = ("alpha", *FACTOR_COLUMNS)
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        coefficient = fit["beta"][i]
        if name == "alpha":
            out["alpha_annual"] = float(coefficient * TRADING_DAYS)
            out["alpha_t"] = float(fit["t"][i])
        else:
            out[name] = float(coefficient)
            out[f"{name}_t"] = float(fit["t"][i])
    out["r_squared"] = float(fit["r_squared"])
    out["n"] = int(fit["n"])
    return out


def attribution_table(
    returns: pd.DataFrame, factors: pd.DataFrame, risk_free: pd.Series, lags: int = 5
) -> pd.DataFrame:
    """One attribution regression per strategy.

    The reader looks first at the alpha_t column: a strategy whose edge is
    real shows a positive, significant alpha (|t| > 2). A strategy whose returns
    are fully explained by factor exposures shows alpha near zero. If every
    strategy has the same loadings and no alpha, they are one bet and the near
    unit correlations between them are fully accounted for.
    """
    excess = returns.sub(risk_free.reindex(returns.index).fillna(0.0), axis=0)
    rows = {}
    for strategy in excess.columns:
        result = attribute(excess[strategy].dropna(), factors, lags=lags)
        if "alpha_annual" in result:
            rows[strategy] = result
    return pd.DataFrame(rows).T

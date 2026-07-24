"""Portfolio construction as convex programs.

The central problem: maximise the Sharpe ratio subject to long-only,
fully-invested constraints.

    max_w  (mu - rf)'w / sqrt(w' Sigma w)   s.t.  w >= 0, sum(w) = 1

As written this is a *ratio* objective and therefore nonconvex. Solved with a
general nonlinear method such as SLSQP it can converge to a local optimum, and
its answer depends on the starting point -- so "deterministic across runs" is
not the same as "correct".

The Schaible transformation removes the difficulty rather than working around
it. Substitute y = w / ((mu - rf)'w), so that the numerator is fixed at 1:

    min_y  y' Sigma y    s.t.  (mu - rf)'y = 1, y >= 0

That is a convex quadratic program with a unique global solution, and the
original weights are recovered as w = y / sum(y). The transformation is exact:
it is the same problem, not an approximation of it.

Every function here returns weights that sum to 1 within 1e-8 and are
non-negative to within solver tolerance. When no asset has positive expected
excess return the max-Sharpe problem is infeasible, and the configured
fallback (GMVP) applies -- silently returning something arbitrary in that case
would be worse than failing.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

WEIGHT_TOLERANCE = 1e-8
SOLVER_MAP = {"clarabel": cp.CLARABEL, "osqp": cp.OSQP}


class OptimizationError(RuntimeError):
    """Raised when a solver fails and no fallback applies."""


def _solve(problem: cp.Problem, solver: str) -> None:
    """Solve in place, raising with the solver's own status on failure."""
    try:
        problem.solve(solver=SOLVER_MAP.get(solver, cp.CLARABEL))
    except cp.error.SolverError as exc:
        raise OptimizationError(f"solver {solver!r} failed: {exc}") from exc
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise OptimizationError(f"solver {solver!r} returned status {problem.status!r}")


def _clean(weights: np.ndarray) -> np.ndarray:
    """Clip solver noise and renormalise so the weights are exactly a portfolio."""
    cleaned = np.clip(np.asarray(weights, dtype=float).ravel(), 0.0, None)
    total = cleaned.sum()
    if total <= 0:
        raise OptimizationError("solver returned a degenerate all-zero solution")
    return cleaned / total


def _cap_constraints(y: cp.Variable, weight_cap: float) -> list:
    """Express a per-asset cap in the transformed space.

    w_i = y_i / sum(y), so w_i <= cap becomes y_i <= cap * sum(y), which stays
    linear in y and therefore keeps the problem convex.
    """
    if weight_cap >= 1.0:
        return []
    return [y <= weight_cap * cp.sum(y)]


def max_sharpe(
    mu: np.ndarray,
    covariance: np.ndarray,
    risk_free: float = 0.0,
    weight_cap: float = 1.0,
    solver: str = "clarabel",
) -> np.ndarray:
    """Long-only maximum-Sharpe weights via the Schaible transformation.

    Raises :class:`OptimizationError` if no asset has positive expected excess
    return; the caller decides the fallback rather than this function guessing.
    """
    excess = np.asarray(mu, dtype=float).ravel() - risk_free
    if excess.max() <= 0:
        raise OptimizationError(
            f"no asset has positive expected excess return (max {excess.max():.6f}); "
            "the max-Sharpe program is infeasible"
        )

    n = len(excess)
    y = cp.Variable(n, nonneg=True)
    constraints = [excess @ y == 1.0, *_cap_constraints(y, weight_cap)]
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(covariance))), constraints)
    _solve(problem, solver)
    return _clean(y.value)


def gmvp(
    covariance: np.ndarray,
    weight_cap: float = 1.0,
    solver: str = "clarabel",
) -> np.ndarray:
    """Global minimum-variance portfolio. Uses no return estimate at all.

    That is the point of it as a benchmark: expected returns are the noisiest
    input in portfolio construction, so a portfolio that ignores them entirely
    is a demanding comparison.
    """
    n = covariance.shape[0]
    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1.0]
    if weight_cap < 1.0:
        constraints.append(w <= weight_cap)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(covariance))), constraints)
    _solve(problem, solver)
    return _clean(w.value)


def mdrp(
    covariance: np.ndarray,
    weight_cap: float = 1.0,
    solver: str = "clarabel",
) -> np.ndarray:
    """Most-diversified portfolio: maximise (w'sigma) / sqrt(w' Sigma w).

    Same fractional structure as max-Sharpe, so the same transformation
    applies with volatilities in place of expected excess returns. Choueifaty's
    interpretation: it maximises the ratio of weighted-average asset risk to
    realised portfolio risk, i.e. the diversification actually achieved.
    """
    volatility = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if volatility.max() <= 0:
        raise OptimizationError("all assets have zero variance")

    n = covariance.shape[0]
    y = cp.Variable(n, nonneg=True)
    constraints = [volatility @ y == 1.0, *_cap_constraints(y, weight_cap)]
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(covariance))), constraints)
    _solve(problem, solver)
    return _clean(y.value)


def robust_max_sharpe(
    mu: np.ndarray,
    covariance: np.ndarray,
    kappa: float,
    uncertainty: np.ndarray | None = None,
    risk_free: float = 0.0,
    weight_cap: float = 1.0,
    solver: str = "clarabel",
    n_obs: int = 252,
) -> np.ndarray:
    """Max-Sharpe against the worst case in an ellipsoidal set around mu.

    The robust objective replaces (mu - rf)'w with its worst case over
    ||S^{-1/2}(m - mu)|| <= kappa, giving (mu - rf)'w - kappa ||S^{1/2} w||.
    That term is concave in w, so requiring it to be at least 1 defines a
    convex set and the Schaible form survives.

    kappa=0 reduces exactly to :func:`max_sharpe`. Larger kappa buys robustness
    with expected performance; the sensitivity grid reports the trade.
    """
    if kappa <= 0:
        return max_sharpe(mu, covariance, risk_free, weight_cap, solver)

    excess = np.asarray(mu, dtype=float).ravel() - risk_free
    if excess.max() <= 0:
        raise OptimizationError("no asset has positive expected excess return")

    n = len(excess)
    if uncertainty is None:
        # Estimation error of the sample mean is Sigma / T, where T is the number
        # of observations -- not the number of assets. kappa is then measured in
        # standard errors, so kappa=1 is a one-sigma adverse move in mu.
        uncertainty = np.diag(np.diag(covariance) / n_obs)
    root = np.linalg.cholesky(uncertainty + np.eye(n) * 1e-12)

    y = cp.Variable(n, nonneg=True)
    constraints = [
        excess @ y - kappa * cp.norm(root.T @ y, 2) >= 1.0,
        *_cap_constraints(y, weight_cap),
    ]
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(covariance))), constraints)
    try:
        _solve(problem, solver)
    except OptimizationError as exc:
        worst_case = float((excess - kappa * np.sqrt(np.diag(uncertainty))).max())
        raise OptimizationError(
            f"robust program infeasible at kappa={kappa}: the best single-asset "
            f"worst-case excess return is {worst_case:.6f}. The uncertainty set is "
            f"wider than the expected returns -- reduce kappa or raise n_obs "
            f"(currently {n_obs})."
        ) from exc
    return _clean(y.value)


def inverse_volatility(covariance: np.ndarray) -> np.ndarray:
    """Weights proportional to 1/sigma. No optimization, no correlations used."""
    volatility = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if volatility.max() <= 0:
        raise OptimizationError("all assets have zero variance")
    inverse = np.where(volatility > 0, 1.0 / np.where(volatility > 0, volatility, 1.0), 0.0)
    return _clean(inverse)


def equal_weight(n: int) -> np.ndarray:
    """1/N. DeMiguel, Garlappi and Uppal (2009) is the reason this is here.

    Their finding -- that 1/N beats most optimised portfolios out of sample --
    is the null hypothesis any paper in this literature has to beat.
    """
    if n <= 0:
        raise OptimizationError("cannot build a portfolio with no assets")
    return np.full(n, 1.0 / n)


def optimize_weights(
    objective: str,
    covariance: np.ndarray,
    mu: np.ndarray | None = None,
    risk_free: float = 0.0,
    weight_cap: float = 1.0,
    solver: str = "clarabel",
    robust_kappa: float = 0.0,
    fallback: str = "gmvp",
    n_obs: int = 252,
) -> tuple[np.ndarray, str]:
    """Dispatch, with the configured fallback when max-Sharpe is infeasible.

    Returns ``(weights, objective_used)``. The second element is not
    decoration: a run where max-Sharpe fell back to GMVP in 30 of 144 months
    is a different strategy from one where it never did, and the backtester
    records the count.
    """
    n = covariance.shape[0]

    if objective == "equal_weight":
        return equal_weight(n), objective
    if objective == "inverse_vol":
        return inverse_volatility(covariance), objective
    if objective == "gmvp":
        return gmvp(covariance, weight_cap, solver), objective
    if objective == "mdrp":
        return mdrp(covariance, weight_cap, solver), objective

    if objective == "max_sharpe":
        if mu is None:
            raise ValueError("max_sharpe requires an expected-return vector")
        try:
            if robust_kappa > 0:
                return (
                    robust_max_sharpe(
                        mu,
                        covariance,
                        robust_kappa,
                        None,
                        risk_free,
                        weight_cap,
                        solver,
                        n_obs,
                    ),
                    objective,
                )
            return max_sharpe(mu, covariance, risk_free, weight_cap, solver), objective
        except OptimizationError:
            if fallback == "gmvp":
                return gmvp(covariance, weight_cap, solver), "gmvp_fallback"
            if fallback == "equal_weight":
                return equal_weight(n), "equal_weight_fallback"
            raise

    raise ValueError(f"unknown objective {objective!r}")


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    """w' Sigma w."""
    w = np.asarray(weights, dtype=float).ravel()
    return float(w @ covariance @ w)


def herfindahl(weights: np.ndarray) -> float:
    """Sum of squared weights. Equal weight over N gives exactly 1/N.

    SPEC S5 anchors: 1/29 = 0.0345 for equal weight; concentrated max-Sharpe
    lands near 0.12-0.32; a well-diversified factor portfolio near 0.05-0.11.
    An HHI equal to 1/N when you expected concentration means the optimizer
    received a constant mu and quietly returned 1/N.
    """
    w = np.asarray(weights, dtype=float).ravel()
    return float(np.sum(w**2))

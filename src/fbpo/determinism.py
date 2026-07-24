from __future__ import annotations

import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_TRACKED_DISTRIBUTIONS: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "cvxpy": "cvxpy",
    "clarabel": "clarabel",
    "arch": "arch",
    "yfinance": "yfinance",
}


def set_thread_limits(n_threads: int = 1) -> None:
    """Pin every BLAS/OpenMP thread-count variable.

    Uses ``setdefault`` so an explicit environment setting always wins. These
    only bind if they are read before the numerical library loads, which is why
    the shell environment sets them too.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n_threads))


def set_deterministic(seed: int = 42) -> None:
    """Pin thread counts and seed the global RNGs.

    Project code should use ``numpy.random.default_rng(seed)`` explicitly; the
    legacy global seed here exists only for third-party libraries that ignore
    the Generator API.
    """
    set_thread_limits(1)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)


def hashseed_is_pinned() -> bool:
    """True when ``PYTHONHASHSEED=0`` was present in the launching environment."""
    return os.environ.get("PYTHONHASHSEED") == "0"


def _distribution_version(dist: str) -> str | None:
    try:
        return _dist_version(dist)
    except PackageNotFoundError:
        return None


def env_fingerprint() -> dict[str, Any]:
    """Everything needed to explain a fifth-decimal disagreement between machines."""
    from threadpoolctl import threadpool_info

    blas = [
        {
            key: info.get(key)
            for key in ("user_api", "internal_api", "prefix", "version", "num_threads")
        }
        for info in threadpool_info()
    ]

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "threads": {var: os.environ.get(var) for var in THREAD_ENV_VARS},
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "packages": {
            name: _distribution_version(dist) for name, dist in _TRACKED_DISTRIBUTIONS.items()
        },
        "blas": blas,
    }


def write_env_fingerprint(path: Path = Path("reports/env_fingerprint.json")) -> Path:
    """Serialise :func:`env_fingerprint` to disk, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(env_fingerprint(), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path
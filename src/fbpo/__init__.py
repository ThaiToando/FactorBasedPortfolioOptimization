"""fbpo: factor-based portfolio optimization.

Reproduction and extension of Auh, J.K. & Cho, W. (2023), "Factor-based
portfolio optimization," Economics Letters 228, 111137.

Research and educational purposes only. Not investment advice.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("fbpo")
except PackageNotFoundError:  # pragma: no cover , only when running from a raw checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
"""WrinkleFE - Finite element analysis of wrinkled composite laminates.

Combines 3D cross-laminated plate theory with advanced composite failure
criteria for modeling fiber waviness defects in composite structures.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wrinklefe")
except PackageNotFoundError:
    # Fallback for when the package isn't installed (e.g. running from source).
    # Keep this in sync with pyproject.toml and CITATION.cff.
    __version__ = "1.0.0"

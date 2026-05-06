"""WrinkleFE - Finite element analysis of wrinkled composite laminates.

Combines 3D cross-laminated plate theory with advanced composite failure
criteria for modeling fiber waviness defects in composite structures.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wrinklefe")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
